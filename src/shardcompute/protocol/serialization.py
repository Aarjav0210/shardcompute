"""Tensor serialization utilities for network transfer."""

import struct
import zlib
from typing import Tuple, Optional, Dict
import numpy as np
import mlx.core as mx
import msgpack

from shardcompute.protocol.messages import TensorMetadata, Message


# MLX dtype to numpy dtype mapping
MLX_TO_NUMPY_DTYPE = {
    mx.float32: np.float32,
    mx.float16: np.float16,
    mx.bfloat16: np.float32,  # numpy doesn't support bfloat16, use float32
    mx.int32: np.int32,
    mx.int64: np.int64,
    mx.int16: np.int16,
    mx.int8: np.int8,
    mx.uint32: np.uint32,
    mx.uint64: np.uint64,
    mx.uint16: np.uint16,
    mx.uint8: np.uint8,
    mx.bool_: np.bool_,
}

# String to MLX dtype mapping
STR_TO_MLX_DTYPE = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "int32": mx.int32,
    "int64": mx.int64,
    "int16": mx.int16,
    "int8": mx.int8,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
    "uint16": mx.uint16,
    "uint8": mx.uint8,
    "bool": mx.bool_,
}

# MLX dtype to string mapping
MLX_DTYPE_TO_STR = {v: k for k, v in STR_TO_MLX_DTYPE.items()}


class TensorSerializer:
    """
    Serializer for MLX tensors with optional compression.

    Protocol (standard):
    - 4 bytes: metadata length (uint32, big-endian)
    - N bytes: msgpack-encoded metadata
    - M bytes: tensor data (raw or compressed)

    Protocol (fast path - when shape is pinned):
    - 1 byte: 0xFF marker (indicates fast path)
    - 1 byte: slot ID (which registered shape to use)
    - M bytes: raw tensor data (no metadata)
    """

    HEADER_SIZE = 4  # uint32 for metadata length
    FAST_PATH_MARKER = 0xFF  # Indicates fast-path message
    COMPRESSION_THRESHOLD = 1024 * 1024  # 1MB default threshold

    def __init__(
        self,
        compression_enabled: bool = False,
        compression_threshold: int = COMPRESSION_THRESHOLD,
        compression_level: int = 1,
    ):
        self.compression_enabled = compression_enabled
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level

        # Fast-path shape registry: slot_id -> (shape, dtype, nbytes)
        self._pinned_shapes: Dict[int, Tuple[Tuple[int, ...], str, int]] = {}
        self._next_slot_id = 0

    def pin_shape(self, shape: Tuple[int, ...], dtype: str) -> int:
        """
        Register a tensor shape for fast-path serialization.

        Once pinned, tensors of this shape can use serialize_fast() which
        skips metadata encoding entirely - just sends raw bytes with a 2-byte header.

        Args:
            shape: Tensor shape tuple
            dtype: Dtype string ('float32', 'float16', etc.)

        Returns:
            slot_id: ID to use with serialize_fast/deserialize_fast
        """
        slot_id = self._next_slot_id
        self._next_slot_id += 1

        # Calculate expected nbytes
        np_dtype = self._get_numpy_dtype(dtype)
        nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize

        self._pinned_shapes[slot_id] = (shape, dtype, nbytes)
        return slot_id

    def serialize_fast(self, tensor: mx.array, slot_id: int) -> bytes:
        """
        Fast-path serialization using a pinned shape.

        Skips msgpack metadata - just sends marker + slot_id + raw bytes.
        ~50x less overhead than standard serialize() for small tensors.

        Args:
            tensor: Tensor to serialize (must match pinned shape)
            slot_id: Slot ID from pin_shape()

        Returns:
            Serialized bytes (2-byte header + raw data)
        """
        if slot_id not in self._pinned_shapes:
            raise ValueError(f"Unknown slot_id: {slot_id}")

        np_array = self._mlx_to_numpy(tensor)
        tensor_bytes = np_array.tobytes()

        # 2-byte header: marker + slot_id
        header = bytes([self.FAST_PATH_MARKER, slot_id & 0xFF])
        return header + tensor_bytes

    def deserialize_fast(self, data: bytes) -> mx.array:
        """
        Fast-path deserialization using pinned shape.

        Expects data from serialize_fast().

        Args:
            data: Serialized bytes with 2-byte header

        Returns:
            Deserialized tensor
        """
        if len(data) < 2 or data[0] != self.FAST_PATH_MARKER:
            raise ValueError("Not a fast-path message")

        slot_id = data[1]
        if slot_id not in self._pinned_shapes:
            raise ValueError(f"Unknown slot_id: {slot_id}")

        shape, dtype_str, expected_nbytes = self._pinned_shapes[slot_id]
        tensor_bytes = data[2:]

        if len(tensor_bytes) != expected_nbytes:
            raise ValueError(f"Expected {expected_nbytes} bytes, got {len(tensor_bytes)}")

        np_dtype = self._get_numpy_dtype(dtype_str)
        np_array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)

        mlx_dtype = STR_TO_MLX_DTYPE.get(dtype_str, mx.float32)
        return mx.array(np_array, dtype=mlx_dtype)

    def is_fast_path(self, data: bytes) -> bool:
        """Check if data uses fast-path encoding."""
        return len(data) >= 2 and data[0] == self.FAST_PATH_MARKER

    def clear_pinned_shapes(self):
        """Clear all pinned shapes (call at end of generation)."""
        self._pinned_shapes.clear()
        self._next_slot_id = 0
    
    def serialize(self, tensor: mx.array) -> bytes:
        """
        Serialize an MLX tensor to bytes.
        
        Args:
            tensor: MLX array to serialize
            
        Returns:
            Serialized bytes including metadata and tensor data
        """
        # Convert to numpy for serialization
        # Note: mx.eval() is NOT called here -- callers (collectives) already
        # evaluate before entering the serialization path.
        np_array = self._mlx_to_numpy(tensor)
        tensor_bytes = np_array.tobytes()
        
        # Optionally compress
        compressed = False
        compression_ratio = 1.0
        
        if self.compression_enabled and len(tensor_bytes) > self.compression_threshold:
            compressed_bytes = zlib.compress(tensor_bytes, self.compression_level)
            if len(compressed_bytes) < len(tensor_bytes) * 0.9:  # Only use if >10% savings
                compression_ratio = len(tensor_bytes) / len(compressed_bytes)
                tensor_bytes = compressed_bytes
                compressed = True
        
        # Create metadata
        dtype_str = MLX_DTYPE_TO_STR.get(tensor.dtype, "float32")
        metadata = TensorMetadata(
            shape=list(tensor.shape),
            dtype=dtype_str,
            nbytes=len(tensor_bytes),
            compressed=compressed,
            compression_ratio=compression_ratio,
        )
        
        # Serialize metadata with msgpack
        metadata_bytes = msgpack.packb(metadata.to_dict())
        metadata_len = struct.pack(">I", len(metadata_bytes))
        
        return metadata_len + metadata_bytes + tensor_bytes
    
    def deserialize(self, data: bytes) -> Tuple[mx.array, TensorMetadata]:
        """
        Deserialize bytes to an MLX tensor.
        
        Args:
            data: Serialized tensor bytes
            
        Returns:
            Tuple of (MLX array, metadata)
        """
        # Parse header
        metadata_len = struct.unpack(">I", data[:self.HEADER_SIZE])[0]
        
        # Parse metadata
        metadata_bytes = data[self.HEADER_SIZE:self.HEADER_SIZE + metadata_len]
        metadata_dict = msgpack.unpackb(metadata_bytes)
        metadata = TensorMetadata.from_dict(metadata_dict)
        
        # Extract tensor data
        tensor_bytes = data[self.HEADER_SIZE + metadata_len:]
        
        # Decompress if needed
        if metadata.compressed:
            tensor_bytes = zlib.decompress(tensor_bytes)
        
        # Convert to numpy then MLX
        np_dtype = self._get_numpy_dtype(metadata.dtype)
        np_array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(metadata.shape)
        
        # Convert to MLX
        mlx_dtype = STR_TO_MLX_DTYPE.get(metadata.dtype, mx.float32)
        tensor = mx.array(np_array, dtype=mlx_dtype)
        
        return tensor, metadata
    
    def serialize_message(self, message: Message) -> bytes:
        """Serialize a protocol message."""
        return msgpack.packb(message.to_dict())
    
    def deserialize_message(self, data: bytes) -> Message:
        """Deserialize a protocol message."""
        msg_dict = msgpack.unpackb(data)
        return Message.from_dict(msg_dict)
    
    def _mlx_to_numpy(self, tensor: mx.array) -> np.ndarray:
        """Convert MLX array to numpy array."""
        # Handle bfloat16 specially - convert to float32
        if tensor.dtype == mx.bfloat16:
            tensor = tensor.astype(mx.float32)

        # Use np.asarray for potential zero-copy when memory layout permits
        np_dtype = MLX_TO_NUMPY_DTYPE.get(tensor.dtype, np.float32)
        return np.asarray(tensor, dtype=np_dtype)
    
    def _get_numpy_dtype(self, dtype_str: str) -> np.dtype:
        """Get numpy dtype from string."""
        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": np.float32,  # Will be converted to bfloat16 after
            "int32": np.int32,
            "int64": np.int64,
            "int16": np.int16,
            "int8": np.int8,
            "uint32": np.uint32,
            "uint64": np.uint64,
            "uint16": np.uint16,
            "uint8": np.uint8,
            "bool": np.bool_,
        }
        return dtype_map.get(dtype_str, np.float32)
    
    @staticmethod
    def get_metadata_from_header(data: bytes) -> Optional[TensorMetadata]:
        """
        Extract metadata from serialized tensor without full deserialization.
        
        Useful for peeking at tensor info before receiving full data.
        """
        if len(data) < TensorSerializer.HEADER_SIZE:
            return None
        
        metadata_len = struct.unpack(">I", data[:TensorSerializer.HEADER_SIZE])[0]
        
        if len(data) < TensorSerializer.HEADER_SIZE + metadata_len:
            return None
        
        metadata_bytes = data[TensorSerializer.HEADER_SIZE:TensorSerializer.HEADER_SIZE + metadata_len]
        metadata_dict = msgpack.unpackb(metadata_bytes)
        return TensorMetadata.from_dict(metadata_dict)


class StreamingTensorSerializer:
    """
    Streaming serializer for large tensors.
    
    Splits large tensors into chunks for streaming transfer,
    reducing memory pressure during network transfer.
    """
    
    DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.base_serializer = TensorSerializer(compression_enabled=False)
    
    def iter_serialize(self, tensor: mx.array):
        """
        Yield serialized chunks of a tensor.
        
        Yields:
            Tuples of (chunk_index, total_chunks, chunk_bytes)
        """
        mx.eval(tensor)
        np_array = np.array(tensor)
        tensor_bytes = np_array.tobytes()
        
        # Create metadata for full tensor
        dtype_str = MLX_DTYPE_TO_STR.get(tensor.dtype, "float32")
        metadata = TensorMetadata(
            shape=list(tensor.shape),
            dtype=dtype_str,
            nbytes=len(tensor_bytes),
            compressed=False,
            compression_ratio=1.0,
        )
        
        # Yield metadata first
        metadata_bytes = msgpack.packb(metadata.to_dict())
        total_chunks = (len(tensor_bytes) + self.chunk_size - 1) // self.chunk_size
        
        # Header chunk
        header = {
            "metadata": metadata.to_dict(),
            "total_chunks": total_chunks,
            "chunk_size": self.chunk_size,
        }
        yield (-1, total_chunks, msgpack.packb(header))
        
        # Data chunks
        for i in range(total_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, len(tensor_bytes))
            chunk_data = tensor_bytes[start:end]
            yield (i, total_chunks, chunk_data)
    
    def reassemble(self, header_data: bytes, chunks: list) -> mx.array:
        """
        Reassemble tensor from streamed chunks.
        
        Args:
            header_data: The header chunk bytes
            chunks: List of data chunk bytes in order
            
        Returns:
            Reassembled MLX array
        """
        header = msgpack.unpackb(header_data)
        metadata = TensorMetadata.from_dict(header["metadata"])
        
        # Concatenate all chunks
        tensor_bytes = b"".join(chunks)
        
        # Convert to array
        np_dtype = np.dtype(metadata.dtype.replace("bfloat16", "float32"))
        np_array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(metadata.shape)
        
        mlx_dtype = STR_TO_MLX_DTYPE.get(metadata.dtype, mx.float32)
        return mx.array(np_array, dtype=mlx_dtype)
