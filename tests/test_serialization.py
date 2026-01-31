"""Tests for tensor serialization."""

import pytest
import numpy as np
import mlx.core as mx

from shardcompute.protocol.serialization import TensorSerializer, StreamingTensorSerializer
from shardcompute.protocol.messages import Message, MessageType, TensorMetadata


class TestTensorSerializer:
    """Tests for TensorSerializer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.serializer = TensorSerializer(compression_enabled=True)
        self.serializer_no_compress = TensorSerializer(compression_enabled=False)
    
    def test_serialize_float32(self):
        """Test serialization of float32 tensor."""
        tensor = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        
        data = self.serializer.serialize(tensor)
        recovered, metadata = self.serializer.deserialize(data)
        
        assert recovered.shape == tensor.shape
        assert recovered.dtype == tensor.dtype
        assert mx.allclose(recovered, tensor).item()
    
    def test_serialize_int32(self):
        """Test serialization of int32 tensor."""
        tensor = mx.array([1, 2, 3, 4], dtype=mx.int32)
        
        data = self.serializer.serialize(tensor)
        recovered, metadata = self.serializer.deserialize(data)
        
        assert recovered.shape == tensor.shape
        assert mx.array_equal(recovered, tensor).item()
    
    def test_serialize_2d_tensor(self):
        """Test serialization of 2D tensor."""
        tensor = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        
        data = self.serializer.serialize(tensor)
        recovered, metadata = self.serializer.deserialize(data)
        
        assert recovered.shape == tensor.shape
        assert mx.allclose(recovered, tensor).item()
    
    def test_serialize_3d_tensor(self):
        """Test serialization of 3D tensor."""
        tensor = mx.random.normal((2, 3, 4))
        
        data = self.serializer.serialize(tensor)
        recovered, metadata = self.serializer.deserialize(data)
        
        assert recovered.shape == tensor.shape
        assert mx.allclose(recovered, tensor, atol=1e-5).item()
    
    def test_serialize_large_tensor(self):
        """Test serialization of large tensor (triggers compression)."""
        tensor = mx.random.normal((1000, 1000))
        
        data = self.serializer.serialize(tensor)
        recovered, metadata = self.serializer.deserialize(data)
        
        assert recovered.shape == tensor.shape
        assert mx.allclose(recovered, tensor, atol=1e-5).item()
    
    def test_no_compression(self):
        """Test serialization without compression."""
        tensor = mx.random.normal((100, 100))
        
        data = self.serializer_no_compress.serialize(tensor)
        recovered, metadata = self.serializer_no_compress.deserialize(data)
        
        assert not metadata.compressed
        assert mx.allclose(recovered, tensor, atol=1e-5).item()
    
    def test_metadata_extraction(self):
        """Test metadata extraction from header."""
        tensor = mx.array([1.0, 2.0, 3.0])
        
        data = self.serializer.serialize(tensor)
        metadata = TensorSerializer.get_metadata_from_header(data)
        
        assert metadata is not None
        assert metadata.shape == [3]
        assert metadata.dtype == "float32"


class TestMessage:
    """Tests for Message serialization."""
    
    def setup_method(self):
        self.serializer = TensorSerializer()
    
    def test_message_roundtrip(self):
        """Test message serialization roundtrip."""
        msg = Message(
            msg_type=MessageType.TENSOR_DATA,
            sender_rank=0,
            sequence_id=123,
            payload={"key": "value", "number": 42},
        )
        
        data = self.serializer.serialize_message(msg)
        recovered = self.serializer.deserialize_message(data)
        
        assert recovered.msg_type == msg.msg_type
        assert recovered.sender_rank == msg.sender_rank
        assert recovered.sequence_id == msg.sequence_id
        assert recovered.payload == msg.payload
    
    def test_message_types(self):
        """Test different message types."""
        for msg_type in MessageType:
            msg = Message(
                msg_type=msg_type,
                sender_rank=1,
                sequence_id=0,
            )
            
            data = self.serializer.serialize_message(msg)
            recovered = self.serializer.deserialize_message(data)
            
            assert recovered.msg_type == msg_type


class TestStreamingSerializer:
    """Tests for StreamingTensorSerializer."""
    
    def test_streaming_roundtrip(self):
        """Test streaming serialization roundtrip."""
        serializer = StreamingTensorSerializer(chunk_size=1024)
        tensor = mx.random.normal((100, 100))
        
        # Collect chunks
        header_data = None
        chunks = []
        
        for chunk_idx, total_chunks, chunk_data in serializer.iter_serialize(tensor):
            if chunk_idx == -1:
                header_data = chunk_data
            else:
                chunks.append(chunk_data)
        
        # Reassemble
        recovered = serializer.reassemble(header_data, chunks)
        
        assert recovered.shape == tensor.shape
        assert mx.allclose(recovered, tensor, atol=1e-5).item()
