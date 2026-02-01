"""Message type definitions for ShardCompute protocol."""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time


class MessageType(IntEnum):
    """Message types for ShardCompute protocol."""
    
    # Worker registration
    WORKER_REGISTER = 1
    WORKER_REGISTER_ACK = 2
    WORKER_DEREGISTER = 3
    
    # Cluster management
    CLUSTER_INFO = 10
    PEER_LIST = 11
    
    # Health monitoring
    HEARTBEAT = 20
    HEARTBEAT_ACK = 21
    
    # Inference
    INFERENCE_REQUEST = 30
    INFERENCE_RESPONSE = 31
    INFERENCE_ERROR = 32
    
    # Collective operations
    TENSOR_DATA = 40
    ALL_REDUCE_START = 41
    ALL_REDUCE_CHUNK = 42
    ALL_REDUCE_DONE = 43
    ALL_GATHER_START = 44
    ALL_GATHER_CHUNK = 45
    ALL_GATHER_DONE = 46
    
    # Synchronization
    BARRIER_ENTER = 50
    BARRIER_RELEASE = 51
    
    # Input broadcast
    BROADCAST_INPUT = 60
    BROADCAST_ACK = 61
    
    # Metrics
    METRICS_REPORT = 70
    METRICS_REQUEST = 71


@dataclass
class Message:
    """Base message container for ShardCompute protocol."""
    
    msg_type: MessageType
    sender_rank: int
    sequence_id: int
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "msg_type": int(self.msg_type),
            "sender_rank": self.sender_rank,
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            msg_type=MessageType(data["msg_type"]),
            sender_rank=data["sender_rank"],
            sequence_id=data["sequence_id"],
            timestamp=data.get("timestamp", time.time()),
            payload=data.get("payload", {}),
        )


@dataclass
class WorkerInfo:
    """Information about a worker node."""
    
    rank: int
    host: str
    port: int
    collective_port: int
    device_info: Dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"
    last_heartbeat: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "host": self.host,
            "port": self.port,
            "collective_port": self.collective_port,
            "device_info": self.device_info,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerInfo":
        """Create from dictionary."""
        return cls(
            rank=data["rank"],
            host=data["host"],
            port=data["port"],
            collective_port=data["collective_port"],
            device_info=data.get("device_info", {}),
            status=data.get("status", "unknown"),
            last_heartbeat=data.get("last_heartbeat", time.time()),
        )


@dataclass
class InferenceRequest:
    """Request for model inference."""

    request_id: str
    input_ids: List[int]
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stop_tokens: List[int] = field(default_factory=lambda: [2])  # Default EOS token for Llama
    stream: bool = False  # Whether to stream tokens as they're generated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "input_ids": self.input_ids,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_tokens": self.stop_tokens,
            "stream": self.stream,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceRequest":
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            input_ids=data["input_ids"],
            max_new_tokens=data.get("max_new_tokens", 100),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            stop_tokens=data.get("stop_tokens", [2]),
            stream=data.get("stream", False),
        )


@dataclass
class InferenceResponse:
    """Response from model inference."""

    request_id: str
    output_ids: List[int]
    timing: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "output_ids": self.output_ids,
            "timing": self.timing,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResponse":
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            output_ids=data["output_ids"],
            timing=data.get("timing", {}),
            error=data.get("error"),
        )


@dataclass
class StreamingToken:
    """A single token streamed during generation."""

    request_id: str
    token_id: int
    token_index: int  # Position in generated sequence
    is_final: bool = False  # True for the last token
    timing: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "token_id": self.token_id,
            "token_index": self.token_index,
            "is_final": self.is_final,
            "timing": self.timing,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingToken":
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            token_id=data["token_id"],
            token_index=data["token_index"],
            is_final=data.get("is_final", False),
            timing=data.get("timing", {}),
            error=data.get("error"),
        )


@dataclass  
class TensorMetadata:
    """Metadata for tensor transfer."""
    
    shape: List[int]
    dtype: str
    nbytes: int
    compressed: bool = False
    compression_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "compressed": self.compressed,
            "compression_ratio": self.compression_ratio,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorMetadata":
        """Create from dictionary."""
        return cls(
            shape=data["shape"],
            dtype=data["dtype"],
            nbytes=data["nbytes"],
            compressed=data.get("compressed", False),
            compression_ratio=data.get("compression_ratio", 1.0),
        )


@dataclass
class CollectiveOperation:
    """Metadata for a collective operation."""
    
    op_id: str
    op_type: str  # "all_reduce", "all_gather", "reduce_scatter"
    tensor_id: str
    reduction_op: str = "sum"  # For all-reduce: sum, mean, max, min
    gather_dim: int = -1  # For all-gather: dimension to concatenate
    world_size: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "tensor_id": self.tensor_id,
            "reduction_op": self.reduction_op,
            "gather_dim": self.gather_dim,
            "world_size": self.world_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectiveOperation":
        """Create from dictionary."""
        return cls(
            op_id=data["op_id"],
            op_type=data["op_type"],
            tensor_id=data["tensor_id"],
            reduction_op=data.get("reduction_op", "sum"),
            gather_dim=data.get("gather_dim", -1),
            world_size=data.get("world_size", 2),
        )
