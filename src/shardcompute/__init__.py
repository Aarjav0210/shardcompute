"""
ShardCompute - Distributed Tensor Parallelism for Apple Silicon

This package implements Megatron-style tensor parallelism across distributed
Apple Silicon devices using MLX for GPU acceleration.
"""

__version__ = "0.1.0"

from shardcompute.protocol.messages import MessageType
from shardcompute.collectives.communicator import Communicator

__all__ = [
    "__version__",
    "MessageType",
    "Communicator",
]
