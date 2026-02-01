"""Protocol layer for tensor serialization and message types."""

from shardcompute.protocol.messages import MessageType, Message

__all__ = ["MessageType", "Message", "TensorSerializer"]


def __getattr__(name: str):
    if name == "TensorSerializer":
        from shardcompute.protocol.serialization import TensorSerializer
        return TensorSerializer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
