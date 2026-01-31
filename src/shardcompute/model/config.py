"""Model and parallelism configuration."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Configuration for transformer model architecture."""
    
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 22
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # For grouped-query attention
    intermediate_size: int = 5632
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    
    # Architecture identifiers
    model_type: str = "llama"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(
            vocab_size=data.get("vocab_size", 32000),
            hidden_size=data.get("hidden_size", 2048),
            num_layers=data.get("num_hidden_layers", data.get("num_layers", 22)),
            num_heads=data.get("num_attention_heads", data.get("num_heads", 32)),
            num_kv_heads=data.get("num_key_value_heads"),
            intermediate_size=data.get("intermediate_size", 5632),
            max_position_embeddings=data.get("max_position_embeddings", 2048),
            rms_norm_eps=data.get("rms_norm_eps", 1e-5),
            rope_theta=data.get("rope_theta", 10000.0),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            model_type=data.get("model_type", "llama"),
        )
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        """Load config from pretrained model directory."""
        config_path = Path(model_path) / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        
        with open(config_path) as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "tie_word_embeddings": self.tie_word_embeddings,
            "model_type": self.model_type,
        }
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads


@dataclass
class ParallelConfig:
    """Configuration for tensor parallelism."""
    
    world_size: int = 2
    rank: int = 0
    
    # Parallelism strategy
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1  # Not used in POC
    
    # For 2D parallelism (future)
    row_parallel_size: int = 1
    col_parallel_size: int = 1
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], rank: int = 0) -> "ParallelConfig":
        """Create config from dictionary."""
        tp_size = data.get("tensor_parallel_size", 2)
        return cls(
            world_size=tp_size,
            rank=rank,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=data.get("pipeline_parallel_size", 1),
            row_parallel_size=data.get("row_parallel_size", 1),
            col_parallel_size=data.get("col_parallel_size", 1),
        )
    
    def validate(self, model_config: ModelConfig):
        """Validate parallelism config against model config."""
        # Check head divisibility
        if model_config.num_heads % self.tensor_parallel_size != 0:
            raise ValueError(
                f"num_heads ({model_config.num_heads}) must be divisible by "
                f"tensor_parallel_size ({self.tensor_parallel_size})"
            )
        
        # Check hidden size divisibility
        if model_config.hidden_size % self.tensor_parallel_size != 0:
            raise ValueError(
                f"hidden_size ({model_config.hidden_size}) must be divisible by "
                f"tensor_parallel_size ({self.tensor_parallel_size})"
            )
        
        # Check intermediate size divisibility
        if model_config.intermediate_size % self.tensor_parallel_size != 0:
            raise ValueError(
                f"intermediate_size ({model_config.intermediate_size}) must be divisible by "
                f"tensor_parallel_size ({self.tensor_parallel_size})"
            )
        
        # Check KV heads for GQA
        if model_config.num_kv_heads:
            if model_config.num_kv_heads % self.tensor_parallel_size != 0:
                raise ValueError(
                    f"num_kv_heads ({model_config.num_kv_heads}) must be divisible by "
                    f"tensor_parallel_size ({self.tensor_parallel_size})"
                )


@dataclass
class ShardingConfig:
    """Configuration for weight sharding."""
    
    model_config: ModelConfig
    parallel_config: ParallelConfig
    
    @property
    def local_num_heads(self) -> int:
        """Number of attention heads per worker."""
        return self.model_config.num_heads // self.parallel_config.tensor_parallel_size
    
    @property
    def local_num_kv_heads(self) -> int:
        """Number of KV heads per worker."""
        kv_heads = self.model_config.num_kv_heads or self.model_config.num_heads
        return kv_heads // self.parallel_config.tensor_parallel_size
    
    @property
    def local_hidden_size(self) -> int:
        """Hidden size per worker (for column parallel)."""
        return self.model_config.hidden_size // self.parallel_config.tensor_parallel_size
    
    @property
    def local_intermediate_size(self) -> int:
        """MLP intermediate size per worker."""
        return self.model_config.intermediate_size // self.parallel_config.tensor_parallel_size
    
    @property
    def local_head_dim(self) -> int:
        """Dimension per head (same as global)."""
        return self.model_config.head_dim
    
    def get_qkv_shard_size(self) -> int:
        """Size of Q, K, V projection shard."""
        return self.local_num_heads * self.local_head_dim
    
    def get_kv_shard_size(self) -> int:
        """Size of K, V projection shard (for GQA)."""
        return self.local_num_kv_heads * self.local_head_dim
