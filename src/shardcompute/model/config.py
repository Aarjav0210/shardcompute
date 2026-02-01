"""Model and parallelism configuration."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
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

    # MLP configuration
    hidden_act: str = "silu"  # Activation function: silu, gelu, gelu_new
    use_gated_mlp: bool = True  # LLaMA uses gated (SwiGLU), Phi-2 does not

    # Architecture identifiers
    model_type: str = "llama"

    # Quantization (will be set after parsing config)
    quantization: Optional["QuantizationConfig"] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        model_type = data.get("model_type", "llama")

        # Auto-detect MLP style based on model type
        # Phi-2 uses non-gated MLP with gelu_new, LLaMA uses gated with silu
        if model_type == "phi":
            default_hidden_act = "gelu_new"
            default_use_gated = False
        else:
            default_hidden_act = "silu"
            default_use_gated = True

        # Parse quantization config if present
        quant_data = data.get("quantization")
        quantization = QuantizationConfig.from_dict(quant_data) if quant_data else None

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
            hidden_act=data.get("hidden_act", default_hidden_act),
            use_gated_mlp=data.get("use_gated_mlp", default_use_gated),
            model_type=model_type,
            quantization=quantization,
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
        result = {
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
            "hidden_act": self.hidden_act,
            "use_gated_mlp": self.use_gated_mlp,
            "model_type": self.model_type,
        }
        if self.quantization:
            result["quantization"] = self.quantization.to_dict()
        return result
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads


@dataclass
class ParallelConfig:
    """Configuration for tensor and pipeline parallelism."""

    world_size: int = 2
    rank: int = 0

    # Parallelism mode: "tensor" or "pipeline"
    mode: str = "tensor"

    # Parallelism strategy
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 2

    # For 2D parallelism (future)
    row_parallel_size: int = 1
    col_parallel_size: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any], rank: int = 0) -> "ParallelConfig":
        """Create config from dictionary."""
        mode = data.get("mode", "tensor")
        tp_size = data.get("tensor_parallel_size", 2)
        pp_size = data.get("pipeline_parallel_size", 2)

        if mode == "pipeline":
            world_size = pp_size
        else:
            world_size = tp_size

        return cls(
            world_size=world_size,
            rank=rank,
            mode=mode,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            row_parallel_size=data.get("row_parallel_size", 1),
            col_parallel_size=data.get("col_parallel_size", 1),
        )

    def validate(self, model_config: "ModelConfig"):
        """Validate parallelism config against model config."""
        if self.mode == "pipeline":
            self._validate_pipeline(model_config)
        else:
            self._validate_tensor(model_config)

    def _validate_tensor(self, model_config: "ModelConfig"):
        """Validate tensor parallelism config against model config."""
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

    def _validate_pipeline(self, model_config: "ModelConfig"):
        """Validate pipeline parallelism config against model config."""
        if model_config.num_layers % self.pipeline_parallel_size != 0:
            raise ValueError(
                f"num_layers ({model_config.num_layers}) must be divisible by "
                f"pipeline_parallel_size ({self.pipeline_parallel_size})"
            )

    def get_pipeline_stage_layers(
        self, rank: int, num_layers: int
    ) -> Tuple[int, int]:
        """
        Get the layer range assigned to a pipeline stage (rank).

        Divides layers evenly across pipeline stages.

        Args:
            rank: Pipeline stage rank
            num_layers: Total number of layers in the model

        Returns:
            (start_layer, end_layer) — exclusive end, i.e. layers [start, end)
        """
        layers_per_stage = num_layers // self.pipeline_parallel_size
        start = rank * layers_per_stage
        end = start + layers_per_stage
        return start, end

    @property
    def is_pipeline(self) -> bool:
        """Whether pipeline parallelism is active."""
        return self.mode == "pipeline"

    @property
    def is_tensor(self) -> bool:
        """Whether tensor parallelism is active."""
        return self.mode == "tensor"


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


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    enabled: bool = False
    bits: int = 4  # 4-bit or 8-bit quantization
    group_size: int = 64  # Quantization group size

    # Which layers to quantize
    quantize_attention: bool = True
    quantize_mlp: bool = True
    quantize_embeddings: bool = False  # Usually not quantized
    quantize_lm_head: bool = False  # Usually not quantized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationConfig":
        """Create config from dictionary."""
        if not data:
            return cls(enabled=False)

        return cls(
            enabled=True,  # If quantization dict exists, it's enabled
            bits=data.get("bits", 4),
            group_size=data.get("group_size", 64),
            quantize_attention=data.get("quantize_attention", True),
            quantize_mlp=data.get("quantize_mlp", True),
            quantize_embeddings=data.get("quantize_embeddings", False),
            quantize_lm_head=data.get("quantize_lm_head", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "bits": self.bits,
            "group_size": self.group_size,
            "quantize_attention": self.quantize_attention,
            "quantize_mlp": self.quantize_mlp,
            "quantize_embeddings": self.quantize_embeddings,
            "quantize_lm_head": self.quantize_lm_head,
        }
