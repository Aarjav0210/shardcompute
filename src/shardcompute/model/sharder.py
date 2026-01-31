"""Weight sharding utilities for tensor parallelism."""

import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import mlx.core as mx
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors

from shardcompute.model.config import ModelConfig, ParallelConfig, ShardingConfig

logger = logging.getLogger(__name__)


class WeightSharder:
    """
    Shards model weights for tensor parallelism.
    
    Handles different sharding strategies for different weight types:
    - Embedding: Column split (by embedding dim)
    - QKV projections: Column split (by heads)
    - Output projection: Row split (by heads)
    - MLP up/gate: Column split (by intermediate dim)
    - MLP down: Row split (by intermediate dim)
    - LayerNorm: Not sharded (replicated)
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ):
        """
        Initialize WeightSharder.
        
        Args:
            model_config: Model architecture config
            parallel_config: Parallelism config
        """
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.sharding_config = ShardingConfig(model_config, parallel_config)
        
        # Validate configs
        parallel_config.validate(model_config)
        
        self.world_size = parallel_config.tensor_parallel_size
    
    def shard_weights(
        self,
        weights: Dict[str, np.ndarray],
        rank: int,
    ) -> Dict[str, np.ndarray]:
        """
        Shard all weights for a specific rank.

        Args:
            weights: Dictionary of weight name -> numpy array
            rank: Worker rank (0 to world_size-1)

        Returns:
            Dictionary of sharded weights for this rank
        """
        sharded = {}

        # Track which quantization components we've already handled
        processed_quant_components = set()

        for name, weight in weights.items():
            # Skip quantization components (scales, biases) - they're handled with their main weight
            if name in processed_quant_components:
                continue

            # Check if this is a quantized weight (has corresponding scales/biases)
            scales_key = name.replace(".weight", ".scales") if ".weight" in name else f"{name}.scales"
            biases_key = name.replace(".weight", ".biases") if ".weight" in name else f"{name}.biases"

            if scales_key in weights and biases_key in weights:
                # This is a quantized weight - shard weight, scales, and biases together
                sharded_components = self._shard_quantized_weight(
                    name, weight, weights[scales_key], weights[biases_key], rank
                )
                if sharded_components:
                    sharded.update(sharded_components)
                    processed_quant_components.add(scales_key)
                    processed_quant_components.add(biases_key)
            elif name.endswith(".scales") or name.endswith(".biases"):
                # Skip standalone quantization components - handled above
                continue
            else:
                # Regular non-quantized weight
                sharded_weight = self._shard_weight(name, weight, rank)
                if sharded_weight is not None:
                    sharded[name] = sharded_weight

        return sharded

    def _shard_quantized_weight(
        self,
        name: str,
        weight: np.ndarray,
        scales: np.ndarray,
        biases: np.ndarray,
        rank: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Shard a quantized weight along with its scales and biases.

        For column-parallel layers (Q, K, V, up, gate projections):
            - Weight is sharded along output dimension
            - Scales and biases are sharded along the same dimension

        For row-parallel layers (O projection, down projection):
            - Weight is sharded along input dimension
            - Scales and biases may need different handling based on group alignment
        """
        result = {}

        # Determine sharding type based on weight name
        is_column_parallel = any(x in name for x in [
            "q_proj", "k_proj", "v_proj", "query", "key", "value",
            "up_proj", "gate_proj", "fc1", "w1", "w3",
            "embed_tokens", "wte", "lm_head"
        ])
        is_row_parallel = any(x in name for x in [
            "o_proj", "out_proj", "down_proj", "fc2", "w2"
        ]) and "mlp" not in name.lower() if "o_proj" in name or "out_proj" in name else any(x in name for x in [
            "down_proj", "fc2", "w2"
        ])

        # For quantized weights, we don't transpose - they're already in the right format
        if is_column_parallel:
            # Shard along output dimension (dim 0 for quantized weights which are [out, in])
            sharded_weight = self._column_shard(weight, rank, dim=0)
            sharded_scales = self._column_shard(scales, rank, dim=0) if len(scales.shape) > 1 else scales.copy()
            sharded_biases = self._column_shard(biases, rank, dim=0) if len(biases.shape) > 1 else biases.copy()
        elif is_row_parallel:
            # Shard along input dimension (dim 1 for quantized weights)
            sharded_weight = self._row_shard(weight, rank, dim=1)
            # Shard scales/biases along the same input-group dimension when possible
            if len(scales.shape) > 1 and scales.shape[1] % self.world_size == 0:
                sharded_scales = self._row_shard(scales, rank, dim=1)
            else:
                if len(scales.shape) > 1:
                    logger.warning(
                        f"Row-parallel quant scales not divisible by world size "
                        f"({scales.shape[1]} vs {self.world_size}); replicating"
                    )
                sharded_scales = scales.copy()

            if len(biases.shape) > 1 and biases.shape[1] % self.world_size == 0:
                sharded_biases = self._row_shard(biases, rank, dim=1)
            else:
                if len(biases.shape) > 1:
                    logger.warning(
                        f"Row-parallel quant biases not divisible by world size "
                        f"({biases.shape[1]} vs {self.world_size}); replicating"
                    )
                sharded_biases = biases.copy()
        else:
            # Unknown - replicate everything
            logger.warning(f"Unknown quantized weight {name}, replicating")
            sharded_weight = weight.copy()
            sharded_scales = scales.copy()
            sharded_biases = biases.copy()

        # Build result with proper keys
        weight_key = name if ".weight" in name else f"{name}.weight"
        scales_key = name.replace(".weight", ".scales") if ".weight" in name else f"{name}.scales"
        biases_key = name.replace(".weight", ".biases") if ".weight" in name else f"{name}.biases"

        result[weight_key] = sharded_weight
        result[scales_key] = sharded_scales
        result[biases_key] = sharded_biases

        logger.debug(f"Sharded quantized weight {name}: {weight.shape} -> {sharded_weight.shape}")

        return result
    
    def _shard_weight(
        self,
        name: str,
        weight: np.ndarray,
        rank: int,
    ) -> Optional[np.ndarray]:
        """
        Shard a single weight tensor.
        
        HuggingFace stores linear weights as [out_features, in_features].
        MLX expects [in_features, out_features].
        We transpose linear weights before sharding to match MLX format.
        """
        # Handle 1D tensors (biases, some layer norms) - always replicate
        if len(weight.shape) == 1:
            logger.debug(f"Replicating 1D tensor: {name} with shape {weight.shape}")
            return weight.copy()
        
        # Handle bias terms explicitly (extra safety check)
        if "bias" in name.lower():
            logger.debug(f"Replicating bias: {name} with shape {weight.shape}")
            return weight.copy()
        
        # Determine sharding strategy based on weight name
        
        # Embeddings - no transpose needed, column shard on embedding dim
        # HuggingFace: [vocab_size, hidden_size], keep as is
        if "embed_tokens" in name or "wte" in name:
            return self._column_shard(weight, rank, dim=1)
        
        # LM head - needs transpose then column shard
        # HuggingFace: [vocab_size, hidden_size] -> transpose to [hidden_size, vocab_size]
        if "lm_head" in name:
            weight = weight.T  # Transpose to MLX format
            return self._column_shard(weight, rank, dim=1)
        
        # Attention weights - need transpose then appropriate sharding
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
            # Q, K, V projections - transpose then column shard
            # HuggingFace: [num_heads * head_dim, hidden_size]
            # MLX format: [hidden_size, num_heads * head_dim]
            weight = weight.T  # Transpose to MLX format
            return self._column_shard(weight, rank, dim=1)
        
        if any(x in name for x in ["o_proj", "out_proj", "dense"]):
            if "mlp" not in name.lower():
                # Attention output projection - transpose then row shard
                # HuggingFace: [hidden_size, num_heads * head_dim]
                # MLX format: [num_heads * head_dim, hidden_size]
                weight = weight.T  # Transpose to MLX format
                return self._row_shard(weight, rank, dim=0)
        
        # MLP weights - need transpose then appropriate sharding
        if any(x in name for x in ["up_proj", "gate_proj", "fc1", "w1", "w3"]):
            # Up and gate projections - transpose then column shard
            # HuggingFace: [intermediate_size, hidden_size]
            # MLX format: [hidden_size, intermediate_size]
            weight = weight.T  # Transpose to MLX format
            return self._column_shard(weight, rank, dim=1)
        
        if any(x in name for x in ["down_proj", "fc2", "w2"]):
            # Down projection - transpose then row shard
            # HuggingFace: [hidden_size, intermediate_size]
            # MLX format: [intermediate_size, hidden_size]
            weight = weight.T  # Transpose to MLX format
            return self._row_shard(weight, rank, dim=0)
        
        # LayerNorm / RMSNorm - replicate (no sharding, no transpose - 1D)
        if any(x in name for x in ["layernorm", "layer_norm", "norm", "ln"]):
            return weight.copy()
        
        # Default: replicate (unknown weights, no transpose)
        logger.warning(f"Unknown weight {name}, replicating")
        return weight.copy()
    
    def _column_shard(
        self,
        weight: np.ndarray,
        rank: int,
        dim: int = 1,
    ) -> np.ndarray:
        """
        Shard weight by columns (output dimension).
        
        Weight shape: [in_features, out_features]
        Sharded shape: [in_features, out_features // world_size]
        """
        # Defensive check: replicate 1D tensors instead of sharding
        if len(weight.shape) == 1:
            logger.warning(f"Attempted to column-shard 1D tensor (shape {weight.shape}), replicating instead")
            return weight.copy()
        
        size = weight.shape[dim]
        shard_size = size // self.world_size
        start = rank * shard_size
        end = start + shard_size
        
        if dim == 0:
            return weight[start:end, ...].copy()
        elif dim == 1:
            return weight[:, start:end, ...].copy()
        else:
            # General case using take
            indices = np.arange(start, end)
            return np.take(weight, indices, axis=dim)
    
    def _row_shard(
        self,
        weight: np.ndarray,
        rank: int,
        dim: int = 0,
    ) -> np.ndarray:
        """
        Shard weight by rows (input dimension).
        
        Weight shape: [in_features, out_features]
        Sharded shape: [in_features // world_size, out_features]
        """
        # Defensive check: replicate 1D tensors instead of sharding
        if len(weight.shape) == 1:
            logger.warning(f"Attempted to row-shard 1D tensor (shape {weight.shape}), replicating instead")
            return weight.copy()
        
        size = weight.shape[dim]
        shard_size = size // self.world_size
        start = rank * shard_size
        end = start + shard_size
        
        if dim == 0:
            return weight[start:end, ...].copy()
        elif dim == 1:
            return weight[:, start:end, ...].copy()
        else:
            indices = np.arange(start, end)
            return np.take(weight, indices, axis=dim)
    
    def save_shards(
        self,
        weights: Dict[str, np.ndarray],
        output_dir: str,
    ):
        """
        Shard weights and save to disk for all ranks.
        
        Creates output_dir/rank_0/, output_dir/rank_1/, etc.
        Each contains sharded weights in safetensors format.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_data = {
            "model": self.model_config.to_dict(),
            "parallel": {
                "world_size": self.world_size,
                "tensor_parallel_size": self.parallel_config.tensor_parallel_size,
            },
        }
        
        with open(output_path / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        # Shard and save for each rank
        for rank in range(self.world_size):
            rank_dir = output_path / f"rank_{rank}"
            rank_dir.mkdir(exist_ok=True)
            
            sharded = self.shard_weights(weights, rank)
            
            # Save using safetensors
            save_safetensors(sharded, str(rank_dir / "model.safetensors"))
            
            # Save metadata
            metadata = {
                "rank": rank,
                "world_size": self.world_size,
                "num_tensors": len(sharded),
                "tensor_shapes": {k: list(v.shape) for k, v in sharded.items()},
            }
            with open(rank_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved shards for rank {rank} to {rank_dir}")
    
    @staticmethod
    def load_shards(
        shard_dir: str,
        rank: int,
    ) -> Tuple[Dict[str, mx.array], Dict]:
        """
        Load sharded weights for a specific rank.
        
        Args:
            shard_dir: Directory containing sharded weights
            rank: Worker rank
            
        Returns:
            Tuple of (weights dict, metadata dict)
        """
        shard_path = Path(shard_dir)
        rank_dir = shard_path / f"rank_{rank}"
        
        if not rank_dir.exists():
            raise FileNotFoundError(f"Shard directory not found: {rank_dir}")
        
        # Load metadata
        with open(rank_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        # Load weights
        weights = {}
        model_file = rank_dir / "model.safetensors"
        
        with safe_open(str(model_file), framework="numpy") as f:
            for key in f.keys():
                weights[key] = mx.array(f.get_tensor(key))
        
        logger.info(f"Loaded {len(weights)} tensors for rank {rank}")
        
        return weights, metadata


def _convert_mlx_to_numpy(tensor: mx.array) -> np.ndarray:
    """Convert MLX array to NumPy, handling bfloat16 conversion."""
    # NumPy doesn't support bfloat16, convert to float16
    if tensor.dtype == mx.bfloat16:
        tensor = tensor.astype(mx.float16)
    return np.array(tensor)


def load_huggingface_weights(model_path: str) -> Dict[str, np.ndarray]:
    """
    Load weights from a HuggingFace model directory.
    
    Supports both safetensors and pytorch formats.
    Handles bfloat16 by converting to float16.
    """
    model_path = Path(model_path)
    weights = {}
    
    # Try safetensors first - use MLX to handle bfloat16
    safetensors_files = list(model_path.glob("*.safetensors"))
    if safetensors_files:
        for sf_file in safetensors_files:
            # Use MLX to load safetensors (handles bfloat16 natively)
            mlx_weights = mx.load(str(sf_file))
            for key, tensor in mlx_weights.items():
                weights[key] = _convert_mlx_to_numpy(tensor)
        logger.info(f"Loaded {len(weights)} tensors from safetensors (via MLX)")
        return weights
    
    # Fall back to pytorch
    try:
        import torch
        pytorch_files = list(model_path.glob("pytorch_model*.bin")) + list(model_path.glob("model*.bin"))
        if pytorch_files:
            for pt_file in pytorch_files:
                state_dict = torch.load(str(pt_file), map_location="cpu", weights_only=True)
                for key, value in state_dict.items():
                    # Handle bfloat16 in PyTorch
                    if value.dtype == torch.bfloat16:
                        value = value.to(torch.float16)
                    weights[key] = value.numpy()
            logger.info(f"Loaded {len(weights)} tensors from pytorch")
            return weights
    except ImportError:
        logger.warning("PyTorch not available for loading .bin files")
    
    raise FileNotFoundError(f"No model weights found in {model_path}")
