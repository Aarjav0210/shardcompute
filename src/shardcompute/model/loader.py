"""Model loading utilities for tensor parallel inference."""

import logging
from typing import Dict, Optional
from pathlib import Path
import json
import mlx.core as mx
from safetensors import safe_open

from shardcompute.model.config import ModelConfig
from shardcompute.parallel.transformer import ParallelTransformer

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads sharded model weights into parallel transformer.
    
    Handles mapping from HuggingFace weight names to our parallel layers.
    """
    
    # Weight name mappings from HuggingFace to our format
    WEIGHT_MAP = {
        # Embeddings
        "model.embed_tokens.weight": "embed_tokens",
        "lm_head.weight": "lm_head",
        
        # Final norm
        "model.norm.weight": "norm",
        
        # Layer patterns (use {layer} as placeholder)
        "model.layers.{layer}.input_layernorm.weight": "layers.{layer}.input_layernorm",
        "model.layers.{layer}.post_attention_layernorm.weight": "layers.{layer}.post_attention_layernorm",
        
        # Attention
        "model.layers.{layer}.self_attn.q_proj.weight": "layers.{layer}.attention.q_proj",
        "model.layers.{layer}.self_attn.k_proj.weight": "layers.{layer}.attention.k_proj",
        "model.layers.{layer}.self_attn.v_proj.weight": "layers.{layer}.attention.v_proj",
        "model.layers.{layer}.self_attn.o_proj.weight": "layers.{layer}.attention.o_proj",
        
        # MLP (LLaMA style with gate)
        "model.layers.{layer}.mlp.gate_proj.weight": "layers.{layer}.mlp.gate_proj",
        "model.layers.{layer}.mlp.up_proj.weight": "layers.{layer}.mlp.up_proj",
        "model.layers.{layer}.mlp.down_proj.weight": "layers.{layer}.mlp.down_proj",
    }
    
    def __init__(self, rank: int, world_size: int):
        """
        Initialize ModelLoader.
        
        Args:
            rank: Worker rank
            world_size: Total number of workers
        """
        self.rank = rank
        self.world_size = world_size
    
    async def load_shards(
        self,
        model: ParallelTransformer,
        shard_dir: Path,
    ):
        """
        Load sharded weights into parallel model.
        
        Args:
            model: ParallelTransformer instance
            shard_dir: Directory containing pre-sharded weights
        """
        rank_dir = shard_dir / f"rank_{self.rank}"
        
        if not rank_dir.exists():
            raise FileNotFoundError(f"Shard directory not found: {rank_dir}")
        
        # Load weights
        weights = self._load_safetensors(rank_dir / "model.safetensors")
        
        # Map and load weights
        self._load_embedding(model, weights)
        self._load_layers(model, weights)
        self._load_norm(model, weights)
        self._load_lm_head(model, weights)
        
        logger.info(f"Rank {self.rank} loaded model weights from {shard_dir}")
    
    def _load_safetensors(self, path: Path) -> Dict[str, mx.array]:
        """Load weights from safetensors file."""
        weights = {}
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                weights[key] = mx.array(f.get_tensor(key))
        return weights
    
    def _load_embedding(
        self,
        model: ParallelTransformer,
        weights: Dict[str, mx.array],
    ):
        """Load embedding weights."""
        # Find embedding weight
        embed_weight = None
        for key in ["model.embed_tokens.weight", "embed_tokens.weight", "wte.weight"]:
            if key in weights:
                embed_weight = weights[key]
                break
        
        if embed_weight is not None:
            model.embed_tokens.load_shard_direct(embed_weight)
            logger.debug(f"Loaded embedding: {embed_weight.shape}")
    
    def _load_norm(
        self,
        model: ParallelTransformer,
        weights: Dict[str, mx.array],
    ):
        """Load final layer norm weights."""
        for key in ["model.norm.weight", "norm.weight", "ln_f.weight"]:
            if key in weights:
                model.norm.load_weights(weights[key])
                logger.debug(f"Loaded final norm: {weights[key].shape}")
                break
    
    def _load_lm_head(
        self,
        model: ParallelTransformer,
        weights: Dict[str, mx.array],
    ):
        """Load LM head weights."""
        if model.lm_head is None:
            return  # Tied embeddings
        
        for key in ["lm_head.weight"]:
            if key in weights:
                # Weights are already transposed during sharding to MLX format
                weight = weights[key]
                model.lm_head.load_shard_direct(weight)
                logger.debug(f"Loaded LM head: {weight.shape}")
                break
    
    def _load_layers(
        self,
        model: ParallelTransformer,
        weights: Dict[str, mx.array],
    ):
        """Load transformer layer weights."""
        for layer_idx, layer in enumerate(model.layers):
            self._load_single_layer(layer_idx, layer, weights)
    
    def _load_single_layer(
        self,
        layer_idx: int,
        layer,
        weights: Dict[str, mx.array],
    ):
        """Load weights for a single transformer layer."""
        prefix = f"model.layers.{layer_idx}"
        
        # Layer norms
        ln_key = f"{prefix}.input_layernorm.weight"
        if ln_key in weights:
            layer.input_layernorm.load_weights(weights[ln_key])
        
        post_ln_key = f"{prefix}.post_attention_layernorm.weight"
        if post_ln_key in weights:
            layer.post_attention_layernorm.load_weights(weights[post_ln_key])
        
        # Attention - weights are already in MLX format from sharding (transposed there)
        attn_weights = {}
        
        q_key = f"{prefix}.self_attn.q_proj.weight"
        if q_key in weights:
            attn_weights["q_weight"] = weights[q_key]
        
        k_key = f"{prefix}.self_attn.k_proj.weight"
        if k_key in weights:
            attn_weights["k_weight"] = weights[k_key]
        
        v_key = f"{prefix}.self_attn.v_proj.weight"
        if v_key in weights:
            attn_weights["v_weight"] = weights[v_key]
        
        o_key = f"{prefix}.self_attn.o_proj.weight"
        if o_key in weights:
            attn_weights["o_weight"] = weights[o_key]
        
        if attn_weights:
            layer.attention.load_shards(
                q_weight=attn_weights.get("q_weight"),
                q_bias=None,
                k_weight=attn_weights.get("k_weight"),
                k_bias=None,
                v_weight=attn_weights.get("v_weight"),
                v_bias=None,
                o_weight=attn_weights.get("o_weight"),
                o_bias=None,
            )
            logger.debug(f"Loaded attention weights for layer {layer_idx}")
        
        # MLP - weights are already in MLX format from sharding (transposed there)
        mlp_weights = {}
        
        gate_key = f"{prefix}.mlp.gate_proj.weight"
        if gate_key in weights:
            mlp_weights["gate_weight"] = weights[gate_key]
        
        up_key = f"{prefix}.mlp.up_proj.weight"
        if up_key in weights:
            mlp_weights["up_weight"] = weights[up_key]
        
        down_key = f"{prefix}.mlp.down_proj.weight"
        if down_key in weights:
            mlp_weights["down_weight"] = weights[down_key]
        
        if mlp_weights:
            layer.mlp.load_shards(
                up_weight=mlp_weights.get("up_weight"),
                up_bias=None,
                down_weight=mlp_weights.get("down_weight"),
                down_bias=None,
                gate_weight=mlp_weights.get("gate_weight"),
                gate_bias=None,
            )
            logger.debug(f"Loaded MLP weights for layer {layer_idx}")


class HuggingFaceConverter:
    """
    Converts HuggingFace models to ShardCompute format.
    
    Downloads from HuggingFace Hub and converts weights.
    """
    
    @staticmethod
    def download_model(
        model_name: str,
        cache_dir: str,
    ) -> Path:
        """
        Download model from HuggingFace Hub.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Local cache directory
            
        Returns:
            Path to downloaded model
        """
        from huggingface_hub import snapshot_download
        
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Download model
        local_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_path),
            local_dir=str(cache_path / model_name.replace("/", "_")),
        )
        
        logger.info(f"Downloaded model to {local_dir}")
        return Path(local_dir)
    
    @staticmethod
    def convert_to_mlx(
        model_dir: Path,
        output_dir: Path,
    ):
        """
        Convert model weights to MLX format.
        
        Args:
            model_dir: Directory with HuggingFace model
            output_dir: Output directory for MLX weights
        """
        import numpy as np
        from safetensors import safe_open
        from safetensors.numpy import save_file
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config
        config_src = model_dir / "config.json"
        if config_src.exists():
            import shutil
            shutil.copy(config_src, output_dir / "config.json")
        
        # Convert weights
        weights = {}
        
        for sf_file in model_dir.glob("*.safetensors"):
            with safe_open(str(sf_file), framework="numpy") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Convert to float32 if needed
                    if tensor.dtype == np.float16:
                        tensor = tensor.astype(np.float32)
                    weights[key] = tensor
        
        # Save as single file
        save_file(weights, str(output_dir / "model.safetensors"))
        
        logger.info(f"Converted {len(weights)} tensors to {output_dir}")
