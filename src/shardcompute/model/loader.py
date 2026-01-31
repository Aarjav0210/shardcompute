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
    
    def __init__(self, rank: int, world_size: int, quantized: bool = False):
        """
        Initialize ModelLoader.

        Args:
            rank: Worker rank
            world_size: Total number of workers
            quantized: Whether to load quantized weights
        """
        self.rank = rank
        self.world_size = world_size
        self.quantized = quantized

    @staticmethod
    def detect_quantization(shard_dir: Path, rank: int = 0) -> bool:
        """
        Detect if the sharded model uses quantized weights.

        Checks for .scales and .biases keys in the weight file.

        Args:
            shard_dir: Directory containing sharded weights
            rank: Rank to check (default 0)

        Returns:
            True if quantized weights detected
        """
        rank_dir = shard_dir / f"rank_{rank}"
        model_file = rank_dir / "model.safetensors"

        if not model_file.exists():
            return False

        with safe_open(str(model_file), framework="numpy") as f:
            keys = f.keys()
            # Look for quantization markers
            has_scales = any(".scales" in k for k in keys)
            has_biases = any(".biases" in k for k in keys)
            return has_scales and has_biases
    
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
        embed_scales = None
        embed_biases = None

        for key in ["model.embed_tokens.weight", "embed_tokens.weight", "wte.weight"]:
            if key in weights:
                embed_weight = weights[key]
                # Check for quantized embedding
                base_key = key.rsplit(".weight", 1)[0]
                embed_scales = weights.get(f"{base_key}.scales")
                embed_biases = weights.get(f"{base_key}.biases")
                break

        if embed_weight is not None:
            if self.quantized and embed_scales is not None and embed_biases is not None:
                # Load quantized embedding
                model.load_quantized_embedding_weights(embed_weight, embed_scales, embed_biases)
                logger.debug(f"Loaded quantized embedding: weight={embed_weight.shape}, scales={embed_scales.shape}")
            else:
                # Load standard embedding
                model.embed_tokens.load_shard_direct(embed_weight)
                logger.debug(f"Loaded embedding: {embed_weight.shape}")
    
    def _load_norm(
        self,
        model: ParallelTransformer,
        weights: Dict[str, mx.array],
    ):
        """Load final layer norm weights."""
        # Try different naming conventions: Llama uses "norm", Phi uses "final_layernorm"
        weight_key = None
        bias_key = None
        
        for key in ["model.norm.weight", "model.final_layernorm.weight", "norm.weight", "ln_f.weight"]:
            if key in weights:
                weight_key = key
                # Check for corresponding bias
                bias_key = key.replace(".weight", ".bias")
                break
        
        if weight_key:
            weight = weights[weight_key]
            bias = weights.get(bias_key)
            model.norm.load_weights(weight, bias)
            logger.debug(f"Loaded final norm: {weight.shape}")
    
    def _load_lm_head(
        self,
        model: ParallelTransformer,
        weights: Dict[str, mx.array],
    ):
        """Load LM head weights."""
        if model.lm_head is None:
            return  # Tied embeddings
        
        if "lm_head.weight" in weights:
            # Weights are already transposed during sharding to MLX format
            weight = weights["lm_head.weight"]
            bias = weights.get("lm_head.bias")
            model.lm_head.load_shard_direct(weight, bias)
            logger.debug(f"Loaded LM head: {weight.shape}")
    
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
        
        # Layer norms (with optional bias for Phi-2 style)
        ln_key = f"{prefix}.input_layernorm.weight"
        ln_bias_key = f"{prefix}.input_layernorm.bias"
        if ln_key in weights:
            layer.input_layernorm.load_weights(
                weights[ln_key],
                weights.get(ln_bias_key)
            )
        
        post_ln_key = f"{prefix}.post_attention_layernorm.weight"
        post_ln_bias_key = f"{prefix}.post_attention_layernorm.bias"
        if post_ln_key in weights:
            layer.post_attention_layernorm.load_weights(
                weights[post_ln_key],
                weights.get(post_ln_bias_key)
            )
        elif ln_key in weights:
            # Phi-2 style: parallel attention+MLP share input_layernorm
            # Use input_layernorm weights for post_attention_layernorm as well
            logger.debug(f"Layer {layer_idx}: No post_attention_layernorm found, "
                        "using input_layernorm (parallel attention+MLP architecture)")
            layer.post_attention_layernorm.load_weights(
                weights[ln_key],
                weights.get(ln_bias_key)
            )
        
        # Attention - weights are already in MLX format from sharding (transposed there)
        # Support both Llama naming (q_proj, etc.) and Phi naming (same for QKV, but "dense" for output)
        attn_weights = {}
        
        # Q projection
        q_key = f"{prefix}.self_attn.q_proj.weight"
        q_bias_key = f"{prefix}.self_attn.q_proj.bias"
        if q_key in weights:
            attn_weights["q_weight"] = weights[q_key]
            attn_weights["q_bias"] = weights.get(q_bias_key)
        
        # K projection
        k_key = f"{prefix}.self_attn.k_proj.weight"
        k_bias_key = f"{prefix}.self_attn.k_proj.bias"
        if k_key in weights:
            attn_weights["k_weight"] = weights[k_key]
            attn_weights["k_bias"] = weights.get(k_bias_key)
        
        # V projection
        v_key = f"{prefix}.self_attn.v_proj.weight"
        v_bias_key = f"{prefix}.self_attn.v_proj.bias"
        if v_key in weights:
            attn_weights["v_weight"] = weights[v_key]
            attn_weights["v_bias"] = weights.get(v_bias_key)
        
        # Output projection - Llama uses "o_proj", Phi-2 uses "dense"
        o_key = f"{prefix}.self_attn.o_proj.weight"
        o_bias_key = f"{prefix}.self_attn.o_proj.bias"
        if o_key not in weights:
            # Try Phi-2 naming
            o_key = f"{prefix}.self_attn.dense.weight"
            o_bias_key = f"{prefix}.self_attn.dense.bias"
        if o_key in weights:
            attn_weights["o_weight"] = weights[o_key]
            attn_weights["o_bias"] = weights.get(o_bias_key)
        
        if attn_weights:
            # Check if this is quantized by looking for scales
            q_scales_key = f"{prefix}.self_attn.q_proj.scales"
            is_quantized = q_scales_key in weights

            if is_quantized and self.quantized:
                # Load quantized attention weights
                layer.attention.load_quantized_shards(
                    q_weight=attn_weights.get("q_weight"),
                    q_scales=weights.get(f"{prefix}.self_attn.q_proj.scales"),
                    q_biases=weights.get(f"{prefix}.self_attn.q_proj.biases"),
                    q_linear_bias=attn_weights.get("q_bias"),
                    k_weight=attn_weights.get("k_weight"),
                    k_scales=weights.get(f"{prefix}.self_attn.k_proj.scales"),
                    k_biases=weights.get(f"{prefix}.self_attn.k_proj.biases"),
                    k_linear_bias=attn_weights.get("k_bias"),
                    v_weight=attn_weights.get("v_weight"),
                    v_scales=weights.get(f"{prefix}.self_attn.v_proj.scales"),
                    v_biases=weights.get(f"{prefix}.self_attn.v_proj.biases"),
                    v_linear_bias=attn_weights.get("v_bias"),
                    o_weight=attn_weights.get("o_weight"),
                    o_scales=weights.get(f"{prefix}.self_attn.o_proj.scales"),
                    o_biases=weights.get(f"{prefix}.self_attn.o_proj.biases"),
                    o_linear_bias=attn_weights.get("o_bias"),
                )
                logger.debug(f"Loaded quantized attention weights for layer {layer_idx}")
            else:
                layer.attention.load_shards(
                    q_weight=attn_weights.get("q_weight"),
                    q_bias=attn_weights.get("q_bias"),
                    k_weight=attn_weights.get("k_weight"),
                    k_bias=attn_weights.get("k_bias"),
                    v_weight=attn_weights.get("v_weight"),
                    v_bias=attn_weights.get("v_bias"),
                    o_weight=attn_weights.get("o_weight"),
                    o_bias=attn_weights.get("o_bias"),
                )
                logger.debug(f"Loaded attention weights for layer {layer_idx}")
        
        # MLP - weights are already in MLX format from sharding (transposed there)
        # Support both Llama naming (gate/up/down_proj) and Phi naming (fc1/fc2)
        mlp_weights = {}
        
        # Check if this is Llama-style (with gate) or Phi-style (fc1/fc2)
        gate_key = f"{prefix}.mlp.gate_proj.weight"
        fc1_key = f"{prefix}.mlp.fc1.weight"
        
        if gate_key in weights:
            # Llama-style: gate_proj, up_proj, down_proj
            gate_bias_key = f"{prefix}.mlp.gate_proj.bias"
            mlp_weights["gate_weight"] = weights[gate_key]
            mlp_weights["gate_bias"] = weights.get(gate_bias_key)
            
            up_key = f"{prefix}.mlp.up_proj.weight"
            up_bias_key = f"{prefix}.mlp.up_proj.bias"
            if up_key in weights:
                mlp_weights["up_weight"] = weights[up_key]
                mlp_weights["up_bias"] = weights.get(up_bias_key)
            
            down_key = f"{prefix}.mlp.down_proj.weight"
            down_bias_key = f"{prefix}.mlp.down_proj.bias"
            if down_key in weights:
                mlp_weights["down_weight"] = weights[down_key]
                mlp_weights["down_bias"] = weights.get(down_bias_key)
                
        elif fc1_key in weights:
            # Phi-style: fc1 (up), fc2 (down), no gate
            fc1_bias_key = f"{prefix}.mlp.fc1.bias"
            mlp_weights["up_weight"] = weights[fc1_key]
            mlp_weights["up_bias"] = weights.get(fc1_bias_key)
            
            fc2_key = f"{prefix}.mlp.fc2.weight"
            fc2_bias_key = f"{prefix}.mlp.fc2.bias"
            if fc2_key in weights:
                mlp_weights["down_weight"] = weights[fc2_key]
                mlp_weights["down_bias"] = weights.get(fc2_bias_key)
            
            # No gate for Phi-style
            mlp_weights["gate_weight"] = None
            mlp_weights["gate_bias"] = None
        
        if mlp_weights:
            # Check if this is quantized by looking for scales
            up_scales_key = f"{prefix}.mlp.up_proj.scales"
            gate_scales_key = f"{prefix}.mlp.gate_proj.scales"
            fc1_scales_key = f"{prefix}.mlp.fc1.scales"
            is_quantized = (up_scales_key in weights or gate_scales_key in weights or fc1_scales_key in weights)

            if is_quantized and self.quantized:
                # Determine which naming convention is used
                if gate_key in weights:
                    # Llama-style naming
                    layer.mlp.load_quantized_shards(
                        up_weight=mlp_weights.get("up_weight"),
                        up_scales=weights.get(f"{prefix}.mlp.up_proj.scales"),
                        up_biases=weights.get(f"{prefix}.mlp.up_proj.biases"),
                        up_linear_bias=mlp_weights.get("up_bias"),
                        down_weight=mlp_weights.get("down_weight"),
                        down_scales=weights.get(f"{prefix}.mlp.down_proj.scales"),
                        down_biases=weights.get(f"{prefix}.mlp.down_proj.biases"),
                        down_linear_bias=mlp_weights.get("down_bias"),
                        gate_weight=mlp_weights.get("gate_weight"),
                        gate_scales=weights.get(f"{prefix}.mlp.gate_proj.scales"),
                        gate_biases=weights.get(f"{prefix}.mlp.gate_proj.biases"),
                        gate_linear_bias=mlp_weights.get("gate_bias"),
                    )
                else:
                    # Phi-style naming (fc1/fc2)
                    layer.mlp.load_quantized_shards(
                        up_weight=mlp_weights.get("up_weight"),
                        up_scales=weights.get(f"{prefix}.mlp.fc1.scales"),
                        up_biases=weights.get(f"{prefix}.mlp.fc1.biases"),
                        up_linear_bias=mlp_weights.get("up_bias"),
                        down_weight=mlp_weights.get("down_weight"),
                        down_scales=weights.get(f"{prefix}.mlp.fc2.scales"),
                        down_biases=weights.get(f"{prefix}.mlp.fc2.biases"),
                        down_linear_bias=mlp_weights.get("down_bias"),
                        gate_weight=None,
                        gate_scales=None,
                        gate_biases=None,
                        gate_linear_bias=None,
                    )
                logger.debug(f"Loaded quantized MLP weights for layer {layer_idx}")
            else:
                layer.mlp.load_shards(
                    up_weight=mlp_weights.get("up_weight"),
                    up_bias=mlp_weights.get("up_bias"),
                    down_weight=mlp_weights.get("down_weight"),
                    down_bias=mlp_weights.get("down_bias"),
                    gate_weight=mlp_weights.get("gate_weight"),
                    gate_bias=mlp_weights.get("gate_bias"),
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
