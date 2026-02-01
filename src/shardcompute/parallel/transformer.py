"""Parallel transformer block and full model."""

import math
import mlx.core as mx
from typing import Optional, List, Tuple, Dict, Any
import logging

from shardcompute.collectives.communicator import Communicator
from shardcompute.parallel.attention import ParallelAttention, RoPE
from shardcompute.parallel.mlp import ParallelMLP
from shardcompute.parallel.embedding import ParallelEmbedding, QuantizedParallelEmbedding
from shardcompute.parallel.column_linear import ColumnParallelLinear

logger = logging.getLogger(__name__)


class RMSNorm:
    """
    Root Mean Square Layer Normalization with optional bias.
    
    Used in LLaMA (RMSNorm, no bias) and Phi-2 (LayerNorm-style, with bias).
    More computationally efficient than LayerNorm as it doesn't require 
    computing mean (when bias is not used).
    
    RMSNorm is applied locally (no communication needed) as it normalizes
    within each sample independently.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight: Optional[mx.array] = None  # [hidden_size]
        self.bias: Optional[mx.array] = None    # [hidden_size] - optional for LayerNorm-style
    
    def load_weights(self, weight: mx.array, bias: Optional[mx.array] = None):
        """Load normalization weights and optional bias."""
        if weight.shape != (self.hidden_size,):
            raise ValueError(f"Expected weight shape ({self.hidden_size},), got {weight.shape}")
        self.weight = weight
        
        if bias is not None:
            if bias.shape != (self.hidden_size,):
                raise ValueError(f"Expected bias shape ({self.hidden_size},), got {bias.shape}")
            self.bias = bias
    
    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization (or LayerNorm-style if bias present)."""
        if self.weight is None:
            raise RuntimeError("Weights not loaded")
        
        # Compute RMS
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        
        # Normalize and scale
        output = (x / rms) * self.weight
        
        # Add bias if present (LayerNorm-style)
        if self.bias is not None:
            output = output + self.bias
        
        return output


class ParallelTransformerBlock:
    """
    Single transformer block with tensor parallelism.
    
    Structure (LLaMA style):
        residual = hidden_states
        hidden_states = RMSNorm(hidden_states)
        hidden_states = Attention(hidden_states) + residual
        
        residual = hidden_states
        hidden_states = RMSNorm(hidden_states)
        hidden_states = MLP(hidden_states) + residual
    
    Communication:
    - RMSNorm: None (local operation)
    - Attention: One all-reduce
    - MLP: One all-reduce
    
    Total: 2 all-reduces per transformer block.
    """
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        world_size: int,
        rank: int,
        communicator: Communicator,
        num_kv_heads: Optional[int] = None,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
        mlp_activation: str = "silu",
        use_gated_mlp: bool = True,
        use_quantized: bool = False,
        quantization_bits: int = 4,
        quantization_group_size: int = 64,
    ):
        """
        Initialize ParallelTransformerBlock.

        Args:
            layer_idx: Layer index in the model
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            intermediate_size: MLP intermediate dimension
            world_size: Number of workers
            rank: This worker's rank
            communicator: Communicator for collectives
            num_kv_heads: Number of KV heads (for GQA)
            rms_norm_eps: Epsilon for RMS normalization
            attention_bias: Whether attention uses bias
            mlp_bias: Whether MLP uses bias
            max_position_embeddings: Maximum sequence length for RoPE
            rope_base: Base frequency for RoPE
            mlp_activation: Activation function for MLP (silu, gelu, gelu_new)
            use_gated_mlp: Whether to use gated MLP (LLaMA style)
            use_quantized: Whether to use quantized linear layers
            quantization_bits: Bit width for quantization (4 or 8)
            quantization_group_size: Group size for quantization
        """
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.world_size = world_size
        self.rank = rank
        self.communicator = communicator
        self.use_quantized = use_quantized

        # Input layernorm (before attention)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Attention
        self.attention = ParallelAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            world_size=world_size,
            rank=rank,
            communicator=communicator,
            num_kv_heads=num_kv_heads,
            bias=attention_bias,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            use_quantized=use_quantized,
            quantization_bits=quantization_bits,
            quantization_group_size=quantization_group_size,
        )

        # Post-attention layernorm (before MLP)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # MLP
        self.mlp = ParallelMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            world_size=world_size,
            rank=rank,
            communicator=communicator,
            activation=mlp_activation,
            bias=mlp_bias,
            use_gated=use_gated_mlp,
            use_quantized=use_quantized,
            quantization_bits=quantization_bits,
            quantization_group_size=quantization_group_size,
        )

        logger.debug(f"ParallelTransformerBlock layer {layer_idx} initialized on rank {rank}")
    
    def load_weights(
        self,
        input_ln_weight: mx.array,
        post_attn_ln_weight: mx.array,
        attn_weights: Dict[str, mx.array],
        mlp_weights: Dict[str, mx.array],
    ):
        """
        Load weights for this transformer block.
        
        Args:
            input_ln_weight: Input layernorm weight
            post_attn_ln_weight: Post-attention layernorm weight
            attn_weights: Dict with q, k, v, o projection weights and biases
            mlp_weights: Dict with up, down, gate projection weights and biases
        """
        self.input_layernorm.load_weights(input_ln_weight)
        self.post_attention_layernorm.load_weights(post_attn_ln_weight)
        
        self.attention.load_shards(
            q_weight=attn_weights["q_weight"],
            q_bias=attn_weights.get("q_bias"),
            k_weight=attn_weights["k_weight"],
            k_bias=attn_weights.get("k_bias"),
            v_weight=attn_weights["v_weight"],
            v_bias=attn_weights.get("v_bias"),
            o_weight=attn_weights["o_weight"],
            o_bias=attn_weights.get("o_bias"),
        )
        
        self.mlp.load_shards(
            up_weight=mlp_weights["up_weight"],
            up_bias=mlp_weights.get("up_bias"),
            down_weight=mlp_weights["down_weight"],
            down_bias=mlp_weights.get("down_bias"),
            gate_weight=mlp_weights.get("gate_weight"),
            gate_bias=mlp_weights.get("gate_bias"),
        )
    
    async def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Forward pass through transformer block.

        Uses standard blocking all-reduce pattern. The async API helps
        reduce latency by yielding control during network I/O.

        Args:
            hidden_states: [batch, seq_len, hidden_size] - REPLICATED
            attention_mask: Optional attention mask
            past_key_value: KV cache from previous steps
            use_cache: Whether to return updated cache

        Returns:
            hidden_states: [batch, seq_len, hidden_size] - REPLICATED
            cache: Updated KV cache if use_cache=True
        """
        # Attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, cache = await self.attention.forward(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = await self.mlp.forward(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, cache

    async def forward_with_pending_ar(
        self,
        hidden_states: mx.array,
        pending_ar_id: Optional[str],
        pending_ar_residual: Optional[mx.array],
        pending_ar_proj,  # The projection layer (for add_bias)
        attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]], Optional[str], Optional[mx.array], Any]:
        """
        Forward pass that accepts a pending all-reduce from the previous layer.

        This enables true inter-layer pipelining: the previous layer's MLP all-reduce
        runs concurrently with this layer's attention computation.

        Args:
            hidden_states: Input hidden states (before previous layer's residual add)
            pending_ar_id: Operation ID for pending all-reduce from previous layer
            pending_ar_residual: Residual to add after pending all-reduce completes
            pending_ar_proj: The projection layer that needs add_bias called
            attention_mask: Optional attention mask
            past_key_value: KV cache from previous steps
            use_cache: Whether to return updated cache

        Returns:
            hidden_states: After this layer's attention (before MLP residual)
            cache: Updated KV cache if use_cache=True
            mlp_ar_id: Pending all-reduce ID for this layer's MLP
            mlp_residual: Residual to add when mlp_ar_id completes
            mlp_proj: The down_proj layer for add_bias
        """
        # === OVERLAP STRATEGY ===
        # While the previous layer's MLP all-reduce runs in background,
        # compute this layer's attention (which doesn't depend on it).

        # Start computing attention on the INPUT hidden_states (not yet updated)
        # This is only valid if we're pipelining correctly from the caller
        if pending_ar_id is not None:
            # Use hidden_states as-is for now (will be updated after attention)
            attn_input = hidden_states
        else:
            attn_input = hidden_states

        # === ATTENTION PHASE ===
        attn_residual = attn_input
        normed_hidden = self.input_layernorm(attn_input)

        # Compute attention partial (overlaps with pending MLP all-reduce if any)
        attn_partial, cache = self.attention.forward_partial_with_cache(
            normed_hidden,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # All-reduce for attention (blocking - we need it for MLP input)
        mx.eval(attn_partial)
        attn_output = await self.communicator.all_reduce(attn_partial, op="sum")
        attn_output = self.attention.o_proj.add_bias(attn_output)
        hidden_states = attn_residual + attn_output

        # === MLP PHASE - start async ===
        mlp_residual = hidden_states
        normed_for_mlp = self.post_attention_layernorm(hidden_states)
        mlp_partial = self.mlp.forward_sync_partial(normed_for_mlp)

        # Start async all-reduce - caller will wait for it
        mx.eval(mlp_partial)
        mlp_ar_id = self.communicator.start_all_reduce(mlp_partial, op="sum")

        # Return with pending MLP all-reduce
        return hidden_states, cache, mlp_ar_id, mlp_residual, self.mlp.down_proj
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters in this shard of the block."""
        params = self.hidden_size * 2  # Two layer norms
        params += self.attention.num_parameters
        params += self.mlp.num_parameters
        return params


class ParallelTransformer:
    """
    Full parallel transformer model.
    
    Structure:
        embed_tokens -> [TransformerBlock] * num_layers -> norm -> lm_head
    
    For tensor parallelism:
    - Embeddings: Column parallel (split by embedding dim)
    - All transformer blocks: Each has parallel attention + MLP
    - Final norm: Local (no communication)
    - LM head: Column parallel (can be tied with embeddings)
    
    Communication per forward pass:
    - Embedding: One all-gather
    - Per layer: Two all-reduces (attention + MLP)
    - LM head: One all-gather
    
    Total collectives: 1 + 2*num_layers + 1 = 2 + 2*num_layers
    For TinyLlama (22 layers): 46 collectives per forward pass
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        world_size: int,
        rank: int,
        communicator: Communicator,
        num_kv_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        rope_base: float = 10000.0,
        mlp_activation: str = "silu",
        use_gated_mlp: bool = True,
        use_quantized: bool = False,
        quantization_bits: int = 4,
        quantization_group_size: int = 64,
    ):
        """
        Initialize ParallelTransformer.

        Args:
            vocab_size: Vocabulary size
            hidden_size: Model hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_size: MLP intermediate dimension
            world_size: Number of workers
            rank: This worker's rank
            communicator: Communicator for collectives
            num_kv_heads: Number of KV heads (for GQA)
            max_position_embeddings: Maximum sequence length
            rms_norm_eps: Epsilon for RMS normalization
            tie_word_embeddings: Whether to tie input/output embeddings
            rope_base: Base frequency for RoPE
            mlp_activation: Activation function for MLP
            use_gated_mlp: Whether to use gated MLP (LLaMA style)
            use_quantized: Whether to use quantized linear layers
            quantization_bits: Bit width for quantization (4 or 8)
            quantization_group_size: Group size for quantization
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.world_size = world_size
        self.rank = rank
        self.communicator = communicator
        self.tie_word_embeddings = tie_word_embeddings
        self.use_quantized = use_quantized
        self.quantization_bits = quantization_bits
        self.quantization_group_size = quantization_group_size

        # Token embeddings
        if use_quantized:
            # Quantized embeddings use vocabulary parallelism
            self.embed_tokens = QuantizedParallelEmbedding(
                vocab_size=vocab_size,
                embedding_dim=hidden_size,
                world_size=world_size,
                rank=rank,
                communicator=communicator,
                group_size=quantization_group_size,
                bits=quantization_bits,
            )
        else:
            # Standard embeddings use column parallelism
            self.embed_tokens = ParallelEmbedding(
                vocab_size=vocab_size,
                embedding_dim=hidden_size,
                world_size=world_size,
                rank=rank,
                gather_output=True,  # Gather to get full embeddings
                communicator=communicator,
            )
        
        # Transformer layers
        self.layers: List[ParallelTransformerBlock] = []
        for layer_idx in range(num_layers):
            layer = ParallelTransformerBlock(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                world_size=world_size,
                rank=rank,
                communicator=communicator,
                num_kv_heads=num_kv_heads,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
                rope_base=rope_base,
                mlp_activation=mlp_activation,
                use_gated_mlp=use_gated_mlp,
                use_quantized=use_quantized,
                quantization_bits=quantization_bits,
                quantization_group_size=quantization_group_size,
            )
            self.layers.append(layer)
        
        # Final layer norm
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # LM head (column parallel, may be tied with embeddings)
        if not tie_word_embeddings:
            self.lm_head = ColumnParallelLinear(
                in_features=hidden_size,
                out_features=vocab_size,
                world_size=world_size,
                rank=rank,
                bias=False,
                gather_output=True,
                communicator=communicator,
            )
        else:
            self.lm_head = None
        
        logger.info(
            f"ParallelTransformer initialized: {num_layers} layers, "
            f"rank {rank}/{world_size}"
        )
    
    def load_embedding_weights(self, weight: mx.array):
        """Load embedding weights (non-quantized)."""
        self.embed_tokens.load_shard_direct(weight)

        # If tied, use same weights for lm_head
        if self.tie_word_embeddings:
            # Transpose for lm_head: [vocab, hidden] -> [hidden, local_vocab]
            # But since we use column parallel, we need [hidden, local_vocab]
            # This is handled by the embedding layer itself
            pass

    def load_quantized_embedding_weights(
        self,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
    ):
        """Load quantized embedding weights."""
        if not self.use_quantized:
            raise RuntimeError("Cannot load quantized embeddings into non-quantized model")

        self.embed_tokens.load_quantized_shard(weight, scales, biases)
    
    def load_lm_head_weights(self, weight: mx.array):
        """Load LM head weights (if not tied)."""
        if self.lm_head is not None:
            self.lm_head.load_shard_direct(weight)
    
    def load_norm_weights(self, weight: mx.array):
        """Load final layer norm weights."""
        self.norm.load_weights(weight)
    
    async def forward(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        """
        Forward pass through the full model.

        Args:
            input_ids: Token IDs [batch, seq_len] - REPLICATED
            attention_mask: Optional attention mask
            past_key_values: KV cache from previous steps
            use_cache: Whether to return updated cache

        Returns:
            logits: [batch, seq_len, vocab_size] - REPLICATED
            cache: Updated KV cache if use_cache=True
        """
        # Embedding lookup with all-gather
        hidden_states = await self.embed_tokens.forward(input_ids)

        # Process through transformer layers
        new_cache = [] if use_cache else None

        # Evaluate every N layers to balance memory vs GPU sync overhead.
        eval_interval = 4

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None

            hidden_states, layer_cache = await layer.forward(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if (i + 1) % eval_interval == 0 or i == len(self.layers) - 1:
                mx.eval(hidden_states)

            if use_cache and layer_cache is not None:
                new_cache.append(layer_cache)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # LM head
        if self.tie_word_embeddings:
            # Use embedding weights transposed
            if self.use_quantized and hasattr(self.embed_tokens, "dequantize_weight"):
                local_weight = self.embed_tokens.dequantize_weight()
                logits = hidden_states @ local_weight.T
            else:
                logits = hidden_states @ self.embed_tokens.weight.T
            # Gather across workers
            logits = await self.communicator.all_gather(logits, dim=-1)
        else:
            logits = await self.lm_head.forward(hidden_states)

        return logits, new_cache
    
    @property
    def num_parameters(self) -> int:
        """Total number of parameters in this shard."""
        params = self.embed_tokens.num_parameters
        params += sum(layer.num_parameters for layer in self.layers)
        params += self.hidden_size  # Final norm
        if self.lm_head is not None:
            params += self.lm_head.num_parameters
        return params
    
    def get_layer(self, idx: int) -> ParallelTransformerBlock:
        """Get a specific transformer layer."""
        return self.layers[idx]


# ---------------------------------------------------------------------------
# Pipeline Parallelism: each worker holds FULL weights for a subset of layers
# ---------------------------------------------------------------------------


class LinearLayer:
    """
    Standard (non-parallel) linear layer for pipeline parallelism.

    Each worker holds the full weight matrix — no sharding, no all-reduce.
    Supports both full-precision and quantized weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, group_size: int = 64, bits: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.weight: Optional[mx.array] = None  # [in_features, out_features] or quantized
        self.scales: Optional[mx.array] = None  # For quantized weights
        self.biases_quant: Optional[mx.array] = None  # For quantized weights
        self.bias: Optional[mx.array] = None  # Linear bias (not quantization bias)
        self.has_bias = bias
        self.is_quantized = False
        self.group_size = group_size
        self.bits = bits

    def load_quantized_weights(
        self,
        quantized_weight: mx.array,
        scales: mx.array,
        biases: mx.array,
    ):
        """Load quantized weights."""
        self.weight = quantized_weight
        self.scales = scales
        self.biases_quant = biases
        self.is_quantized = True

    def __call__(self, x: mx.array) -> mx.array:
        if self.weight is None:
            raise RuntimeError("Weight not loaded")

        if self.is_quantized:
            # Quantized linear: use quantized_matmul with transpose=True
            # This matches the working QuantizedColumnParallelLinear implementation
            # transpose=True handles the weight format correctly for MLX quantized weights
            output = mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                self.biases_quant,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
            )
        else:
            output = x @ self.weight

        if self.has_bias and self.bias is not None:
            output = output + self.bias
        return output

    @property
    def num_parameters(self) -> int:
        params = self.in_features * self.out_features
        if self.has_bias:
            params += self.out_features
        return params


class PipelineAttention:
    """
    Full (non-parallel) multi-head attention for pipeline parallelism.

    All heads are local — no weight splitting, no all-reduce needed.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        bias: bool = False,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = LinearLayer(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = LinearLayer(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = LinearLayer(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = LinearLayer(num_heads * self.head_dim, hidden_size, bias=bias)

        self.rope = RoPE(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
        )

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        position_offset = 0
        if past_key_value is not None:
            position_offset = past_key_value[0].shape[2]

        q, k = self.rope(q, k, position_offset=position_offset)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)

        new_cache = (k, v) if use_cache else None
        kv_seq_len = k.shape[2]

        # Handle GQA
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, repeat_factor, axis=1)
            v = mx.repeat(v, repeat_factor, axis=1)

        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        if seq_len > 1:
            causal_mask = mx.triu(
                mx.full((seq_len, kv_seq_len), float('-inf')),
                k=kv_seq_len - seq_len + 1,
            )
            attn_scores = attn_scores + causal_mask

        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_output = attn_weights @ v

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        output = self.o_proj(attn_output)
        return output, new_cache

    @property
    def num_parameters(self) -> int:
        return (
            self.q_proj.num_parameters
            + self.k_proj.num_parameters
            + self.v_proj.num_parameters
            + self.o_proj.num_parameters
        )


class PipelineMLP:
    """
    Full (non-parallel) MLP for pipeline parallelism.

    No weight splitting, no all-reduce.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = False,
        use_gated: bool = True,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_gated = use_gated

        self.up_proj = LinearLayer(hidden_size, intermediate_size, bias=bias)
        self.down_proj = LinearLayer(intermediate_size, hidden_size, bias=bias)
        self.gate_proj = (
            LinearLayer(hidden_size, intermediate_size, bias=bias) if use_gated else None
        )

        if activation in ("silu", "swish"):
            self.activation = lambda x: x * mx.sigmoid(x)
        elif activation == "gelu":
            self.activation = lambda x: x * mx.sigmoid(1.702 * x)
        elif activation == "gelu_new":
            self.activation = lambda x: 0.5 * x * (
                1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))
            )
        elif activation == "relu":
            self.activation = lambda x: mx.maximum(x, 0)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, hidden_states: mx.array) -> mx.array:
        if self.use_gated and self.gate_proj is not None:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
            intermediate = self.activation(gate) * up
        else:
            intermediate = self.up_proj(hidden_states)
            intermediate = self.activation(intermediate)
        return self.down_proj(intermediate)

    @property
    def num_parameters(self) -> int:
        params = self.up_proj.num_parameters + self.down_proj.num_parameters
        if self.gate_proj is not None:
            params += self.gate_proj.num_parameters
        return params


class PipelineTransformerBlock:
    """
    Single transformer block for pipeline parallelism.

    Identical structure to ParallelTransformerBlock but uses full
    (non-parallel) attention and MLP — no all-reduces.
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: Optional[int] = None,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
        mlp_activation: str = "silu",
        use_gated_mlp: bool = True,
    ):
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.attention = PipelineAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=attention_bias,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
        )

        self.mlp = PipelineMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=mlp_activation,
            bias=mlp_bias,
            use_gated=use_gated_mlp,
        )

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass — fully local, no communication."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, cache = self.attention.forward(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp.forward(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, cache

    @property
    def num_parameters(self) -> int:
        params = self.hidden_size * 2  # Two layer norms
        params += self.attention.num_parameters
        params += self.mlp.num_parameters
        return params


class Embedding:
    """Simple (non-parallel) embedding lookup."""

    def __init__(self, vocab_size: int, embedding_dim: int, group_size: int = 64, bits: int = 4):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.bits = bits
        self.weight: Optional[mx.array] = None  # [vocab_size, embedding_dim] or quantized
        self.scales: Optional[mx.array] = None  # For quantized embeddings
        self.biases: Optional[mx.array] = None  # For quantized embeddings
        self.is_quantized = False

    def load_quantized_weights(
        self,
        quantized_weight: mx.array,
        scales: mx.array,
        biases: mx.array,
    ):
        """Load quantized embedding weights."""
        self.weight = quantized_weight
        self.scales = scales
        self.biases = biases
        self.is_quantized = True

    def forward(self, input_ids: mx.array) -> mx.array:
        if self.weight is None:
            raise RuntimeError("Embedding weights not loaded")

        flat_ids = input_ids.reshape(-1)
        batch_size, seq_len = input_ids.shape

        if self.is_quantized:
            # Dequantize embeddings for the needed indices
            flat_quant_weight = mx.take(self.weight, flat_ids, axis=0)
            flat_scales = mx.take(self.scales, flat_ids, axis=0)
            flat_biases = mx.take(self.biases, flat_ids, axis=0)

            # Dequantize
            embeddings = mx.dequantize(
                flat_quant_weight,
                flat_scales,
                flat_biases,
                group_size=self.group_size,
                bits=self.bits,
            )

            return embeddings.reshape(batch_size, seq_len, self.embedding_dim)
        else:
            embeddings = mx.take(self.weight, flat_ids, axis=0)
            return embeddings.reshape(batch_size, seq_len, self.embedding_dim)

    @property
    def num_parameters(self) -> int:
        return self.vocab_size * self.embedding_dim


class PipelineParallelTransformer:
    """
    Transformer model for pipeline parallelism.

    Each pipeline stage (rank) holds:
    - A contiguous subset of transformer layers with full weights
    - First stage also holds the embedding layer
    - Last stage also holds the final norm and LM head

    Communication:
    - 1 point-to-point send per forward pass (hidden states between stages)
    - Compare with TP: 2 * num_layers all-reduces per forward pass

    Layer assignment is dynamic: num_layers // pipeline_parallel_size per stage.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        world_size: int,
        rank: int,
        communicator: Communicator,
        num_kv_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        rope_base: float = 10000.0,
        mlp_activation: str = "silu",
        use_gated_mlp: bool = True,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.world_size = world_size
        self.rank = rank
        self.communicator = communicator
        self.tie_word_embeddings = tie_word_embeddings

        # Determine layer assignment
        layers_per_stage = num_layers // world_size
        self.start_layer = rank * layers_per_stage
        self.end_layer = self.start_layer + layers_per_stage
        self.has_embedding = rank == 0
        self.has_lm_head = rank == world_size - 1

        # Embedding (first stage only)
        if self.has_embedding:
            self.embed_tokens = Embedding(vocab_size, hidden_size)
        else:
            self.embed_tokens = None

        # Transformer layers (only this stage's subset)
        self.layers: List[PipelineTransformerBlock] = []
        for i in range(self.start_layer, self.end_layer):
            block = PipelineTransformerBlock(
                layer_idx=i,
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                num_kv_heads=num_kv_heads,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
                rope_base=rope_base,
                mlp_activation=mlp_activation,
                use_gated_mlp=use_gated_mlp,
            )
            self.layers.append(block)

        # Final norm + LM head (last stage only)
        if self.has_lm_head:
            self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            if not tie_word_embeddings:
                self.lm_head = LinearLayer(hidden_size, vocab_size, bias=False)
            else:
                self.lm_head = None
        else:
            self.norm = None
            self.lm_head = None

        logger.info(
            f"PipelineParallelTransformer rank {rank}/{world_size}: "
            f"layers [{self.start_layer}, {self.end_layer}), "
            f"embedding={self.has_embedding}, lm_head={self.has_lm_head}"
        )

    async def forward(
        self,
        input_ids: Optional[mx.array] = None,
        hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        """
        Forward pass for this pipeline stage.

        First stage: takes input_ids, produces hidden states
        Middle stages: takes hidden_states, produces hidden states
        Last stage: takes hidden_states, produces logits

        Args:
            input_ids: Token IDs (first stage only)
            hidden_states: Hidden states from previous stage (non-first stages)
            attention_mask: Optional attention mask
            past_key_values: KV cache (indexed by local layer index)
            use_cache: Whether to return updated cache

        Returns:
            (output, cache) where output is hidden_states or logits
        """
        # Embedding (first stage)
        if self.has_embedding:
            if input_ids is None:
                raise ValueError("First stage requires input_ids")
            hidden_states = self.embed_tokens.forward(input_ids)
        else:
            if hidden_states is None:
                raise ValueError("Non-first stages require hidden_states")

        # Process local layers
        new_cache = [] if use_cache else None
        eval_interval = 4

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, layer_cache = layer.forward(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if (i + 1) % eval_interval == 0 or i == len(self.layers) - 1:
                mx.eval(hidden_states)
            if use_cache and layer_cache is not None:
                new_cache.append(layer_cache)

        # Final norm + LM head (last stage)
        if self.has_lm_head:
            hidden_states = self.norm(hidden_states)
            if self.tie_word_embeddings and self.embed_tokens is not None:
                logits = hidden_states @ self.embed_tokens.weight.T
            elif self.lm_head is not None:
                logits = self.lm_head(hidden_states)
            else:
                logits = hidden_states
            return logits, new_cache

        return hidden_states, new_cache

    @property
    def num_parameters(self) -> int:
        params = 0
        if self.embed_tokens is not None:
            params += self.embed_tokens.num_parameters
        params += sum(layer.num_parameters for layer in self.layers)
        if self.norm is not None:
            params += self.hidden_size
        if self.lm_head is not None:
            params += self.lm_head.num_parameters
        return params
