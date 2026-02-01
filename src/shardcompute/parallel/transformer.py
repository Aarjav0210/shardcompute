"""Parallel transformer block and full model."""

import mlx.core as mx
from typing import Optional, List, Tuple, Dict, Any
import logging

from shardcompute.collectives.communicator import Communicator
from shardcompute.parallel.attention import ParallelAttention
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
        Forward pass through transformer block with pipelined communication.

        Overlaps attention all-reduce with MLP computation for better latency.

        Args:
            hidden_states: [batch, seq_len, hidden_size] - REPLICATED
            attention_mask: Optional attention mask
            past_key_value: KV cache from previous steps
            use_cache: Whether to return updated cache

        Returns:
            hidden_states: [batch, seq_len, hidden_size] - REPLICATED
            cache: Updated KV cache if use_cache=True
        """
        # === ATTENTION PHASE ===
        attn_residual = hidden_states
        normed_hidden = self.input_layernorm(hidden_states)

        # Compute attention (QKV + attention + O_proj partial) without all-reduce
        attn_partial, cache = self.attention.forward_partial_with_cache(
            normed_hidden,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # Start async all-reduce for attention output
        mx.eval(attn_partial)  # Ensure computation is done before network op
        attn_ar_id = self.communicator.start_all_reduce(attn_partial, op="sum")

        # === MLP PHASE (overlapped with attention all-reduce) ===
        # While attention all-reduce is in flight, prepare MLP input
        # We need the attention output for the residual, so we'll compute
        # MLP on the pre-attention hidden states normalized differently
        #
        # However, the correct MLP input depends on attention output:
        #   mlp_input = post_attn_norm(attn_residual + attn_output)
        #
        # We CAN still overlap by computing MLP's column-parallel projections
        # (gate_proj, up_proj) which don't depend on the all-reduce result.
        # But the full pipeline requires attention output first.
        #
        # For now, wait for attention and overlap with NEXT layer instead.

        # Wait for attention all-reduce
        attn_output = await self.communicator.wait_all_reduce(attn_ar_id)

        # Add bias if needed (bias is added after all-reduce in row-parallel)
        attn_output = self.attention.o_proj.add_bias(attn_output)

        # Attention residual connection
        hidden_states = attn_residual + attn_output

        # === MLP with pipelined all-reduce ===
        mlp_residual = hidden_states
        normed_for_mlp = self.post_attention_layernorm(hidden_states)

        # Compute MLP partial (gate + up + activation + down_proj partial)
        mlp_partial = self.mlp.forward_sync_partial(normed_for_mlp)

        # Start async all-reduce for MLP output
        mx.eval(mlp_partial)
        mlp_ar_id = self.communicator.start_all_reduce(mlp_partial, op="sum")

        # Wait for MLP all-reduce
        mlp_output = await self.communicator.wait_all_reduce(mlp_ar_id)

        # Add bias if needed
        mlp_output = self.mlp.down_proj.add_bias(mlp_output)

        # MLP residual connection
        hidden_states = mlp_residual + mlp_output

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
        # First, complete the pending all-reduce from previous layer if any
        if pending_ar_id is not None:
            prev_output = await self.communicator.wait_all_reduce(pending_ar_id)
            prev_output = pending_ar_proj.add_bias(prev_output)
            hidden_states = pending_ar_residual + prev_output

        # === ATTENTION PHASE ===
        attn_residual = hidden_states
        normed_hidden = self.input_layernorm(hidden_states)

        # Compute attention partial
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
        use_pipelined: bool = True,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        """
        Forward pass through the full model.

        Args:
            input_ids: Token IDs [batch, seq_len] - REPLICATED
            attention_mask: Optional attention mask
            past_key_values: KV cache from previous steps
            use_cache: Whether to return updated cache
            use_pipelined: Use pipelined communication (overlaps MLP all-reduce
                          with next layer's attention). Default True.

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

        if use_pipelined and len(self.layers) > 1:
            # === PIPELINED EXECUTION ===
            # Overlap layer N's MLP all-reduce with layer N+1's attention compute

            pending_ar_id = None
            pending_ar_residual = None
            pending_ar_proj = None

            for i, layer in enumerate(self.layers):
                past_kv = past_key_values[i] if past_key_values else None

                (
                    hidden_states,
                    layer_cache,
                    pending_ar_id,
                    pending_ar_residual,
                    pending_ar_proj,
                ) = await layer.forward_with_pending_ar(
                    hidden_states,
                    pending_ar_id,
                    pending_ar_residual,
                    pending_ar_proj,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

                # Evaluate periodically to prevent memory buildup
                if (i + 1) % eval_interval == 0:
                    mx.eval(hidden_states)

                if use_cache and layer_cache is not None:
                    new_cache.append(layer_cache)

            # Complete the final layer's pending MLP all-reduce
            if pending_ar_id is not None:
                final_mlp = await self.communicator.wait_all_reduce(pending_ar_id)
                final_mlp = pending_ar_proj.add_bias(final_mlp)
                hidden_states = pending_ar_residual + final_mlp

            mx.eval(hidden_states)

        else:
            # === SEQUENTIAL EXECUTION (fallback) ===
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
