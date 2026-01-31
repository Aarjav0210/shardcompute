"""Parallel multi-head attention for tensor parallelism."""

import mlx.core as mx
import math
from typing import Optional, Tuple
import logging

from shardcompute.collectives.communicator import Communicator
from shardcompute.parallel.column_linear import ColumnParallelLinear
from shardcompute.parallel.row_linear import RowParallelLinear

logger = logging.getLogger(__name__)


class RoPE:
    """
    Rotary Position Embeddings (RoPE) for LLaMA-style models.
    
    RoPE encodes position information by rotating query and key vectors
    in 2D subspaces. This allows the model to learn relative positions
    through the rotation angles.
    
    Key properties:
    - Position information is encoded in the angle of rotation
    - Relative positions are captured by the dot product of rotated vectors
    - Extrapolates well to longer sequences than seen during training
    """
    
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        """
        Initialize RoPE.
        
        Args:
            head_dim: Dimension per attention head
            max_position_embeddings: Maximum sequence length
            base: Base for the frequency computation
            scaling_factor: Scaling factor for extended context (e.g., for NTK-aware scaling)
        """
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Precompute inverse frequencies
        # theta_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        self.inv_freq = inv_freq
        
        # Cache for cos/sin values
        self._cos_cache: Optional[mx.array] = None
        self._sin_cache: Optional[mx.array] = None
        self._cache_seq_len: int = 0
    
    def _update_cache(self, seq_len: int):
        """Update the cos/sin cache if needed."""
        if seq_len <= self._cache_seq_len:
            return
        
        # Extend cache to handle this sequence length (with some buffer)
        new_cache_len = max(seq_len, self._cache_seq_len * 2, 256)
        
        # Position indices
        t = mx.arange(new_cache_len, dtype=mx.float32)
        
        # Apply scaling factor for extended context
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor
        
        # Compute frequencies: [seq_len, head_dim/2]
        freqs = mx.outer(t, self.inv_freq)
        
        # Compute cos and sin: [seq_len, head_dim/2]
        # Then repeat to get [seq_len, head_dim]
        cos_vals = mx.cos(freqs)
        sin_vals = mx.sin(freqs)
        
        # Repeat for both halves of head_dim
        self._cos_cache = mx.concatenate([cos_vals, cos_vals], axis=-1)
        self._sin_cache = mx.concatenate([sin_vals, sin_vals], axis=-1)
        self._cache_seq_len = new_cache_len
    
    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        position_offset: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """
        Apply rotary position embeddings to query and key.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            position_offset: Offset for positions (used with KV cache)
            
        Returns:
            Rotated (q, k) tuple
        """
        seq_len = q.shape[2]
        
        # Update cache if needed
        self._update_cache(position_offset + seq_len)
        
        # Get cos/sin for current positions
        cos = self._cos_cache[position_offset:position_offset + seq_len]
        sin = self._sin_cache[position_offset:position_offset + seq_len]
        
        # Reshape for broadcasting: [1, 1, seq_len, head_dim]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        
        # Apply rotation
        q_rotated = self._rotate(q, cos, sin)
        k_rotated = self._rotate(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _rotate(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        """
        Apply rotation using the "rotate half" method.
        
        For each pair of dimensions (x0, x1), compute:
            x0' = x0 * cos - x1 * sin
            x1' = x0 * sin + x1 * cos
        """
        # Split into two halves
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        
        # Create rotated version: [-x2, x1]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        
        # Apply rotation: x * cos + rotated * sin
        return x * cos + rotated * sin


class ParallelAttention:
    """
    Multi-head attention with heads distributed across workers.
    
    Each worker computes attention for num_heads // world_size heads.
    Uses column-parallel for QKV projection (split by heads),
    row-parallel for output projection (combines head outputs).
    
    Communication pattern:
    - QKV projection: No communication (column parallel)
    - Attention computation: No communication (local per-head)
    - Output projection: All-reduce to combine head outputs
    
    Total: ONE all-reduce per attention block.
    
    For LLaMA-style attention:
    - Separate Q, K, V projections
    - RoPE positional embeddings (applied after projection)
    - Optional KV cache for generation
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        world_size: int,
        rank: int,
        communicator: Communicator,
        num_kv_heads: Optional[int] = None,  # For grouped-query attention
        head_dim: Optional[int] = None,
        bias: bool = False,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling_factor: float = 1.0,
    ):
        """
        Initialize ParallelAttention.
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Total number of attention heads
            world_size: Number of workers
            rank: This worker's rank
            communicator: Communicator for all-reduce
            num_kv_heads: Number of KV heads (for grouped-query attention)
            head_dim: Dimension per head (default: hidden_size // num_heads)
            bias: Whether to use bias in projections
            attention_dropout: Dropout probability (not used in inference)
            max_position_embeddings: Maximum sequence length for RoPE
            rope_base: Base frequency for RoPE
            rope_scaling_factor: Scaling factor for extended context
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        self.communicator = communicator
        
        # Validate head distribution
        if num_heads % world_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            )
        
        self.local_num_heads = num_heads // world_size
        self.head_dim = head_dim if head_dim else hidden_size // num_heads
        self.local_hidden = self.local_num_heads * self.head_dim
        
        # Handle grouped-query attention (GQA)
        # In GQA, num_kv_heads < num_heads, multiple Q heads share same KV
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        if self.num_kv_heads % world_size != 0:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be divisible by world_size"
            )
        self.local_num_kv_heads = self.num_kv_heads // world_size
        self.local_kv_hidden = self.local_num_kv_heads * self.head_dim
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize RoPE for positional embeddings
        self.rope = RoPE(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
            scaling_factor=rope_scaling_factor,
        )
        
        # Q projection: column parallel
        # Full: [hidden_size, num_heads * head_dim]
        # Local: [hidden_size, local_num_heads * head_dim]
        self.q_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=num_heads * self.head_dim,
            world_size=world_size,
            rank=rank,
            bias=bias,
            gather_output=False,
        )
        
        # K projection: column parallel (may be smaller for GQA)
        self.k_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            world_size=world_size,
            rank=rank,
            bias=bias,
            gather_output=False,
        )
        
        # V projection: column parallel
        self.v_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            world_size=world_size,
            rank=rank,
            bias=bias,
            gather_output=False,
        )
        
        # Output projection: row parallel (all-reduce at end)
        self.o_proj = RowParallelLinear(
            in_features=num_heads * self.head_dim,
            out_features=hidden_size,
            world_size=world_size,
            rank=rank,
            bias=bias,
            input_is_partitioned=True,
            communicator=communicator,
        )
        
        logger.debug(
            f"ParallelAttention rank {rank}: "
            f"{self.local_num_heads}/{num_heads} heads, "
            f"{self.local_num_kv_heads}/{self.num_kv_heads} KV heads"
        )
    
    def load_shards(
        self,
        q_weight: mx.array,
        q_bias: Optional[mx.array],
        k_weight: mx.array,
        k_bias: Optional[mx.array],
        v_weight: mx.array,
        v_bias: Optional[mx.array],
        o_weight: mx.array,
        o_bias: Optional[mx.array],
    ):
        """Load pre-sharded weights for all projections."""
        self.q_proj.load_shard_direct(q_weight, q_bias)
        self.k_proj.load_shard_direct(k_weight, k_bias)
        self.v_proj.load_shard_direct(v_weight, v_bias)
        self.o_proj.load_shard_direct(o_weight, o_bias)
    
    async def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Forward pass with tensor-parallel attention.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size] - REPLICATED
            attention_mask: Optional causal/padding mask
            position_ids: Position IDs for RoPE (if used)
            past_key_value: Cached K, V from previous steps
            use_cache: Whether to return updated cache
            
        Returns:
            output: [batch, seq_len, hidden_size] - REPLICATED (after all-reduce)
            cache: Updated (K, V) cache if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Column-parallel QKV projections (no communication)
        q = self.q_proj.forward_sync(hidden_states)  # [batch, seq, local_hidden]
        k = self.k_proj.forward_sync(hidden_states)  # [batch, seq, local_kv_hidden]
        v = self.v_proj.forward_sync(hidden_states)  # [batch, seq, local_kv_hidden]
        
        # Reshape for multi-head attention
        # Q: [batch, seq, local_num_heads, head_dim] -> [batch, local_num_heads, seq, head_dim]
        q = q.reshape(batch_size, seq_len, self.local_num_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        
        k = k.reshape(batch_size, seq_len, self.local_num_kv_heads, self.head_dim)
        k = k.transpose(0, 2, 1, 3)
        
        v = v.reshape(batch_size, seq_len, self.local_num_kv_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)
        
        # Calculate position offset for RoPE (from KV cache)
        position_offset = 0
        if past_key_value is not None:
            position_offset = past_key_value[0].shape[2]
        
        # Apply RoPE to Q and K (before KV cache concatenation for K)
        q, k = self.rope(q, k, position_offset=position_offset)
        
        # Handle KV cache (after RoPE is applied)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        
        new_cache = (k, v) if use_cache else None
        kv_seq_len = k.shape[2]
        
        # Handle grouped-query attention: repeat K, V for Q heads
        if self.local_num_kv_heads < self.local_num_heads:
            repeat_factor = self.local_num_heads // self.local_num_kv_heads
            k = mx.repeat(k, repeat_factor, axis=1)
            v = mx.repeat(v, repeat_factor, axis=1)
        
        # Scaled dot-product attention
        # [batch, local_heads, seq, head_dim] @ [batch, local_heads, head_dim, kv_seq]
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Create causal mask if needed
        if seq_len > 1:
            causal_mask = mx.triu(
                mx.full((seq_len, kv_seq_len), float('-inf')),
                k=kv_seq_len - seq_len + 1
            )
            attn_scores = attn_scores + causal_mask
        
        # Softmax and attention output
        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_output = attn_weights @ v  # [batch, local_heads, seq, head_dim]
        
        # Reshape: [batch, seq, local_hidden]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.local_hidden)
        
        # Row-parallel output projection with all-reduce
        output = await self.o_proj.forward(attn_output)
        
        return output, new_cache
    
    def forward_sync_partial(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_offset: int = 0,
    ) -> mx.array:
        """
        Compute attention without all-reduce (for fused operations).
        
        Returns partial output that needs all-reduce.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj.forward_sync(hidden_states)
        k = self.k_proj.forward_sync(hidden_states)
        v = self.v_proj.forward_sync(hidden_states)
        
        # Reshape and transpose
        q = q.reshape(batch_size, seq_len, self.local_num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.local_num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.local_num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        q, k = self.rope(q, k, position_offset=position_offset)
        
        # Handle GQA
        if self.local_num_kv_heads < self.local_num_heads:
            repeat_factor = self.local_num_heads // self.local_num_kv_heads
            k = mx.repeat(k, repeat_factor, axis=1)
            v = mx.repeat(v, repeat_factor, axis=1)
        
        # Attention
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        if seq_len > 1:
            causal_mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
            attn_scores = attn_scores + causal_mask
        
        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_output = attn_weights @ v
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.local_hidden)
        
        # Return partial (before all-reduce)
        return self.o_proj.forward_partial(attn_output)
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters in this shard."""
        return (
            self.q_proj.num_parameters +
            self.k_proj.num_parameters +
            self.v_proj.num_parameters +
            self.o_proj.num_parameters
        )
