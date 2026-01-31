"""Row-parallel linear layer for tensor parallelism."""

import mlx.core as mx
from typing import Optional
import logging

from shardcompute.collectives.communicator import Communicator

logger = logging.getLogger(__name__)


class RowParallelLinear:
    """
    Linear layer with weights split by rows across workers.
    
    Full weight shape: [in_features, out_features]
    Local weight shape: [in_features // world_size, out_features]
    
    In row parallelism:
    - Each worker holds a horizontal slice of the weight matrix (rows)
    - Input X is partitioned across workers (each gets X_local)
    - Each worker computes Y_partial = X_local @ W_local
    - All-reduce sums partials to get full output: Y = sum(Y_partial)
    
    This is used for:
    - Second linear in MLP (down-projection)
    - Output projection in attention
    
    Communication:
    - Forward: All-reduce to combine partial outputs
    - Bias is only added on rank 0 to avoid double-counting
    
    Key insight (Megatron trick):
    When ColumnParallel is followed by RowParallel, the partitioned output
    of ColumnParallel naturally matches the partitioned input expected by
    RowParallel - no communication needed between them!
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        input_is_partitioned: bool = True,
        communicator: Optional[Communicator] = None,
    ):
        """
        Initialize RowParallelLinear.
        
        Args:
            in_features: Input dimension (full, before sharding)
            out_features: Output dimension
            world_size: Number of workers in tensor parallel group
            rank: This worker's rank
            bias: Whether to use bias
            input_is_partitioned: Whether input is already partitioned
                If True, expects X_local of shape [..., in_features // world_size]
                If False, input will be scattered internally (not implemented)
            communicator: Communicator for all-reduce
        """
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.input_is_partitioned = input_is_partitioned
        self.communicator = communicator
        
        # Validate dimensions
        if in_features % world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by world_size ({world_size})"
            )
        
        # Calculate local slice dimensions
        self.local_in_features = in_features // world_size
        self.row_start = rank * self.local_in_features
        self.row_end = self.row_start + self.local_in_features
        
        # Initialize weights (will be loaded from checkpoint)
        # Shape: [local_in_features, out_features]
        self.weight: Optional[mx.array] = None
        
        # Bias: only on rank 0 to avoid double-adding after all-reduce
        # Shape: [out_features] on rank 0, None on other ranks
        self.bias: Optional[mx.array] = None
        self.has_bias = bias
        
        logger.debug(
            f"RowParallelLinear rank {rank}: "
            f"[{in_features}, {out_features}] -> local [{self.local_in_features}, {out_features}]"
        )
    
    def load_shard(
        self,
        full_weight: mx.array,
        full_bias: Optional[mx.array] = None,
    ):
        """
        Load this worker's slice from full weight matrix.
        
        Args:
            full_weight: Full weight matrix [in_features, out_features]
            full_bias: Full bias vector [out_features] or None
        """
        # Extract row slice
        self.weight = full_weight[self.row_start:self.row_end, :]
        
        # Bias only on rank 0 to avoid double-counting after all-reduce
        if self.has_bias and full_bias is not None and self.rank == 0:
            self.bias = full_bias
        
        logger.debug(f"Rank {self.rank} loaded weight shard: {self.weight.shape}")
    
    def load_shard_direct(
        self,
        weight_shard: mx.array,
        bias_shard: Optional[mx.array] = None,
    ):
        """
        Load pre-sharded weights directly.
        
        Args:
            weight_shard: Pre-sharded weight [local_in_features, out_features]
            bias_shard: Bias [out_features], only used on rank 0
        """
        expected_shape = (self.local_in_features, self.out_features)
        if weight_shard.shape != expected_shape:
            raise ValueError(
                f"Weight shard shape {weight_shard.shape} doesn't match "
                f"expected {expected_shape}"
            )
        
        self.weight = weight_shard
        
        # Only rank 0 gets bias
        if self.has_bias and bias_shard is not None and self.rank == 0:
            self.bias = bias_shard
    
    async def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass with row-parallel weight.
        
        Args:
            x: Input tensor - PARTITIONED if input_is_partitioned=True
               Shape: [batch, seq_len, local_in_features]
               
        Returns:
            Output tensor [batch, seq_len, out_features] - REPLICATED
            (same value on all workers after all-reduce)
        """
        if self.weight is None:
            raise RuntimeError("Weight not loaded")
        
        # Local matmul produces partial sum
        # [batch, seq, local_in_features] @ [local_in_features, out_features]
        output_partial = x @ self.weight  # [batch, seq, out_features]
        
        # All-reduce to sum partials across workers
        if self.communicator is not None:
            output = await self.communicator.all_reduce(output_partial, op='sum')
        else:
            output = output_partial
        
        # Add bias after reduction (only rank 0 has bias)
        if self.has_bias and self.bias is not None:
            output = output + self.bias
        
        return output
    
    def forward_partial(self, x: mx.array) -> mx.array:
        """
        Compute partial output without all-reduce.
        
        Useful when you want to manually control when all-reduce happens,
        such as fusing multiple all-reduces.
        
        Args:
            x: Partitioned input [batch, seq, local_in_features]
            
        Returns:
            Partial output [batch, seq, out_features] - NOT YET REDUCED
        """
        if self.weight is None:
            raise RuntimeError("Weight not loaded")
        
        return x @ self.weight
    
    def add_bias(self, x: mx.array) -> mx.array:
        """Add bias to reduced output (call only after all-reduce)."""
        if self.has_bias and self.bias is not None:
            return x + self.bias
        return x
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters in this shard."""
        params = self.local_in_features * self.out_features
        if self.has_bias and self.rank == 0:
            params += self.out_features
        return params
    
    def __repr__(self) -> str:
        return (
            f"RowParallelLinear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"local_in_features={self.local_in_features}, "
            f"rank={self.rank}/{self.world_size}, "
            f"bias={self.has_bias})"
        )


class FusedColumnRowLinear:
    """
    Fused Column-Row linear for MLP pattern.
    
    Combines ColumnParallelLinear -> activation -> RowParallelLinear
    with a single all-reduce at the end.
    
    This is the Megatron-style MLP pattern that minimizes communication.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        activation: str = "gelu",
        bias: bool = True,
        communicator: Optional[Communicator] = None,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.communicator = communicator
        
        # Column parallel up-projection (no communication)
        self.up_proj = ColumnParallelLinear(
            in_features=in_features,
            out_features=hidden_features,
            world_size=world_size,
            rank=rank,
            bias=bias,
            gather_output=False,  # Don't gather - row parallel will consume partitioned
            communicator=None,  # No comm needed
        )
        
        # Row parallel down-projection (all-reduce at end)
        self.down_proj = RowParallelLinear(
            in_features=hidden_features,
            out_features=out_features,
            world_size=world_size,
            rank=rank,
            bias=bias,
            input_is_partitioned=True,
            communicator=communicator,
        )
        
        # Activation function
        self.activation_name = activation
        if activation == "gelu":
            self.activation = lambda x: x * mx.sigmoid(1.702 * x)  # Approximate GeLU
        elif activation == "silu" or activation == "swish":
            self.activation = lambda x: x * mx.sigmoid(x)
        elif activation == "relu":
            self.activation = lambda x: mx.maximum(x, 0)
        else:
            self.activation = lambda x: x  # Identity
    
    def load_shards(
        self,
        up_weight: mx.array,
        up_bias: Optional[mx.array],
        down_weight: mx.array,
        down_bias: Optional[mx.array],
    ):
        """Load pre-sharded weights for both projections."""
        self.up_proj.load_shard_direct(up_weight, up_bias)
        self.down_proj.load_shard_direct(down_weight, down_bias)
    
    async def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass: up_proj -> activation -> down_proj -> all_reduce.
        
        Args:
            x: Input [batch, seq, in_features] - REPLICATED
            
        Returns:
            Output [batch, seq, out_features] - REPLICATED
        """
        # Column-parallel up projection (no comm)
        hidden = self.up_proj.forward_sync(x)
        
        # Activation on partitioned data (no comm)
        hidden = self.activation(hidden)
        
        # Row-parallel down projection with all-reduce
        output = await self.down_proj.forward(hidden)
        
        return output
