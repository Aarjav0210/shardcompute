"""Column-parallel linear layer for tensor parallelism."""

import mlx.core as mx
from typing import Optional
import logging

from shardcompute.collectives.communicator import Communicator

logger = logging.getLogger(__name__)


class ColumnParallelLinear:
    """
    Linear layer with weights split by columns across workers.
    
    Full weight shape: [in_features, out_features]
    Local weight shape: [in_features, out_features // world_size]
    
    In column parallelism:
    - Each worker holds a vertical slice of the weight matrix (columns)
    - Input X is replicated across all workers
    - Output Y is partitioned: Y_local = X @ W_local
    - Consumer must all-gather or use row-parallel to combine
    
    This is used for:
    - First linear in MLP (up-projection)
    - Q, K, V projections in attention (split by heads)
    - Embedding layer (vocabulary embedding)
    
    Communication:
    - Forward: No communication required
    - Output is partitioned and may need all-gather depending on use case
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        gather_output: bool = False,
        communicator: Optional[Communicator] = None,
    ):
        """
        Initialize ColumnParallelLinear.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension (full, before sharding)
            world_size: Number of workers in tensor parallel group
            rank: This worker's rank
            bias: Whether to use bias
            gather_output: Whether to all-gather output to produce full tensor
            communicator: Communicator for all-gather (required if gather_output=True)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        self.communicator = communicator
        
        # Validate dimensions
        if out_features % world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by world_size ({world_size})"
            )
        
        # Calculate local slice dimensions
        self.local_out_features = out_features // world_size
        self.col_start = rank * self.local_out_features
        self.col_end = self.col_start + self.local_out_features
        
        # Initialize weights (will be loaded from checkpoint)
        # Shape: [in_features, local_out_features]
        self.weight: Optional[mx.array] = None
        
        # Bias shape: [local_out_features] if bias else None
        self.bias: Optional[mx.array] = None
        self.has_bias = bias
        
        logger.debug(
            f"ColumnParallelLinear rank {rank}: "
            f"[{in_features}, {out_features}] -> local [{in_features}, {self.local_out_features}]"
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
        # Extract column slice
        self.weight = full_weight[:, self.col_start:self.col_end]
        
        if self.has_bias and full_bias is not None:
            self.bias = full_bias[self.col_start:self.col_end]
        
        logger.debug(f"Rank {self.rank} loaded weight shard: {self.weight.shape}")
    
    def load_shard_direct(
        self,
        weight_shard: Optional[mx.array],
        bias_shard: Optional[mx.array] = None,
    ):
        """
        Load pre-sharded weights directly.
        
        Use this when weights are pre-sharded during model preparation.
        
        Args:
            weight_shard: Pre-sharded weight [in_features, local_out_features]
            bias_shard: Pre-sharded bias [local_out_features] or None
        """
        # Handle None weight gracefully (skip loading)
        if weight_shard is None:
            logger.warning("load_shard_direct called with None weight, skipping")
            return
        
        expected_shape = (self.in_features, self.local_out_features)
        if weight_shard.shape != expected_shape:
            raise ValueError(
                f"Weight shard shape {weight_shard.shape} doesn't match "
                f"expected {expected_shape}"
            )
        
        self.weight = weight_shard
        
        if self.has_bias and bias_shard is not None:
            self.bias = bias_shard
    
    async def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass with column-parallel weight.
        
        Args:
            x: Input tensor [batch, seq_len, in_features] - REPLICATED across workers
            
        Returns:
            Output tensor:
            - If gather_output=False: [batch, seq_len, local_out_features] - PARTITIONED
            - If gather_output=True: [batch, seq_len, out_features] - REPLICATED
        """
        if self.weight is None:
            raise RuntimeError("Weight not loaded. Call load_shard() or load_shard_direct() first.")
        
        # Local matmul: [batch, seq, in_features] @ [in_features, local_out_features]
        output = x @ self.weight  # [batch, seq, local_out_features]
        
        # Add local bias
        if self.has_bias and self.bias is not None:
            output = output + self.bias
        
        # Optionally gather to produce full output
        if self.gather_output:
            if self.communicator is None:
                raise RuntimeError("Communicator required for gather_output=True")
            output = await self.communicator.all_gather(output, dim=-1)
        
        return output
    
    def forward_sync(self, x: mx.array) -> mx.array:
        """
        Synchronous forward pass (no gathering).
        
        Use this when you know the next layer will consume partitioned output,
        such as when followed by RowParallelLinear.
        
        Args:
            x: Input tensor [batch, seq_len, in_features] - REPLICATED
            
        Returns:
            Output tensor [batch, seq_len, local_out_features] - PARTITIONED
        """
        if self.weight is None:
            raise RuntimeError("Weight not loaded")
        
        output = x @ self.weight
        
        if self.has_bias and self.bias is not None:
            output = output + self.bias
        
        return output
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters in this shard."""
        params = self.in_features * self.local_out_features
        if self.has_bias:
            params += self.local_out_features
        return params
    
    def __repr__(self) -> str:
        return (
            f"ColumnParallelLinear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"local_out_features={self.local_out_features}, "
            f"rank={self.rank}/{self.world_size}, "
            f"bias={self.has_bias})"
        )
