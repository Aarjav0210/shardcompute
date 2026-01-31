"""Quantized parallel linear layers for tensor parallelism.

These layers combine MLX quantization with tensor parallelism for maximum
performance. They use quantized_matmul for efficient inference while
maintaining proper sharding across workers.
"""

import mlx.core as mx
from typing import Optional, Tuple
import logging

from shardcompute.collectives.communicator import Communicator

logger = logging.getLogger(__name__)


class QuantizedColumnParallelLinear:
    """
    Column-parallel linear layer with quantized weights.
    
    Combines the benefits of:
    - Tensor parallelism: Each worker holds out_features // world_size columns
    - Quantization: 4-bit or 8-bit weights for memory/speed efficiency
    
    Weight sharding:
    - Full weight shape: [in_features, out_features]
    - Local weight shape: [in_features, out_features // world_size]
    - Each shard is quantized independently
    
    Communication:
    - Forward: No communication required (same as non-quantized)
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
        group_size: int = 64,
        bits: int = 4,
    ):
        """
        Initialize QuantizedColumnParallelLinear.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension (full, before sharding)
            world_size: Number of workers in tensor parallel group
            rank: This worker's rank
            bias: Whether to use bias
            gather_output: Whether to all-gather output
            communicator: Communicator for all-gather
            group_size: Quantization group size
            bits: Quantization bit width (4 or 8)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        self.communicator = communicator
        self.group_size = group_size
        self.bits = bits
        
        # Validate dimensions
        if out_features % world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by world_size ({world_size})"
            )
        
        # Calculate local dimensions
        self.local_out_features = out_features // world_size
        self.col_start = rank * self.local_out_features
        self.col_end = self.col_start + self.local_out_features
        
        # Quantized weight components (will be loaded from checkpoint)
        self.quantized_weight: Optional[mx.array] = None
        self.scales: Optional[mx.array] = None
        self.biases_quant: Optional[mx.array] = None  # Quantization biases
        
        # Linear layer bias
        self.bias: Optional[mx.array] = None
        self.has_bias = bias
        
        logger.debug(
            f"QuantizedColumnParallelLinear rank {rank}: "
            f"[{in_features}, {out_features}] -> local [{in_features}, {self.local_out_features}], "
            f"{bits}-bit quantized"
        )
    
    def load_shard_and_quantize(
        self,
        weight_shard: mx.array,
        bias_shard: Optional[mx.array] = None,
    ):
        """
        Load pre-sharded full-precision weights and quantize them.
        
        Args:
            weight_shard: Pre-sharded weight [in_features, local_out_features]
            bias_shard: Pre-sharded bias [local_out_features] or None
        """
        expected_shape = (self.in_features, self.local_out_features)
        if weight_shard.shape != expected_shape:
            raise ValueError(
                f"Weight shard shape {weight_shard.shape} doesn't match "
                f"expected {expected_shape}"
            )
        
        # Quantize the shard
        self.quantized_weight, self.scales, self.biases_quant = mx.quantize(
            weight_shard.astype(mx.float32),
            group_size=self.group_size,
            bits=self.bits,
        )
        
        if self.has_bias and bias_shard is not None:
            self.bias = bias_shard
        
        logger.debug(f"Rank {self.rank} quantized weight shard: {weight_shard.shape} -> {self.bits}-bit")
    
    def load_quantized_shard(
        self,
        quantized_weight: mx.array,
        scales: mx.array,
        biases: mx.array,
        linear_bias: Optional[mx.array] = None,
    ):
        """
        Load pre-quantized sharded weights directly.
        
        Use this when weights are pre-quantized during model preparation.
        
        Args:
            quantized_weight: Pre-quantized weight shard
            scales: Scale factors for quantization
            biases: Bias values for quantization
            linear_bias: Linear layer bias
        """
        self.quantized_weight = quantized_weight
        self.scales = scales
        self.biases_quant = biases

        if self.has_bias and linear_bias is not None:
            self.bias = linear_bias
    
    async def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass with quantized column-parallel weight.
        
        Args:
            x: Input tensor [batch, seq_len, in_features] - REPLICATED across workers
            
        Returns:
            Output tensor:
            - If gather_output=False: [batch, seq_len, local_out_features] - PARTITIONED
            - If gather_output=True: [batch, seq_len, out_features] - REPLICATED
        """
        if self.quantized_weight is None:
            raise RuntimeError("Weight not loaded. Call load_shard_and_quantize() first.")
        
        # Quantized matmul: [batch, seq, in_features] @ quantized[in_features, local_out_features]
        output = mx.quantized_matmul(
            x,
            self.quantized_weight,
            self.scales,
            self.biases_quant,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        
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
        such as when followed by QuantizedRowParallelLinear.
        
        Args:
            x: Input tensor [batch, seq_len, in_features] - REPLICATED
            
        Returns:
            Output tensor [batch, seq_len, local_out_features] - PARTITIONED
        """
        if self.quantized_weight is None:
            raise RuntimeError("Weight not loaded")
        
        output = mx.quantized_matmul(
            x,
            self.quantized_weight,
            self.scales,
            self.biases_quant,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        
        if self.has_bias and self.bias is not None:
            output = output + self.bias
        
        return output
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters in this shard (full precision equivalent)."""
        params = self.in_features * self.local_out_features
        if self.has_bias:
            params += self.local_out_features
        return params
    
    @property
    def memory_bytes(self) -> int:
        """Actual memory usage in bytes."""
        if self.quantized_weight is None:
            return 0
        total = (
            self.quantized_weight.nbytes +
            self.scales.nbytes +
            self.biases_quant.nbytes
        )
        if self.bias is not None:
            total += self.bias.nbytes
        return total
    
    def __repr__(self) -> str:
        return (
            f"QuantizedColumnParallelLinear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"local_out_features={self.local_out_features}, "
            f"rank={self.rank}/{self.world_size}, "
            f"bits={self.bits})"
        )


class QuantizedRowParallelLinear:
    """
    Row-parallel linear layer with quantized weights.
    
    Combines the benefits of:
    - Tensor parallelism: Each worker holds in_features // world_size rows
    - Quantization: 4-bit or 8-bit weights for memory/speed efficiency
    
    Weight sharding:
    - Full weight shape: [in_features, out_features]
    - Local weight shape: [in_features // world_size, out_features]
    - Each shard is quantized independently
    
    Communication:
    - Forward: All-reduce to combine partial outputs
    - Bias is only added on rank 0 to avoid double-counting
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
        group_size: int = 64,
        bits: int = 4,
    ):
        """
        Initialize QuantizedRowParallelLinear.
        
        Args:
            in_features: Input dimension (full, before sharding)
            out_features: Output dimension
            world_size: Number of workers in tensor parallel group
            rank: This worker's rank
            bias: Whether to use bias
            input_is_partitioned: Whether input is already partitioned
            communicator: Communicator for all-reduce
            group_size: Quantization group size
            bits: Quantization bit width (4 or 8)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.input_is_partitioned = input_is_partitioned
        self.communicator = communicator
        self.group_size = group_size
        self.bits = bits
        
        # Validate dimensions
        if in_features % world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by world_size ({world_size})"
            )
        
        # Calculate local dimensions
        self.local_in_features = in_features // world_size
        self.row_start = rank * self.local_in_features
        self.row_end = self.row_start + self.local_in_features
        
        # Quantized weight components
        self.quantized_weight: Optional[mx.array] = None
        self.scales: Optional[mx.array] = None
        self.biases_quant: Optional[mx.array] = None
        
        # Linear layer bias (only on rank 0)
        self.bias: Optional[mx.array] = None
        self.has_bias = bias
        
        logger.debug(
            f"QuantizedRowParallelLinear rank {rank}: "
            f"[{in_features}, {out_features}] -> local [{self.local_in_features}, {out_features}], "
            f"{bits}-bit quantized"
        )
    
    def load_shard_and_quantize(
        self,
        weight_shard: mx.array,
        bias_shard: Optional[mx.array] = None,
    ):
        """
        Load pre-sharded full-precision weights and quantize them.
        
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
        
        # Quantize the shard
        self.quantized_weight, self.scales, self.biases_quant = mx.quantize(
            weight_shard.astype(mx.float32),
            group_size=self.group_size,
            bits=self.bits,
        )
        
        # Only rank 0 gets the linear bias
        if self.has_bias and bias_shard is not None and self.rank == 0:
            self.bias = bias_shard
        
        logger.debug(f"Rank {self.rank} quantized weight shard: {weight_shard.shape} -> {self.bits}-bit")
    
    def load_quantized_shard(
        self,
        quantized_weight: mx.array,
        scales: mx.array,
        biases: mx.array,
        linear_bias: Optional[mx.array] = None,
    ):
        """
        Load pre-quantized sharded weights directly.
        
        Args:
            quantized_weight: Pre-quantized weight shard
            scales: Scale factors
            biases: Quantization biases
            linear_bias: Linear layer bias (only used on rank 0)
        """
        self.quantized_weight = quantized_weight
        self.scales = scales
        self.biases_quant = biases

        # Some pre-sharded quantized models keep full scales/biases for row-parallel
        # weights. Slice them to the local input shard if needed.
        if (
            self.quantized_weight is not None
            and self.scales is not None
            and len(self.scales.shape) == 2
            and self.quantized_weight.shape[0] == self.out_features
        ):
            pack_factor = 32 // self.bits
            input_shard = self.quantized_weight.shape[1] * pack_factor
            if input_shard % self.group_size == 0:
                expected_groups = input_shard // self.group_size
                full_groups = self.scales.shape[1]
                if full_groups == expected_groups * self.world_size:
                    start = self.rank * expected_groups
                    end = start + expected_groups
                    self.scales = self.scales[:, start:end]
                    if self.biases_quant is not None and len(self.biases_quant.shape) == 2:
                        self.biases_quant = self.biases_quant[:, start:end]
                elif full_groups != expected_groups:
                    logger.warning(
                        f"QuantizedRowParallelLinear scales shape {self.scales.shape} "
                        f"does not match expected groups {expected_groups}; "
                        "leaving scales/biases unchanged"
                    )
            else:
                logger.warning(
                    f"QuantizedRowParallelLinear input shard {input_shard} not divisible by "
                    f"group_size {self.group_size}; leaving scales/biases unchanged"
                )
        
        if self.has_bias and linear_bias is not None and self.rank == 0:
            self.bias = linear_bias
    
    async def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass with quantized row-parallel weight.
        
        Args:
            x: Input tensor - PARTITIONED if input_is_partitioned=True
               Shape: [batch, seq_len, local_in_features]
               
        Returns:
            Output tensor [batch, seq_len, out_features] - REPLICATED
            (same value on all workers after all-reduce)
        """
        if self.quantized_weight is None:
            raise RuntimeError("Weight not loaded")
        
        # Quantized matmul produces partial sum
        output_partial = mx.quantized_matmul(
            x,
            self.quantized_weight,
            self.scales,
            self.biases_quant,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        
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
        
        Useful when you want to manually control when all-reduce happens.
        
        Args:
            x: Partitioned input [batch, seq, local_in_features]
            
        Returns:
            Partial output [batch, seq, out_features] - NOT YET REDUCED
        """
        if self.quantized_weight is None:
            raise RuntimeError("Weight not loaded")
        
        return mx.quantized_matmul(
            x,
            self.quantized_weight,
            self.scales,
            self.biases_quant,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
    
    def add_bias(self, x: mx.array) -> mx.array:
        """Add bias to reduced output (call only after all-reduce)."""
        if self.has_bias and self.bias is not None:
            return x + self.bias
        return x
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters in this shard (full precision equivalent)."""
        params = self.local_in_features * self.out_features
        if self.has_bias and self.rank == 0:
            params += self.out_features
        return params
    
    @property
    def memory_bytes(self) -> int:
        """Actual memory usage in bytes."""
        if self.quantized_weight is None:
            return 0
        total = (
            self.quantized_weight.nbytes +
            self.scales.nbytes +
            self.biases_quant.nbytes
        )
        if self.bias is not None:
            total += self.bias.nbytes
        return total
    
    def __repr__(self) -> str:
        return (
            f"QuantizedRowParallelLinear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"local_in_features={self.local_in_features}, "
            f"rank={self.rank}/{self.world_size}, "
            f"bits={self.bits})"
        )
