"""Parallel MLP (feed-forward network) for tensor parallelism."""

import mlx.core as mx
from typing import Optional
import logging

from shardcompute.collectives.communicator import Communicator
from shardcompute.parallel.column_linear import ColumnParallelLinear
from shardcompute.parallel.row_linear import RowParallelLinear
from shardcompute.parallel.quantized_linear import (
    QuantizedColumnParallelLinear,
    QuantizedRowParallelLinear,
)

logger = logging.getLogger(__name__)


class ParallelMLP:
    """
    Feed-forward network with tensor parallelism.
    
    Standard MLP structure:
        hidden -> up_proj (ColPar) -> activation -> down_proj (RowPar) -> hidden
    
    LLaMA-style gated MLP (SwiGLU):
        hidden -> gate_proj (ColPar) ----+
                                         |-> gate * up -> down_proj (RowPar) -> hidden
        hidden -> up_proj (ColPar) ------+
    
    Communication pattern:
    - Up projection: No communication (column parallel)
    - Activation: No communication (applied locally)
    - Down projection: All-reduce to combine partials
    
    Total: ONE all-reduce per MLP block.
    
    This is the Megatron-style communication reduction - the partitioned
    output of column-parallel perfectly matches the partitioned input
    expected by row-parallel.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        world_size: int,
        rank: int,
        communicator: Communicator,
        activation: str = "silu",
        bias: bool = False,
        use_gated: bool = True,  # LLaMA uses gated MLP
        use_quantized: bool = False,
        quantization_bits: int = 4,
        quantization_group_size: int = 64,
    ):
        """
        Initialize ParallelMLP.

        Args:
            hidden_size: Model hidden dimension
            intermediate_size: MLP intermediate dimension
            world_size: Number of workers
            rank: This worker's rank
            communicator: Communicator for all-reduce
            activation: Activation function ('silu', 'gelu', 'relu')
            bias: Whether to use bias
            use_gated: Whether to use gated MLP (LLaMA style)
            use_quantized: Whether to use quantized linear layers
            quantization_bits: Bit width for quantization (4 or 8)
            quantization_group_size: Group size for quantization
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.world_size = world_size
        self.rank = rank
        self.communicator = communicator
        self.use_gated = use_gated
        self.use_quantized = use_quantized

        # Validate dimensions
        if intermediate_size % world_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by world_size"
            )

        self.local_intermediate = intermediate_size // world_size

        if use_quantized:
            # Quantized projections
            self.up_proj = QuantizedColumnParallelLinear(
                in_features=hidden_size,
                out_features=intermediate_size,
                world_size=world_size,
                rank=rank,
                bias=bias,
                gather_output=False,
                bits=quantization_bits,
                group_size=quantization_group_size,
            )

            if use_gated:
                self.gate_proj = QuantizedColumnParallelLinear(
                    in_features=hidden_size,
                    out_features=intermediate_size,
                    world_size=world_size,
                    rank=rank,
                    bias=bias,
                    gather_output=False,
                    bits=quantization_bits,
                    group_size=quantization_group_size,
                )
            else:
                self.gate_proj = None

            self.down_proj = QuantizedRowParallelLinear(
                in_features=intermediate_size,
                out_features=hidden_size,
                world_size=world_size,
                rank=rank,
                bias=bias,
                input_is_partitioned=True,
                communicator=communicator,
                bits=quantization_bits,
                group_size=quantization_group_size,
            )
        else:
            # Standard (non-quantized) projections
            self.up_proj = ColumnParallelLinear(
                in_features=hidden_size,
                out_features=intermediate_size,
                world_size=world_size,
                rank=rank,
                bias=bias,
                gather_output=False,
            )

            if use_gated:
                self.gate_proj = ColumnParallelLinear(
                    in_features=hidden_size,
                    out_features=intermediate_size,
                    world_size=world_size,
                    rank=rank,
                    bias=bias,
                    gather_output=False,
                )
            else:
                self.gate_proj = None

            self.down_proj = RowParallelLinear(
                in_features=intermediate_size,
                out_features=hidden_size,
                world_size=world_size,
                rank=rank,
                bias=bias,
                input_is_partitioned=True,
                communicator=communicator,
            )
        
        # Activation function
        self.activation_name = activation
        if activation == "silu" or activation == "swish":
            self.activation = lambda x: x * mx.sigmoid(x)
        elif activation == "gelu":
            # Approximate GeLU (faster than exact)
            self.activation = lambda x: x * mx.sigmoid(1.702 * x)
        elif activation == "gelu_new":
            # GELU approximation used by Phi-2 and GPT-2
            self.activation = lambda x: 0.5 * x * (1.0 + mx.tanh(
                0.7978845608028654 * (x + 0.044715 * x * x * x)
            ))
        elif activation == "relu":
            self.activation = lambda x: mx.maximum(x, 0)
        elif activation == "gelu_exact":
            # Exact GeLU
            import math
            self.activation = lambda x: 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        logger.debug(
            f"ParallelMLP rank {rank}: "
            f"[{hidden_size} -> {intermediate_size} -> {hidden_size}], "
            f"local intermediate: {self.local_intermediate}, "
            f"gated: {use_gated}"
        )
    
    def load_shards(
        self,
        up_weight: mx.array,
        up_bias: Optional[mx.array],
        down_weight: mx.array,
        down_bias: Optional[mx.array],
        gate_weight: Optional[mx.array] = None,
        gate_bias: Optional[mx.array] = None,
    ):
        """Load pre-sharded weights."""
        self.up_proj.load_shard_direct(up_weight, up_bias)
        self.down_proj.load_shard_direct(down_weight, down_bias)

        if self.use_gated and gate_weight is not None:
            self.gate_proj.load_shard_direct(gate_weight, gate_bias)

    def load_quantized_shards(
        self,
        up_weight: mx.array,
        up_scales: mx.array,
        up_biases: mx.array,
        up_linear_bias: Optional[mx.array],
        down_weight: mx.array,
        down_scales: mx.array,
        down_biases: mx.array,
        down_linear_bias: Optional[mx.array],
        gate_weight: Optional[mx.array] = None,
        gate_scales: Optional[mx.array] = None,
        gate_biases: Optional[mx.array] = None,
        gate_linear_bias: Optional[mx.array] = None,
    ):
        """Load pre-quantized sharded weights."""
        if not self.use_quantized:
            raise RuntimeError("Cannot load quantized shards into non-quantized MLP")

        self.up_proj.load_quantized_shard(up_weight, up_scales, up_biases, up_linear_bias)
        self.down_proj.load_quantized_shard(down_weight, down_scales, down_biases, down_linear_bias)

        if self.use_gated and gate_weight is not None:
            self.gate_proj.load_quantized_shard(gate_weight, gate_scales, gate_biases, gate_linear_bias)
    
    async def forward(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass with tensor-parallel MLP.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size] - REPLICATED
            
        Returns:
            [batch, seq_len, hidden_size] - REPLICATED (after all-reduce)
        """
        if self.use_gated:
            # Gated MLP (LLaMA style): gate * activation(up)
            # Both projections run in parallel (column parallel, no comm)
            gate = self.gate_proj.forward_sync(hidden_states)
            up = self.up_proj.forward_sync(hidden_states)
            
            # Apply activation to gate, multiply with up
            intermediate = self.activation(gate) * up
        else:
            # Standard MLP: activation(up)
            intermediate = self.up_proj.forward_sync(hidden_states)
            intermediate = self.activation(intermediate)
        
        # Row-parallel down projection with all-reduce
        output = await self.down_proj.forward(intermediate)
        
        return output
    
    def forward_sync_partial(self, hidden_states: mx.array) -> mx.array:
        """
        Compute MLP without all-reduce (for fused operations).
        
        Returns partial output that needs all-reduce.
        """
        if self.use_gated:
            gate = self.gate_proj.forward_sync(hidden_states)
            up = self.up_proj.forward_sync(hidden_states)
            intermediate = self.activation(gate) * up
        else:
            intermediate = self.up_proj.forward_sync(hidden_states)
            intermediate = self.activation(intermediate)
        
        # Return partial (before all-reduce)
        return self.down_proj.forward_partial(intermediate)
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters in this shard."""
        params = self.up_proj.num_parameters + self.down_proj.num_parameters
        if self.gate_proj is not None:
            params += self.gate_proj.num_parameters
        return params


class FusedAttentionMLP:
    """
    Fused attention + MLP with combined all-reduce.
    
    Instead of two separate all-reduces (one for attention, one for MLP),
    we can fuse them by summing the partials before the all-reduce.
    
    This reduces communication from 2 all-reduces to 1 per transformer block.
    
    Note: This changes the computation order slightly (attention + MLP partial
    are summed before reduction), which may affect numerical precision.
    
    NOT IMPLEMENTED IN POC - included for future optimization.
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FusedAttentionMLP not implemented in POC. "
            "Use separate ParallelAttention and ParallelMLP."
        )
