"""Parallel embedding layer for tensor parallelism."""

import mlx.core as mx
from typing import Optional
import logging

from shardcompute.collectives.communicator import Communicator

logger = logging.getLogger(__name__)


class ParallelEmbedding:
    """
    Embedding layer split by embedding dimension (column parallel).
    
    Full embedding shape: [vocab_size, embedding_dim]
    Local embedding shape: [vocab_size, embedding_dim // world_size]
    
    Each worker holds a slice of the embedding dimension for ALL vocabulary tokens.
    After lookup, requires all-gather to reconstruct full embeddings.
    
    This is the simplest parallelism strategy for embeddings:
    - Every worker can look up any token
    - Output is partitioned by embedding dimension
    - Requires all-gather after lookup
    
    Alternative (not implemented): Vocabulary parallelism
    - Each worker holds embeddings for vocab_size // world_size tokens
    - Requires scatter of token IDs, all-reduce of embeddings
    - More complex but may be better for very large vocabularies
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        world_size: int,
        rank: int,
        gather_output: bool = True,
        communicator: Optional[Communicator] = None,
        padding_idx: Optional[int] = None,
    ):
        """
        Initialize ParallelEmbedding.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Full embedding dimension
            world_size: Number of workers
            rank: This worker's rank
            gather_output: Whether to all-gather output
            communicator: Communicator for all-gather
            padding_idx: Optional padding token index (embeddings will be zeros)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        self.communicator = communicator
        self.padding_idx = padding_idx
        
        # Validate dimensions
        if embedding_dim % world_size != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by world_size ({world_size})"
            )
        
        # Calculate local dimensions
        self.local_embedding_dim = embedding_dim // world_size
        self.col_start = rank * self.local_embedding_dim
        self.col_end = self.col_start + self.local_embedding_dim
        
        # Embedding weights: [vocab_size, local_embedding_dim]
        self.weight: Optional[mx.array] = None
        
        logger.debug(
            f"ParallelEmbedding rank {rank}: "
            f"[{vocab_size}, {embedding_dim}] -> local [{vocab_size}, {self.local_embedding_dim}]"
        )
    
    def load_shard(self, full_weight: mx.array):
        """
        Load this worker's slice from full embedding matrix.
        
        Args:
            full_weight: Full embedding matrix [vocab_size, embedding_dim]
        """
        self.weight = full_weight[:, self.col_start:self.col_end]
        logger.debug(f"Rank {self.rank} loaded embedding shard: {self.weight.shape}")
    
    def load_shard_direct(self, weight_shard: Optional[mx.array]):
        """
        Load pre-sharded embedding weights.
        
        Args:
            weight_shard: Pre-sharded weights [vocab_size, local_embedding_dim]
        """
        # Handle None weight gracefully
        if weight_shard is None:
            logger.warning("load_shard_direct called with None weight, skipping")
            return
            
        expected_shape = (self.vocab_size, self.local_embedding_dim)
        if weight_shard.shape != expected_shape:
            raise ValueError(
                f"Weight shard shape {weight_shard.shape} doesn't match "
                f"expected {expected_shape}"
            )
        self.weight = weight_shard
    
    async def forward(self, input_ids: mx.array) -> mx.array:
        """
        Forward pass: lookup embeddings and optionally gather.
        
        Args:
            input_ids: Token IDs [batch, seq_len] - REPLICATED across workers
            
        Returns:
            Embeddings:
            - If gather_output=False: [batch, seq_len, local_embedding_dim] - PARTITIONED
            - If gather_output=True: [batch, seq_len, embedding_dim] - REPLICATED
        """
        if self.weight is None:
            raise RuntimeError("Embedding weights not loaded")
        
        # Embedding lookup using take (equivalent to indexing)
        # input_ids: [batch, seq_len]
        # weight: [vocab_size, local_embedding_dim]
        # output: [batch, seq_len, local_embedding_dim]
        flat_ids = input_ids.reshape(-1)
        embeddings = mx.take(self.weight, flat_ids, axis=0)
        
        # Reshape back
        batch_size, seq_len = input_ids.shape
        embeddings = embeddings.reshape(batch_size, seq_len, self.local_embedding_dim)
        
        # Handle padding
        if self.padding_idx is not None:
            mask = (input_ids == self.padding_idx)[..., None]
            embeddings = mx.where(mask, mx.zeros_like(embeddings), embeddings)
        
        # Optionally gather to get full embeddings
        if self.gather_output:
            if self.communicator is None:
                raise RuntimeError("Communicator required for gather_output=True")
            embeddings = await self.communicator.all_gather(embeddings, dim=-1)
        
        return embeddings
    
    def forward_sync(self, input_ids: mx.array) -> mx.array:
        """
        Synchronous embedding lookup (no gathering).
        
        Args:
            input_ids: Token IDs [batch, seq_len] - REPLICATED
            
        Returns:
            Partial embeddings [batch, seq_len, local_embedding_dim] - PARTITIONED
        """
        if self.weight is None:
            raise RuntimeError("Embedding weights not loaded")
        
        flat_ids = input_ids.reshape(-1)
        embeddings = mx.take(self.weight, flat_ids, axis=0)
        
        batch_size, seq_len = input_ids.shape
        embeddings = embeddings.reshape(batch_size, seq_len, self.local_embedding_dim)
        
        if self.padding_idx is not None:
            mask = (input_ids == self.padding_idx)[..., None]
            embeddings = mx.where(mask, mx.zeros_like(embeddings), embeddings)
        
        return embeddings
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters in this shard."""
        return self.vocab_size * self.local_embedding_dim
    
    def __repr__(self) -> str:
        return (
            f"ParallelEmbedding("
            f"vocab_size={self.vocab_size}, "
            f"embedding_dim={self.embedding_dim}, "
            f"local_embedding_dim={self.local_embedding_dim}, "
            f"rank={self.rank}/{self.world_size})"
        )


class QuantizedParallelEmbedding:
    """
    Quantized embedding layer with vocabulary parallelism.

    For MLX pre-quantized models, embeddings are stored in quantized format
    and sharded by vocabulary dimension.

    Each worker holds embeddings for vocab_size // world_size tokens.
    Token IDs are mapped to local indices, lookup performed, then all-reduce
    combines results across workers.

    Weight shapes (for 4-bit, group_size=64):
    - quantized_weight: [local_vocab_size, embedding_dim // 8]  (8 values per uint32)
    - scales: [local_vocab_size, embedding_dim // group_size]
    - biases: [local_vocab_size, embedding_dim // group_size]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        world_size: int,
        rank: int,
        communicator: Optional[Communicator] = None,
        group_size: int = 64,
        bits: int = 4,
    ):
        """
        Initialize QuantizedParallelEmbedding.

        Args:
            vocab_size: Full vocabulary size
            embedding_dim: Embedding dimension
            world_size: Number of workers
            rank: This worker's rank
            communicator: Communicator for all-reduce
            group_size: Quantization group size
            bits: Quantization bit width (4 or 8)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.world_size = world_size
        self.rank = rank
        self.communicator = communicator
        self.group_size = group_size
        self.bits = bits

        # Validate dimensions
        if vocab_size % world_size != 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be divisible by world_size ({world_size})"
            )

        # Calculate local dimensions
        self.local_vocab_size = vocab_size // world_size
        self.vocab_start = rank * self.local_vocab_size
        self.vocab_end = self.vocab_start + self.local_vocab_size

        # Quantized weight components
        self.quantized_weight: Optional[mx.array] = None
        self.scales: Optional[mx.array] = None
        self.biases_quant: Optional[mx.array] = None

        logger.debug(
            f"QuantizedParallelEmbedding rank {rank}: "
            f"vocab [{self.vocab_start}:{self.vocab_end}] of {vocab_size}, "
            f"embedding_dim={embedding_dim}, {bits}-bit quantized"
        )

    def load_quantized_shard(
        self,
        quantized_weight: mx.array,
        scales: mx.array,
        biases: mx.array,
    ):
        """
        Load pre-quantized vocab-sharded embeddings.

        Args:
            quantized_weight: Quantized embedding weights [local_vocab, packed_dim]
            scales: Quantization scales [local_vocab, num_groups]
            biases: Quantization biases [local_vocab, num_groups]
        """
        self.quantized_weight = quantized_weight
        self.scales = scales
        self.biases_quant = biases

        logger.debug(
            f"Rank {self.rank} loaded quantized embedding: "
            f"weight={quantized_weight.shape}, scales={scales.shape}"
        )

    def load_shard_direct(self, weight_shard: Optional[mx.array]):
        """
        For compatibility - detect if weight is quantized and handle appropriately.

        If the weight shape indicates quantization (packed), we cannot use it directly.
        This method is here for API compatibility but will raise an error for quantized weights.
        """
        if weight_shard is None:
            logger.warning("load_shard_direct called with None weight, skipping")
            return

        # Check if this looks like a quantized weight (vocab is sharded, embedding dim is packed)
        expected_fp_shape = (self.local_vocab_size, self.embedding_dim)
        if weight_shard.shape != expected_fp_shape:
            raise ValueError(
                f"Weight shard shape {weight_shard.shape} doesn't match expected {expected_fp_shape}. "
                f"For quantized embeddings, use load_quantized_shard() with weight, scales, and biases."
            )

        # If it's the expected full-precision shape, quantize it
        self.quantized_weight, self.scales, self.biases_quant = mx.quantize(
            weight_shard.astype(mx.float32),
            group_size=self.group_size,
            bits=self.bits,
        )

    async def forward(self, input_ids: mx.array) -> mx.array:
        """
        Forward pass: lookup embeddings with vocab parallelism.

        Args:
            input_ids: Token IDs [batch, seq_len] - REPLICATED across workers

        Returns:
            Embeddings [batch, seq_len, embedding_dim] - REPLICATED
        """
        if self.quantized_weight is None:
            raise RuntimeError("Embedding weights not loaded")

        batch_size, seq_len = input_ids.shape

        # Create local indices (offset by vocab_start)
        local_ids = input_ids - self.vocab_start

        # Mask for tokens in this worker's vocab range
        in_range = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)

        # Clamp to valid range for lookup (out-of-range will be zeroed later)
        local_ids = mx.clip(local_ids, 0, self.local_vocab_size - 1)

        # Flatten for lookup
        flat_local_ids = local_ids.reshape(-1)

        # Dequantize embeddings for the needed indices
        # First, gather the quantized values
        flat_quant_weight = mx.take(self.quantized_weight, flat_local_ids, axis=0)
        flat_scales = mx.take(self.scales, flat_local_ids, axis=0)
        flat_biases = mx.take(self.biases_quant, flat_local_ids, axis=0)

        # Dequantize: output = (quantized - bias) * scale
        embeddings = mx.dequantize(
            flat_quant_weight,
            flat_scales,
            flat_biases,
            group_size=self.group_size,
            bits=self.bits,
        )

        # Reshape back to [batch, seq_len, embedding_dim]
        embeddings = embeddings.reshape(batch_size, seq_len, self.embedding_dim)

        # Zero out embeddings for tokens not in this worker's range
        in_range_expanded = in_range[..., None]  # [batch, seq_len, 1]
        embeddings = mx.where(in_range_expanded, embeddings, mx.zeros_like(embeddings))

        # All-reduce to combine embeddings from all workers
        if self.communicator is not None:
            embeddings = await self.communicator.all_reduce(embeddings, op='sum')

        return embeddings

    @property
    def weight(self) -> Optional[mx.array]:
        """Return quantized weight for compatibility."""
        return self.quantized_weight

    def dequantize_weight(self) -> mx.array:
        """Return dequantized embedding weights for this rank's vocab shard."""
        if self.quantized_weight is None or self.scales is None or self.biases_quant is None:
            raise RuntimeError("Quantized embedding weights not loaded")
        return mx.dequantize(
            self.quantized_weight,
            self.scales,
            self.biases_quant,
            group_size=self.group_size,
            bits=self.bits,
        )

    @property
    def num_parameters(self) -> int:
        """Number of parameters (full precision equivalent)."""
        return self.local_vocab_size * self.embedding_dim

    def __repr__(self) -> str:
        return (
            f"QuantizedParallelEmbedding("
            f"vocab_size={self.vocab_size}, "
            f"local_vocab_size={self.local_vocab_size}, "
            f"embedding_dim={self.embedding_dim}, "
            f"rank={self.rank}/{self.world_size}, "
            f"bits={self.bits})"
        )


class VocabParallelEmbedding:
    """
    Embedding layer split by vocabulary (alternative strategy).

    Each worker holds embeddings for vocab_size // world_size tokens.
    More complex but can be more efficient for very large vocabularies.

    NOT IMPLEMENTED IN POC - included for future reference.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        world_size: int,
        rank: int,
        communicator: Optional[Communicator] = None,
    ):
        raise NotImplementedError(
            "VocabParallelEmbedding not implemented in POC. "
            "Use ParallelEmbedding (column parallel) instead."
        )
