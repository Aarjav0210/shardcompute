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
    
    def load_shard_direct(self, weight_shard: mx.array):
        """
        Load pre-sharded embedding weights.
        
        Args:
            weight_shard: Pre-sharded weights [vocab_size, local_embedding_dim]
        """
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
