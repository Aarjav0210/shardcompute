"""Tests for parallel layers."""

import pytest
import mlx.core as mx

from shardcompute.parallel.column_linear import ColumnParallelLinear
from shardcompute.parallel.row_linear import RowParallelLinear
from shardcompute.parallel.embedding import ParallelEmbedding


class TestColumnParallelLinear:
    """Tests for ColumnParallelLinear."""
    
    def test_shard_dimensions(self):
        """Test that shard dimensions are correct."""
        in_features = 256
        out_features = 512
        world_size = 2
        
        layer_0 = ColumnParallelLinear(in_features, out_features, world_size, rank=0)
        layer_1 = ColumnParallelLinear(in_features, out_features, world_size, rank=1)
        
        assert layer_0.local_out_features == out_features // world_size
        assert layer_1.local_out_features == out_features // world_size
        assert layer_0.col_start == 0
        assert layer_0.col_end == 256
        assert layer_1.col_start == 256
        assert layer_1.col_end == 512
    
    def test_load_shard(self):
        """Test loading sharded weights."""
        in_features = 256
        out_features = 512
        world_size = 2
        
        # Create full weight
        full_weight = mx.random.normal((in_features, out_features))
        full_bias = mx.random.normal((out_features,))
        
        # Create layers and load shards
        layer_0 = ColumnParallelLinear(in_features, out_features, world_size, rank=0)
        layer_1 = ColumnParallelLinear(in_features, out_features, world_size, rank=1)
        
        layer_0.load_shard(full_weight, full_bias)
        layer_1.load_shard(full_weight, full_bias)
        
        # Verify shapes
        assert layer_0.weight.shape == (in_features, out_features // 2)
        assert layer_1.weight.shape == (in_features, out_features // 2)
        
        # Verify content
        assert mx.allclose(layer_0.weight, full_weight[:, :256]).item()
        assert mx.allclose(layer_1.weight, full_weight[:, 256:]).item()
    
    def test_forward_sync(self):
        """Test synchronous forward pass."""
        in_features = 256
        out_features = 512
        world_size = 2
        batch_size = 2
        seq_len = 4
        
        # Create input and full weight
        x = mx.random.normal((batch_size, seq_len, in_features))
        full_weight = mx.random.normal((in_features, out_features))
        full_bias = mx.random.normal((out_features,))
        
        # Full computation
        full_output = x @ full_weight + full_bias
        
        # Parallel computation
        layer_0 = ColumnParallelLinear(in_features, out_features, world_size, rank=0)
        layer_1 = ColumnParallelLinear(in_features, out_features, world_size, rank=1)
        
        layer_0.load_shard(full_weight, full_bias)
        layer_1.load_shard(full_weight, full_bias)
        
        output_0 = layer_0.forward_sync(x)
        output_1 = layer_1.forward_sync(x)
        
        # Concatenate outputs (simulating all-gather)
        parallel_output = mx.concatenate([output_0, output_1], axis=-1)
        
        # Compare
        assert mx.allclose(parallel_output, full_output, atol=1e-5).item()


class TestRowParallelLinear:
    """Tests for RowParallelLinear."""
    
    def test_shard_dimensions(self):
        """Test that shard dimensions are correct."""
        in_features = 512
        out_features = 256
        world_size = 2
        
        layer_0 = RowParallelLinear(in_features, out_features, world_size, rank=0)
        layer_1 = RowParallelLinear(in_features, out_features, world_size, rank=1)
        
        assert layer_0.local_in_features == in_features // world_size
        assert layer_1.local_in_features == in_features // world_size
    
    def test_forward_partial(self):
        """Test forward pass produces correct partial sums."""
        in_features = 512
        out_features = 256
        world_size = 2
        batch_size = 2
        seq_len = 4
        
        # Create full weight and partitioned input
        full_weight = mx.random.normal((in_features, out_features))
        full_bias = mx.random.normal((out_features,))
        full_x = mx.random.normal((batch_size, seq_len, in_features))
        
        # Full computation
        full_output = full_x @ full_weight + full_bias
        
        # Parallel computation
        layer_0 = RowParallelLinear(in_features, out_features, world_size, rank=0)
        layer_1 = RowParallelLinear(in_features, out_features, world_size, rank=1)
        
        layer_0.load_shard(full_weight, full_bias)
        layer_1.load_shard(full_weight, full_bias)
        
        # Partition input
        x_0 = full_x[..., :256]
        x_1 = full_x[..., 256:]
        
        # Get partials
        partial_0 = layer_0.forward_partial(x_0)
        partial_1 = layer_1.forward_partial(x_1)
        
        # Sum partials (simulating all-reduce)
        parallel_output = partial_0 + partial_1 + full_bias  # Add bias after sum
        
        # Compare
        assert mx.allclose(parallel_output, full_output, atol=1e-5).item()


class TestColumnRowPattern:
    """Test the Megatron pattern: Column -> activation -> Row."""
    
    def test_megatron_mlp_pattern(self):
        """Test that column-row pattern produces correct output."""
        hidden = 256
        intermediate = 512
        world_size = 2
        batch_size = 2
        seq_len = 4
        
        # Input
        x = mx.random.normal((batch_size, seq_len, hidden))
        
        # Full weights
        up_weight = mx.random.normal((hidden, intermediate))
        down_weight = mx.random.normal((intermediate, hidden))
        
        # Full computation
        intermediate_full = x @ up_weight
        activated_full = mx.maximum(intermediate_full, 0)  # ReLU
        full_output = activated_full @ down_weight
        
        # Parallel computation
        col_0 = ColumnParallelLinear(hidden, intermediate, world_size, rank=0, bias=False)
        col_1 = ColumnParallelLinear(hidden, intermediate, world_size, rank=1, bias=False)
        row_0 = RowParallelLinear(intermediate, hidden, world_size, rank=0, bias=False)
        row_1 = RowParallelLinear(intermediate, hidden, world_size, rank=1, bias=False)
        
        col_0.load_shard(up_weight)
        col_1.load_shard(up_weight)
        row_0.load_shard(down_weight)
        row_1.load_shard(down_weight)
        
        # Rank 0: column -> activate -> row partial
        inter_0 = col_0.forward_sync(x)
        act_0 = mx.maximum(inter_0, 0)
        partial_0 = row_0.forward_partial(act_0)
        
        # Rank 1: column -> activate -> row partial
        inter_1 = col_1.forward_sync(x)
        act_1 = mx.maximum(inter_1, 0)
        partial_1 = row_1.forward_partial(act_1)
        
        # All-reduce (sum)
        parallel_output = partial_0 + partial_1
        
        # Compare
        assert mx.allclose(parallel_output, full_output, atol=1e-5).item()


class TestParallelEmbedding:
    """Tests for ParallelEmbedding."""
    
    def test_shard_dimensions(self):
        """Test embedding shard dimensions."""
        vocab_size = 1000
        embed_dim = 256
        world_size = 2
        
        embed_0 = ParallelEmbedding(vocab_size, embed_dim, world_size, rank=0, gather_output=False)
        embed_1 = ParallelEmbedding(vocab_size, embed_dim, world_size, rank=1, gather_output=False)
        
        assert embed_0.local_embedding_dim == embed_dim // world_size
        assert embed_1.local_embedding_dim == embed_dim // world_size
    
    def test_lookup(self):
        """Test embedding lookup produces correct partial results."""
        vocab_size = 1000
        embed_dim = 256
        world_size = 2
        batch_size = 2
        seq_len = 4
        
        # Full embedding
        full_weight = mx.random.normal((vocab_size, embed_dim))
        input_ids = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32)
        
        # Full lookup
        flat_ids = input_ids.reshape(-1)
        full_embed = mx.take(full_weight, flat_ids, axis=0)
        full_embed = full_embed.reshape(batch_size, seq_len, embed_dim)
        
        # Parallel lookup
        embed_0 = ParallelEmbedding(vocab_size, embed_dim, world_size, rank=0, gather_output=False)
        embed_1 = ParallelEmbedding(vocab_size, embed_dim, world_size, rank=1, gather_output=False)
        
        embed_0.load_shard(full_weight)
        embed_1.load_shard(full_weight)
        
        partial_0 = embed_0.forward_sync(input_ids)
        partial_1 = embed_1.forward_sync(input_ids)
        
        # Concatenate (simulating all-gather)
        parallel_embed = mx.concatenate([partial_0, partial_1], axis=-1)
        
        # Compare
        assert mx.allclose(parallel_embed, full_embed, atol=1e-5).item()
