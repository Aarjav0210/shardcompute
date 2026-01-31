"""Integration tests for ShardCompute."""

import pytest
import asyncio
import aiohttp
import mlx.core as mx
import numpy as np
from pathlib import Path
import tempfile
import json

from shardcompute.model.config import ModelConfig, ParallelConfig
from shardcompute.model.sharder import WeightSharder


class TestWeightSharding:
    """Integration tests for weight sharding."""
    
    def test_shard_and_verify(self):
        """Test that sharding and reconstruction produces original."""
        # Small test model config
        model_config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
        )
        
        parallel_config = ParallelConfig(world_size=2, tensor_parallel_size=2)
        sharder = WeightSharder(model_config, parallel_config)
        
        # Create test weights (simplified)
        weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 256).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(256, 256).astype(np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(256, 256).astype(np.float32),
            "model.layers.0.self_attn.v_proj.weight": np.random.randn(256, 256).astype(np.float32),
            "model.layers.0.self_attn.o_proj.weight": np.random.randn(256, 256).astype(np.float32),
            "model.layers.0.mlp.up_proj.weight": np.random.randn(256, 512).astype(np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(256, 512).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": np.random.randn(512, 256).astype(np.float32),
            "model.layers.0.input_layernorm.weight": np.random.randn(256).astype(np.float32),
            "model.norm.weight": np.random.randn(256).astype(np.float32),
        }
        
        # Shard for both ranks
        shards_0 = sharder.shard_weights(weights, rank=0)
        shards_1 = sharder.shard_weights(weights, rank=1)
        
        # Verify embedding sharding
        embed_key = "model.embed_tokens.weight"
        full_embed = weights[embed_key]
        
        # Reconstruct
        reconstructed = np.concatenate([
            shards_0[embed_key],
            shards_1[embed_key]
        ], axis=1)
        
        np.testing.assert_allclose(reconstructed, full_embed)
        
        # Verify QKV sharding (column parallel)
        for proj in ["q_proj", "k_proj", "v_proj"]:
            key = f"model.layers.0.self_attn.{proj}.weight"
            full = weights[key]
            reconstructed = np.concatenate([shards_0[key], shards_1[key]], axis=1)
            np.testing.assert_allclose(reconstructed, full)
        
        # Verify O projection sharding (row parallel)
        o_key = "model.layers.0.self_attn.o_proj.weight"
        full_o = weights[o_key]
        reconstructed_o = np.concatenate([shards_0[o_key], shards_1[o_key]], axis=0)
        np.testing.assert_allclose(reconstructed_o, full_o)
        
        # Verify MLP sharding
        for proj in ["up_proj", "gate_proj"]:
            key = f"model.layers.0.mlp.{proj}.weight"
            full = weights[key]
            reconstructed = np.concatenate([shards_0[key], shards_1[key]], axis=1)
            np.testing.assert_allclose(reconstructed, full)
        
        down_key = "model.layers.0.mlp.down_proj.weight"
        full_down = weights[down_key]
        reconstructed_down = np.concatenate([shards_0[down_key], shards_1[down_key]], axis=0)
        np.testing.assert_allclose(reconstructed_down, full_down)
        
        # Verify LayerNorm is replicated
        ln_key = "model.layers.0.input_layernorm.weight"
        np.testing.assert_allclose(shards_0[ln_key], weights[ln_key])
        np.testing.assert_allclose(shards_1[ln_key], weights[ln_key])
    
    def test_save_and_load_shards(self):
        """Test saving and loading shards."""
        model_config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=1,
            num_heads=2,
            intermediate_size=128,
        )
        
        parallel_config = ParallelConfig(world_size=2, tensor_parallel_size=2)
        sharder = WeightSharder(model_config, parallel_config)
        
        # Create minimal test weights
        weights = {
            "model.embed_tokens.weight": np.random.randn(100, 64).astype(np.float32),
            "model.norm.weight": np.random.randn(64).astype(np.float32),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save shards
            sharder.save_shards(weights, tmpdir)
            
            # Verify directory structure
            output_path = Path(tmpdir)
            assert (output_path / "config.json").exists()
            assert (output_path / "rank_0" / "model.safetensors").exists()
            assert (output_path / "rank_1" / "model.safetensors").exists()
            
            # Load and verify shards
            loaded_0, meta_0 = WeightSharder.load_shards(tmpdir, rank=0)
            loaded_1, meta_1 = WeightSharder.load_shards(tmpdir, rank=1)
            
            assert meta_0["rank"] == 0
            assert meta_1["rank"] == 1
            assert len(loaded_0) > 0
            assert len(loaded_1) > 0


class TestParallelMatmulCorrectness:
    """Test that parallel matmul produces same result as regular matmul."""
    
    def test_column_row_equivalence(self):
        """Test Column -> Row pattern equals full matmul."""
        # Dimensions
        batch = 2
        seq = 4
        hidden = 64
        intermediate = 128
        world_size = 2
        
        # Input
        x = mx.random.normal((batch, seq, hidden))
        
        # Full weights
        W1 = mx.random.normal((hidden, intermediate))  # up projection
        W2 = mx.random.normal((intermediate, hidden))  # down projection
        
        # Full computation
        full_inter = x @ W1
        full_output = full_inter @ W2
        
        # Simulate 2-way tensor parallel
        # Worker 0: W1[:, :64], W2[:64, :]
        # Worker 1: W1[:, 64:], W2[64:, :]
        
        W1_0 = W1[:, :intermediate // 2]
        W1_1 = W1[:, intermediate // 2:]
        W2_0 = W2[:intermediate // 2, :]
        W2_1 = W2[intermediate // 2:, :]
        
        # Parallel computation
        inter_0 = x @ W1_0  # [batch, seq, intermediate/2]
        inter_1 = x @ W1_1  # [batch, seq, intermediate/2]
        
        partial_0 = inter_0 @ W2_0  # [batch, seq, hidden]
        partial_1 = inter_1 @ W2_1  # [batch, seq, hidden]
        
        # All-reduce (sum)
        parallel_output = partial_0 + partial_1
        
        # Compare
        assert mx.allclose(parallel_output, full_output, atol=1e-5).item()
    
    def test_attention_head_split(self):
        """Test that splitting attention by heads is equivalent."""
        batch = 2
        seq = 4
        hidden = 64
        num_heads = 4
        head_dim = 16
        world_size = 2
        
        # Input
        x = mx.random.normal((batch, seq, hidden))
        
        # Full QKV projections
        W_q = mx.random.normal((hidden, hidden))
        W_k = mx.random.normal((hidden, hidden))
        W_v = mx.random.normal((hidden, hidden))
        W_o = mx.random.normal((hidden, hidden))
        
        # Full attention
        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v
        
        Q = Q.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)
        
        scores = Q @ K.transpose(0, 1, 3, 2) / (head_dim ** 0.5)
        attn = mx.softmax(scores, axis=-1)
        attn_out = attn @ V
        
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, hidden)
        full_output = attn_out @ W_o
        
        # Parallel: split by heads
        # Worker 0: heads 0, 1
        # Worker 1: heads 2, 3
        local_heads = num_heads // world_size
        local_hidden = local_heads * head_dim
        
        # Shard QKV (column parallel)
        W_q_0 = W_q[:, :local_hidden]
        W_q_1 = W_q[:, local_hidden:]
        W_k_0 = W_k[:, :local_hidden]
        W_k_1 = W_k[:, local_hidden:]
        W_v_0 = W_v[:, :local_hidden]
        W_v_1 = W_v[:, local_hidden:]
        
        # Shard O (row parallel)
        W_o_0 = W_o[:local_hidden, :]
        W_o_1 = W_o[local_hidden:, :]
        
        # Worker 0 computation
        Q_0 = (x @ W_q_0).reshape(batch, seq, local_heads, head_dim).transpose(0, 2, 1, 3)
        K_0 = (x @ W_k_0).reshape(batch, seq, local_heads, head_dim).transpose(0, 2, 1, 3)
        V_0 = (x @ W_v_0).reshape(batch, seq, local_heads, head_dim).transpose(0, 2, 1, 3)
        scores_0 = Q_0 @ K_0.transpose(0, 1, 3, 2) / (head_dim ** 0.5)
        attn_0 = mx.softmax(scores_0, axis=-1)
        attn_out_0 = attn_0 @ V_0
        attn_out_0 = attn_out_0.transpose(0, 2, 1, 3).reshape(batch, seq, local_hidden)
        partial_0 = attn_out_0 @ W_o_0
        
        # Worker 1 computation
        Q_1 = (x @ W_q_1).reshape(batch, seq, local_heads, head_dim).transpose(0, 2, 1, 3)
        K_1 = (x @ W_k_1).reshape(batch, seq, local_heads, head_dim).transpose(0, 2, 1, 3)
        V_1 = (x @ W_v_1).reshape(batch, seq, local_heads, head_dim).transpose(0, 2, 1, 3)
        scores_1 = Q_1 @ K_1.transpose(0, 1, 3, 2) / (head_dim ** 0.5)
        attn_1 = mx.softmax(scores_1, axis=-1)
        attn_out_1 = attn_1 @ V_1
        attn_out_1 = attn_out_1.transpose(0, 2, 1, 3).reshape(batch, seq, local_hidden)
        partial_1 = attn_out_1 @ W_o_1
        
        # All-reduce
        parallel_output = partial_0 + partial_1
        
        # Compare
        assert mx.allclose(parallel_output, full_output, atol=1e-4).item()
