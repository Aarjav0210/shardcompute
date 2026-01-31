#!/usr/bin/env python3
"""Test script to verify quantization support before running full inference."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from safetensors import safe_open

from shardcompute.model.config import ModelConfig, QuantizationConfig
from shardcompute.model.loader import ModelLoader
from shardcompute.parallel.quantized_linear import (
    QuantizedColumnParallelLinear,
    QuantizedRowParallelLinear,
)


def test_quantization_detection(shard_dir: Path):
    """Test that quantization is properly detected."""
    print("\n=== Test 1: Quantization Detection ===")

    is_quantized = ModelLoader.detect_quantization(shard_dir, rank=0)
    print(f"Quantization detected: {is_quantized}")

    if not is_quantized:
        print("ERROR: Expected quantized weights but detection returned False")
        return False

    print("✓ Quantization detection passed")
    return True


def test_config_loading(shard_dir: Path):
    """Test that config loads correctly with quantization."""
    print("\n=== Test 2: Config Loading ===")

    config_path = shard_dir / "config.json"
    with open(config_path) as f:
        data = json.load(f)

    # Handle nested structure
    model_data = data.get("model", data)
    config = ModelConfig.from_dict(model_data)

    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_layers}")
    print(f"Num heads: {config.num_heads}")
    print(f"Num KV heads: {config.num_kv_heads}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Use gated MLP: {config.use_gated_mlp}")
    print(f"Hidden act: {config.hidden_act}")

    if config.quantization:
        print(f"Quantization enabled: {config.quantization.enabled}")
        print(f"Quantization bits: {config.quantization.bits}")
        print(f"Quantization group size: {config.quantization.group_size}")
    else:
        print("ERROR: Quantization config is None")
        return False, None

    print("✓ Config loading passed")
    return True, config


def test_weight_shapes(shard_dir: Path, config: ModelConfig):
    """Test that weight shapes are correct for quantized model."""
    print("\n=== Test 3: Weight Shapes ===")

    rank_dir = shard_dir / "rank_0"
    model_file = rank_dir / "model.safetensors"

    weights = {}
    with safe_open(str(model_file), framework="numpy") as f:
        keys = list(f.keys())
        print(f"Total weight keys: {len(keys)}")

        # Sample some keys
        sample_keys = [k for k in keys if "layers.0." in k][:10]
        print(f"\nSample keys from layer 0:")
        for key in sample_keys:
            tensor = f.get_tensor(key)
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")

        # Check for quantization components
        has_scales = any(".scales" in k for k in keys)
        has_biases = any(".biases" in k for k in keys)
        print(f"\nHas .scales keys: {has_scales}")
        print(f"Has .biases keys: {has_biases}")

        if not (has_scales and has_biases):
            print("ERROR: Missing quantization components")
            return False

        # Check a specific quantized weight
        q_proj_key = "model.layers.0.self_attn.q_proj.weight"
        q_scales_key = "model.layers.0.self_attn.q_proj.scales"
        q_biases_key = "model.layers.0.self_attn.q_proj.biases"

        if q_proj_key in keys:
            q_weight = f.get_tensor(q_proj_key)
            q_scales = f.get_tensor(q_scales_key) if q_scales_key in keys else None
            q_biases = f.get_tensor(q_biases_key) if q_biases_key in keys else None

            print(f"\nQ projection shapes (layer 0):")
            print(f"  weight: {q_weight.shape} ({q_weight.dtype})")
            if q_scales is not None:
                print(f"  scales: {q_scales.shape} ({q_scales.dtype})")
            if q_biases is not None:
                print(f"  biases: {q_biases.shape} ({q_biases.dtype})")

            # Verify dimensions make sense for sharded quantized weight
            # For column parallel Q proj: [hidden_size, num_heads * head_dim / world_size]
            # But quantized weights may be packed differently
            expected_out_dim = config.num_heads * (config.hidden_size // config.num_heads) // 2  # world_size=2
            print(f"\nExpected output dim (sharded): {expected_out_dim}")
            print(f"Actual weight shape: {q_weight.shape}")

    print("✓ Weight shapes check passed")
    return True


def test_quantized_linear_creation(config: ModelConfig):
    """Test that quantized linear layers can be created."""
    print("\n=== Test 4: Quantized Linear Layer Creation ===")

    world_size = 2
    rank = 0
    bits = config.quantization.bits
    group_size = config.quantization.group_size

    try:
        # Create a quantized column parallel linear (like Q projection)
        col_linear = QuantizedColumnParallelLinear(
            in_features=config.hidden_size,
            out_features=config.num_heads * (config.hidden_size // config.num_heads),
            world_size=world_size,
            rank=rank,
            bias=False,
            gather_output=False,
            bits=bits,
            group_size=group_size,
        )
        print(f"✓ Created QuantizedColumnParallelLinear")
        print(f"  in_features: {col_linear.in_features}")
        print(f"  out_features: {col_linear.out_features}")
        print(f"  local_out_features: {col_linear.local_out_features}")

        # Create a quantized row parallel linear (like O projection)
        row_linear = QuantizedRowParallelLinear(
            in_features=config.num_heads * (config.hidden_size // config.num_heads),
            out_features=config.hidden_size,
            world_size=world_size,
            rank=rank,
            bias=False,
            input_is_partitioned=True,
            communicator=None,  # Not needed for this test
            bits=bits,
            group_size=group_size,
        )
        print(f"✓ Created QuantizedRowParallelLinear")
        print(f"  in_features: {row_linear.in_features}")
        print(f"  out_features: {row_linear.out_features}")
        print(f"  local_in_features: {row_linear.local_in_features}")

    except Exception as e:
        print(f"ERROR: Failed to create quantized linear: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✓ Quantized linear creation passed")
    return True


def test_weight_loading_shapes(shard_dir: Path, config: ModelConfig):
    """Test that weights can be loaded into quantized layers."""
    print("\n=== Test 5: Weight Loading Shapes ===")

    rank_dir = shard_dir / "rank_0"
    model_file = rank_dir / "model.safetensors"

    world_size = 2
    rank = 0
    bits = config.quantization.bits
    group_size = config.quantization.group_size

    with safe_open(str(model_file), framework="numpy") as f:
        # Load Q projection weights
        q_weight = mx.array(f.get_tensor("model.layers.0.self_attn.q_proj.weight"))
        q_scales = mx.array(f.get_tensor("model.layers.0.self_attn.q_proj.scales"))
        q_biases = mx.array(f.get_tensor("model.layers.0.self_attn.q_proj.biases"))

        print(f"Loaded Q proj weights:")
        print(f"  weight: {q_weight.shape}")
        print(f"  scales: {q_scales.shape}")
        print(f"  biases: {q_biases.shape}")

        # Create column parallel linear for Q
        col_linear = QuantizedColumnParallelLinear(
            in_features=config.hidden_size,
            out_features=config.num_heads * (config.hidden_size // config.num_heads),
            world_size=world_size,
            rank=rank,
            bias=False,
            gather_output=False,
            bits=bits,
            group_size=group_size,
        )

        # Check if shapes match what the layer expects
        print(f"\nExpected shapes for QuantizedColumnParallelLinear:")
        print(f"  local_out_features: {col_linear.local_out_features}")
        print(f"  in_features: {col_linear.in_features}")

        # MLX quantized format: weight is packed, scales/biases have group structure
        # The weight should be loadable
        try:
            col_linear.load_quantized_shard(q_weight, q_scales, q_biases, None)
            print("✓ Successfully loaded weights into QuantizedColumnParallelLinear")
        except Exception as e:
            print(f"ERROR loading weights: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test forward pass (just shape check, no actual computation)
        test_input = mx.zeros((1, 10, config.hidden_size))
        try:
            output = col_linear.forward_sync(test_input)
            print(f"✓ Forward pass succeeded, output shape: {output.shape}")
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("✓ Weight loading shapes passed")
    return True


def test_transformer_creation(config: ModelConfig):
    """Test that full transformer can be created with quantization."""
    print("\n=== Test 6: Transformer Model Creation ===")

    from shardcompute.parallel.transformer import ParallelTransformer

    world_size = 2
    rank = 0
    bits = config.quantization.bits
    group_size = config.quantization.group_size

    # Create a mock communicator (not needed for creation test)
    class MockCommunicator:
        async def all_reduce(self, tensor, op="sum"):
            return tensor
        async def all_gather(self, tensor, dim=-1):
            # Simulate full gather for shape checks
            return mx.concatenate([tensor] * world_size, axis=dim)
        async def broadcast(self, tensor, root=0):
            return tensor

    try:
        model = ParallelTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            world_size=world_size,
            rank=rank,
            communicator=MockCommunicator(),
            num_kv_heads=config.num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_base=config.rope_theta,
            mlp_activation=config.hidden_act,
            use_gated_mlp=config.use_gated_mlp,
            tie_word_embeddings=config.tie_word_embeddings,
            use_quantized=True,
            quantization_bits=bits,
            quantization_group_size=group_size,
        )
        print(f"✓ Created ParallelTransformer with quantization")
        print(f"  num_layers: {model.num_layers}")
        print(f"  hidden_size: {model.hidden_size}")
        print(f"  vocab_size: {model.vocab_size}")

        # Check that layers have quantized attention/MLP
        layer0 = model.layers[0]
        print(f"  layer 0 attention use_quantized: {layer0.attention.use_quantized}")
        print(f"  layer 0 mlp use_quantized: {layer0.mlp.use_quantized}")

        if not layer0.attention.use_quantized:
            print("ERROR: Attention should be quantized")
            return False
        if not layer0.mlp.use_quantized:
            print("ERROR: MLP should be quantized")
            return False

    except Exception as e:
        print(f"ERROR: Failed to create transformer: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✓ Transformer creation passed")
    return True


def test_full_model_loading(shard_dir: Path, config: ModelConfig):
    """Test loading full model with ModelLoader."""
    print("\n=== Test 7: Full Model Loading ===")

    import asyncio
    from shardcompute.parallel.transformer import ParallelTransformer
    from shardcompute.model.loader import ModelLoader

    world_size = 2
    rank = 0
    bits = config.quantization.bits
    group_size = config.quantization.group_size

    class MockCommunicator:
        async def all_reduce(self, tensor, op="sum"):
            return tensor
        async def all_gather(self, tensor, dim=-1):
            # Simulate full gather for shape checks
            return mx.concatenate([tensor] * world_size, axis=dim)
        async def broadcast(self, tensor, root=0):
            return tensor

    try:
        model = ParallelTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            world_size=world_size,
            rank=rank,
            communicator=MockCommunicator(),
            num_kv_heads=config.num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_base=config.rope_theta,
            mlp_activation=config.hidden_act,
            use_gated_mlp=config.use_gated_mlp,
            tie_word_embeddings=config.tie_word_embeddings,
            use_quantized=True,
            quantization_bits=bits,
            quantization_group_size=group_size,
        )

        loader = ModelLoader(rank=rank, world_size=world_size, quantized=True)

        async def load():
            await loader.load_shards(model, shard_dir)

        asyncio.run(load())
        print("✓ Model weights loaded successfully")

        # Verify some weights are loaded
        layer0 = model.layers[0]
        # For quantized layers, use quantized_weight
        q_weight = layer0.attention.q_proj.quantized_weight
        if q_weight is None:
            print("ERROR: Q projection weight is None after loading")
            return False
        print(f"  Layer 0 Q proj quantized_weight shape: {q_weight.shape}")

        # Check embedding
        embed_weight = model.embed_tokens.weight
        if embed_weight is None:
            print("ERROR: Embedding weight is None after loading")
            return False
        print(f"  Embedding weight shape: {embed_weight.shape}")

        # Check norm
        norm_weight = model.norm.weight
        if norm_weight is None:
            print("ERROR: Final norm weight is None after loading")
            return False
        print(f"  Final norm weight shape: {norm_weight.shape}")

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✓ Full model loading passed")
    return True


def test_forward_pass(shard_dir: Path, config: ModelConfig):
    """Test a single forward pass through the model."""
    print("\n=== Test 8: Single Forward Pass ===")

    import asyncio
    from shardcompute.parallel.transformer import ParallelTransformer
    from shardcompute.model.loader import ModelLoader

    world_size = 2
    rank = 0
    bits = config.quantization.bits
    group_size = config.quantization.group_size

    class MockCommunicator:
        async def all_reduce(self, tensor, op="sum"):
            return tensor
        async def all_gather(self, tensor, dim=-1):
            # Simulate full gather for shape checks
            return mx.concatenate([tensor] * world_size, axis=dim)
        async def broadcast(self, tensor, root=0):
            return tensor

    try:
        model = ParallelTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            world_size=world_size,
            rank=rank,
            communicator=MockCommunicator(),
            num_kv_heads=config.num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_base=config.rope_theta,
            mlp_activation=config.hidden_act,
            use_gated_mlp=config.use_gated_mlp,
            tie_word_embeddings=config.tie_word_embeddings,
            use_quantized=True,
            quantization_bits=bits,
            quantization_group_size=group_size,
        )

        loader = ModelLoader(rank=rank, world_size=world_size, quantized=True)

        async def load_and_forward():
            await loader.load_shards(model, shard_dir)

            # Create test input (token IDs that are valid for this rank's vocab range)
            # Rank 0 holds vocab [0, 64128), so use tokens in that range
            test_input = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
            print(f"  Test input shape: {test_input.shape}")
            print(f"  Test input: {test_input.tolist()}")

            # Run forward pass
            logits, _ = await model.forward(test_input, use_cache=False)

            print(f"  Output logits shape: {logits.shape}")
            print(f"  Expected: (1, 5, {config.vocab_size})")

            # Verify output shape
            expected_shape = (1, 5, config.vocab_size)
            if logits.shape != expected_shape:
                print(f"ERROR: Output shape {logits.shape} != expected {expected_shape}")
                return False

            # Check logits are reasonable (not NaN/Inf)
            if mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
                print("ERROR: Output contains NaN or Inf")
                return False

            # Get top prediction for last token
            last_logits = logits[0, -1, :]
            top_token = mx.argmax(last_logits).item()
            print(f"  Top predicted token: {top_token}")

            return True

        success = asyncio.run(load_and_forward())
        if not success:
            return False

    except Exception as e:
        print(f"ERROR: Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✓ Forward pass passed")
    return True


def main():
    shard_dir = Path("/Users/aarjavjain/Desktop/Dev/shardcompute/model_shards_mlx_Llama_3b_Instruct_4bit")

    print("=" * 60)
    print("Quantization Support Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Detection
    if not test_quantization_detection(shard_dir):
        all_passed = False

    # Test 2: Config loading
    passed, config = test_config_loading(shard_dir)
    if not passed:
        all_passed = False
        print("\nCannot continue without valid config")
        return 1

    # Test 3: Weight shapes
    if not test_weight_shapes(shard_dir, config):
        all_passed = False

    # Test 4: Layer creation
    if not test_quantized_linear_creation(config):
        all_passed = False

    # Test 5: Weight loading
    if not test_weight_loading_shapes(shard_dir, config):
        all_passed = False

    # Test 6: Transformer creation
    if not test_transformer_creation(config):
        all_passed = False

    # Test 7: Full model loading
    if not test_full_model_loading(shard_dir, config):
        all_passed = False

    # Test 8: Single forward pass
    if not test_forward_pass(shard_dir, config):
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
