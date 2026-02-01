# Quantization Error Analysis

## Error Summary
```
ValueError: [dequantize] Shape of scales and biases does not match the matrix
```

**Location**: `LinearLayer.__call__()` in `transformer.py:637`
**Call Stack**:
- Inference request → PipelineExecutor.generate() → forward() → model.forward()
- PipelineTransformerBlock.forward() → PipelineAttention.forward()
- `q_proj(hidden_states)` → LinearLayer.__call__() → **mx.dequantize() fails**

## Problem Statement

The `mx.dequantize()` function is rejecting the quantized weights because the shapes of `scales` and `biases` don't match the expected dimensions for the quantized weight matrix.

## Quantization Format Analysis

### Expected MLX Quantization Format
MLX's 4-bit quantization stores weights in a packed format:
- **Quantized weight**: `[out_features, in_features // 2]` (4-bit packed into uint8)
- **Scales**: `[out_features, in_features // group_size]`
- **Biases**: `[out_features, in_features // group_size]`

For a `LinearLayer(in_features=3072, out_features=3072)` with `group_size=64, bits=4`:
- Expected quantized weight: `[3072, 1536]` (3072 / 2)
- Expected scales: `[3072, 48]` (3072 / 64)
- Expected biases: `[3072, 48]`

### What We're Seeing

The error indicates the shapes don't match. Likely issues:

1. **Transposition Issue**: HuggingFace stores weights as `[in_features, out_features]` but MLX expects `[out_features, in_features]` for linear layers.

2. **Quantized Weight Shape**: The quantized weight might be stored as:
   - HF format: `[in_features // 2, out_features]` (packed along input dimension)
   - MLX expects: `[out_features, in_features // 2]` (packed along input dimension, but transposed)

3. **Scales/Biases Shape**: Scales and biases might be stored as:
   - HF format: `[in_features // group_size, out_features]`
   - MLX expects: `[out_features, in_features // group_size]`

## Hypotheses

### Hypothesis 1: Missing Transpose on Quantized Weights
**Problem**: We're loading quantized weights directly without transposing them.

**Evidence**:
- Standard (non-quantized) HF weights are transposed: `weights[key].T`
- Quantized weights are loaded as-is: `weights[w_key]` (no transpose)

**Solution**: Need to transpose quantized weights, scales, and biases during loading.

### Hypothesis 2: Incorrect Quantization Parameter Detection
**Problem**: Group size or bits parameters don't match the actual quantization format.

**Evidence**:
- We're using default `group_size=64, bits=4`
- Actual quantization might use different parameters
- MLX LLama model uses different quantization config

**Solution**: Detect and extract quantization parameters from the model config.

### Hypothesis 3: Incompatible Quantization Format
**Problem**: The quantized weights from HuggingFace might use a different quantization scheme than MLX's `mx.dequantize()` expects.

**Evidence**:
- HF quantization formats vary (GPTQ, AWQ, bitsandbytes, etc.)
- MLX has its own quantization format
- We're directly loading HF quantized weights into MLX

**Solution**: Need to either:
- Convert HF quantization format to MLX format during sharding
- Use MLX's quantization utilities to re-quantize
- Detect the quantization method and handle accordingly

## Investigation Steps

### Step 1: Inspect Actual Shapes
Check the actual shapes of quantized tensors in the sharded weights:

```python
# Load a shard file and inspect shapes
from safetensors.torch import safe_open

with safe_open("sharded_weights/rank_0/model.safetensors", framework="numpy") as f:
    for key in f.keys():
        if "q_proj" in key and "layer.0" in key:
            tensor = f.get_tensor(key)
            print(f"{key}: {tensor.shape}")
```

Expected keys for layer 0 q_proj:
- `model.layers.0.self_attn.q_proj.weight`: `[?, ?]`
- `model.layers.0.self_attn.q_proj.scales`: `[?, ?]`
- `model.layers.0.self_attn.q_proj.biases`: `[?, ?]`

### Step 2: Check Model Config
Examine the model config for quantization parameters:

```python
import json
with open("sharded_weights/rank_0/config.json") as f:
    config = json.load(f)
    print(json.dumps(config.get("quantization", {}), indent=2))
```

### Step 3: Test Dequantization Manually
Try dequantizing a single tensor with different transpose combinations:

```python
import mlx.core as mx

# Load tensors
weight_q = mx.array(weight_tensor)
scales = mx.array(scales_tensor)
biases = mx.array(biases_tensor)

# Try different transpose combinations
try:
    # As-is
    w = mx.dequantize(weight_q, scales, biases, group_size=64, bits=4)
except Exception as e:
    print(f"No transpose: {e}")

try:
    # Transpose all
    w = mx.dequantize(weight_q.T, scales.T, biases.T, group_size=64, bits=4)
except Exception as e:
    print(f"Transpose all: {e}")
```

## Likely Solution

Based on the error pattern, the most likely fix is:

**In `PipelineModelLoader._load_single_layer()`**:

```python
# Current (incorrect):
if scales_key in weights and biases_key in weights:
    proj.load_quantized_weights(
        weights[w_key],          # ← No transpose
        weights[scales_key],     # ← No transpose
        weights[biases_key],     # ← No transpose
    )

# Fixed (with transpose):
if scales_key in weights and biases_key in weights:
    proj.load_quantized_weights(
        weights[w_key].T,         # ← Transpose quantized weight
        weights[scales_key].T,    # ← Transpose scales
        weights[biases_key].T,    # ← Transpose biases
    )
```

Similarly for embeddings and all other quantized layers.

## Action Plan

1. **Immediate**: Read the actual shard file and inspect tensor shapes
2. **Verify**: Check if model config contains quantization parameters
3. **Test**: Try transpose fix on LinearLayer loading
4. **Apply**: Update all quantized weight loading locations if transpose works
5. **Validate**: Re-run inference and verify it proceeds past q_proj

## Files to Modify

- `/Users/essajan/Desktop/arjav_app/shardcompute/src/shardcompute/model/loader.py`
  - `PipelineModelLoader._load_single_layer()` (attention projections)
  - `PipelineModelLoader._load_single_layer()` (MLP projections)
  - `PipelineModelLoader._load_embedding()` (embedding)
  - `PipelineModelLoader._load_lm_head()` (LM head)
