# Pipeline Parallelism Implementation - Plan of Action

**Date:** 2026-02-01
**Status:** Fixed - Ready for Testing
**Last Updated:** 2026-02-01 19:30

---

## Overview

A previous implementation added Pipeline Parallelism (PP) as an alternative to Tensor Parallelism (TP). The implementation is architecturally sound but encountered issues with quantized weight handling and was not fully tested end-to-end.

---

## Background: TP vs PP

### Tensor Parallelism (Original)
- ✅ All workers have ALL layers
- ✅ Each layer's weights are SPLIT across workers
- ❌ Requires all-reduce after each layer → high communication overhead
- **Communication:** 2 × num_layers all-reduces per forward pass

### Pipeline Parallelism (New)
- ✅ Each worker has SUBSET of layers
- ✅ Each layer's weights are FULL (not split)
- ✅ Communication only between pipeline stages → lower overhead
- **Communication:** (world_size - 1) point-to-point transfers per forward pass

---

## Issues Identified and Fixed

### Issue #1: Wrong Quantization API ✅ **[FIXED]**

**Problem:**
- `LinearLayer.__call__()` used `mx.dequantize()` + regular matmul
- Existing working code uses `mx.quantized_matmul()` with `transpose=True`
- These have different shape expectations and behavior

**Error:**
```
ValueError: [dequantize] Shape of scales and biases does not match the matrix
```

**Root Cause:**
The new `LinearLayer` class didn't follow the same pattern as the working `QuantizedColumnParallelLinear`:
- ❌ Wrong: `mx.dequantize(weight, scales, biases)` then `x @ weight_full`
- ✅ Correct: `mx.quantized_matmul(x, weight, scales, biases, transpose=True)`

**Fix Applied:**
Changed [transformer.py:631-650](src/shardcompute/parallel/transformer.py#L631-L650) to use `mx.quantized_matmul()` with `transpose=True` parameter, matching the working tensor parallelism implementation.

---

### Issue #2: MLX Format vs HuggingFace Format ✅ **[FIXED]**

**Problem:**
- Initial fix attempted to transpose quantized weights assuming HuggingFace format source
- BUT `mlx-community/Llama-3.2-3B-Instruct-4bit` model is **already in MLX format**
- Transposing MLX-format weights made them incorrect!

**Discovery:**
```python
# Original MLX model (already correct):
weight: (3072, 384)  # [out_features, in_features // 8]
scales: (3072, 48)   # [out_features, in_features // group_size]
biases: (3072, 48)   # [out_features, in_features // group_size]

# After incorrect transpose (broken):
weight: [384, 3072]  # ← Wrong direction!
scales: [48, 3072]   # ← Wrong direction!
biases: [48, 3072]   # ← Wrong direction!
```

**Fix Applied:**
- Modified `PipelineWeightSharder.shard_weights()` to **NOT transpose** quantized weights
- MLX-format models are already in the correct format
- Standard (non-quantized) weights still get transposed as needed
- [sharder.py:588-596](src/shardcompute/model/sharder.py#L588-L596)

**Files Modified:**
- `src/shardcompute/model/sharder.py` (lines 530-599)

---

---

### Issue #3: No Pipeline Shards Exist ✅ **[COMPLETED]**

**Problem:**
- All existing shard directories use tensor parallelism format:
  - `model_shards_phi2/config.json`: `"tensor_parallel_size": 2`
  - `model_shards_mlx_Llama_3b_Instruct_4bit/config.json`: `"tensor_parallel_size": 2`
- The `default.yaml` config has `mode: "pipeline"` but points to tensor-parallel shards
- Workers will load split weights but expect full weights → dimension mismatches

**Impact:**
- Cannot test pipeline parallelism without regenerating shards
- Runtime failures due to weight dimension mismatches

**Resolution:**
- ✅ Pipeline shards regenerated with fixed code
- ✅ Located at `./model_shards_mlx_Llama_3b_Instruct_4bit_pipeline/`
- ✅ Config shows `"mode": "pipeline"` correctly
- ✅ Weights in correct MLX format (no unwanted transpose)

---

### Issue #4: Incomplete End-to-End Testing ⚠️ **[PENDING]**

**Problem:**
- Implementation appears untested with:
  - Quantized models in pipeline mode
  - Multi-worker pipeline execution
  - Inference with pipeline parallelism

**Resolution Required:**
- Run full inference test after regenerating shards
- Validate correctness of generated output
- Performance profiling to confirm reduced communication overhead

---

## Action Plan

### ✅ Step 1: Fix Wrong Quantization API (COMPLETED)

**What was done:**
- Changed `LinearLayer.__call__()` to use `mx.quantized_matmul()` instead of `mx.dequantize()`
- Added `transpose=True` parameter to match working `QuantizedColumnParallelLinear`
- This matches exactly how tensor parallelism handles quantized weights

**Code changes:**
```python
# Before (WRONG - caused shape mismatch):
if self.is_quantized:
    weight_full = mx.dequantize(
        self.weight, self.scales, self.biases_quant,
        group_size=self.group_size, bits=self.bits,
    )
    output = x @ weight_full

# After (CORRECT - matches working TP code):
if self.is_quantized:
    output = mx.quantized_matmul(
        x, self.weight, self.scales, self.biases_quant,
        transpose=True,  # ← Key parameter
        group_size=self.group_size, bits=self.bits,
    )
```

**File:** `src/shardcompute/parallel/transformer.py:631-650`

---

### ✅ Step 2: Fix MLX Format Handling (COMPLETED)

**What was done:**
- Removed incorrect transpose for quantized weights in `PipelineWeightSharder`
- MLX-format models (`mlx-community/*`) are already in correct format
- Only non-quantized standard weights get transposed

**Code changes:**
```python
# For quantized weights: copy as-is (no transpose)
# MLX-format models are already [out_features, in_features//2]
sharded[name] = weight.copy()
sharded[scales_key] = weights[scales_key].copy()
sharded[biases_key] = weights[biases_key].copy()
```

**File:** `src/shardcompute/model/sharder.py:588-596`

---

### ✅ Step 3: Regenerate Pipeline Shards (COMPLETED)

**Objective:** Create properly formatted pipeline-parallel weight shards

**Command Used:**
```bash
python scripts/shard_weights.py \
    --model ./model_cache_mlx_Llama_3b_Instruct_4bit \
    --output ./model_shards_mlx_Llama_3b_Instruct_4bit_pipeline \
    --world-size 2 \
    --mode pipeline
```

**Validation Results:**
- ✅ `config.json` has `"mode": "pipeline"`
- ✅ `rank_0/` and `rank_1/` directories created
- ✅ Quantized weights have correct shapes (not transposed from MLX format):
  - `weight: [3072, 384]` - MLX format preserved
  - `scales: [3072, 48]` - MLX format preserved
  - `biases: [3072, 48]` - MLX format preserved

---

### ✅ Step 4: Update Configuration (COMPLETED)

**File:** `config/default.yaml`

**Changes made:**
```yaml
parallelism:
  mode: "pipeline"  # ✅ Set
  pipeline_parallel_size: 2  # ✅ Set

model:
  name: "mlx-community/Llama-3.2-3B-Instruct-4bit"  # ✅ MLX model
  # ... all model params configured

inference:
  eos_token_id: 128009  # ✅ Set for Llama 3.2
```

**Note:** Workers are started with `--shard-dir` command line arg, not from config

---

### 🔲 Step 5: Test Inference (READY TO TEST)

**Objective:** Validate end-to-end pipeline parallelism with quantized weights

#### 4.1 Start Coordinator
```bash
python -m shardcompute.coordinator.server
```

#### 5.2 Start Workers (2 terminals)
```bash
# Terminal 1 - Worker 0
python scripts/start_worker.py \
    --rank 0 \
    --coordinator-url http://localhost:8000 \
    --shard-dir ./model_shards_mlx_Llama_3b_Instruct_4bit_pipeline \
    --config config/default.yaml

# Terminal 2 - Worker 1
python scripts/start_worker.py \
    --rank 1 \
    --coordinator-url http://localhost:8000 \
    --shard-dir ./model_shards_mlx_Llama_3b_Instruct_4bit_pipeline \
    --config config/default.yaml
```

#### 5.3 Send Test Request
```bash
curl -X POST http://localhost:8000/inference/text \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 20
  }'
```

**Expected behavior:**
- No `ValueError: [dequantize] Shape of scales and biases does not match`
- Workers successfully exchange hidden states between stages
- Coherent text generation output

**Success criteria:**
- ✅ No shape mismatch errors
- ✅ Correct output generation
- ✅ Both workers participate in generation
- ✅ Communication happens only between stages (not all-reduce)

---

### 🔲 Step 6: Performance Validation (Optional)

**Objective:** Confirm pipeline parallelism reduces communication overhead vs tensor parallelism

**Metrics to compare:**
- Communication time per forward pass
- Tokens per second
- Memory usage per worker

**Test approach:**
1. Run inference with pipeline parallelism (current)
2. Regenerate tensor-parallel shards and test
3. Compare `comm_time_ms` and `tokens_per_second` metrics

**Expected:**
- Pipeline: Lower communication overhead
- Pipeline: Potentially lower throughput for small batches (pipeline bubbles)
- Pipeline: Better scaling for larger models with more stages

---

## Risk Assessment

### Low Risk
- ✅ Fix is localized to weight sharding logic
- ✅ Doesn't affect existing tensor parallelism code
- ✅ Helper methods have clear, single responsibilities

### Medium Risk
- ⚠️ Quantization format assumptions may vary across models
- ⚠️ Untested with models beyond Phi-2/Llama family
- ⚠️ Edge cases (e.g., partial quantization) not explicitly handled

### Mitigation
- Test with both quantized and non-quantized models
- Add logging for quantized weight detection
- Validate shard metadata after generation

---

## Summary of Fixes

### Root Causes Identified

1. **Wrong API Usage**: Used `mx.dequantize()` instead of `mx.quantized_matmul(transpose=True)`
2. **Format Confusion**: Tried to transpose MLX-format weights that were already correct
3. **Code Duplication**: New `LinearLayer` reimplemented logic instead of reusing existing patterns

### Files Modified

1. **`src/shardcompute/parallel/transformer.py`**
   - `LinearLayer.__call__()` - Changed to use `mx.quantized_matmul()` (lines 631-650)

2. **`src/shardcompute/model/sharder.py`**
   - `PipelineWeightSharder._is_quantized_weight()` - NEW (helper)
   - `PipelineWeightSharder._is_linear_weight()` - NEW (helper)
   - `PipelineWeightSharder.shard_weights()` - MODIFIED (no transpose for quantized)

### Key Files (Not Modified, Already Correct)

1. **`src/shardcompute/parallel/transformer.py`**
   - `LinearLayer.__call__()` - Dequantization logic (line 637)
   - `PipelineAttention`, `PipelineMLP`, `PipelineTransformerBlock` - NEW

2. **`src/shardcompute/model/loader.py`**
   - `PipelineModelLoader._load_single_layer()` - Loads quantized weights
   - `detect_parallelism_mode()` - Detects shard format

3. **`src/shardcompute/worker/executor.py`**
   - `PipelineExecutor` - Stage-to-stage communication

4. **`src/shardcompute/worker/node.py`**
   - `_load_model_pipeline()` - Pipeline model initialization

---

## Testing Checklist

- [ ] Regenerate Phi-2 pipeline shards (non-quantized)
- [ ] Regenerate Llama 3B pipeline shards (quantized 4-bit)
- [ ] Update config to use pipeline shards
- [ ] Start coordinator
- [ ] Start 2 workers
- [ ] Send inference request
- [ ] Verify no quantization errors
- [ ] Verify correct output generation
- [ ] Check logs for proper stage assignment
- [ ] Validate communication pattern (send/recv, not all-reduce)
- [ ] Optional: Profile performance vs tensor parallelism

---

## References

- **Debug Analysis:** `debug_quantization_error.md`
- **Main Implementation:** Commits `e2be95c`, `56e19f6`, `d1d81cc`
- **MLX Quantization Docs:** https://ml-explore.github.io/mlx/build/html/usage/quantization.html
- **HuggingFace Weight Format:** Standard `[out_features, in_features]` for linear layers

---

## Notes

### Quantization Format Expectations

**HuggingFace format (input):**
- Linear weight: `[out_features, in_features]` or `[out_features, in_features // 2]` if quantized
- Scales: `[out_features, in_features // group_size]`
- Biases: `[out_features, in_features // group_size]`

**MLX format (required):**
- Linear weight: `[in_features, out_features]` or `[in_features // 2, out_features]` if quantized
- Scales: `[in_features // group_size, out_features]`
- Biases: `[in_features // group_size, out_features]`

**Transpose operation:** `.T` applied to all three components together

---

## Commit Plan

After successful testing:

```bash
git add src/shardcompute/model/sharder.py
git commit -m "Fix quantized weight transpose in pipeline parallelism

- Add quantized weight detection in PipelineWeightSharder
- Transpose weight, scales, and biases together for quantized linear layers
- Preserve embedding format (no transpose)
- Handle quantized lm_head properly

Fixes ValueError: [dequantize] Shape of scales and biases does not match
the matrix error in pipeline parallelism mode with quantized models."
```

---

**Status Summary:**
- ✅ Issue #1: Wrong quantization API - FIXED
- ✅ Issue #2: MLX format confusion - FIXED
- ✅ Issue #3: Pipeline shards generated - COMPLETE
- ✅ Issue #4: Configuration updated - COMPLETE
- 🔲 Issue #5: End-to-end testing - READY

**Next Steps:**
1. Start coordinator and workers with pipeline shards
2. Send test inference request
3. Verify successful generation with no errors
4. Optional: Performance comparison vs tensor parallelism

**Last Updated:** 2026-02-01 19:30
**Next Review:** After successful inference test
