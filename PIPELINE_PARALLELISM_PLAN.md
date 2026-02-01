# Pipeline Parallelism Implementation - Plan of Action

**Date:** 2026-02-01
**Status:** In Progress - Fix Applied, Testing Required

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

## Issues Identified

### Issue #1: Quantized Weight Transpose Missing ⚠️ **[FIXED]**

**Problem:**
- `PipelineWeightSharder` transposes standard 2D linear weights from HuggingFace `[out, in]` to MLX `[in, out]` format
- BUT quantized weights (with separate `.scales` and `.biases`) were NOT being transposed
- MLX's `mx.dequantize()` expects `[out, in]` format with matching scales/biases dimensions

**Result:**
```
ValueError: [dequantize] Shape of scales and biases does not match the matrix
```
at `transformer.py:637` in `LinearLayer.__call__()`

**Fix Applied:**
- Modified `PipelineWeightSharder.shard_weights()` in `src/shardcompute/model/sharder.py`
- Added quantized weight detection and group processing
- Now transposes all three components (weight, scales, biases) together for quantized linear layers
- Commit: *[To be created after testing]*

**Files Modified:**
- `src/shardcompute/model/sharder.py`

---

### Issue #2: No Pipeline Shards Exist ⚠️ **[PENDING]**

**Problem:**
- All existing shard directories use tensor parallelism format:
  - `model_shards_phi2/config.json`: `"tensor_parallel_size": 2`
  - `model_shards_mlx_Llama_3b_Instruct_4bit/config.json`: `"tensor_parallel_size": 2`
- The `default.yaml` config has `mode: "pipeline"` but points to tensor-parallel shards
- Workers will load split weights but expect full weights → dimension mismatches

**Impact:**
- Cannot test pipeline parallelism without regenerating shards
- Runtime failures due to weight dimension mismatches

**Resolution Required:**
- Regenerate shards using the fixed `PipelineWeightSharder`
- Update configuration to point to new shard directories

---

### Issue #3: Incomplete End-to-End Testing ⚠️ **[PENDING]**

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

### ✅ Step 1: Fix Quantized Weight Transpose (COMPLETED)

**What was done:**
- Added `_is_quantized_weight()` helper to detect quantization components
- Added `_is_linear_weight()` helper to identify layers needing transpose
- Modified `shard_weights()` to:
  - Track processed quantization components
  - Transpose weight, scales, and biases together for linear layers
  - Preserve embedding format (no transpose)
  - Handle quantized lm_head properly

**Code changes:**
```python
# Before (INCORRECT):
if len(weight.shape) == 2 and not any(x in name for x in ["layernorm", ...]):
    sharded[name] = weight.T.copy()
# Scales and biases were processed separately without transpose awareness

# After (CORRECT):
if self._is_quantized_weight(name, weights):
    if self._is_linear_weight(name) and len(weight.shape) == 2:
        sharded[name] = weight.T.copy()
        sharded[scales_key] = weights[scales_key].T.copy()  # ← Fixed
        sharded[biases_key] = weights[biases_key].T.copy()  # ← Fixed
```

---

### 🔲 Step 2: Regenerate Pipeline Shards

**Objective:** Create properly formatted pipeline-parallel weight shards

**Commands:**

#### For Phi-2 (2.7B, non-quantized):
```bash
python scripts/shard_weights.py \
    --model-dir ./model_cache_phi2 \
    --output-dir ./model_shards_phi2_pipeline \
    --world-size 2 \
    --mode pipeline
```

#### For Llama 3.2 3B (quantized 4-bit):
```bash
python scripts/shard_weights.py \
    --model-dir ./model_cache_mlx_Llama_3b_Instruct_4bit \
    --output-dir ./model_shards_llama3_3b_pipeline \
    --world-size 2 \
    --mode pipeline
```

**Expected output:**
- `model_shards_*_pipeline/config.json` with `"mode": "pipeline"`
- `model_shards_*_pipeline/rank_0/` with embedding + first N/2 layers
- `model_shards_*_pipeline/rank_1/` with last N/2 layers + norm + lm_head
- All linear weights transposed to MLX format
- Quantized components (if present) transposed together

**Validation:**
- Check `config.json` has correct parallelism mode
- Verify `metadata.json` shows expected layer distribution
- Confirm tensor shapes match expected dimensions

---

### 🔲 Step 3: Update Configuration

**File:** `config/default.yaml`

**Changes needed:**
```yaml
parallelism:
  mode: "pipeline"  # Already set
  pipeline_parallel_size: 2

model:
  # Update to use pipeline shards:
  shards_dir: "./model_shards_phi2_pipeline"
  # OR for quantized:
  # shards_dir: "./model_shards_llama3_3b_pipeline"
```

**Verification:**
- Ensure `mode` matches shard directory format
- Confirm `shards_dir` points to regenerated pipeline shards

---

### 🔲 Step 4: Test Inference

**Objective:** Validate end-to-end pipeline parallelism with quantized weights

#### 4.1 Start Coordinator
```bash
python -m shardcompute.coordinator.server
```

#### 4.2 Start Workers (2 terminals)
```bash
# Terminal 1 - Worker 0
python -m shardcompute.worker.node --rank 0

# Terminal 2 - Worker 1
python -m shardcompute.worker.node --rank 1
```

#### 4.3 Send Test Request
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

### 🔲 Step 5: Performance Validation (Optional)

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

## Implementation Details

### Key Files Modified

1. **`src/shardcompute/model/sharder.py`**
   - `PipelineWeightSharder._is_quantized_weight()` - NEW
   - `PipelineWeightSharder._is_linear_weight()` - NEW
   - `PipelineWeightSharder.shard_weights()` - MODIFIED

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

**Last Updated:** 2026-02-01
**Next Review:** After Step 4 (Test Inference) completion
