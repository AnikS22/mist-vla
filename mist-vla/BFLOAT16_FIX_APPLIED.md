# BFloat16 Dtype Fix Applied - 2026-01-15

## Problem

**Error in Job 3747108:**
```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```

**Location:** `scripts/collect_phase1_data.py:111`

**Root Cause:**
- OpenVLA model loaded with `torch_dtype=torch.bfloat16` (line 198)
- Input pixel_values remained as `float32` after processor (line 106-107)
- Forward pass failed due to dtype mismatch in conv2d layer

## Solution Applied

**File:** `scripts/collect_phase1_data.py`

**Lines 109-111 (NEW):**
```python
# Convert pixel_values to match model dtype (bfloat16 on GPU)
if 'pixel_values' in inputs:
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model.dtype)
```

**Complete context (lines 105-115):**
```python
# Process inputs
inputs = processor(images=image, text=instruction, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}

# Convert pixel_values to match model dtype (bfloat16 on GPU)
if 'pixel_values' in inputs:
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model.dtype)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
```

## Why This Fix Works

1. **Model dtype:** Model is `torch.bfloat16` on CUDA (line 198)
2. **Input dtype:** Processor outputs `torch.float32` by default
3. **Conversion:** Explicitly convert pixel_values to match `model.dtype`
4. **Verified pattern:** Same fix used in `test_real_quick.py` lines 94-97

## Verification

This fix matches the working pattern in `test_real_quick.py`:
```python
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(
        model.device, dtype=model.dtype
    )
```

## Testing Status

- ✅ Fix applied to `scripts/collect_phase1_data.py`
- ⏳ **Needs HPC testing** (local GPU has insufficient memory)
- ✅ Code pattern verified against working test script

## Next Steps on HPC

### 1. Transfer Updated Code
```bash
bash transfer_to_hpc.sh
```

### 2. SSH to HPC
```bash
ssh asahai2024@athene-login.fau.edu
cd ~/mist-vla
```

### 3. Submit Quick Test (5 rollouts)
```bash
sbatch test_quick.slurm
# OR directly test the fixed script:
sbatch mist_vla_quick_test.slurm
```

### 4. Monitor Results
```bash
# Find your job ID
squeue -u asahai2024

# Watch output
tail -f logs/quick_test_JOBID.out
# OR
tail -f logs/mist_vla_quick_JOBID.out
```

### 5. Expected Success Output
```
[3/5] Running data collection...
  Collecting 5 rollouts with 50 steps max
  Task 0: pick up the black bowl...
    Rollout 0/5...
      Step 0/50... ✓
      Step 1/50... ✓
      ...
    Rollout 1/5...
      ...
  ✓ Data saved to data/phase1/test_rollouts.pkl

✅ ALL TESTS PASSED!
```

## Previous Fixes Applied

This is fix #10 in the series:

1. ✅ LIBERO PYTHONPATH configuration
2. ✅ LIBERO config file creation
3. ✅ `len(task_suite)` → `get_num_tasks()`
4. ✅ robosuite installation
5. ✅ NumPy 2.x → 1.26.4 downgrade
6. ✅ Wrong import path for `get_libero_path`
7. ✅ Invalid OffScreenRenderEnv parameters
8. ✅ Indentation errors
9. ✅ CollisionDetector sim access (env.env.sim)
10. ✅ **BFloat16 dtype conversion (THIS FIX)**

## Confidence Level

**95% confident this fixes the issue**

**Reasoning:**
- Exact same pattern used in working test script
- Root cause clearly identified in error traceback
- Fix addresses the dtype mismatch directly
- Standard practice for mixed-precision models

## Files Modified

- `scripts/collect_phase1_data.py` - Lines 109-111 (added dtype conversion)

## Estimated Resolution

- **Transfer time:** 2-3 minutes (rsync over VPN)
- **Job submission:** Instant
- **Test job runtime:** 30-45 minutes (5 rollouts on A5000)
- **Total time to verification:** ~45-50 minutes

---

**Status:** Ready for HPC testing (requires VPN connection)
