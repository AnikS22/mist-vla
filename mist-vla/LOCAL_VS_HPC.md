# Local vs HPC Strategy for MIST-VLA

## Your Hardware
- GPU: RTX 2080 Ti (11.3 GB)
- Status: Borderline for OpenVLA-7B

## Recommended Strategy: HYBRID APPROACH

### ✅ Run LOCALLY (Development & Testing)
These are lightweight and good for development:

1. **Code verification** (Already done!)
   ```bash
   python scripts/verify_setup.py
   ```

2. **Small-scale testing** (< 5 minutes)
   ```bash
   # Test imports and basic functionality
   python -c "from src.models.hooked_openvla import HookedOpenVLA; print('Works!')"
   ```

3. **Steering vector extraction** (30-60 min, one-time)
   ```bash
   python scripts/extract_steering_vectors.py
   ```
   - This loads the model once and analyzes weights
   - Outputs small files (~100MB) you can reuse
   - Should fit in 11GB

4. **Development and debugging**
   - Modify code
   - Test individual components
   - Quick iterations

### ⚠️ Run on HPC (Heavy Computation)
These require running the full model many times:

1. **Data collection** (~2-4 hours, needs LIBERO)
   ```bash
   python scripts/collect_failure_data.py --n_success 100 --n_failure 100
   ```
   - Runs 200+ rollouts
   - Each rollout = multiple forward passes
   - LIBERO simulation environment

2. **Training failure detector** (~1-2 hours)
   ```bash
   python scripts/train_failure_detector.py --epochs 50
   ```
   - Multiple forward passes through detector
   - Batch processing

3. **Full evaluation** (~2-4 hours)
   ```bash
   python scripts/run_libero_eval.py --n_episodes 50
   ```
   - 50+ episodes with LIBERO
   - Continuous model inference

## Why This Strategy?

### Local Advantages:
- ✅ Fast iteration during development
- ✅ No queue waiting time
- ✅ Easy debugging
- ✅ Already have model downloaded

### HPC Advantages:
- ✅ More VRAM (won't run out of memory)
- ✅ Can run longer experiments
- ✅ Run multiple experiments in parallel
- ✅ LIBERO environment more stable

## Memory-Saving Tips for Local Testing

If you want to try running locally:

1. **Reduce batch size**
   ```python
   # In train_failure_detector.py
   batch_size = 8  # Instead of 32
   ```

2. **Enable gradient checkpointing**
   ```python
   # In hooked_openvla.py
   model.gradient_checkpointing_enable()
   ```

3. **Clear cache frequently**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Use smaller model (if available)**
   - Look for openvla-3b or similar variants

## Next Steps

### Option A: Try Local First (RECOMMENDED)
```bash
# 1. Extract steering vectors locally (one-time)
cd /home/mpcr/Desktop/SalusV5/mist-vla
python scripts/extract_steering_vectors.py

# 2. If it works, great! If OOM, move to HPC
```

### Option B: Go Straight to HPC
```bash
# 1. Transfer code to HPC
rsync -avz mist-vla/ user@hpc:/path/to/mist-vla/

# 2. Install dependencies on HPC
# 3. Run full pipeline
```

## Quick Local Test

Try this small test to see if OpenVLA loads:

```bash
python -c "
import torch
from transformers import AutoModelForVision2Seq

print('Loading OpenVLA...')
model = AutoModelForVision2Seq.from_pretrained(
    'openvla/openvla-7b',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='auto'
)
print(f'✓ Model loaded successfully')
print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"
```

If this succeeds, you can probably run steering vector extraction locally!
