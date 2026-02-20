# Plan to Improve Risk Predictor Signals

## Current Status

### Training Results (v3 with multi-signal labels)
- **Overall MSE:** 0.00149 (good)
- **Overall MAE:** 0.0128 (good)
- **Per-dimension AUC:** All 0.5 (random - **needs improvement**)
- **AP (Average Precision):** 0.05-0.06 (very low)

### Problem Analysis
1. **Label sparsity:** Only 5.9% of samples have non-zero risk
2. **Small risk values:** Mean risk per dimension: 0.003-0.029
3. **No actual collisions:** 0 collisions detected in failure rollouts
4. **Class imbalance:** 94% of samples have zero risk

## Action Plan

### Phase 1: Collect Data with Actual Collisions (IMMEDIATE)

**Goal:** Get real collision data to create better labels

**Script:** `scripts/collect_diverse_failures.py`
- Apply controlled perturbations (30% of episodes)
- Perturbation types: noise, bias, override
- Target: 100-200 rollouts with actual collisions

**Command:**
```bash
python scripts/collect_diverse_failures.py \
    --model-name openvla/openvla-7b \
    --n-rollouts 200 \
    --perturbation-prob 0.4 \
    --perturbation-strength 0.6 \
    --save-dir data/diverse_failures
```

**Expected outcome:**
- 30-40% of rollouts should have collisions
- Better ground truth for per-dimension labels
- More diverse failure modes

### Phase 2: Improve Labeling Strategy

**Current issues:**
- Labels too conservative (only near failure)
- No collision geometry data
- Risk values too small

**Improvements:**
1. **Lower thresholds:** Use 0.05 instead of 0.1 for risk
2. **Expand time window:** Look back 20 steps instead of 10
3. **Use collision geometry:** When collisions occur, use normal vectors
4. **Action pattern analysis:** Identify risky action sequences

**Script:** `scripts/improve_labels_multi_signal.py` (already created)
- Enhanced with collision geometry
- Better failure direction inference
- Velocity/acceleration patterns

### Phase 3: Retrain with Better Data

**Steps:**
1. Collect diverse failures (Phase 1)
2. Label with improved strategy (Phase 2)
3. Combine with existing data
4. Retrain with:
   - Weighted loss (already implemented)
   - Focal loss (for extreme class imbalance)
   - Balanced sampling

### Phase 4: Alternative Approaches

If AUCs still don't improve:

1. **Focus on working dimensions:**
   - Train separate models for gripper/z (already working)
   - Collect more data for x, y, roll, pitch, yaw

2. **Use regression instead of classification:**
   - Predict continuous risk values
   - Use MSE/MAE as primary metrics
   - AUC is secondary

3. **Multi-task learning:**
   - Predict both time-to-failure and per-dimension risk
   - Share representation

## Immediate Next Steps

1. **Collect diverse failure data** (30 min - 1 hour on HPC)
   ```bash
   sbatch scripts/hpc/collect_diverse_failures.slurm
   ```

2. **Label the new data** with improved strategy
   ```bash
   python scripts/improve_labels_multi_signal.py \
       --input data/diverse_failures/diverse_failure_rollouts.pkl \
       --output data/diverse_failures/diverse_failure_rollouts_labeled.pkl
   ```

3. **Combine datasets and retrain**
   ```bash
   python scripts/prepare_training_data.py \
       --success data/rollouts_oft_eval_big/seed_0/success_rollouts.pkl \
       --failure data/diverse_failures/diverse_failure_rollouts_labeled.pkl \
       --output data/training_datasets/openvla_oft_dataset_v4.pkl \
       --require-labels
   ```

4. **Retrain with combined data**
   ```bash
   sbatch scripts/hpc/train_risk_predictor.slurm
   ```

## Success Criteria

- **Per-dimension AUC > 0.65** for at least 4 dimensions
- **Mean AUC > 0.60** across all dimensions
- **AP > 0.15** for at least 3 dimensions

## Notes

- Current model is learning (MSE/MAE decreasing)
- Issue is binary classification threshold, not regression
- Need more positive samples (collisions) for better AUC
- Gripper and z-axis showed promise in v2 training (0.997, 0.965 AUC)
