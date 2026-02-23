# MLP Training Data Optimization Guide

## 🔴 Core Problems Identified

### 1. **Label Sparsity (Root Cause)**
```
Dataset v3 statistics:
- Total samples: 101,720
- Non-zero risk: 5,984 (5.9%)
- Zero risk: 95,736 (94.1%)  ← Model learns to output zeros!
```

The model minimizes MSE by predicting zeros for everything. This is mathematically optimal given the data distribution.

### 2. **No Ground Truth Collisions**
```python
# From HPC analysis:
Failure rollouts with collisions: 0 / 374
Steps with collision flag: 0
Steps with collision geometry: 0
```

Without actual collision data, labels are purely heuristic and may not correlate with hidden states.

### 3. **Binary Classification on Wrong Data**
```python
# Old approach (line 185-187 in improve_labels_multi_signal.py):
if not fail_within_k and not collision_within_k:
    per_dim_risk = np.zeros(7)  # 94% of samples!
```

This creates:
- 94% "negative" samples with zero labels
- 6% "positive" samples with non-zero labels
- Model optimally predicts zeros for everything

---

## 🟢 Fixes Implemented

### Fix 1: **Balanced Binary Classification Dataset**
New script: `prepare_training_data_v2.py`

```python
# Now uses:
- Positive: Last 20 steps of failure trajectories
- Negative: All steps of success trajectories
- Balance: 1:1 ratio (undersample negatives)
```

### Fix 2: **Dual-Head Architecture**
New model: `train_risk_predictor_v2.py`

```
Input (4096D hidden state)
    ↓
Shared Backbone (LayerNorm + GELU + Dropout)
    ↓
┌───────────────┬───────────────┐
│ Binary Head   │ Regression Head│
│ (will fail?)  │ (per-dim risk)│
└───────────────┴───────────────┘
```

### Fix 3: **Focal Loss for Class Imbalance**
```python
class FocalLoss:
    # Down-weights easy (zero) samples
    # Up-weights hard (positive) samples
    focal_loss = alpha * (1 - pt)^gamma * BCE_loss
```

### Fix 4: **Better Metrics**
```python
# Old: AUC with arbitrary 0.5 threshold
# New: 
- Binary AUC (will trajectory fail?)
- Binary AP (precision-recall)
- Balanced Accuracy
- Per-dimension AUC with adaptive threshold
```

---

## 📊 Expected Results

### Before (v3):
```
Per-dimension AUC: All 0.5 (random)
Binary classification: N/A
```

### After (v2):
```
Binary AUC: 0.75-0.85 (expected)
Binary Balanced Acc: 0.70-0.80 (expected)
Per-dimension AUC: 0.55-0.65 (some dimensions)
```

---

## 🚀 How to Use

### Local Test:
```bash
cd mist-vla
bash scripts/test_training_locally.sh
```

### HPC:
```bash
# 1. Prepare balanced dataset
python scripts/prepare_training_data_v2.py \
    --success data/rollouts_oft_eval_big/seed_0/success_rollouts.pkl \
    --failure data/rollouts_oft_eval_big/seed_0/failure_rollouts.pkl \
    --output data/training_datasets/openvla_oft_balanced.pkl \
    --mode binary \
    --k-before-failure 20 \
    --balance-ratio 1.0

# 2. Train improved model
python scripts/train_risk_predictor_v2.py \
    --data data/training_datasets/openvla_oft_balanced.pkl \
    --output-dir checkpoints/risk_predictor_v2 \
    --epochs 100 \
    --batch-size 256 \
    --lr 5e-4
```

---

## 🔬 Why This Works

### The Key Insight:

**Old approach:** Predict per-step risk → 94% zeros → model predicts zeros

**New approach:** 
1. **Binary classification:** "Will this trajectory fail?" 
   - Clear signal: success vs failure
   - Balanced classes: 50/50 split
   - Model learns to distinguish hidden states

2. **Regression only on positives:**
   - Only compute per-dim loss when binary_label=1
   - Avoids learning to predict zeros

3. **Focal loss:**
   - Handles any remaining class imbalance
   - Focuses on hard-to-classify samples

---

## 📈 Next Steps

1. **Test locally** with small data
2. **Sync to HPC** and run on full dataset
3. **Analyze results** - if binary AUC > 0.7, the model is learning
4. **Collect more diverse failures** with perturbations (for better per-dim labels)
5. **LIBERO-90** - expand beyond LIBERO-Spatial

---

## 📁 New Files

| File | Purpose |
|------|---------|
| `scripts/prepare_training_data_v2.py` | Creates balanced binary dataset |
| `scripts/train_risk_predictor_v2.py` | Dual-head model with focal loss |
| `scripts/test_training_locally.sh` | Local test script |
| `MLP_OPTIMIZATION_GUIDE.md` | This documentation |
