# Research Methods Implemented for Risk Classification

This document summarizes the research-backed techniques implemented to address the sparse label classification problem.

## Problem Summary

**Original Issue:**
- 101,720 samples, only 5,984 (5.9%) with non-zero risk labels
- Model learned to predict zeros (optimal MSE solution)
- AUC = 0.5 (random performance)

---

## Research Methods Implemented

### 1. Contrastive Learning (NT-Xent Loss)
**Source:** SimCLR, MoCo, CLIP

**Implementation:** `ContrastiveEncoder` in `train_risk_predictor_research.py`

```python
# Pull same-class (success/failure) embeddings together
# Push different-class embeddings apart
contrastive_loss = NT-Xent(projections, labels, temperature=0.1)
```

**Why it helps:**
- Learns meaningful embeddings that separate success from failure
- Works with limited labels by leveraging trajectory-level supervision
- Creates a better feature space for downstream classification

---

### 2. Uncertainty Estimation (Epistemic Uncertainty)
**Source:** RACER (ICML 2025), Bayesian Deep Learning

**Implementation:** `UncertaintyHead` + ensemble + MC dropout

```python
# Predict mean AND variance
mean, log_var = uncertainty_head(features)

# Ensemble for epistemic uncertainty
ensemble_preds = [head(features) for head in ensemble_heads]
uncertainty = ensemble_preds.std()

# MC dropout for additional uncertainty
model.train()  # Keep dropout active
mc_samples = [model(x) for _ in range(20)]
```

**Why it helps:**
- Distinguishes "confident negative" from "uncertain"
- Higher uncertainty on wrong predictions
- Enables calibrated predictions

---

### 3. Multi-Task Learning
**Source:** Safe Learning Survey (2024)

**Implementation:** Three heads in `ResearchRiskPredictor`

```python
outputs = {
    'binary_logits': binary_head(features),      # Will fail?
    'ttf_pred': ttf_head(features),              # Time to failure
    'risk_pred': risk_head(features),            # Per-dim risk
}

loss = binary_loss + risk_loss + ttf_weight * ttf_loss + contrastive_loss
```

**Why it helps:**
- Shared representations learn more generalizable features
- Multiple signals reinforce each other
- Auxiliary tasks (TTF) provide additional supervision

---

### 4. Focal Loss for Class Imbalance
**Source:** RetinaNet (ICCV 2017)

**Implementation:**

```python
focal_loss = alpha * (1 - pt)^gamma * BCE_loss
# alpha = 0.25, gamma = 2.0
```

**Why it helps:**
- Down-weights easy negatives (94% of data)
- Focuses learning on hard positives
- Prevents model from ignoring minority class

---

### 5. Data Augmentation (SMOTE-like)
**Source:** SMOTE, Data Augmentation for Imbalanced Learning

**Implementation:** `ResearchDataset`

```python
# Augment positive samples with Gaussian noise
if binary_label > 0.5:
    for _ in range(augment_factor):
        augmented = hidden_state + noise * noise_std
        dataset.append(augmented)
```

**Why it helps:**
- Increases effective positive sample count
- Regularizes model against overfitting
- Creates more diverse training signal

---

### 6. Weighted Random Sampling
**Source:** Class-balanced Learning

**Implementation:**

```python
# Compute inverse class frequency weights
weights = np.where(labels > 0.5, 1/n_pos, 1/n_neg)
sampler = WeightedRandomSampler(weights, len(dataset))
```

**Why it helps:**
- Each batch has balanced classes
- Model sees equal positive/negative examples
- Prevents gradient bias toward majority class

---

### 7. Temporal Risk Decay
**Source:** Safe RL, CMDP formulation

**Implementation:** `prepare_comprehensive_dataset.py`

```python
# Risk decays exponentially as we move away from failure
decay = risk_decay ** steps_from_failure
per_dim_risk = base_risk * decay * action_contribution
```

**Why it helps:**
- ALL failure steps get non-zero labels (not just last k)
- Risk correlates with proximity to failure
- More training signal from limited data

---

### 8. Action-Based Risk Attribution
**Source:** SAFE, SafeVLA

**Implementation:**

```python
# Risk proportional to action contribution
action_mag = np.abs(action)
action_contrib = action_mag / (action_max + 1e-8)

# Alignment with failure direction
alignment = np.abs(action * failure_direction)

# Velocity contribution (sudden changes = risk)
vel_contrib = velocity / max_velocity

per_dim_risk = 0.4*action_contrib + 0.3*alignment + 0.2*vel_contrib + 0.1
```

**Why it helps:**
- Per-dimension labels based on physics
- Larger actions in failure-aligned directions = higher risk
- Sudden velocity changes indicate risky behavior

---

## Expected Results

### Before (v3 - Sparse Labels):
```
Binary AUC: N/A
Per-dim AUC: All 0.5
```

### After (Research Methods):
```
Binary AUC: 0.80-0.90 (expected)
Binary AP: 0.70-0.85 (expected)
Balanced Accuracy: 0.75-0.85 (expected)
Per-dim AUC: 0.60-0.75 (expected for some dims)
```

---

## Files Created

| File | Purpose |
|------|---------|
| `train_risk_predictor_research.py` | Main training with all methods |
| `prepare_comprehensive_dataset.py` | Create balanced, richly-labeled data |
| `train_research_model.slurm` | HPC submission script |

---

## Usage

### Local Test:
```bash
python scripts/prepare_comprehensive_dataset.py \
    --success data/rollouts_oft_eval_test/success_rollouts.pkl \
    --failure data/rollouts_oft_eval_test/failure_rollouts.pkl \
    --output data/training_datasets/local_research.pkl \
    --k-before-failure 30 \
    --include-all-failure-steps

python scripts/train_risk_predictor_research.py \
    --data data/training_datasets/local_research.pkl \
    --epochs 20 \
    --device cpu
```

### HPC:
```bash
sbatch scripts/hpc/train_research_model.slurm
```

---

## References

1. RACER: Epistemic Risk-Sensitive RL (ICML 2025)
2. Safe Learning Survey (arxiv:2512.11908)
3. Deep Collision Encoding (RSS 2024)
4. Latent Activation Editing (NeurIPS 2025)
5. Focal Loss (RetinaNet, ICCV 2017)
6. SimCLR, MoCo (Contrastive Learning)
