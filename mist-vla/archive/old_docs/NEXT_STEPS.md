# MIST-VLA Next Steps for Paper

This document outlines the remaining steps to complete the MIST-VLA paper.

## ✅ Completed

1. **Data Collection**
   - ✅ Collected 163+ success and 237+ failure rollouts using OpenVLA-OFT
   - ✅ Extracted 4096D hidden states from VLA transformer
   - ✅ Collected 7D actions, robot states, collision information
   - ✅ Generated per-dimension risk labels using collision geometry

2. **Analysis**
   - ✅ Analyzed internal states and correlations with actions
   - ✅ Identified failure patterns per action dimension
   - ✅ Found strong correlations: y, z, pitch (r > 0.8)
   - ✅ Discovered yaw shows positive correlation with failure

3. **Infrastructure**
   - ✅ Risk predictor MLP architecture
   - ✅ Training pipeline scripts
   - ✅ Multi-model training support
   - ✅ HPC training scripts

## 🚀 Next Steps

### Phase 1: Train Risk Predictor (Current Priority)

#### Step 1.1: Prepare Training Data
```bash
# On HPC
python scripts/prepare_training_data.py \
    --success data/rollouts_oft_eval_big/seed_0/success_rollouts.pkl \
    --failure data/rollouts_oft_eval_big/seed_0/failure_rollouts.pkl \
    --output data/training_datasets/openvla_oft_dataset.pkl
```

#### Step 1.2: Train Initial Model
```bash
# On HPC
sbatch scripts/hpc/train_risk_predictor.slurm
```

**Expected Results:**
- Per-dimension AUC-ROC > 0.7 for y, z, pitch
- Per-dimension AUC-ROC > 0.6 for x
- Lower AUC for roll, yaw (expected from analysis)

#### Step 1.3: Evaluate and Iterate
- Check test metrics
- Adjust hyperparameters if needed
- Collect more data if performance is low

### Phase 2: Collect Data from Other Models

#### Step 2.1: Collect from Base OpenVLA
- Use `scripts/collect_failure_data.py` with `--model-type openvla`
- Target: 20 successes, 40 failures
- Compare hidden state distributions with OpenVLA-OFT

#### Step 2.2: Collect from Other VLA Models (Optional)
- RT-1-X, RT-2, etc. (if available)
- Goal: Demonstrate generalization across VLA architectures

#### Step 2.3: Multi-Model Training
```bash
# After collecting from multiple models
python scripts/train_multi_model.py \
    --data-config configs/multi_model_training.json \
    --output-dir checkpoints/risk_predictor_multi
```

### Phase 3: Activation Steering Integration

#### Step 3.1: Integrate Trained Predictor
- Update `src/steering/steering_module.py` to use trained model
- Load checkpoint and replace heuristic risk prediction

#### Step 3.2: Test Steering in Simulation
- Run evaluation with steering enabled
- Measure: success rate, collision rate, recovery rate
- Compare: baseline vs. steering

#### Step 3.3: Ablation Studies
- Test different steering layers
- Test different steering strengths
- Test per-dimension vs. generic steering

### Phase 4: Evaluation and Metrics

#### Step 4.1: Compute SafeVLA-Style Metrics
- **Success Rate (SR)**: Task completion rate
- **Collision Rate (CR)**: % episodes with collisions
- **Recovery Rate (RR)**: Successful interventions / triggered interventions
- **Per-Dimension AUC-ROC**: Risk prediction accuracy

#### Step 4.2: Baselines Comparison
- **No Intervention**: Baseline VLA performance
- **SAFE-Stop**: Stop on failure prediction (0% recovery)
- **Generic Slow**: Reduce all actions uniformly
- **MIST-VLA**: Per-dimension targeted steering

#### Step 4.3: Robustness Evaluation
- Test on LIBERO-Plus perturbations
- Test on unseen tasks
- Test on unseen VLA architectures

### Phase 5: Paper Writing

#### Step 5.1: Results Tables
- **Table 1**: Main results (SR, CR, RR, Per-Dim AUC)
- **Table 2**: Ablation studies
- **Table 3**: Robustness (OOD evaluation)

#### Step 5.2: Figures
- **Figure 1**: System overview
- **Figure 2**: Hidden state analysis (from `analysis_output/`)
- **Figure 3**: Per-dimension failure patterns
- **Figure 4**: Steering visualization
- **Figure 5**: Comparison with baselines

#### Step 5.3: Key Contributions
1. **Per-dimension risk prediction** (vs. binary in SAFE)
2. **Targeted activation steering** (vs. stopping in SAFE)
3. **Recovery rate** (vs. 0% in SAFE-Stop)
4. **Multi-model generalization** (if collected)

## 📊 Current Data Status

### Available Data
- **OpenVLA-OFT**: 163 successes, 237 failures (~7 GB)
- **Hidden states**: 4096D per step
- **Labels**: Per-dimension risk (from collision geometry)

### Data Quality
- ✅ High success rate (~40% matches expected)
- ✅ Diverse failure modes
- ✅ Accurate collision detection
- ✅ Per-dimension labels from geometry

## 🎯 Success Criteria

### Risk Predictor
- [ ] Per-dimension AUC-ROC > 0.7 for at least 4 dimensions
- [ ] Mean per-dimension AUC-ROC > 0.65
- [ ] Test MSE < 0.1

### Steering Performance
- [ ] Recovery Rate > 0.7
- [ ] Collision Rate reduction > 30%
- [ ] Success Rate maintained or improved

### Paper Metrics
- [ ] All SafeVLA-style metrics computed
- [ ] Baselines compared
- [ ] Ablation studies complete

## 🔧 Quick Commands

### Prepare Data
```bash
python scripts/prepare_training_data.py \
    --success data/rollouts_oft_eval_big/seed_0/success_rollouts.pkl \
    --failure data/rollouts_oft_eval_big/seed_0/failure_rollouts.pkl \
    --output data/training_datasets/openvla_oft_dataset.pkl
```

### Train Model
```bash
python scripts/train_risk_predictor.py \
    --data data/training_datasets/openvla_oft_dataset.pkl \
    --output-dir checkpoints/risk_predictor_openvla_oft \
    --epochs 50 --normalize
```

### Evaluate Model
```python
import torch
checkpoint = torch.load('checkpoints/risk_predictor_openvla_oft/best_model.pt')
print(checkpoint['val_metrics'])
```

## 📝 Notes

- **Training time**: ~1-2 hours on A100 for 50 epochs
- **Data size**: ~500MB-1GB per dataset
- **Model size**: ~10M parameters
- **Memory**: ~8GB GPU memory needed

## 🚨 Potential Issues

1. **Low AUC scores**: May need better labels or more data
2. **Overfitting**: Use dropout, early stopping
3. **Class imbalance**: Weighted loss or sampling
4. **Hidden state mismatch**: Normalize across models

## 📚 References

- See `TRAINING_GUIDE.md` for detailed training instructions
- See `RESEARCH_DATA_INVENTORY.md` for data locations
- See `analysis_output/` for analysis visualizations

---

**Last Updated**: January 30, 2025
**Status**: Ready for Phase 1 (Training)
