# Next Steps - Ready for HPC Execution

## Current Status ✓

- ✅ All implementation complete (Phases 0-5)
- ✅ All tests passing locally
- ✅ Documentation updated
- ✅ Old code removed
- ✅ Ready for HPC transfer

## Quick Test

```bash
python test_implementation.py
```

Expected output: `✅ All tests passed!`

## Transfer to HPC

```bash
# From local machine
cd /home/mpcr/Desktop/SalusV5/mist-vla

rsync -avz --progress \
    --exclude='data/' \
    --exclude='models/' \
    --exclude='results/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    . asahai2024@athene-login.fau.edu:~/mist-vla/
```

## On HPC: Full Pipeline

### Option 1: Interactive Testing (Recommended First)

```bash
# SSH to HPC
ssh asahai2024@athene-login.fau.edu

# Navigate to project
cd ~/mist-vla

# Activate environment
conda activate mist-vla

# Verify setup
python scripts/verify_phase0.py

# Test small data collection (5 rollouts)
python scripts/collect_phase1_data.py \
    --output data/phase1/test_rollouts.pkl \
    --num-rollouts 5 \
    --max-steps 50
```

### Option 2: Full SLURM Job

```bash
# Submit full pipeline job
sbatch run_hpc.slurm
```

## Pipeline Stages

### Stage 1: Data Collection
```bash
python scripts/collect_phase1_data.py \
    --output data/phase1/rollouts.pkl \
    --num-rollouts 2000 \
    --benchmark libero_spatial \
    --max-steps 200
```
- **Time**: ~6-12 hours
- **Output**: `data/phase1/rollouts.pkl` (~10-20 GB)

### Stage 2: Risk Labels
```bash
python scripts/compute_risk_labels.py \
    --input data/phase1/rollouts.pkl \
    --output data/phase1/labeled_data.pkl \
    --K 10
```
- **Time**: ~5-10 minutes
- **Output**: `data/phase1/labeled_data.pkl`

### Stage 3: Train Risk Predictor
```bash
python scripts/train_risk_predictor.py \
    --data data/phase1/labeled_data.pkl \
    --output-dir models/risk_predictor \
    --epochs 50 \
    --batch-size 256
```
- **Time**: ~2-4 hours
- **Output**: `models/risk_predictor/best_model.pt`
- **Success**: Per-dimension AUC > 0.75

### Stage 4: Extract Steering Vectors
```bash
python scripts/extract_steering_vectors.py \
    --output data/phase3/steering_vectors.pkl \
    --layers 16 20 24 \
    --top-k 20
```
- **Time**: ~1-2 hours
- **Output**: `data/phase3/steering_vectors.pkl`

### Stage 5: Full Evaluation
```bash
python scripts/run_evaluation.py \
    --risk-predictor models/risk_predictor/best_model.pt \
    --steering-vectors data/phase3/steering_vectors.pkl \
    --output-dir results/evaluation \
    --num-episodes 10 \
    --baselines none safe_stop random_steer generic_slow mist
```
- **Time**: ~4-8 hours
- **Output**: `results/evaluation/results.json`, `results/evaluation/results.png`

## Expected Results

### Collision Rate (↓ better)
- none (vanilla): ~30-40%
- safe_stop: ~10-20%
- random_steer: ~25-35%
- generic_slow: ~15-25%
- **mist: ~5-15%** ← Lowest

### Success Rate (↑ better)
- none (vanilla): ~60-70%
- safe_stop: ~40-50% (too much stopping)
- random_steer: ~50-60%
- generic_slow: ~55-65%
- **mist: ~65-75%** ← Highest or second-highest

### Recovery Rate (↑ better)
- none (vanilla): ~20-30%
- safe_stop: ~30-40%
- random_steer: ~35-45%
- generic_slow: ~40-50%
- **mist: ~60-70%** ← Highest

## Monitoring Progress

### Check SLURM job status
```bash
squeue -u $USER
```

### View job output
```bash
tail -f slurm-JOBID.out
```

### Check intermediate results
```bash
# After data collection
ls -lh data/phase1/

# After training
ls -lh models/risk_predictor/
cat models/risk_predictor/training_history.pkl

# After evaluation
cat results/evaluation/results.json
```

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size
```bash
python scripts/train_risk_predictor.py --batch-size 128
```

### Issue: Data collection too slow
**Solution**: Reduce rollouts or max steps
```bash
python scripts/collect_phase1_data.py --num-rollouts 1000 --max-steps 100
```

### Issue: AUC < 0.75
**Solutions**:
1. Collect more data (3000-5000 rollouts)
2. Train longer (100 epochs)
3. Tune hyperparameters (larger hidden dims)

### Issue: Steering vectors all None
**Solutions**:
1. Lower threshold (`--threshold 0.05`)
2. More layers (`--layers 12 16 20 24 28`)
3. Different model checkpoint

## Quick Commands Reference

```bash
# Verify setup
python scripts/verify_phase0.py

# Quick test (5 rollouts)
python scripts/collect_phase1_data.py --num-rollouts 5 --max-steps 50

# Full pipeline
python scripts/collect_phase1_data.py --num-rollouts 2000
python scripts/compute_risk_labels.py
python scripts/train_risk_predictor.py --epochs 50
python scripts/extract_steering_vectors.py
python scripts/run_evaluation.py

# Check results
cat results/evaluation/results.json
```

## File Sizes to Expect

- `rollouts.pkl`: ~10-20 GB (2000 rollouts)
- `labeled_data.pkl`: ~10-20 GB
- `best_model.pt`: ~10 MB
- `steering_vectors.pkl`: ~50-100 MB
- `results.json`: ~10 KB

## Total Time Estimate

- Data collection: 6-12 hours
- Risk label computation: 5-10 minutes
- Risk predictor training: 2-4 hours
- Steering extraction: 1-2 hours
- Full evaluation: 4-8 hours

**Total: ~14-26 hours (~1-2 days)**

## Success Indicators

✅ Phase 1: `labeled_data.pkl` exists, contains 2000 trajectories
✅ Phase 2: AUC > 0.75 for all dimensions
✅ Phase 3: Steering vectors have norms > 0.01
✅ Phase 5: MIST has lowest collision rate

## Documentation

- `README.md` - Quick start overview
- `claude.md` - Complete specification
- `REAL_PROJECT_SPEC.md` - Detailed phase guide
- `IMPLEMENTATION_SUMMARY.md` - What was built
- `test_implementation.py` - Local verification

## Questions?

See `claude.md` for complete implementation details.
