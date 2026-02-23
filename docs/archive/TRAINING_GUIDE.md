# MIST-VLA Risk Predictor Training Guide

This guide covers training the per-dimension risk predictor MLP on collected rollout data.

## Overview

The risk predictor is an MLP that takes VLA hidden states (4096D) and predicts 7-dimensional risk vectors, one for each action dimension: [x, y, z, roll, pitch, yaw, gripper].

## Quick Start

### Step 1: Prepare Training Data

First, prepare the dataset from collected rollouts:

```bash
python scripts/prepare_training_data.py \
    --success data/rollouts_oft_eval_big/seed_0/success_rollouts.pkl \
    --failure data/rollouts_oft_eval_big/seed_0/failure_rollouts.pkl \
    --output data/training_datasets/openvla_oft_dataset.pkl
```

If your rollouts already have `per_dim_risk` labels, use `--require-labels` to skip heuristic labeling.

### Step 2: Train Single Model

Train on data from one VLA model:

```bash
python scripts/train_risk_predictor.py \
    --data data/training_datasets/openvla_oft_dataset.pkl \
    --output-dir checkpoints/risk_predictor_openvla_oft \
    --model-name openvla_oft \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-3 \
    --normalize
```

### Step 3: Train Multi-Model (Optional)

To train on data from multiple VLA models:

1. Create a config file (`configs/multi_model_training.json`):
```json
{
  "datasets": [
    {
      "model_name": "openvla_oft",
      "data_path": "data/training_datasets/openvla_oft_dataset.pkl"
    },
    {
      "model_name": "openvla",
      "data_path": "data/training_datasets/openvla_dataset.pkl"
    }
  ]
}
```

2. Run training:
```bash
python scripts/train_multi_model.py \
    --data-config configs/multi_model_training.json \
    --output-dir checkpoints/risk_predictor_multi \
    --epochs 50
```

## Training on HPC

### SLURM Script

Use `scripts/hpc/train_risk_predictor.slurm`:

```bash
sbatch scripts/hpc/train_risk_predictor.slurm
```

Or customize:
```bash
sbatch --job-name=train_risk \
       --time=4:00:00 \
       --gres=gpu:1 \
       scripts/hpc/train_risk_predictor.slurm
```

## Training Parameters

### Key Hyperparameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden-dims`: Hidden layer dimensions (default: [512, 256])
- `--dropout`: Dropout rate (default: 0.1)
- `--loss-type`: Loss function - 'mse', 'mae', or 'huber' (default: 'mse')
- `--normalize`: Normalize hidden states (recommended)

### Data Splits

- Training: 70% (default)
- Validation: 20% (default)
- Test: 10% (default)

Adjust with `--val-split` and `--test-split`.

## Evaluation Metrics

The training script computes:

### Overall Metrics
- **MSE**: Mean squared error
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error

### Per-Dimension Metrics
- **MSE per dimension**: Error for each action dimension
- **MAE per dimension**: Absolute error per dimension
- **AUC-ROC per dimension**: Area under ROC curve (binary classification)
- **AP per dimension**: Average precision (binary classification)

### Expected Performance

Based on analysis:
- **Strong correlations** expected for: y, z, pitch (r > 0.8)
- **Moderate correlations** for: x (r ~ 0.67)
- **Yaw** may show positive correlation with failure (larger yaw → higher risk)

## Output Files

Training produces:

- `best_model.pt`: Best model checkpoint (lowest validation loss)
- `latest.pt`: Latest checkpoint
- `results.json`: Final test metrics and training history

### Checkpoint Contents

```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'val_loss': float,
    'val_metrics': dict,
    'model_config': dict,
    'normalization': {
        'mean': list,
        'std': list
    }
}
```

## Using Trained Model

### Load and Use

```python
import torch
from src.training.risk_predictor import RiskPredictor

# Load checkpoint
checkpoint = torch.load('checkpoints/risk_predictor_openvla_oft/best_model.pt')

# Create model
model = RiskPredictor(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict risk
hidden_state = torch.randn(1, 4096)  # Your VLA hidden state
risk = model(hidden_state)  # [1, 7] risk vector
```

### Integration with Steering

The trained risk predictor is used in `src/steering/steering_module.py` to:
1. Predict per-dimension risk from hidden states
2. Identify riskiest dimension
3. Apply targeted activation steering

## Troubleshooting

### Low AUC Scores

- **Check label quality**: Ensure `per_dim_risk` labels are accurate
- **Increase training data**: More diverse rollouts help
- **Adjust loss function**: Try 'huber' for robustness
- **Check data balance**: Ensure both success and failure samples

### Training Instability

- **Reduce learning rate**: Try `--lr 1e-4`
- **Increase batch size**: If memory allows
- **Add gradient clipping**: Already included (max_norm=1.0)
- **Check normalization**: Use `--normalize` flag

### Memory Issues

- **Reduce batch size**: `--batch-size 128`
- **Reduce hidden dimensions**: `--hidden-dims 256 128`
- **Use CPU**: `--device cpu` (slower but works)

## Next Steps

After training:

1. **Evaluate on held-out data**: Use test set metrics
2. **Deploy in steering module**: Integrate with activation steering
3. **Collect more data**: From other VLA models for multi-model training
4. **Ablation studies**: Test different architectures, loss functions

## Paper Integration

For the paper, report:
- Per-dimension AUC-ROC scores
- Comparison: single-model vs. multi-model training
- Ablation: different loss functions, architectures
- Correlation with actual failure rates (from analysis)
