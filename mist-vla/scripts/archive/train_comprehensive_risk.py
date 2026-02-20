#!/usr/bin/env python3
"""Train comprehensive risk predictor for course correction.

Predicts multiple targets from VLA hidden states:
1. Time-to-failure (TTF): How many steps until failure? (regression)
2. Per-dimension risk: Which dimensions are risky? (regression, 0-1)
3. Risk magnitude: How severe is the overall risk? (regression)
4. Failure trajectory: Is this from a failure episode? (classification - auxiliary)

This enables:
- Early warning of impending failures
- Identifying which action dimensions need correction
- Estimating severity for prioritization
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import json


class ComprehensiveRiskPredictor(nn.Module):
    """Multi-task MLP for comprehensive risk prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        
        # Shared encoder with residual connections
        layers = []
        curr_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(curr_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            ])
            curr_dim = out_dim
        self.encoder = nn.Sequential(*layers)
        
        feat_dim = hidden_dim // 2
        
        # Head 1: Time-to-failure prediction (regression)
        self.ttf_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus()  # TTF is non-negative
        )
        
        # Head 2: Per-dimension risk (7 dimensions, regression 0-1)
        self.per_dim_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 7),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Head 3: Overall risk magnitude (regression 0-1)
        self.magnitude_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 4: Failure trajectory classifier (auxiliary)
        self.failure_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        ttf = self.ttf_head(features).squeeze(-1)
        per_dim_risk = self.per_dim_head(features)
        magnitude = self.magnitude_head(features).squeeze(-1)
        failure_logit = self.failure_head(features).squeeze(-1)
        
        return {
            'ttf': ttf,
            'per_dim_risk': per_dim_risk,
            'magnitude': magnitude,
            'failure_logit': failure_logit
        }


def create_comprehensive_dataset(data_path: str, decay: float = 50.0):
    """
    Create dataset with properly scaled labels for all prediction targets.
    
    Labels:
    - ttf: Time-to-failure (0 for success, actual TTF for failure)
    - per_dim_risk: Per-dimension risk scores (scaled)
    - magnitude: Overall risk magnitude
    - is_failure: Binary indicator
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['dataset']
    
    hidden_states = []
    ttf_labels = []
    per_dim_labels = []
    magnitude_labels = []
    is_failure_labels = []
    
    for s in samples:
        hidden_states.append(s['hidden_state'])
        
        # TTF: Use actual time-to-failure, capped and normalized
        raw_ttf = s.get('time_to_failure', -1)
        if raw_ttf < 0:
            ttf = 0.0  # Success trajectory
        else:
            ttf = min(raw_ttf, 200) / 200.0  # Normalize to [0, 1], cap at 200
        ttf_labels.append(ttf)
        
        # Per-dimension risk: Scale up for better learning
        risk = s['risk_label']
        # Apply sqrt to spread out the small values
        scaled_risk = np.sqrt(risk) * 2  # Scale up
        scaled_risk = np.clip(scaled_risk, 0, 1)
        per_dim_labels.append(scaled_risk)
        
        # Magnitude: Overall risk level
        magnitude = np.sqrt(risk.sum()) if risk.sum() > 0 else 0.0
        magnitude = min(magnitude, 1.0)
        magnitude_labels.append(magnitude)
        
        # Is failure
        is_failure_labels.append(float(s['is_failure']))
    
    hidden_states = np.array(hidden_states, dtype=np.float32)
    ttf_labels = np.array(ttf_labels, dtype=np.float32)
    per_dim_labels = np.array(per_dim_labels, dtype=np.float32)
    magnitude_labels = np.array(magnitude_labels, dtype=np.float32)
    is_failure_labels = np.array(is_failure_labels, dtype=np.float32)
    
    print(f'\n=== Dataset Statistics ===')
    print(f'Total samples: {len(samples)}')
    print(f'\nTTF labels:')
    print(f'  Range: [{ttf_labels.min():.4f}, {ttf_labels.max():.4f}]')
    print(f'  Mean: {ttf_labels.mean():.4f}')
    print(f'  Non-zero: {(ttf_labels > 0).sum()} ({100*(ttf_labels > 0).mean():.1f}%)')
    
    print(f'\nPer-dim risk labels (scaled):')
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for i, name in enumerate(dim_names):
        col = per_dim_labels[:, i]
        print(f'  {name}: mean={col.mean():.4f}, max={col.max():.4f}, >0={100*(col>0).mean():.1f}%')
    
    print(f'\nMagnitude labels:')
    print(f'  Range: [{magnitude_labels.min():.4f}, {magnitude_labels.max():.4f}]')
    print(f'  Mean: {magnitude_labels.mean():.4f}')
    
    print(f'\nFailure labels:')
    print(f'  Positive: {is_failure_labels.sum():.0f} ({100*is_failure_labels.mean():.1f}%)')
    
    return hidden_states, ttf_labels, per_dim_labels, magnitude_labels, is_failure_labels


def compute_loss(outputs, targets, weights):
    """Compute weighted multi-task loss."""
    ttf_pred = outputs['ttf']
    per_dim_pred = outputs['per_dim_risk']
    magnitude_pred = outputs['magnitude']
    failure_logit = outputs['failure_logit']
    
    ttf_target, per_dim_target, magnitude_target, failure_target = targets
    
    # TTF loss: Smooth L1 for robustness
    ttf_loss = F.smooth_l1_loss(ttf_pred, ttf_target)
    
    # Per-dim loss: MSE with higher weight on positive samples
    per_dim_weight = 1.0 + 4.0 * per_dim_target  # Higher weight for risky samples
    per_dim_loss = (per_dim_weight * (per_dim_pred - per_dim_target) ** 2).mean()
    
    # Magnitude loss: MSE
    magnitude_loss = F.mse_loss(magnitude_pred, magnitude_target)
    
    # Failure loss: BCE with logits
    failure_loss = F.binary_cross_entropy_with_logits(failure_logit, failure_target)
    
    # Weighted combination
    total_loss = (
        weights['ttf'] * ttf_loss +
        weights['per_dim'] * per_dim_loss +
        weights['magnitude'] * magnitude_loss +
        weights['failure'] * failure_loss
    )
    
    return total_loss, {
        'ttf': ttf_loss.item(),
        'per_dim': per_dim_loss.item(),
        'magnitude': magnitude_loss.item(),
        'failure': failure_loss.item()
    }


def train_epoch(model, loader, optimizer, device, loss_weights):
    model.train()
    total_losses = {'total': 0, 'ttf': 0, 'per_dim': 0, 'magnitude': 0, 'failure': 0}
    n_batches = 0
    
    for batch in tqdm(loader, desc='Training', leave=False):
        hidden, ttf, per_dim, magnitude, failure = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(hidden)
        
        loss, loss_dict = compute_loss(
            outputs, (ttf, per_dim, magnitude, failure), loss_weights
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_losses['total'] += loss.item()
        for k, v in loss_dict.items():
            total_losses[k] += v
        n_batches += 1
    
    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    
    all_preds = {'ttf': [], 'per_dim': [], 'magnitude': [], 'failure': []}
    all_targets = {'ttf': [], 'per_dim': [], 'magnitude': [], 'failure': []}
    
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        hidden, ttf, per_dim, magnitude, failure = [b.to(device) for b in batch]
        
        outputs = model(hidden)
        
        all_preds['ttf'].append(outputs['ttf'].cpu().numpy())
        all_preds['per_dim'].append(outputs['per_dim_risk'].cpu().numpy())
        all_preds['magnitude'].append(outputs['magnitude'].cpu().numpy())
        all_preds['failure'].append(torch.sigmoid(outputs['failure_logit']).cpu().numpy())
        
        all_targets['ttf'].append(ttf.cpu().numpy())
        all_targets['per_dim'].append(per_dim.cpu().numpy())
        all_targets['magnitude'].append(magnitude.cpu().numpy())
        all_targets['failure'].append(failure.cpu().numpy())
    
    # Concatenate
    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k])
        all_targets[k] = np.concatenate(all_targets[k])
    
    metrics = {}
    
    # TTF metrics
    metrics['ttf_mae'] = mean_absolute_error(all_targets['ttf'], all_preds['ttf'])
    metrics['ttf_r2'] = r2_score(all_targets['ttf'], all_preds['ttf'])
    
    # Per-dimension metrics
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    per_dim_maes = []
    per_dim_aucs = []
    per_dim_r2s = []
    
    for i, name in enumerate(dim_names):
        pred = all_preds['per_dim'][:, i]
        target = all_targets['per_dim'][:, i]
        
        mae = mean_absolute_error(target, pred)
        per_dim_maes.append(mae)
        metrics[f'{name}_mae'] = mae
        
        # R2 score
        r2 = r2_score(target, pred) if target.std() > 0 else 0.0
        per_dim_r2s.append(r2)
        metrics[f'{name}_r2'] = r2
        
        # AUC for binary thresholded
        binary_target = (target > 0.1).astype(float)
        if binary_target.sum() > 0 and binary_target.sum() < len(binary_target):
            try:
                auc = roc_auc_score(binary_target, pred)
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5
        per_dim_aucs.append(auc)
        metrics[f'{name}_auc'] = auc
    
    metrics['mean_per_dim_mae'] = np.mean(per_dim_maes)
    metrics['mean_per_dim_auc'] = np.mean(per_dim_aucs)
    metrics['mean_per_dim_r2'] = np.mean(per_dim_r2s)
    
    # Magnitude metrics
    metrics['magnitude_mae'] = mean_absolute_error(all_targets['magnitude'], all_preds['magnitude'])
    metrics['magnitude_r2'] = r2_score(all_targets['magnitude'], all_preds['magnitude'])
    
    # Failure classification metrics
    try:
        metrics['failure_auc'] = roc_auc_score(all_targets['failure'], all_preds['failure'])
    except ValueError:
        metrics['failure_auc'] = 0.5
    
    return metrics, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description='Train comprehensive risk predictor')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='checkpoints/comprehensive_risk',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Loss weights
    parser.add_argument('--w-ttf', type=float, default=1.0, help='TTF loss weight')
    parser.add_argument('--w-per-dim', type=float, default=2.0, help='Per-dim loss weight')
    parser.add_argument('--w-magnitude', type=float, default=1.0, help='Magnitude loss weight')
    parser.add_argument('--w-failure', type=float, default=0.5, help='Failure loss weight')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print('Loading and preparing data...')
    hidden_states, ttf_labels, per_dim_labels, magnitude_labels, is_failure_labels = \
        create_comprehensive_dataset(args.data)
    
    # Split data
    indices = np.arange(len(hidden_states))
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=args.seed)
    
    def make_dataset(idx):
        return TensorDataset(
            torch.FloatTensor(hidden_states[idx]),
            torch.FloatTensor(ttf_labels[idx]),
            torch.FloatTensor(per_dim_labels[idx]),
            torch.FloatTensor(magnitude_labels[idx]),
            torch.FloatTensor(is_failure_labels[idx])
        )
    
    train_dataset = make_dataset(train_idx)
    val_dataset = make_dataset(val_idx)
    test_dataset = make_dataset(test_idx)
    
    print(f'\nTrain: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
    
    # Data loaders with weighted sampling for training
    # Weight samples with higher risk more
    sample_weights = 1.0 + 2.0 * is_failure_labels[train_idx]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = ComprehensiveRiskPredictor(
        input_dim=hidden_states.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    loss_weights = {
        'ttf': args.w_ttf,
        'per_dim': args.w_per_dim,
        'magnitude': args.w_magnitude,
        'failure': args.w_failure
    }
    
    # Training
    best_score = -float('inf')
    print('\nStarting training...\n')
    
    for epoch in range(args.epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device, loss_weights)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        
        # Composite score: prioritize per-dim prediction
        score = val_metrics['mean_per_dim_r2'] + 0.5 * val_metrics['ttf_r2']
        
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'  Train Loss: total={train_losses["total"]:.4f}, ttf={train_losses["ttf"]:.4f}, '
              f'per_dim={train_losses["per_dim"]:.4f}, mag={train_losses["magnitude"]:.4f}')
        print(f'  Val TTF: MAE={val_metrics["ttf_mae"]:.4f}, R²={val_metrics["ttf_r2"]:.4f}')
        print(f'  Val Per-dim: MAE={val_metrics["mean_per_dim_mae"]:.4f}, R²={val_metrics["mean_per_dim_r2"]:.4f}, AUC={val_metrics["mean_per_dim_auc"]:.4f}')
        print(f'  Val Magnitude: MAE={val_metrics["magnitude_mae"]:.4f}, R²={val_metrics["magnitude_r2"]:.4f}')
        print(f'  Val Failure AUC: {val_metrics["failure_auc"]:.4f}')
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f'  ✓ Saved best model (score={score:.4f})')
        print()
    
    # Final evaluation
    print('=' * 70)
    print('FINAL TEST EVALUATION')
    print('=' * 70)
    
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)
    
    print(f'\n📊 Time-to-Failure Prediction:')
    print(f'   MAE: {test_metrics["ttf_mae"]:.4f}')
    print(f'   R²:  {test_metrics["ttf_r2"]:.4f}')
    
    print(f'\n📊 Per-Dimension Risk Prediction:')
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for name in dim_names:
        print(f'   {name.upper():8s}: MAE={test_metrics[f"{name}_mae"]:.4f}, '
              f'R²={test_metrics[f"{name}_r2"]:.4f}, AUC={test_metrics[f"{name}_auc"]:.4f}')
    print(f'   {"MEAN":8s}: MAE={test_metrics["mean_per_dim_mae"]:.4f}, '
          f'R²={test_metrics["mean_per_dim_r2"]:.4f}, AUC={test_metrics["mean_per_dim_auc"]:.4f}')
    
    print(f'\n📊 Risk Magnitude:')
    print(f'   MAE: {test_metrics["magnitude_mae"]:.4f}')
    print(f'   R²:  {test_metrics["magnitude_r2"]:.4f}')
    
    print(f'\n📊 Failure Classification:')
    print(f'   AUC: {test_metrics["failure_auc"]:.4f}')
    
    # Save results
    results = {
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_val_score': float(best_score),
        'args': vars(args)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model info
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': hidden_states.shape[1],
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers
    }, output_dir / 'model_complete.pt')
    
    print(f'\n✅ Training complete! Results saved to {output_dir}')


if __name__ == '__main__':
    main()
