#!/usr/bin/env python3
"""Train failure predictor using VLA hidden states.

Research-grade model that predicts:
1. Time-to-failure (TTF): Regression - how many steps until episode ends?
2. Failure probability: Classification - will this episode fail?
3. Action anomaly scores: Per-dimension - how anomalous is each action dimension?

Key insight: We train on ground truth labels (TTF, failure) rather than 
fabricated per-dimension risk. The model learns to detect failure patterns
in the hidden states that correlate with these outcomes.

For course correction, we use gradient-based attribution to identify which
action dimensions the model associates with failure risk.
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import json


class FailurePredictor(nn.Module):
    """
    Multi-task model for failure prediction.
    
    Architecture:
    - Shared encoder processes hidden states
    - TTF head predicts time-to-failure (regression)
    - Failure head predicts failure probability (classification)
    - Action head predicts expected action (for anomaly detection)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, action_dim: int = 7):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        feat_dim = hidden_dim // 2
        
        # TTF prediction
        self.ttf_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Failure classification
        self.failure_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Action prediction (for anomaly detection)
        self.action_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )
        
        self.action_dim = action_dim
    
    def forward(self, x):
        features = self.encoder(x)
        
        ttf = self.ttf_head(features).squeeze(-1)
        failure_logit = self.failure_head(features).squeeze(-1)
        predicted_action = self.action_head(features)
        
        return {
            'ttf': ttf,
            'failure_logit': failure_logit,
            'predicted_action': predicted_action,
            'features': features
        }
    
    def get_action_anomaly(self, x, actual_action):
        """Compute per-dimension action anomaly scores."""
        with torch.no_grad():
            outputs = self.forward(x)
            predicted = outputs['predicted_action']
            
            # Anomaly = squared deviation from prediction
            anomaly = (predicted - actual_action) ** 2
            
            # Normalize per-dimension
            anomaly = anomaly / (anomaly.mean(dim=0, keepdim=True) + 1e-6)
            
            return anomaly


def prepare_data(fail_rollouts, succ_rollouts):
    """Prepare training data from rollouts."""
    
    samples = []
    
    # Process failure rollouts
    for rollout in fail_rollouts:
        features = np.array(rollout['features'])
        actions = np.array(rollout['actions'])
        T = len(features)
        
        for t in range(T):
            ttf = (T - 1 - t) / 200.0  # Normalize TTF
            
            samples.append({
                'hidden_state': features[t].astype(np.float32),
                'action': actions[t].astype(np.float32),
                'ttf': float(ttf),
                'is_failure': 1.0
            })
    
    # Process success rollouts
    for rollout in succ_rollouts:
        features = np.array(rollout['features'])
        actions = np.array(rollout['actions'])
        T = len(features)
        
        for t in range(T):
            samples.append({
                'hidden_state': features[t].astype(np.float32),
                'action': actions[t].astype(np.float32),
                'ttf': 0.0,  # No failure
                'is_failure': 0.0
            })
    
    return samples


def train_epoch(model, loader, optimizer, device, action_weight=0.5):
    model.train()
    losses = {'total': 0, 'ttf': 0, 'failure': 0, 'action': 0}
    n_batches = 0
    
    for batch in tqdm(loader, desc='Training', leave=False):
        hidden, action, ttf, is_failure = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(hidden)
        
        # TTF loss: only for failure samples
        ttf_mask = is_failure > 0.5
        if ttf_mask.any():
            ttf_loss = F.smooth_l1_loss(outputs['ttf'][ttf_mask], ttf[ttf_mask])
        else:
            ttf_loss = torch.tensor(0.0, device=device)
        
        # Failure loss: focal loss for imbalance
        p = torch.sigmoid(outputs['failure_logit'])
        pt = p * is_failure + (1 - p) * (1 - is_failure)
        focal_weight = (1 - pt) ** 2
        failure_loss = (focal_weight * F.binary_cross_entropy_with_logits(
            outputs['failure_logit'], is_failure, reduction='none'
        )).mean()
        
        # Action prediction loss
        action_loss = F.mse_loss(outputs['predicted_action'], action)
        
        # Total loss
        total_loss = ttf_loss + failure_loss + action_weight * action_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses['total'] += total_loss.item()
        losses['ttf'] += ttf_loss.item()
        losses['failure'] += failure_loss.item()
        losses['action'] += action_loss.item()
        n_batches += 1
    
    return {k: v / n_batches for k, v in losses.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    
    all_ttf_pred, all_ttf_target = [], []
    all_fail_pred, all_fail_target = [], []
    all_action_pred, all_action_target = [], []
    
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        hidden, action, ttf, is_failure = [b.to(device) for b in batch]
        
        outputs = model(hidden)
        
        all_ttf_pred.append(outputs['ttf'].cpu().numpy())
        all_ttf_target.append(ttf.cpu().numpy())
        all_fail_pred.append(torch.sigmoid(outputs['failure_logit']).cpu().numpy())
        all_fail_target.append(is_failure.cpu().numpy())
        all_action_pred.append(outputs['predicted_action'].cpu().numpy())
        all_action_target.append(action.cpu().numpy())
    
    ttf_pred = np.concatenate(all_ttf_pred)
    ttf_target = np.concatenate(all_ttf_target)
    fail_pred = np.concatenate(all_fail_pred)
    fail_target = np.concatenate(all_fail_target)
    action_pred = np.concatenate(all_action_pred)
    action_target = np.concatenate(all_action_target)
    
    # TTF metrics (only for failures)
    fail_mask = fail_target > 0.5
    metrics = {}
    
    if fail_mask.sum() > 0:
        metrics['ttf_mae'] = mean_absolute_error(ttf_target[fail_mask], ttf_pred[fail_mask])
        metrics['ttf_r2'] = r2_score(ttf_target[fail_mask], ttf_pred[fail_mask])
    else:
        metrics['ttf_mae'] = 0.0
        metrics['ttf_r2'] = 0.0
    
    # Failure classification
    metrics['failure_auc'] = roc_auc_score(fail_target, fail_pred)
    metrics['failure_acc'] = accuracy_score(fail_target > 0.5, fail_pred > 0.5)
    
    # Action prediction MAE per dimension
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    action_mae = np.abs(action_pred - action_target).mean(axis=0)
    for i, name in enumerate(dim_names):
        metrics[f'action_{name}_mae'] = action_mae[i]
    metrics['action_mean_mae'] = action_mae.mean()
    
    # Compute action anomaly (deviation from prediction) correlation with failure
    action_anomaly = (action_pred - action_target) ** 2
    anomaly_per_sample = action_anomaly.mean(axis=1)
    
    # Check if anomaly correlates with failure
    try:
        metrics['anomaly_failure_corr'] = np.corrcoef(anomaly_per_sample, fail_target)[0, 1]
    except:
        metrics['anomaly_failure_corr'] = 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train failure predictor')
    parser.add_argument('--failure-data', type=str, required=True)
    parser.add_argument('--success-data', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='checkpoints/failure_predictor')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--action-weight', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print('Loading data...')
    with open(args.failure_data, 'rb') as f:
        fail_rollouts = pickle.load(f)
    with open(args.success_data, 'rb') as f:
        succ_rollouts = pickle.load(f)
    
    print(f'Failure rollouts: {len(fail_rollouts)}')
    print(f'Success rollouts: {len(succ_rollouts)}')
    
    # Prepare samples
    samples = prepare_data(fail_rollouts, succ_rollouts)
    print(f'Total samples: {len(samples)}')
    
    # Extract arrays
    hidden_states = np.array([s['hidden_state'] for s in samples])
    actions = np.array([s['action'] for s in samples])
    ttf = np.array([s['ttf'] for s in samples])
    is_failure = np.array([s['is_failure'] for s in samples])
    
    # Rollout-level split to prevent data leakage
    # (consecutive steps within a rollout are highly correlated)
    n_fail = len(fail_rollouts)
    n_succ = len(succ_rollouts)
    fail_perm = np.random.permutation(n_fail)
    succ_perm = np.random.permutation(n_succ)
    
    n_fail_test = max(1, int(0.15 * n_fail))
    n_fail_val = max(1, int(0.15 * n_fail))
    n_succ_test = max(1, int(0.15 * n_succ))
    n_succ_val = max(1, int(0.15 * n_succ))
    
    test_rollout_ids = set(fail_perm[:n_fail_test].tolist() + [n_fail + i for i in succ_perm[:n_succ_test]])
    val_rollout_ids = set(fail_perm[n_fail_test:n_fail_test+n_fail_val].tolist() + [n_fail + i for i in succ_perm[n_succ_test:n_succ_test+n_succ_val]])
    
    # Each sample knows which rollout it came from
    rollout_ids = []
    for i, rollout in enumerate(fail_rollouts):
        rollout_ids.extend([i] * len(rollout['features']))
    for i, rollout in enumerate(succ_rollouts):
        rollout_ids.extend([n_fail + i] * len(rollout['features']))
    rollout_ids = np.array(rollout_ids)
    
    # Apply balance ONLY to training set
    train_mask = ~np.isin(rollout_ids, list(test_rollout_ids | val_rollout_ids))
    test_mask = np.isin(rollout_ids, list(test_rollout_ids))
    val_mask = np.isin(rollout_ids, list(val_rollout_ids))
    
    train_is_fail = is_failure[train_mask]
    train_fail_idx = np.where(train_mask)[0][train_is_fail > 0.5]
    train_succ_idx = np.where(train_mask)[0][train_is_fail < 0.5]
    n_each = min(len(train_fail_idx), len(train_succ_idx))
    
    train_idx = np.concatenate([
        np.random.choice(train_fail_idx, n_each, replace=len(train_fail_idx) < n_each),
        np.random.choice(train_succ_idx, n_each, replace=len(train_succ_idx) < n_each)
    ])
    np.random.shuffle(train_idx)
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    print(f'Rollout-level split (prevents data leakage):')
    print(f'  Train: {len(train_idx)} samples from {n_fail - n_fail_test - n_fail_val}F + {n_succ - n_succ_test - n_succ_val}S rollouts (balanced to {n_each} each)')
    print(f'  Val:   {len(val_idx)} samples from {n_fail_val}F + {n_succ_val}S rollouts')
    print(f'  Test:  {len(test_idx)} samples from {n_fail_test}F + {n_succ_test}S rollouts')
    
    def make_loader(idx, shuffle=True):
        dataset = TensorDataset(
            torch.FloatTensor(hidden_states[idx]),
            torch.FloatTensor(actions[idx]),
            torch.FloatTensor(ttf[idx]),
            torch.FloatTensor(is_failure[idx])
        )
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    
    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)
    
    print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
    
    # Model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = FailurePredictor(
        input_dim=hidden_states.shape[1],
        hidden_dim=args.hidden_dim,
        action_dim=actions.shape[1]
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    best_auc = 0
    print('\nStarting training...\n')
    
    for epoch in range(args.epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device, args.action_weight)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'  Train: total={train_losses["total"]:.4f}, ttf={train_losses["ttf"]:.4f}, '
              f'fail={train_losses["failure"]:.4f}, action={train_losses["action"]:.4f}')
        print(f'  Val: Failure AUC={val_metrics["failure_auc"]:.4f}, Acc={val_metrics["failure_acc"]:.4f}')
        print(f'       TTF MAE={val_metrics["ttf_mae"]:.4f}, R²={val_metrics["ttf_r2"]:.4f}')
        print(f'       Action MAE={val_metrics["action_mean_mae"]:.4f}')
        
        if val_metrics['failure_auc'] > best_auc:
            best_auc = val_metrics['failure_auc']
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f'  ✓ Saved best model (AUC={best_auc:.4f})')
        print()
    
    # Final evaluation
    print('=' * 70)
    print('FINAL TEST EVALUATION')
    print('=' * 70)
    
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_metrics = evaluate(model, test_loader, device)
    
    print(f'\n📊 Failure Classification:')
    print(f'   AUC: {test_metrics["failure_auc"]:.4f}')
    print(f'   Accuracy: {test_metrics["failure_acc"]:.4f}')
    
    print(f'\n📊 Time-to-Failure Prediction:')
    print(f'   MAE: {test_metrics["ttf_mae"]:.4f} (normalized)')
    print(f'   R²:  {test_metrics["ttf_r2"]:.4f}')
    
    print(f'\n📊 Action Prediction (for anomaly detection):')
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for name in dim_names:
        print(f'   {name.upper():8s}: MAE={test_metrics[f"action_{name}_mae"]:.4f}')
    print(f'   {"MEAN":8s}: MAE={test_metrics["action_mean_mae"]:.4f}')
    
    print(f'\n📊 Anomaly-Failure Correlation: {test_metrics["anomaly_failure_corr"]:.4f}')
    print('   (Higher = action anomaly correlates with failure)')
    
    # Save results
    results = {
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_val_auc': float(best_auc),
        'args': vars(args)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': hidden_states.shape[1],
        'hidden_dim': args.hidden_dim,
        'action_dim': actions.shape[1]
    }, output_dir / 'model_complete.pt')
    
    print(f'\n✅ Training complete! Results saved to {output_dir}')


if __name__ == '__main__':
    main()
