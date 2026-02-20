"""
Improved risk predictor training with:
1. Focal loss for class imbalance
2. Binary classification head
3. Per-dimension regression head
4. Better metrics (balanced accuracy, PR-AUC)
"""
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    balanced_accuracy_score, f1_score, precision_recall_curve
)
from tqdm import tqdm


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class ImprovedRiskPredictor(nn.Module):
    """
    Risk predictor with dual heads:
    1. Binary classification: will this lead to failure?
    2. Per-dimension regression: risk in each dimension
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [1024, 512, 256],
        output_dim: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Shared backbone
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1),
        )
        
        # Per-dimension regression head
        self.regression_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], output_dim),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        binary_logits = self.binary_head(features)
        risk_scores = torch.sigmoid(self.regression_head(features))
        return binary_logits.squeeze(-1), risk_scores


class RiskDatasetV2(Dataset):
    """Dataset with binary and regression labels."""
    
    def __init__(self, samples: List[Dict], normalize: bool = True):
        hidden_states = np.array([s['hidden_state'] for s in samples], dtype=np.float32)
        risk_labels = np.array([s.get('risk_label', np.zeros(7)) for s in samples], dtype=np.float32)
        binary_labels = np.array([s.get('binary_label', 0) for s in samples], dtype=np.float32)
        
        # Normalize
        if normalize:
            self.mean = hidden_states.mean(axis=0)
            self.std = hidden_states.std(axis=0) + 1e-8
            hidden_states = (hidden_states - self.mean) / self.std
        else:
            self.mean = np.zeros(hidden_states.shape[1])
            self.std = np.ones(hidden_states.shape[1])
        
        self.hidden_states = torch.FloatTensor(hidden_states)
        self.risk_labels = torch.FloatTensor(risk_labels)
        self.binary_labels = torch.FloatTensor(binary_labels)
    
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        return {
            'hidden_state': self.hidden_states[idx],
            'risk_label': self.risk_labels[idx],
            'binary_label': self.binary_labels[idx],
        }


def compute_metrics(
    binary_preds: np.ndarray,
    binary_targets: np.ndarray,
    risk_preds: np.ndarray,
    risk_targets: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    metrics = {}
    
    # Binary classification metrics
    binary_probs = 1 / (1 + np.exp(-binary_preds))  # Sigmoid
    binary_pred_labels = (binary_probs > 0.5).astype(int)
    
    try:
        metrics['binary_auc'] = roc_auc_score(binary_targets, binary_probs)
        metrics['binary_ap'] = average_precision_score(binary_targets, binary_probs)
    except:
        metrics['binary_auc'] = 0.5
        metrics['binary_ap'] = 0.5
    
    metrics['binary_accuracy'] = (binary_pred_labels == binary_targets).mean()
    metrics['binary_balanced_acc'] = balanced_accuracy_score(binary_targets, binary_pred_labels)
    
    if len(np.unique(binary_targets)) > 1:
        metrics['binary_f1'] = f1_score(binary_targets, binary_pred_labels)
    else:
        metrics['binary_f1'] = 0.0
    
    # Regression metrics
    metrics['risk_mse'] = float(np.mean((risk_preds - risk_targets) ** 2))
    metrics['risk_mae'] = float(np.mean(np.abs(risk_preds - risk_targets)))
    
    # Per-dimension AUC (for steps with non-zero targets)
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for i, dim in enumerate(dim_names):
        target_dim = risk_targets[:, i]
        pred_dim = risk_preds[:, i]
        
        # Use median as threshold for binary
        threshold = np.median(target_dim[target_dim > 0]) if (target_dim > 0).any() else 0.1
        threshold = max(threshold, 0.01)  # Minimum threshold
        
        target_binary = (target_dim > threshold).astype(int)
        
        if len(np.unique(target_binary)) > 1:
            try:
                metrics[f'auc_{dim}'] = roc_auc_score(target_binary, pred_dim)
            except:
                metrics[f'auc_{dim}'] = 0.5
        else:
            metrics[f'auc_{dim}'] = 0.5
    
    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    focal_loss: FocalLoss,
    regression_weight: float = 0.5,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()
    total_binary_loss = 0.0
    total_regression_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        hidden_state = batch['hidden_state'].to(device)
        binary_label = batch['binary_label'].to(device)
        risk_label = batch['risk_label'].to(device)
        
        # Forward
        binary_logits, risk_scores = model(hidden_state)
        
        # Binary loss (focal)
        binary_loss = focal_loss(binary_logits, binary_label)
        
        # Regression loss (only on positive samples)
        positive_mask = binary_label > 0.5
        if positive_mask.any():
            regression_loss = F.mse_loss(
                risk_scores[positive_mask], 
                risk_label[positive_mask]
            )
        else:
            regression_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        loss = binary_loss + regression_weight * regression_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_binary_loss += binary_loss.item()
        total_regression_loss += regression_loss.item()
        n_batches += 1
    
    return {
        'binary_loss': total_binary_loss / n_batches,
        'regression_loss': total_regression_loss / n_batches,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model."""
    model.eval()
    all_binary_preds = []
    all_binary_targets = []
    all_risk_preds = []
    all_risk_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            hidden_state = batch['hidden_state'].to(device)
            binary_label = batch['binary_label']
            risk_label = batch['risk_label']
            
            binary_logits, risk_scores = model(hidden_state)
            
            all_binary_preds.append(binary_logits.cpu().numpy())
            all_binary_targets.append(binary_label.numpy())
            all_risk_preds.append(risk_scores.cpu().numpy())
            all_risk_targets.append(risk_label.numpy())
    
    binary_preds = np.concatenate(all_binary_preds)
    binary_targets = np.concatenate(all_binary_targets)
    risk_preds = np.concatenate(all_risk_preds)
    risk_targets = np.concatenate(all_risk_targets)
    
    metrics = compute_metrics(binary_preds, binary_targets, risk_preds, risk_targets)
    
    return metrics, risk_preds, risk_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/risk_predictor_v2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[1024, 512, 256])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading {args.data}...")
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['dataset']
    print(f"Loaded {len(samples)} samples")
    
    # Dataset
    dataset = RiskDatasetV2(samples, normalize=True)
    input_dim = dataset.hidden_states.shape[1]
    
    # Split
    n_total = len(dataset)
    n_test = int(0.1 * n_total)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val - n_test
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = ImprovedRiskPredictor(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Training
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    focal_loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    
    best_val_auc = 0.0
    history = []
    
    print("\nTraining...")
    for epoch in range(args.epochs):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device, focal_loss, args.regression_weight
        )
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        
        history.append({
            'epoch': epoch,
            'train_losses': train_losses,
            'val_metrics': val_metrics,
        })
        
        # Print
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train: binary_loss={train_losses['binary_loss']:.4f}, reg_loss={train_losses['regression_loss']:.4f}")
        print(f"  Val: AUC={val_metrics['binary_auc']:.4f}, AP={val_metrics['binary_ap']:.4f}, "
              f"Balanced Acc={val_metrics['binary_balanced_acc']:.4f}")
        
        dim_aucs = [val_metrics.get(f'auc_{d}', 0.5) for d in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']]
        print(f"  Per-dim AUC: {' | '.join([f'{a:.3f}' for a in dim_aucs])}")
        
        # Save best
        if val_metrics['binary_auc'] > best_val_auc:
            best_val_auc = val_metrics['binary_auc']
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'epoch': epoch,
                'config': {
                    'input_dim': input_dim,
                    'hidden_dims': args.hidden_dims,
                    'dropout': args.dropout,
                },
                'normalization': {
                    'mean': dataset.mean.tolist(),
                    'std': dataset.std.tolist(),
                }
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (AUC={best_val_auc:.4f})")
    
    # Final test
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, _, _ = evaluate(model, test_loader, device)
    
    print(f"\nBinary Classification:")
    print(f"  AUC: {test_metrics['binary_auc']:.4f}")
    print(f"  AP: {test_metrics['binary_ap']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['binary_balanced_acc']:.4f}")
    print(f"  F1: {test_metrics['binary_f1']:.4f}")
    
    print(f"\nPer-Dimension AUC:")
    for dim in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']:
        print(f"  {dim.upper()}: {test_metrics.get(f'auc_{dim}', 0.5):.4f}")
    
    print(f"\nRegression:")
    print(f"  MSE: {test_metrics['risk_mse']:.6f}")
    print(f"  MAE: {test_metrics['risk_mae']:.6f}")
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'config': vars(args),
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}")
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
