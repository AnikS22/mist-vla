"""
Train per-dimension risk predictor MLP on collected rollout data.

This script trains an MLP to predict 7-dimensional risk vectors from VLA hidden states.
Supports training on data from multiple VLA models (OpenVLA, OpenVLA-OFT, etc.).
"""
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.training.risk_predictor import RiskPredictor, compute_loss


class RiskDataset(Dataset):
    """Dataset for risk prediction training."""
    
    def __init__(
        self,
        samples: List[Dict],
        normalize: bool = True,
        use_time_to_failure: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of dicts with 'hidden_state', 'risk_label', 'action', etc.
            normalize: Whether to normalize hidden states
            use_time_to_failure: Whether to include time-to-failure as additional target
        """
        self.samples = samples
        self.use_time_to_failure = use_time_to_failure
        
        # Extract arrays
        hidden_states = np.array([s['hidden_state'] for s in samples], dtype=np.float32)
        risk_labels = np.array([s.get('risk_label', np.zeros(7)) for s in samples], dtype=np.float32)
        actions = np.array([s.get('action', np.zeros(7)) for s in samples], dtype=np.float32)
        
        # Normalize hidden states
        if normalize:
            self.hidden_mean = hidden_states.mean(axis=0)
            self.hidden_std = hidden_states.std(axis=0) + 1e-8
            hidden_states = (hidden_states - self.hidden_mean) / self.hidden_std
        else:
            self.hidden_mean = np.zeros(hidden_states.shape[1])
            self.hidden_std = np.ones(hidden_states.shape[1])
        
        self.hidden_states = torch.FloatTensor(hidden_states)
        self.risk_labels = torch.FloatTensor(risk_labels)
        self.actions = torch.FloatTensor(actions)
        
        # Optional: time-to-failure
        if use_time_to_failure:
            time_to_failure = np.array([
                s.get('time_to_failure', -1) for s in samples
            ], dtype=np.float32)
            # Normalize time-to-failure (clip at max_steps)
            max_ttf = time_to_failure[time_to_failure >= 0].max() if (time_to_failure >= 0).any() else 220
            time_to_failure = np.clip(time_to_failure, -1, max_ttf) / max_ttf
            self.time_to_failure = torch.FloatTensor(time_to_failure)
        else:
            self.time_to_failure = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = {
            'hidden_state': self.hidden_states[idx],
            'risk_label': self.risk_labels[idx],
            'action': self.actions[idx],
        }
        if self.time_to_failure is not None:
            item['time_to_failure'] = self.time_to_failure[idx]
        return item


def load_dataset(data_path: Path) -> List[Dict]:
    """Load dataset from pickle file."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'dataset' in data:
        return data['dataset']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown data format in {data_path}")


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    per_dimension: bool = True
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: [N, 7] predicted risks
        targets: [N, 7] target risks
        per_dimension: Whether to compute per-dimension metrics
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Overall metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    metrics['mse'] = float(mse)
    metrics['mae'] = float(mae)
    metrics['rmse'] = float(np.sqrt(mse))
    
    # Per-dimension metrics
    if per_dimension:
        dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        per_dim_mse = np.mean((predictions - targets) ** 2, axis=0)
        per_dim_mae = np.mean(np.abs(predictions - targets), axis=0)
        
        for i, dim_name in enumerate(dim_names):
            metrics[f'mse_{dim_name}'] = float(per_dim_mse[i])
            metrics[f'mae_{dim_name}'] = float(per_dim_mae[i])
        
        # AUC-ROC per dimension (for binary classification)
        # Use adaptive threshold: 10th percentile of non-zero targets
        for i, dim_name in enumerate(dim_names):
            target_dim = targets[:, i]
            pred_dim = predictions[:, i]
            
            # Find adaptive threshold (10th percentile of non-zero targets)
            non_zero_targets = target_dim[target_dim > 0]
            if len(non_zero_targets) > 0:
                threshold = np.percentile(non_zero_targets, 10)
            else:
                threshold = 0.1  # Fallback
            
            # Use lower threshold if still too high
            threshold = min(threshold, 0.1)
            
            target_binary = (target_dim > threshold).astype(int)
            
            if len(np.unique(target_binary)) > 1:  # Need both classes
                try:
                    auc = roc_auc_score(target_binary, pred_dim)
                    ap = average_precision_score(target_binary, pred_dim)
                    metrics[f'auc_{dim_name}'] = float(auc)
                    metrics[f'ap_{dim_name}'] = float(ap)
                except Exception:
                    metrics[f'auc_{dim_name}'] = 0.5
                    metrics[f'ap_{dim_name}'] = 0.0
            else:
                metrics[f'auc_{dim_name}'] = 0.5
                metrics[f'ap_{dim_name}'] = 0.0
    
    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str = 'mse',
    use_weighted_loss: bool = True
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        hidden_state = batch['hidden_state'].to(device)
        risk_label = batch['risk_label'].to(device)
        
        # Forward pass
        predictions = model(hidden_state)
        
        # Compute loss with weighting for positive samples
        if use_weighted_loss:
            # Weight samples with non-zero risk higher
            weights = (risk_label.sum(dim=1) > 0).float() * 10.0 + 1.0  # 10x weight for positive samples
            weights = weights.to(device)
        else:
            weights = None
        
        # Compute loss
        if loss_type == 'mse':
            loss = F.mse_loss(predictions, risk_label, reduction='none')
        elif loss_type == 'mae':
            loss = F.l1_loss(predictions, risk_label, reduction='none')
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(predictions, risk_label, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Average over dimensions
        loss = loss.mean(dim=1)  # [batch]
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        loss = loss.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            hidden_state = batch['hidden_state'].to(device)
            risk_label = batch['risk_label'].to(device)
            
            predictions = model(hidden_state)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(risk_label.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_metrics(predictions, targets)
    
    return metrics, predictions, targets


def main():
    parser = argparse.ArgumentParser(description="Train per-dimension risk predictor")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset pickle file")
    parser.add_argument("--output-dir", type=str, default="checkpoints/risk_predictor", help="Output directory")
    parser.add_argument("--model-name", type=str, default="openvla_oft", help="Model name (for logging)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[512, 256], help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "mae", "huber"], help="Loss type")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--normalize", action="store_true", help="Normalize hidden states")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.data}...")
    samples = load_dataset(Path(args.data))
    print(f"Loaded {len(samples)} samples")
    
    # Create dataset
    dataset = RiskDataset(samples, normalize=args.normalize)
    input_dim = dataset.hidden_states.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Split dataset
    n_total = len(dataset)
    n_test = int(args.test_split * n_total)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val - n_test
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RiskPredictor(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=7,
        dropout=args.dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.loss_type, use_weighted_loss=True)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_loss = val_metrics['mse']
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Val MSE: {val_metrics['mse']:.6f}, Val MAE: {val_metrics['mae']:.6f}")
        
        # Print per-dimension AUC
        dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        aucs = [val_metrics.get(f'auc_{dim}', 0.5) for dim in dim_names]
        print(f"Per-dimension AUC: {' - '.join([f'{d}: {a:.4f}' for d, a in zip(dim_names, aucs)])}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'model_config': {
                'input_dim': input_dim,
                'hidden_dims': args.hidden_dims,
                'output_dim': 7,
                'dropout': args.dropout,
            },
            'normalization': {
                'mean': dataset.hidden_mean.tolist(),
                'std': dataset.hidden_std.tolist(),
            }
        }
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"✓ Saved best model (val_loss={val_loss:.6f})")
        
        # Save latest checkpoint
        torch.save(checkpoint, output_dir / 'latest.pt')
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("Final Test Evaluation")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_predictions, test_targets = evaluate(model, test_loader, device)
    
    print(f"\nTest Metrics:")
    print(f"  MSE: {test_metrics['mse']:.6f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    
    print(f"\nPer-Dimension Test Metrics:")
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for dim in dim_names:
        mse = test_metrics.get(f'mse_{dim}', 0.0)
        mae = test_metrics.get(f'mae_{dim}', 0.0)
        auc = test_metrics.get(f'auc_{dim}', 0.5)
        ap = test_metrics.get(f'ap_{dim}', 0.0)
        print(f"  {dim.upper():8s}: MSE={mse:.6f}, MAE={mae:.6f}, AUC={auc:.4f}, AP={ap:.4f}")
    
    # Save final results
    results = {
        'test_metrics': test_metrics,
        'history': history,
        'config': vars(args),
        'model_name': args.model_name,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'results.json'}")
    print(f"✓ Best model saved to {output_dir / 'best_model.pt'}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
