#!/usr/bin/env python3
"""
Phase 2.3: Train per-dimension risk predictor and validate AUC.

This script trains the risk predictor MLP and validates that per-dimension
AUC exceeds the target threshold (0.75).

Success criteria:
- AUC > 0.75 for each dimension
- If not achieved, need to collect more data or tune hyperparameters
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.dataset import RiskPredictionDataset, create_dataloaders, balance_dataset
from src.training.risk_predictor import RiskPredictor, compute_loss


def train_epoch(model, train_loader, optimizer, device, loss_type='mse'):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        hidden = batch['hidden_state'].to(device)
        target = batch['risk_label'].to(device)

        # Forward pass
        pred = model(hidden)
        loss = compute_loss(pred, target, loss_type=loss_type)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': total_loss / n_batches})

    return total_loss / n_batches


def validate(model, val_loader, device):
    """Validate and compute per-dimension AUC."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            hidden = batch['hidden_state'].to(device)
            target = batch['risk_label'].to(device)

            pred = model(hidden)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)  # [n_samples, 7]
    all_targets = np.concatenate(all_targets, axis=0)  # [n_samples, 7]

    # Compute per-dimension AUC
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    aucs = {}

    for i, dim_name in enumerate(dim_names):
        # Binarize targets (risk > 0 means positive)
        binary_targets = (all_targets[:, i] > 0).astype(int)

        # Check if we have both classes
        if len(np.unique(binary_targets)) < 2:
            aucs[dim_name] = None
            continue

        # Compute AUC
        try:
            auc = roc_auc_score(binary_targets, all_preds[:, i])
            aucs[dim_name] = auc
        except Exception as e:
            print(f"  ! Error computing AUC for {dim_name}: {e}")
            aucs[dim_name] = None

    # Compute mean squared error
    mse = np.mean((all_preds - all_targets) ** 2)

    return aucs, mse, all_preds, all_targets


def plot_roc_curves(all_targets, all_preds, save_path):
    """Plot ROC curves for each dimension."""
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, dim_name in enumerate(dim_names):
        ax = axes[i]

        # Binarize targets
        binary_targets = (all_targets[:, i] > 0).astype(int)

        if len(np.unique(binary_targets)) < 2:
            ax.text(0.5, 0.5, 'No positive samples', ha='center', va='center')
            ax.set_title(f'{dim_name}: N/A')
            continue

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(binary_targets, all_preds[:, i])
        auc = roc_auc_score(binary_targets, all_preds[:, i])

        # Plot
        ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{dim_name}: AUC = {auc:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[7])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curves saved to {save_path}")


def train_risk_predictor(
    data_path,
    output_dir,
    input_dim=4096,
    hidden_dims=[512, 256],
    epochs=50,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    balance_data=True,
    target_auc=0.75,
    device='cuda'
):
    """
    Train risk predictor.

    Args:
        data_path: Path to labeled data
        output_dir: Directory to save model and results
        input_dim: Hidden state dimension
        hidden_dims: List of hidden layer dimensions
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        balance_data: Whether to balance positive/negative samples
        target_auc: Target AUC threshold
        device: Device for training
    """
    print("=" * 60)
    print("Phase 2.3: Train Risk Predictor")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"\n[1/6] Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['dataset']
    print(f"  ✓ Loaded {len(samples)} samples")

    # Balance dataset if requested
    if balance_data:
        print("\n[2/6] Balancing dataset...")
        samples = balance_dataset(samples, target_positive_ratio=0.3)
    else:
        print("\n[2/6] Skipping dataset balancing")

    # Create dataset and dataloaders
    print("\n[3/6] Creating dataset and dataloaders...")
    dataset = RiskPredictionDataset(samples, normalize_hidden=True)
    stats = dataset.get_stats()

    print(f"  Dataset statistics:")
    print(f"    Samples: {stats['num_samples']}")
    print(f"    Hidden dim: {stats['hidden_dim']}")
    print(f"    Risk dims: {stats['risk_dims']}")
    print(f"    Positive rate per dim: {[f'{r:.3f}' for r in stats['positive_rate']]}")

    train_loader, val_loader = create_dataloaders(
        dataset,
        train_ratio=0.8,
        batch_size=batch_size,
        num_workers=4
    )
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")

    # Create model
    print("\n[4/6] Creating model...")
    model = RiskPredictor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=7,
        dropout=0.1
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created with {n_params:,} parameters")
    print(f"  Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 7")

    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\n[5/6] Training for {epochs} epochs...")
    best_mean_auc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_auc': [], 'val_mse': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        history['train_loss'].append(train_loss)

        # Validate
        aucs, val_mse, val_preds, val_targets = validate(model, val_loader, device)
        history['val_mse'].append(val_mse)

        # Print metrics
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val MSE: {val_mse:.4f}")
        print(f"  Per-dimension AUC:")

        valid_aucs = [auc for auc in aucs.values() if auc is not None]
        mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        history['val_auc'].append(mean_auc)

        for dim_name, auc in aucs.items():
            if auc is not None:
                status = "✓" if auc > target_auc else "✗"
                print(f"    {status} {dim_name:8s}: {auc:.4f}")
            else:
                print(f"    - {dim_name:8s}: N/A")

        print(f"  Mean AUC: {mean_auc:.4f} (target: {target_auc:.4f})")

        # Save best model
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'aucs': aucs,
                'mean_auc': mean_auc,
                'val_mse': val_mse,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  💾 Saved best model (AUC: {mean_auc:.4f})")

        # Step scheduler
        scheduler.step()

    # Final evaluation
    print(f"\n[6/6] Final Evaluation")
    print(f"  Best epoch: {best_epoch + 1}")
    print(f"  Best mean AUC: {best_mean_auc:.4f}")

    # Check if target met
    all_dims_pass = all(auc > target_auc for auc in aucs.values() if auc is not None)
    mean_pass = best_mean_auc > target_auc

    if mean_pass and all_dims_pass:
        print("\n  ✅ SUCCESS: All dimensions exceed target AUC!")
    elif mean_pass:
        print(f"\n  ⚠️  WARNING: Mean AUC passes but some dimensions below target")
    else:
        print(f"\n  ❌ FAILURE: Mean AUC below target")
        print(f"     Recommendation: Collect more data or tune hyperparameters")

    # Plot ROC curves
    print("\n[7/7] Generating visualizations...")
    plot_roc_curves(
        val_targets, val_preds,
        os.path.join(output_dir, 'roc_curves.png')
    )

    # Save history
    with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f"  ✓ Training history saved")

    print("\n" + "=" * 60)
    print("✅ Phase 2.3 Complete - Phase 2 Done!")
    print("=" * 60)
    print("\nRisk predictor trained! Next steps:")
    print("  1. Extract token-neuron alignments (Phase 3.1)")
    print("  2. Find semantic concept neurons (Phase 3.2)")
    print("\nNext command:")
    print("  python scripts/extract_steering_vectors.py")


def main():
    parser = argparse.ArgumentParser(description="Train per-dimension risk predictor")
    parser.add_argument('--data', type=str, default='data/phase1/labeled_data.pkl',
                        help='Path to labeled data')
    parser.add_argument('--output-dir', type=str, default='models/risk_predictor',
                        help='Directory to save model')
    parser.add_argument('--input-dim', type=int, default=4096,
                        help='Hidden state dimension')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256],
                        help='Hidden layer dimensions')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--no-balance', action='store_true',
                        help='Disable dataset balancing')
    parser.add_argument('--target-auc', type=float, default=0.75,
                        help='Target AUC threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for training')

    args = parser.parse_args()

    train_risk_predictor(
        data_path=args.data,
        output_dir=args.output_dir,
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        balance_data=not args.no_balance,
        target_auc=args.target_auc,
        device=args.device,
    )


if __name__ == "__main__":
    main()
