"""
Train risk predictor on data from multiple VLA models.

This script orchestrates training on data collected from different VLA models
(OpenVLA, OpenVLA-OFT, etc.) to create a unified risk predictor.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

from train_risk_predictor import load_dataset, RiskDataset, train_epoch, evaluate
from src.training.risk_predictor import RiskPredictor, compute_loss
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset, random_split


def main():
    parser = argparse.ArgumentParser(description="Train risk predictor on multiple models")
    parser.add_argument("--data-config", type=str, required=True, help="JSON config with data paths")
    parser.add_argument("--output-dir", type=str, default="checkpoints/risk_predictor_multi", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "mae", "huber"])
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Load data configuration
    with open(args.data_config, 'r') as f:
        data_config = json.load(f)
    
    print("="*80)
    print("Multi-Model Risk Predictor Training")
    print("="*80)
    print(f"\nData sources: {len(data_config['datasets'])}")
    
    # Load datasets from all sources
    all_datasets = []
    model_names = []
    
    for dataset_info in data_config['datasets']:
        model_name = dataset_info['model_name']
        data_path = Path(dataset_info['data_path'])
        
        print(f"\nLoading {model_name} data from {data_path}...")
        samples = load_dataset(data_path)
        print(f"  Loaded {len(samples)} samples")
        
        dataset = RiskDataset(samples, normalize=True)
        all_datasets.append(dataset)
        model_names.append(model_name)
    
    # Concatenate datasets
    print(f"\nConcatenating {len(all_datasets)} datasets...")
    combined_dataset = ConcatDataset(all_datasets)
    print(f"Total samples: {len(combined_dataset)}")
    
    # Get input dimension (should be same across all)
    input_dim = all_datasets[0].hidden_states.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Split dataset
    n_total = len(combined_dataset)
    n_test = int(args.test_split * n_total)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val - n_test
    
    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
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
    print(f"\nUsing device: {device}")
    
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
    
    # Training loop (same as single-model training)
    print("\nStarting training...")
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.loss_type)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_loss = val_metrics['mse']
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
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
            'data_sources': model_names,
        }
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"✓ Saved best model")
        
        torch.save(checkpoint, output_dir / 'latest.pt')
    
    # Final test evaluation
    print("\n" + "="*80)
    print("Final Test Evaluation")
    print("="*80)
    
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, _, _ = evaluate(model, test_loader, device)
    
    print(f"\nTest Metrics:")
    print(f"  MSE: {test_metrics['mse']:.6f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    print(f"\nPer-Dimension AUC:")
    for dim in dim_names:
        auc = test_metrics.get(f'auc_{dim}', 0.5)
        print(f"  {dim.upper():8s}: {auc:.4f}")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'history': history,
        'data_sources': model_names,
        'config': vars(args),
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'results.json'}")
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
