"""
Research-backed risk predictor combining:
1. Contrastive Learning - separate success/failure embeddings
2. Uncertainty Estimation - bootstrap ensemble with MC dropout
3. Multi-task Learning - binary + time-to-failure + per-dimension
4. Temporal Context - sequence of hidden states
5. Class-balanced sampling with SMOTE-like augmentation

Based on:
- RACER: Epistemic Risk-Sensitive RL (ICML 2025)
- Deep Collision Encoding (RSS 2024)
- Latent Activation Editing (NeurIPS 2025)
"""
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    balanced_accuracy_score, f1_score
)
from tqdm import tqdm


# ============================================================================
# CONTRASTIVE LEARNING: Learn embeddings that separate success from failure
# ============================================================================

class ContrastiveEncoder(nn.Module):
    """Encodes hidden states into risk-aware embeddings using contrastive learning."""
    
    def __init__(self, input_dim=4096, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, x):
        embed = self.encoder(x)
        proj = self.projector(embed)
        return F.normalize(embed, dim=-1), F.normalize(proj, dim=-1)


def contrastive_loss(proj1, proj2, labels, temperature=0.1):
    """
    NT-Xent loss: pull same-class embeddings together, push different apart.
    labels: 0=success, 1=failure
    """
    batch_size = proj1.shape[0]
    
    # Concatenate projections
    proj = torch.cat([proj1, proj2], dim=0)
    labels_cat = torch.cat([labels, labels], dim=0)
    
    # Compute similarity matrix
    sim = torch.matmul(proj, proj.T) / temperature
    
    # Create mask for positive pairs (same label)
    mask = (labels_cat.unsqueeze(0) == labels_cat.unsqueeze(1)).float()
    mask = mask - torch.eye(2 * batch_size, device=mask.device)  # Remove diagonal
    
    # InfoNCE loss
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Average log prob for positive pairs
    pos_loss = -(mask * log_prob).sum() / mask.sum().clamp(min=1)
    
    return pos_loss


# ============================================================================
# UNCERTAINTY ESTIMATION: Bootstrap ensemble with MC dropout
# ============================================================================

class UncertaintyHead(nn.Module):
    """Predicts both mean and variance for uncertainty estimation."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mean_head = nn.Linear(input_dim, output_dim)
        self.log_var_head = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        return mean, log_var
    
    def sample(self, x, n_samples=10):
        """Monte Carlo sampling for uncertainty."""
        mean, log_var = self.forward(x)
        std = torch.exp(0.5 * log_var)
        samples = [mean + std * torch.randn_like(mean) for _ in range(n_samples)]
        return torch.stack(samples).mean(dim=0), torch.stack(samples).std(dim=0)


def gaussian_nll_loss(pred_mean, pred_log_var, target):
    """Negative log likelihood loss for Gaussian with learned variance."""
    precision = torch.exp(-pred_log_var)
    nll = 0.5 * (precision * (target - pred_mean) ** 2 + pred_log_var)
    return nll.mean()


# ============================================================================
# MULTI-TASK MODEL: Combines all components
# ============================================================================

class ResearchRiskPredictor(nn.Module):
    """
    Multi-task risk predictor with:
    - Contrastive encoder (embeddings)
    - Binary classification (will fail?)
    - Time-to-failure regression
    - Per-dimension risk with uncertainty
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 7,
        dropout: float = 0.3,
        n_ensemble: int = 3,
    ):
        super().__init__()
        
        # Contrastive encoder
        self.encoder = ContrastiveEncoder(input_dim, embed_dim, hidden_dim)
        
        # Shared backbone after embedding
        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Task 1: Binary classification (will this trajectory fail?)
        self.binary_head = nn.Linear(hidden_dim // 2, 1)
        
        # Task 2: Time-to-failure regression
        self.ttf_head = nn.Linear(hidden_dim // 2, 1)
        
        # Task 3: Per-dimension risk with uncertainty
        self.risk_head = UncertaintyHead(hidden_dim // 2, output_dim)
        
        # Ensemble members for uncertainty (bootstrap)
        self.ensemble_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, output_dim) for _ in range(n_ensemble)
        ])
        
        self.embed_dim = embed_dim
        self.n_ensemble = n_ensemble
    
    def forward(self, x, return_embeddings=False):
        # Encode
        embed, proj = self.encoder(x)
        
        # Backbone
        features = self.backbone(embed)
        
        # Predictions
        binary_logits = self.binary_head(features).squeeze(-1)
        ttf_pred = F.softplus(self.ttf_head(features).squeeze(-1))  # Non-negative
        risk_mean, risk_log_var = self.risk_head(features)
        risk_pred = torch.sigmoid(risk_mean)  # [0, 1]
        
        # Ensemble predictions for uncertainty
        ensemble_preds = [torch.sigmoid(head(features)) for head in self.ensemble_heads]
        ensemble_mean = torch.stack(ensemble_preds).mean(dim=0)
        ensemble_std = torch.stack(ensemble_preds).std(dim=0)
        
        outputs = {
            'binary_logits': binary_logits,
            'ttf_pred': ttf_pred,
            'risk_pred': risk_pred,
            'risk_log_var': risk_log_var,
            'ensemble_mean': ensemble_mean,
            'ensemble_std': ensemble_std,
            'proj': proj,
        }
        
        if return_embeddings:
            outputs['embed'] = embed
        
        return outputs
    
    def predict_with_uncertainty(self, x, n_mc_samples=20):
        """Predict with MC dropout for uncertainty."""
        self.train()  # Enable dropout
        
        preds = []
        for _ in range(n_mc_samples):
            with torch.no_grad():
                out = self.forward(x)
                preds.append(out['risk_pred'])
        
        preds = torch.stack(preds)
        return preds.mean(dim=0), preds.std(dim=0)


# ============================================================================
# TEMPORAL CONTEXT: Use sequence of hidden states
# ============================================================================

class TemporalRiskPredictor(nn.Module):
    """Uses temporal context (sequence of hidden states) for better prediction."""
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 256,
        output_dim: int = 7,
        seq_len: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Per-step encoder
        self.step_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Temporal modeling (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Heads
        self.binary_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, output_dim)
        
        self.seq_len = seq_len
    
    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        """
        batch_size = x.shape[0]
        
        # Encode each step
        encoded = self.step_encoder(x)  # [batch, seq_len, hidden_dim]
        
        # Temporal modeling
        temporal = self.temporal_encoder(encoded)  # [batch, seq_len, hidden_dim]
        
        # Aggregate
        aggregated = self.aggregator(temporal.reshape(batch_size, -1))
        
        # Predictions
        binary_logits = self.binary_head(aggregated).squeeze(-1)
        risk_pred = torch.sigmoid(self.risk_head(aggregated))
        
        return {
            'binary_logits': binary_logits,
            'risk_pred': risk_pred,
        }


# ============================================================================
# DATASET WITH AUGMENTATION
# ============================================================================

class ResearchDataset(Dataset):
    """
    Dataset with:
    - Trajectory-level features
    - Augmentation for positive samples
    - Class balancing
    """
    
    def __init__(
        self,
        samples: List[Dict],
        augment_positives: bool = True,
        augment_factor: int = 3,
        noise_std: float = 0.1,
    ):
        self.samples = []
        self.augment_positives = augment_positives
        
        for s in samples:
            hidden = np.asarray(s['hidden_state'], dtype=np.float32)
            risk = np.asarray(s.get('risk_label', np.zeros(7)), dtype=np.float32)
            binary = float(s.get('binary_label', 0))
            ttf = float(s.get('time_to_failure', -1))
            
            if len(hidden) < 10:
                continue
            
            self.samples.append({
                'hidden_state': hidden,
                'risk_label': risk,
                'binary_label': binary,
                'time_to_failure': ttf,
            })
            
            # Augment positive samples
            if augment_positives and binary > 0.5:
                for _ in range(augment_factor):
                    noise = np.random.randn(*hidden.shape).astype(np.float32) * noise_std
                    self.samples.append({
                        'hidden_state': hidden + noise,
                        'risk_label': risk,
                        'binary_label': binary,
                        'time_to_failure': ttf,
                    })
        
        # Normalize
        hidden_states = np.array([s['hidden_state'] for s in self.samples])
        self.mean = hidden_states.mean(axis=0)
        self.std = hidden_states.std(axis=0) + 1e-8
        
        for s in self.samples:
            s['hidden_state'] = (s['hidden_state'] - self.mean) / self.std
        
        # Compute class weights for balanced sampling
        labels = np.array([s['binary_label'] for s in self.samples])
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        self.weights = np.where(labels > 0.5, 1/n_pos, 1/n_neg)
        self.weights = self.weights / self.weights.sum()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'hidden_state': torch.FloatTensor(s['hidden_state']),
            'risk_label': torch.FloatTensor(s['risk_label']),
            'binary_label': torch.tensor(s['binary_label']),
            'time_to_failure': torch.tensor(s['time_to_failure']),
        }
    
    def get_sampler(self):
        """Get weighted random sampler for class balance."""
        return WeightedRandomSampler(self.weights, len(self.samples), replacement=True)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    contrastive_weight: float = 0.1,
    ttf_weight: float = 0.2,
) -> Dict[str, float]:
    """Train one epoch with multi-task loss."""
    model.train()
    
    losses = {'binary': 0, 'risk': 0, 'ttf': 0, 'contrastive': 0, 'total': 0}
    n_batches = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        hidden = batch['hidden_state'].to(device)
        binary_target = batch['binary_label'].float().to(device)
        risk_target = batch['risk_label'].to(device)
        ttf_target = batch['time_to_failure'].float().to(device)
        
        # Forward
        outputs = model(hidden)
        
        # Binary classification loss (focal)
        bce = F.binary_cross_entropy_with_logits(
            outputs['binary_logits'], binary_target, reduction='none'
        )
        pt = torch.exp(-bce)
        focal = 0.25 * (1 - pt) ** 2 * bce
        binary_loss = focal.mean()
        
        # Risk regression loss (only on positive samples)
        pos_mask = binary_target > 0.5
        if pos_mask.any():
            risk_loss = gaussian_nll_loss(
                outputs['risk_pred'][pos_mask],
                outputs['risk_log_var'][pos_mask],
                risk_target[pos_mask]
            )
        else:
            risk_loss = torch.tensor(0.0, device=device)
        
        # Time-to-failure loss (only on samples with valid TTF)
        ttf_mask = ttf_target >= 0
        if ttf_mask.any():
            ttf_loss = F.smooth_l1_loss(
                outputs['ttf_pred'][ttf_mask],
                ttf_target[ttf_mask] / 50.0  # Normalize to ~1
            )
        else:
            ttf_loss = torch.tensor(0.0, device=device)
        
        # Contrastive loss
        proj = outputs['proj']
        # Create augmented view (dropout is already applied)
        proj_aug = model.encoder.projector(model.encoder.encoder(hidden))
        proj_aug = F.normalize(proj_aug, dim=-1)
        contrastive = contrastive_loss(proj, proj_aug, binary_target)
        
        # Total loss
        total_loss = (
            binary_loss + 
            risk_loss + 
            ttf_weight * ttf_loss + 
            contrastive_weight * contrastive
        )
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses['binary'] += binary_loss.item()
        losses['risk'] += risk_loss.item()
        losses['ttf'] += ttf_loss.item()
        losses['contrastive'] += contrastive.item()
        losses['total'] += total_loss.item()
        n_batches += 1
    
    return {k: v / n_batches for k, v in losses.items()}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    
    all_binary_logits = []
    all_binary_targets = []
    all_risk_preds = []
    all_risk_targets = []
    all_uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            hidden = batch['hidden_state'].to(device)
            binary_target = batch['binary_label']
            risk_target = batch['risk_label']
            
            outputs = model(hidden)
            
            all_binary_logits.append(outputs['binary_logits'].cpu().numpy())
            all_binary_targets.append(binary_target.numpy())
            all_risk_preds.append(outputs['ensemble_mean'].cpu().numpy())
            all_risk_targets.append(risk_target.numpy())
            all_uncertainties.append(outputs['ensemble_std'].cpu().numpy())
    
    binary_logits = np.concatenate(all_binary_logits)
    binary_targets = np.concatenate(all_binary_targets)
    risk_preds = np.concatenate(all_risk_preds)
    risk_targets = np.concatenate(all_risk_targets)
    uncertainties = np.concatenate(all_uncertainties)
    
    # Metrics
    binary_probs = 1 / (1 + np.exp(-binary_logits))
    binary_pred = (binary_probs > 0.5).astype(int)
    
    metrics = {}
    
    try:
        metrics['binary_auc'] = roc_auc_score(binary_targets, binary_probs)
        metrics['binary_ap'] = average_precision_score(binary_targets, binary_probs)
    except:
        metrics['binary_auc'] = 0.5
        metrics['binary_ap'] = 0.5
    
    metrics['binary_acc'] = (binary_pred == binary_targets).mean()
    metrics['binary_balanced_acc'] = balanced_accuracy_score(binary_targets, binary_pred)
    
    if len(np.unique(binary_targets)) > 1:
        metrics['binary_f1'] = f1_score(binary_targets, binary_pred)
    else:
        metrics['binary_f1'] = 0.0
    
    # Per-dimension AUC
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for i, dim in enumerate(dim_names):
        target = risk_targets[:, i]
        pred = risk_preds[:, i]
        
        # Binary threshold (25th percentile of positives)
        pos_vals = target[target > 0]
        thresh = np.percentile(pos_vals, 25) if len(pos_vals) > 0 else 0.01
        target_binary = (target > thresh).astype(int)
        
        if len(np.unique(target_binary)) > 1:
            try:
                metrics[f'auc_{dim}'] = roc_auc_score(target_binary, pred)
            except:
                metrics[f'auc_{dim}'] = 0.5
        else:
            metrics[f'auc_{dim}'] = 0.5
    
    # Uncertainty quality (should be higher for wrong predictions)
    wrong_mask = binary_pred != binary_targets
    metrics['uncertainty_mean'] = uncertainties.mean()
    if wrong_mask.any():
        metrics['uncertainty_on_errors'] = uncertainties[wrong_mask].mean()
    else:
        metrics['uncertainty_on_errors'] = 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/risk_predictor_research")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-ensemble", type=int, default=3)
    parser.add_argument("--augment-factor", type=int, default=3)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--ttf-weight", type=float, default=0.2)
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
    
    # Create dataset with augmentation
    dataset = ResearchDataset(
        samples,
        augment_positives=True,
        augment_factor=args.augment_factor,
    )
    
    input_dim = len(dataset.samples[0]['hidden_state'])
    print(f"Input dim: {input_dim}")
    print(f"After augmentation: {len(dataset)} samples")
    
    # Split
    n_total = len(dataset)
    n_test = int(0.1 * n_total)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val - n_test
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Use weighted sampler for training
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=dataset.get_sampler(),
        num_workers=4,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = ResearchRiskPredictor(
        input_dim=input_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_ensemble=args.n_ensemble,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Training
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_auc = 0.0
    history = []
    
    print("\nTraining...")
    for epoch in range(args.epochs):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device,
            contrastive_weight=args.contrastive_weight,
            ttf_weight=args.ttf_weight,
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
        history.append({
            'epoch': epoch,
            'train_losses': train_losses,
            'val_metrics': val_metrics,
        })
        
        # Print
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss: total={train_losses['total']:.4f}, binary={train_losses['binary']:.4f}, "
              f"risk={train_losses['risk']:.4f}, contrastive={train_losses['contrastive']:.4f}")
        print(f"  Val: AUC={val_metrics['binary_auc']:.4f}, AP={val_metrics['binary_ap']:.4f}, "
              f"Balanced Acc={val_metrics['binary_balanced_acc']:.4f}, F1={val_metrics['binary_f1']:.4f}")
        
        dim_aucs = [val_metrics.get(f'auc_{d}', 0.5) for d in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']]
        print(f"  Per-dim AUC: x={dim_aucs[0]:.3f} y={dim_aucs[1]:.3f} z={dim_aucs[2]:.3f} "
              f"roll={dim_aucs[3]:.3f} pitch={dim_aucs[4]:.3f} yaw={dim_aucs[5]:.3f} grip={dim_aucs[6]:.3f}")
        
        # Save best
        if val_metrics['binary_auc'] > best_val_auc:
            best_val_auc = val_metrics['binary_auc']
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'epoch': epoch,
                'config': vars(args),
                'normalization': {'mean': dataset.mean.tolist(), 'std': dataset.std.tolist()},
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Best model (AUC={best_val_auc:.4f})")
    
    # Final test
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nBinary Classification:")
    print(f"  AUC-ROC: {test_metrics['binary_auc']:.4f}")
    print(f"  Average Precision: {test_metrics['binary_ap']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['binary_balanced_acc']:.4f}")
    print(f"  F1 Score: {test_metrics['binary_f1']:.4f}")
    
    print(f"\nPer-Dimension AUC-ROC:")
    for dim in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']:
        print(f"  {dim.upper():8s}: {test_metrics.get(f'auc_{dim}', 0.5):.4f}")
    
    print(f"\nUncertainty Quality:")
    print(f"  Mean uncertainty: {test_metrics['uncertainty_mean']:.4f}")
    print(f"  Uncertainty on errors: {test_metrics['uncertainty_on_errors']:.4f}")
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'test_metrics': test_metrics, 'config': vars(args)}, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}")
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
