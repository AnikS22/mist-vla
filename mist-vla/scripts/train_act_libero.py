#!/usr/bin/env python3
"""
Train ACT (Action Chunking with Transformers) on LIBERO demonstration data.

ACT (Zhao et al. 2023) uses a CVAE with a transformer backbone to predict
action chunks. This is a self-contained implementation for LIBERO.

This script:
1. Loads LIBERO expert demonstrations (HDF5 files)
2. Trains an ACT model
3. Saves the best checkpoint for downstream evaluation
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
import h5py


# ══════════════════════════════════════════════════════════════════════════
#  DATASET (shared with DP — same loader)
# ══════════════════════════════════════════════════════════════════════════

class LIBERODemoDataset(Dataset):
    """Load LIBERO expert demonstrations for imitation learning."""

    def __init__(self, dataset_dir, benchmark_name, obs_horizon=1,
                 action_horizon=8, img_size=128):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        benchmark = get_benchmark(benchmark_name)(0)
        self.samples = []

        for task_id in range(benchmark.n_tasks):
            demo_file = os.path.join(
                dataset_dir, benchmark.get_task_demonstration(task_id))

            if not os.path.exists(demo_file):
                print(f"  ⚠ Missing: {demo_file}")
                continue

            print(f"  Loading task {task_id}: {demo_file}")
            with h5py.File(demo_file, 'r') as f:
                demos = f['data']
                for demo_key in sorted(demos.keys()):
                    demo = demos[demo_key]
                    actions = demo['actions'][:]
                    T = len(actions)

                    # Find image key
                    img_key = None
                    for k in ['obs/agentview_rgb', 'obs/agentview_image']:
                        if k in demo:
                            img_key = k
                            break
                    if img_key is None and 'obs' in demo:
                        for k in demo['obs'].keys():
                            if 'image' in k.lower() or 'rgb' in k.lower():
                                img_key = f'obs/{k}'
                                break
                    if img_key is None:
                        continue

                    images = demo[img_key][:]

                    # Find proprio
                    proprio = None
                    for pk in ['obs/ee_pos', 'obs/robot0_eef_pos',
                               'obs/joint_states', 'obs/robot0_joint_pos']:
                        if pk in demo:
                            proprio = demo[pk][:]
                            break
                    if proprio is None and 'obs' in demo:
                        for k in demo['obs'].keys():
                            if 'joint' in k.lower() or 'proprio' in k.lower():
                                proprio = demo[f'obs/{k}'][:]
                                break
                    if proprio is None:
                        proprio = np.zeros((T, 8), dtype=np.float32)

                    for t in range(T - action_horizon):
                        self.samples.append({
                            'image': images[t],
                            'proprio': proprio[t].astype(np.float32),
                            'actions': actions[t:t+action_horizon].astype(
                                np.float32),
                        })

        print(f"  Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self.transform(s['image'])
        return {
            'image': img,
            'proprio': torch.FloatTensor(s['proprio']),
            'actions': torch.FloatTensor(s['actions']),
        }


# ══════════════════════════════════════════════════════════════════════════
#  ACT MODEL
# ══════════════════════════════════════════════════════════════════════════

class ACTPolicy(nn.Module):
    """Action Chunking with Transformers (Zhao et al. 2023).

    Architecture:
      - CNN image encoder
      - CVAE encoder (training only): encodes action sequence into latent z
      - Transformer decoder: conditions on obs + z to predict action chunk
    """

    def __init__(self, obs_dim, action_dim=7, action_horizon=8,
                 hidden_dim=256, n_heads=4, n_layers=4, latent_dim=32):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Image encoder
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        img_feat_dim = 64

        # Observation projection
        self.obs_proj = nn.Linear(img_feat_dim + obs_dim, hidden_dim)

        # CVAE encoder (training only)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.cvae_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2)
        self.z_mu = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)

        # Latent projection
        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        # Action query tokens
        self.action_queries = nn.Embedding(action_horizon, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers)

        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def encode_obs(self, image, proprio):
        img_feat = self.img_encoder(image)
        obs = torch.cat([img_feat, proprio], dim=-1)
        return self.obs_proj(obs)  # (B, hidden_dim)

    def encode_actions(self, actions, obs_feat):
        """CVAE encoder: encode action sequence + obs into latent z."""
        B = actions.shape[0]
        act_emb = self.action_encoder(actions)  # (B, H, hidden_dim)
        # Prepend obs as first token
        obs_token = obs_feat.unsqueeze(1)  # (B, 1, hidden_dim)
        seq = torch.cat([obs_token, act_emb], dim=1)  # (B, H+1, hidden_dim)
        encoded = self.cvae_encoder(seq)
        # Use first token output for latent
        z_input = encoded[:, 0]
        mu = self.z_mu(z_input)
        logvar = self.z_logvar(z_input)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, obs_feat, z):
        """Decode: obs + z → action chunk."""
        B = obs_feat.shape[0]

        # Memory = [obs_token, z_token]
        z_feat = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([obs_feat.unsqueeze(1), z_feat], dim=1)

        # Queries
        queries = self.action_queries.weight.unsqueeze(0).expand(
            B, -1, -1)  # (B, action_horizon, hidden_dim)

        out = self.transformer_decoder(queries, memory)
        actions = self.action_head(out)  # (B, action_horizon, action_dim)
        return actions

    def forward(self, image, proprio, actions=None):
        """
        Training: returns predicted actions, mu, logvar
        Inference (actions=None): returns predicted actions with z=0
        """
        obs_feat = self.encode_obs(image, proprio)

        if actions is not None:
            # Training: CVAE encode
            mu, logvar = self.encode_actions(actions, obs_feat)
            z = self.reparameterize(mu, logvar)
            pred_actions = self.decode(obs_feat, z)
            return pred_actions, mu, logvar
        else:
            # Inference: z = 0
            z = torch.zeros(image.shape[0], self.latent_dim,
                            device=image.device)
            pred_actions = self.decode(obs_feat, z)
            return pred_actions

    @torch.no_grad()
    def predict(self, image, proprio):
        """Inference wrapper."""
        self.eval()
        return self.forward(image, proprio, actions=None)


# ══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading LIBERO demonstrations...")
    dataset = LIBERODemoDataset(
        args.dataset_dir, args.benchmark,
        obs_horizon=1, action_horizon=8, img_size=128)

    if len(dataset) == 0:
        print("ERROR: No training data loaded!")
        return

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    print(f"Train: {n_train}  Val: {n_val}")

    proprio_dim = dataset.samples[0]['proprio'].shape[-1]
    model = ACTPolicy(
        obs_dim=proprio_dim, action_dim=7, action_horizon=8).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ACT params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    kl_weight = 10.0  # KL weight for CVAE

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0
        for batch in train_loader:
            image = batch['image'].to(device)
            proprio = batch['proprio'].to(device)
            actions = batch['actions'].to(device)

            pred_actions, mu, logvar = model(image, proprio, actions)

            # Reconstruction loss
            recon_loss = F.l1_loss(pred_actions, actions)

            # KL divergence
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                proprio = batch['proprio'].to(device)
                actions = batch['actions'].to(device)
                pred, mu, logvar = model(image, proprio, actions)
                recon = F.l1_loss(pred, actions)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss_sum += (recon + kl_weight * kl).item()
        val_loss = val_loss_sum / max(len(val_loader), 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:>3} | train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'proprio_dim': proprio_dim,
                'action_dim': 7,
                'action_horizon': 8,
                'model_type': 'act',
            }, output_dir / "best_model.pt")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--benchmark", default="LIBERO_SPATIAL")
    parser.add_argument("--output-dir", default="checkpoints/act")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
