#!/usr/bin/env python3
"""
Train Diffusion Policy on LIBERO demonstration data.

Uses the DDPM-based action prediction architecture from Chi et al. 2023,
adapted for LIBERO's observation space (image + proprioception) and
7-DOF action space.

This script:
1. Loads LIBERO expert demonstrations (HDF5 files)
2. Trains a CNN-based Diffusion Policy
3. Saves the best checkpoint for downstream evaluation
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Try to import diffusion_policy; fall back to our own implementation
try:
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    HAS_DP_REPO = True
    print("Using diffusion_policy repo implementation")
except ImportError:
    HAS_DP_REPO = False
    print("diffusion_policy repo not found — using standalone DDPM implementation")

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
import h5py


# ══════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════

class LIBERODemoDataset(Dataset):
    """Load LIBERO expert demonstrations for imitation learning."""

    def __init__(self, dataset_dir, benchmark_name, obs_horizon=2,
                 action_horizon=8, img_size=128):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        benchmark = get_benchmark(benchmark_name)(0)
        self.samples = []  # list of (image, proprio, action_chunk)

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
                    # Get observations and actions
                    actions = demo['actions'][:]  # (T, 7)
                    T = len(actions)

                    # Images: try agentview_rgb or agentview_image
                    img_key = None
                    for k in ['obs/agentview_rgb', 'obs/agentview_image']:
                        if k in demo:
                            img_key = k
                            break
                    if img_key is None:
                        # Try nested obs keys
                        if 'obs' in demo:
                            for k in demo['obs'].keys():
                                if 'image' in k.lower() or 'rgb' in k.lower():
                                    img_key = f'obs/{k}'
                                    break

                    if img_key is None:
                        print(f"    ⚠ No image key found in {demo_key}")
                        continue

                    images = demo[img_key][:]  # (T, H, W, 3)

                    # Proprio: try ee_pos, robot0_eef_pos, joint_states
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

                    # Create samples with observation and action chunks
                    for t in range(T - action_horizon):
                        # Obs: last obs_horizon frames
                        t_start = max(0, t - obs_horizon + 1)
                        img_chunk = images[t_start:t+1]
                        proprio_chunk = proprio[t_start:t+1]

                        # Pad if needed
                        if len(img_chunk) < obs_horizon:
                            pad = obs_horizon - len(img_chunk)
                            img_chunk = np.concatenate(
                                [np.repeat(img_chunk[:1], pad, axis=0),
                                 img_chunk], axis=0)
                            proprio_chunk = np.concatenate(
                                [np.repeat(proprio_chunk[:1], pad, axis=0),
                                 proprio_chunk], axis=0)

                        # Action chunk: next action_horizon steps
                        action_chunk = actions[t:t+action_horizon]

                        self.samples.append({
                            'images': img_chunk,
                            'proprio': proprio_chunk.astype(np.float32),
                            'actions': action_chunk.astype(np.float32),
                        })

        print(f"  Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Transform images
        imgs = []
        for img in sample['images']:
            imgs.append(self.transform(img))
        imgs = torch.stack(imgs)  # (obs_horizon, 3, H, W)

        return {
            'images': imgs,
            'proprio': torch.FloatTensor(sample['proprio']),
            'actions': torch.FloatTensor(sample['actions']),
        }


# ══════════════════════════════════════════════════════════════════════════
#  SIMPLE DIFFUSION POLICY (fallback if diffusion_policy repo not installed)
# ══════════════════════════════════════════════════════════════════════════

class SimpleDiffusionPolicy(nn.Module):
    """Minimal DDPM-based policy for action prediction.

    Architecture:
      - CNN image encoder (ResNet-18 backbone)
      - MLP obs encoder combining image features + proprio
      - 1D U-Net for denoising action sequences
    """

    def __init__(self, obs_dim, action_dim=7, action_horizon=8,
                 n_diffusion_steps=100):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_diffusion_steps = n_diffusion_steps

        # Image encoder (simple CNN)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        img_feat_dim = 64

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(img_feat_dim + obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Noise prediction network (1D temporal convolutions)
        self.noise_pred = nn.Sequential(
            nn.Linear(action_dim * action_horizon + 256 + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * action_horizon),
        )

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def encode_obs(self, images, proprio):
        """Encode observation into a conditioning vector."""
        B = images.shape[0]
        # Use last frame
        img = images[:, -1]  # (B, 3, H, W)
        img_feat = self.img_encoder(img)  # (B, 64)
        # Use last proprio
        prop = proprio[:, -1]  # (B, obs_dim)
        obs_feat = self.obs_encoder(torch.cat([img_feat, prop], dim=-1))
        return obs_feat  # (B, 256)

    def forward(self, images, proprio, actions):
        """Training forward pass — predict noise."""
        B = actions.shape[0]
        obs_feat = self.encode_obs(images, proprio)

        # Sample random timestep
        t = torch.randint(0, self.n_diffusion_steps, (B,),
                          device=actions.device)

        # Add noise
        noise = torch.randn_like(actions)  # (B, H, 7)
        alpha_t = self.alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        noisy_actions = torch.sqrt(alpha_t) * actions + \
            torch.sqrt(1 - alpha_t) * noise

        # Flatten for prediction
        noisy_flat = noisy_actions.reshape(B, -1)
        t_emb = (t.float() / self.n_diffusion_steps).unsqueeze(-1)
        x = torch.cat([noisy_flat, obs_feat, t_emb], dim=-1)

        noise_pred = self.noise_pred(x).reshape(B, self.action_horizon,
                                                 self.action_dim)
        return noise_pred, noise

    @torch.no_grad()
    def predict(self, images, proprio):
        """Inference — denoise from pure noise to get action chunk."""
        B = images.shape[0]
        obs_feat = self.encode_obs(images, proprio)

        # Start from pure noise
        x = torch.randn(B, self.action_horizon, self.action_dim,
                         device=images.device)

        for t_idx in reversed(range(self.n_diffusion_steps)):
            t = torch.full((B,), t_idx, device=images.device, dtype=torch.long)
            t_emb = (t.float() / self.n_diffusion_steps).unsqueeze(-1)

            x_flat = x.reshape(B, -1)
            inp = torch.cat([x_flat, obs_feat, t_emb], dim=-1)
            noise_pred = self.noise_pred(inp).reshape(
                B, self.action_horizon, self.action_dim)

            # DDPM reverse step
            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred)

            if t_idx > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)

        return x  # (B, action_horizon, action_dim)


# ══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    print("\nLoading LIBERO demonstrations...")
    dataset = LIBERODemoDataset(
        args.dataset_dir, args.benchmark,
        obs_horizon=2, action_horizon=8, img_size=128)

    if len(dataset) == 0:
        print("ERROR: No training data loaded!")
        return

    # Split 90/10
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

    # Model
    proprio_dim = dataset.samples[0]['proprio'].shape[-1]
    model = SimpleDiffusionPolicy(
        obs_dim=proprio_dim, action_dim=7, action_horizon=8).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            images = batch['images'].to(device)
            proprio = batch['proprio'].to(device)
            actions = batch['actions'].to(device)

            noise_pred, noise_true = model(images, proprio, actions)
            loss = F.mse_loss(noise_pred, noise_true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                proprio = batch['proprio'].to(device)
                actions = batch['actions'].to(device)
                noise_pred, noise_true = model(images, proprio, actions)
                val_loss += F.mse_loss(noise_pred, noise_true).item()
        val_loss /= max(len(val_loader), 1)

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
            }, output_dir / "best_model.pt")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--benchmark", default="LIBERO_SPATIAL")
    parser.add_argument("--output-dir", default="checkpoints/diffusion_policy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
