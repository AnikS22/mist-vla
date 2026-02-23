#!/usr/bin/env python3
"""
ACT Steering Evaluation — Category 2: Cross-Architecture Proof
================================================================

Compares:
  Mode A: Vanilla ACT       — raw, unsteered baseline
  Mode B: ACT + Steering    — MLP-guided correction from 256-dim hidden states

For each task, runs N episodes (default 20) per mode and reports:
  - Success Rate (%)
  - Intervention Rate (IR)
  - Δ vs Vanilla (percentage points)

This proves the MLP works on a non-VLA architecture (CVAE-based ACT),
using only 256-dim latent states instead of 4096-dim VLA embeddings.

Usage
-----
  python scripts/eval_act_steering.py \
      --act-checkpoint checkpoints/act/best_model.pt \
      --mlp-checkpoint checkpoints/eef_correction_mlp_act/best_model.pt \
      --env libero_spatial \
      --episodes-per-task 20 \
      --modes vanilla steering \
      --save-dir results/eval_act_steering
"""

import argparse
import gc
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

from libero.libero import benchmark as libero_benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ══════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING (must match training)
# ══════════════════════════════════════════════════════════════════════════

IMG_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════
#  SAFETY MLP (must match train_eef_correction_mlp.py v4)
# ══════════════════════════════════════════════════════════════════════════

class EEFCorrectionMLP(nn.Module):
    """v4 architecture — must match train_eef_correction_mlp.py exactly."""
    HIDDEN_DIM = 256

    def __init__(self, input_dim=256):
        super().__init__()
        h = self.HIDDEN_DIM
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(h // 2, h // 4),
            nn.LayerNorm(h // 4),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        feat = h // 4
        self.fail_head = nn.Linear(feat, 1)
        self.ttf_head = nn.Linear(feat, 1)
        self.correction_head = nn.Linear(feat, 3)

    def forward(self, x):
        x = self.input_norm(x)
        feat = self.encoder(x)
        return {
            "will_fail": self.fail_head(feat).squeeze(-1),
            "ttf": self.ttf_head(feat).squeeze(-1),
            "correction": self.correction_head(feat),
        }


# ══════════════════════════════════════════════════════════════════════════
#  ACT MODEL (must match train_act_libero.py)
# ══════════════════════════════════════════════════════════════════════════

class ACTPolicy(nn.Module):
    """Action Chunking with Transformers (Zhao et al. 2023)."""

    def __init__(self, obs_dim, action_dim=7, action_horizon=8,
                 hidden_dim=256, n_heads=4, n_layers=4, latent_dim=32):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        img_feat_dim = 64

        self.obs_proj = nn.Linear(img_feat_dim + obs_dim, hidden_dim)

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.cvae_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2)
        self.z_mu = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)

        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.action_queries = nn.Embedding(action_horizon, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers)

        self.action_head = nn.Linear(hidden_dim, action_dim)

    def encode_obs(self, image, proprio):
        img_feat = self.img_encoder(image)
        obs = torch.cat([img_feat, proprio], dim=-1)
        return self.obs_proj(obs)

    def decode(self, obs_feat, z):
        B = obs_feat.shape[0]
        z_feat = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([obs_feat.unsqueeze(1), z_feat], dim=1)
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.transformer_decoder(queries, memory)
        actions = self.action_head(out)
        return actions

    def forward(self, image, proprio, actions=None):
        obs_feat = self.encode_obs(image, proprio)
        if actions is not None:
            B = actions.shape[0]
            act_emb = self.action_encoder(actions)
            obs_token = obs_feat.unsqueeze(1)
            seq = torch.cat([obs_token, act_emb], dim=1)
            encoded = self.cvae_encoder(seq)
            z_input = encoded[:, 0]
            mu = self.z_mu(z_input)
            logvar = self.z_logvar(z_input)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            pred_actions = self.decode(obs_feat, z)
            return pred_actions, mu, logvar
        else:
            z = torch.zeros(image.shape[0], self.latent_dim,
                            device=image.device)
            pred_actions = self.decode(obs_feat, z)
            return pred_actions

    @torch.no_grad()
    def predict(self, image, proprio):
        self.eval()
        return self.forward(image, proprio, actions=None)


# ══════════════════════════════════════════════════════════════════════════
#  STEERED AGENT (same logic as eval_tuning.py)
# ══════════════════════════════════════════════════════════════════════════

class SteeredAgent:
    """Steering controller: clamp + magnitude gate."""

    def __init__(self, mlp, scaler, *,
                 alpha=1.0, ema_beta=0.7, action_scale=0.05,
                 correction_threshold=0.005, max_correction=0.01,
                 device="cpu"):
        self.mlp = mlp
        self.scaler = scaler
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.action_scale = action_scale
        self.correction_threshold = correction_threshold
        self.max_correction = max_correction
        self.device = device
        self.prev_correction = None
        self._steps = 0
        self._interventions = 0
        self._corr_mags = []

    def reset(self):
        self.prev_correction = None
        self._steps = 0
        self._interventions = 0
        self._corr_mags = []

    @property
    def intervention_rate(self):
        return self._interventions / max(self._steps, 1)

    @property
    def mean_corr_mag(self):
        return float(np.mean(self._corr_mags)) if self._corr_mags else 0.0

    def apply(self, action, features):
        """Apply gated, clamped MLP correction to action[:3]."""
        self._steps += 1

        if features is None or np.prod(features.shape) < 2:
            return action, False

        scaled = self.scaler.transform(features.reshape(1, -1))
        x = torch.FloatTensor(scaled).to(self.device)

        with torch.no_grad():
            out = self.mlp(x)
        raw = out["correction"].cpu().numpy()[0]  # (3,) meters

        # EMA smoothing
        if self.prev_correction is not None:
            smoothed = (self.ema_beta * self.prev_correction
                        + (1.0 - self.ema_beta) * raw)
        else:
            smoothed = raw.copy()
        self.prev_correction = smoothed.copy()

        # Clamp magnitude
        mag = float(np.linalg.norm(smoothed))
        if mag > self.max_correction and mag > 1e-8:
            smoothed = smoothed * (self.max_correction / mag)
            mag = self.max_correction

        self._corr_mags.append(mag)

        # Gate: only intervene if correction is meaningful
        if mag > self.correction_threshold:
            self._interventions += 1
            action[:3] += (self.alpha * smoothed / self.action_scale)
            return action, True
        return action, False


# ══════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_eef_pos(env):
    """Extract end-effector position from LIBERO environment."""
    try:
        sim = getattr(env, 'sim', None)
        if sim is None:
            inner = getattr(env, 'env', None)
            if inner is not None:
                sim = getattr(inner, 'sim', None)
        if sim is not None:
            site_id = sim.model.site_name2id("gripper0_grip_site")
            return sim.data.site_xpos[site_id].copy()
    except Exception:
        pass
    return np.zeros(3)


# ══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_episode(env, act_model, device, init_state, max_steps=300,
                mode="vanilla", steered_agent=None,
                noise_sigma=0.05, ema_only_beta=0.9):
    """Run one ACT episode.

    mode ∈ {"vanilla", "noise", "ema_only", "steering"}
    """
    env.reset()
    if init_state is not None:
        try:
            obs = env.set_init_state(init_state)
        except Exception:
            obs = env.reset()
    else:
        obs = env.reset()

    if obs is None:
        obs = env.reset()

    if mode == "steering" and steered_agent is not None:
        steered_agent.reset()

    action_queue = []
    last_features = None
    total_steps = 0
    success = False
    ema_action = None

    for step in range(max_steps):
        # Prepare observation
        img = obs.get('agentview_image', obs.get('agentview_rgb'))
        if img is None:
            for k in obs:
                if 'image' in k.lower() or 'rgb' in k.lower():
                    img = obs[k]
                    break
        if img is None:
            break

        img_tensor = IMG_TRANSFORM(img).unsqueeze(0).to(device)

        proprio = obs.get('robot0_eef_pos', obs.get('ee_pos'))
        if proprio is None:
            proprio = obs.get('robot0_joint_pos', np.zeros(8))
        proprio_tensor = torch.FloatTensor(proprio).unsqueeze(0).to(device)

        # Get action from ACT (with chunking)
        if len(action_queue) == 0:
            with torch.no_grad():
                # Extract features (256-dim)
                features = act_model.encode_obs(img_tensor, proprio_tensor)
                last_features = features[0].cpu().numpy()

                # Get action chunk (8 steps)
                actions = act_model.predict(img_tensor, proprio_tensor)
                action_queue = list(actions[0].cpu().numpy())

        action = action_queue.pop(0)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ── Apply mode-specific modification ──
        if mode == "noise":
            action[:3] += np.random.normal(0, noise_sigma, size=3)

        elif mode == "ema_only":
            if ema_action is None:
                ema_action = action[:3].copy()
            else:
                ema_action = ema_only_beta * ema_action + (1 - ema_only_beta) * action[:3]
            action[:3] = ema_action

        elif mode == "steering" and steered_agent is not None:
            action, _ = steered_agent.apply(action, last_features)

        action = np.clip(action, -1.0, 1.0)

        # Step environment
        try:
            obs, reward, done, info = env.step(action.tolist())
        except Exception as e:
            print(f"    ⚠ env.step error at step {step}: {e}", flush=True)
            break

        total_steps += 1

        # Check success
        if done:
            success = True
            break
        if isinstance(info, dict):
            if info.get("success", False) or info.get("is_success", False):
                success = True
                break

    ir = steered_agent.intervention_rate if (
        mode == "steering" and steered_agent) else 0.0
    cm = steered_agent.mean_corr_mag if (
        mode == "steering" and steered_agent) else 0.0

    return {"success": success, "total_steps": total_steps,
            "intervention_rate": round(ir, 4),
            "mean_corr_mag": round(cm, 6)}


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ACT Steering Evaluation — Cross-Architecture Proof")
    parser.add_argument("--act-checkpoint", required=True,
                        help="Path to ACT model checkpoint")
    parser.add_argument("--mlp-checkpoint", required=True,
                        help="Path to ACT safety MLP checkpoint")
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--tasks", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--modes", nargs="+",
                        default=["vanilla", "steering"],
                        choices=["vanilla", "noise", "ema_only", "steering"])
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ema-beta", type=float, default=0.7,
                        help="EMA β for steering correction smoothing")
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument("--ema-only-beta", type=float, default=0.9)
    parser.add_argument("--action-scale", type=float, default=0.05)
    parser.add_argument("--correction-threshold", type=float, default=0.005)
    parser.add_argument("--max-correction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="results/eval_act_steering")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    max_pert = args.alpha * args.max_correction / args.action_scale

    MODE_LABELS = {
        "vanilla":  "Vanilla ACT (baseline)",
        "noise":    f"Random Noise (σ={args.noise_sigma})",
        "ema_only": f"EMA Smoothing Only (β={args.ema_only_beta})",
        "steering": f"Latent Steering (α={args.alpha}, clamp={args.max_correction}m)",
    }

    # ── Banner ──
    print("=" * 70, flush=True)
    print("ACT STEERING EVALUATION — Cross-Architecture Proof", flush=True)
    print("=" * 70, flush=True)
    print(f"  Tasks:       {args.tasks}", flush=True)
    print(f"  Episodes:    {args.episodes_per_task} per task per mode",
          flush=True)
    print(f"  Modes:       {args.modes}", flush=True)
    for m in args.modes:
        print(f"    → {MODE_LABELS.get(m, m)}", flush=True)
    print(f"  Steering:    α={args.alpha}  clamp={args.max_correction}m  "
          f"gate=‖c‖>{args.correction_threshold}m", flush=True)
    print(f"  Max Δaction: {max_pert:.4f} units "
          f"({max_pert * 100:.1f}% of range)", flush=True)
    print(f"  Device:      {device}", flush=True)
    print(flush=True)

    # ─── 1. Load ACT Model ───────────────────────────────────────
    print("[1/3] Loading ACT model...", flush=True)
    act_ckpt = torch.load(args.act_checkpoint, map_location=device,
                          weights_only=False)
    act_model = ACTPolicy(
        obs_dim=act_ckpt['proprio_dim'],
        action_dim=act_ckpt['action_dim'],
        action_horizon=act_ckpt['action_horizon'],
    ).to(device)
    act_model.load_state_dict(act_ckpt['model_state_dict'])
    act_model.eval()
    n_act_params = sum(p.numel() for p in act_model.parameters())
    print(f"  ✓ ACT loaded  ({n_act_params:,} params)  "
          f"action_horizon={act_ckpt['action_horizon']}", flush=True)

    # ─── 2. Load Safety MLP ──────────────────────────────────────
    print("[2/3] Loading Safety MLP...", flush=True)
    mlp_ckpt = torch.load(args.mlp_checkpoint, map_location=device,
                          weights_only=False)
    mlp = EEFCorrectionMLP(input_dim=mlp_ckpt["input_dim"]).to(device)
    mlp.load_state_dict(mlp_ckpt["model_state_dict"])
    mlp.eval()

    scaler = StandardScaler()
    scaler.mean_ = mlp_ckpt["scaler_mean"]
    scaler.scale_ = mlp_ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    agent = SteeredAgent(
        mlp, scaler,
        alpha=args.alpha,
        ema_beta=args.ema_beta,
        action_scale=args.action_scale,
        correction_threshold=args.correction_threshold,
        max_correction=args.max_correction,
        device=device,
    )
    n_mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"  ✓ MLP loaded  ({n_mlp_params:,} params)  "
          f"input_dim={mlp_ckpt['input_dim']}  "
          f"arch={mlp_ckpt.get('arch_version', 'unknown')}", flush=True)

    # ─── 3. Setup LIBERO ─────────────────────────────────────────
    print("[3/3] Setting up LIBERO environment...", flush=True)
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.env]()
    print(f"  ✓ {args.env}: {task_suite.n_tasks} tasks", flush=True)
    print(flush=True)

    # ═══════════════════════════════════════════════════════════════
    #  RUN EVALUATION
    # ═══════════════════════════════════════════════════════════════
    results = {}
    t0_all = time.time()

    for task_id in args.tasks:
        task = task_suite.get_task(task_id)
        task_name = task.language
        init_states = task_suite.get_task_init_states(task_id)

        # Create environment
        task_bddl_file = str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env = OffScreenRenderEnv(**env_args)

        print(f"━━━ Task {task_id}: {task_name[:60]}... ━━━", flush=True)

        task_results = {}
        for mode in args.modes:
            successes = 0
            ir_vals, cm_vals = [], []
            print(f"  {mode:>10}  ", end="", flush=True)

            for ep in range(args.episodes_per_task):
                init_state = (init_states[ep % len(init_states)]
                              if init_states is not None and len(init_states) > 0
                              else None)
                r = run_episode(
                    env, act_model, device,
                    init_state=init_state,
                    max_steps=300,
                    mode=mode,
                    steered_agent=agent,
                    noise_sigma=args.noise_sigma,
                    ema_only_beta=args.ema_only_beta,
                )
                print("✓" if r["success"] else "✗", end="", flush=True)
                if r["success"]:
                    successes += 1
                ir_vals.append(r["intervention_rate"])
                cm_vals.append(r["mean_corr_mag"])

            rate = successes / args.episodes_per_task * 100
            avg_ir = np.mean(ir_vals) if ir_vals else 0.0
            avg_cm = np.mean(cm_vals) if cm_vals else 0.0
            suffix = ""
            if mode == "steering":
                suffix = (f"  IR={avg_ir:.0%}  "
                          f"‖c‖={avg_cm:.4f}m")
                if avg_ir > 0.50:
                    suffix += "  ⚠ OVER-CORR"
            print(f"  {successes}/{args.episodes_per_task} "
                  f"({rate:.0f}%){suffix}", flush=True)

            task_results[mode] = {
                "success_rate_pct": round(rate, 1),
                "n_successes": successes,
                "n_episodes": args.episodes_per_task,
                "mean_ir": round(avg_ir, 4),
                "mean_corr_mag_m": round(avg_cm, 6),
            }

        # Compute Δ for all modes vs vanilla
        v_rate = task_results.get("vanilla", {}).get("success_rate_pct", 0)
        for mode in args.modes:
            if mode == "vanilla":
                continue
            delta = task_results[mode]["success_rate_pct"] - v_rate
            task_results[f"delta_{mode}_pp"] = round(delta, 1)

        # Print deltas
        for mode in args.modes:
            if mode == "vanilla":
                continue
            d = task_results.get(f"delta_{mode}_pp", 0)
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
            print(f"  {'Δ'+mode:>14}  {arrow}{abs(d):+.0f}pp", flush=True)

        results[task_id] = task_results
        env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(flush=True)

    total_time = time.time() - t0_all

    # ═══════════════════════════════════════════════════════════════
    #  RESULTS SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70, flush=True)
    print("RESULTS SUMMARY — ACT STEERING EVALUATION", flush=True)
    print("=" * 70, flush=True)

    hdr = f"  {'Task':>4}"
    for m in args.modes:
        hdr += f"  {m:>10}"
    print(f"\n{hdr}", flush=True)
    print(f"  {'─' * (6 + 12 * len(args.modes))}", flush=True)

    mode_avgs = {m: [] for m in args.modes}
    for tid in args.tasks:
        if tid not in results:
            continue
        r = results[tid]
        row = f"  {tid:>4}"
        for m in args.modes:
            rate = r.get(m, {}).get("success_rate_pct", 0)
            mode_avgs[m].append(rate)
            row += f"  {rate:>9.0f}%"
        print(row, flush=True)

    print(f"  {'─' * (6 + 12 * len(args.modes))}", flush=True)
    avg_row = f"  {'AVG':>4}"
    for m in args.modes:
        avg = np.mean(mode_avgs[m]) if mode_avgs[m] else 0
        avg_row += f"  {avg:>9.1f}%"
    print(avg_row, flush=True)

    # ── Verdict vs vanilla ──
    v_avg = np.mean(mode_avgs.get("vanilla", [0]))
    print(flush=True)
    print("  ╔══════════════════════════════════════════════════════════╗",
          flush=True)
    for m in args.modes:
        if m == "vanilla":
            continue
        m_avg = np.mean(mode_avgs.get(m, [0]))
        diff = m_avg - v_avg
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        status = "BETTER" if diff > 0 else ("SAME" if diff == 0 else "WORSE")
        print(f"  ║  {m:>10}: {m_avg:.1f}% vs Vanilla {v_avg:.1f}%  "
              f"→ {arrow}{abs(diff):.1f}pp ({status})", flush=True)
    print("  ╚══════════════════════════════════════════════════════════╝",
          flush=True)

    if "steering" in mode_avgs and "vanilla" in mode_avgs:
        s_avg = np.mean(mode_avgs["steering"])
        v_avg_f = np.mean(mode_avgs["vanilla"])
        if s_avg > v_avg_f:
            print(f"\n  ✅ CROSS-ARCHITECTURE PROOF: ACT+Steering ({s_avg:.1f}%) "
                  f"> Vanilla ACT ({v_avg_f:.1f}%)", flush=True)
            print(f"     MLP works on 256-dim CVAE latents, not just 4096-dim VLA!",
                  flush=True)
        else:
            print(f"\n  ⚠ ACT+Steering ({s_avg:.1f}%) ≤ Vanilla ACT ({v_avg_f:.1f}%)",
                  flush=True)
            print(f"     256-dim latent may lack spatial geometry for correction",
                  flush=True)

    print(f"\n  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)",
          flush=True)

    # ─── Save ─────────────────────────────────────────────────────
    summary = {}
    for m in args.modes:
        avg = np.mean(mode_avgs.get(m, [0]))
        summary[f"avg_{m}_pct"] = round(avg, 1)
        if m != "vanilla":
            summary[f"delta_{m}_vs_vanilla_pp"] = round(avg - v_avg, 1)

    report = {
        "config": {
            "env": args.env,
            "tasks": args.tasks,
            "modes": args.modes,
            "episodes_per_task": args.episodes_per_task,
            "alpha": args.alpha,
            "max_correction_m": args.max_correction,
            "correction_threshold_m": args.correction_threshold,
            "ema_beta": args.ema_beta,
            "action_scale": args.action_scale,
            "seed": args.seed,
            "arch_version": "v4",
            "model_type": "act",
            "act_checkpoint": args.act_checkpoint,
            "mlp_checkpoint": args.mlp_checkpoint,
            "act_params": n_act_params,
            "mlp_params": n_mlp_params,
            "mlp_input_dim": mlp_ckpt["input_dim"],
        },
        "per_task": {str(k): v for k, v in results.items()},
        "summary": summary,
    }
    out_path = save_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print("DONE", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
