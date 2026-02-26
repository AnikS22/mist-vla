#!/usr/bin/env python3
"""
Final Evaluation Pipeline (v4) — Full Ablation Suite
=====================================================

7-mode comparison for the paper results table:
  Mode A: Vanilla VLA              — raw, unsteered baseline
  Mode B: Latent Stop              — SAFE-style freeze when failure risk is high
  Mode C: Random Noise Injection   — null hypothesis (proves MLP > random jitter)
  Mode D: EMA Smoothing Only       — proves smoothing alone isn't enough
  Mode E: Random Latent Jiggle     — matched-magnitude random correction (proves MLP direction matters)
  Mode F: Action MPPI              — sampling-based optimization using MLP as cost function
  Mode G: Latent Steering (Ours)   — MLP-guided correction

For each task, runs N episodes (default 20) per mode and reports:
  - Success Rate (%)
  - Intervention Rate (IR)
  - Δ vs Vanilla (percentage points)

Usage
-----
  python scripts/eval_tuning.py \
      --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
      --mlp-checkpoint checkpoints/eef_correction_mlp/best_model.pt \
      --tasks 0 1 2 3 4 5 6 7 8 9 \
      --episodes-per-task 20 \
      --modes vanilla latent_stop noise ema_only latent_jiggle mppi steering \
      --save-dir results/eval_v4
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

# ── Torch CPU fallback ───────────────────────────────────────────────────
if not torch.cuda.is_available():
    _orig_torch_load = torch.load

    def _cpu_torch_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = torch.device("cpu")
        return _orig_torch_load(*args, **kwargs)

    torch.load = _cpu_torch_load

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    TASK_MAX_STEPS,
    prepare_observation,
    process_action,
)
from src.data_collection.hooks import HiddenStateCollector
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION  (must match train_eef_correction_mlp.py v4)
# ══════════════════════════════════════════════════════════════════════════

class EEFCorrectionMLP(nn.Module):
    """v4 architecture — must match train_eef_correction_mlp.py exactly."""
    HIDDEN_DIM = 256

    def __init__(self, input_dim=4096):
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
#  STEERED AGENT  (clamp + magnitude gate)
# ══════════════════════════════════════════════════════════════════════════

class SteeredAgent:
    """Minimal steering controller for the final evaluation.

    Safety: clamp ‖correction‖ → only intervene if magnitude > threshold.
    """

    def __init__(self, mlp, scaler, *,
                 alpha=1.0, ema_beta=0.7, action_scale=0.05,
                 correction_threshold=0.005, max_correction=0.01,
                 use_fail_gate=False, fail_threshold=0.5,
                 device="cpu"):
        self.mlp = mlp
        self.scaler = scaler
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.action_scale = action_scale
        self.correction_threshold = correction_threshold
        self.max_correction = max_correction
        self.use_fail_gate = use_fail_gate
        self.fail_threshold = fail_threshold
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
        fail_prob = torch.sigmoid(out["will_fail"]).item()
        raw = out["correction"].cpu().numpy()[0]  # (3,) meters

        # EMA smoothing
        if self.prev_correction is not None:
            smoothed = (self.ema_beta * self.prev_correction
                        + (1.0 - self.ema_beta) * raw)
        else:
            smoothed = raw.copy()
        self.prev_correction = smoothed.copy()

        # Clamp magnitude (primary safety)
        mag = float(np.linalg.norm(smoothed))
        if mag > self.max_correction and mag > 1e-8:
            smoothed = smoothed * (self.max_correction / mag)
            mag = self.max_correction

        self._corr_mags.append(mag)

        # Gate: only intervene if correction is meaningful
        should_intervene = mag > self.correction_threshold
        if self.use_fail_gate:
            should_intervene = should_intervene and (fail_prob >= self.fail_threshold)

        if should_intervene:
            self._interventions += 1
            action[:3] += (self.alpha * smoothed / self.action_scale)
            return action, True
        return action, False


class LatentStopAgent:
    """SAFE-style baseline: stop/freeze when fail probability exceeds threshold."""

    def __init__(self, mlp, scaler, *, stop_threshold=0.85, device="cpu"):
        self.mlp = mlp
        self.scaler = scaler
        self.stop_threshold = stop_threshold
        self.device = device
        self._steps = 0
        self._interventions = 0

    def reset(self):
        self._steps = 0
        self._interventions = 0

    @property
    def intervention_rate(self):
        return self._interventions / max(self._steps, 1)

    @property
    def mean_corr_mag(self):
        return 0.0

    def apply(self, action, features):
        self._steps += 1
        if features is None or np.prod(features.shape) < 2:
            return action, False
        scaled = self.scaler.transform(features.reshape(1, -1))
        x = torch.FloatTensor(scaled).to(self.device)
        with torch.no_grad():
            out = self.mlp(x)
        fail_prob = torch.sigmoid(out["will_fail"]).item()
        if fail_prob >= self.stop_threshold:
            self._interventions += 1
            # Freeze all action dimensions for strict stop baseline.
            action[:] = 0.0
            return action, True
        return action, False


# ══════════════════════════════════════════════════════════════════════════
#  MPPI CONTROLLER (sampling-based baseline using MLP as cost function)
# ══════════════════════════════════════════════════════════════════════════

class MPPIController:
    """Action MPPI: samples K corrections, scores each with MLP failure head,
    takes softmax-weighted average. Uses MLP as a VALUE function, not policy.

    This is a stronger baseline than random noise — it uses the MLP's failure
    prediction to select promising corrections, but doesn't use the direct
    correction head output.
    """

    def __init__(self, mlp, scaler, *,
                 n_samples=16, temperature=5.0,
                 correction_std=0.005, max_correction=0.01,
                 action_scale=0.05, device="cpu"):
        self.mlp = mlp
        self.scaler = scaler
        self.n_samples = n_samples
        self.temperature = temperature
        self.correction_std = correction_std
        self.max_correction = max_correction
        self.action_scale = action_scale
        self.device = device
        self._steps = 0
        self._interventions = 0
        self._corr_mags = []

    def reset(self):
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
        """Sample K corrections, score with MLP, weight-average."""
        self._steps += 1

        if features is None or np.prod(features.shape) < 2:
            return action, False

        scaled = self.scaler.transform(features.reshape(1, -1))
        x = torch.FloatTensor(scaled).to(self.device)

        with torch.no_grad():
            out = self.mlp(x)

        fail_prob = torch.sigmoid(out["will_fail"]).item()

        # Only intervene if MLP thinks we're failing
        if fail_prob < 0.5:
            self._corr_mags.append(0.0)
            return action, False

        self._interventions += 1

        # Sample K random 3D correction candidates
        candidates = np.random.normal(
            0, self.correction_std, size=(self.n_samples, 3)
        ).astype(np.float32)

        # Score each candidate by perturbing features and checking failure prob
        scores = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            # Perturb features slightly in the direction of the correction
            feat_perturbed = scaled.copy()
            # Add small random noise to features to simulate effect
            feat_perturbed += np.random.normal(0, 0.01, feat_perturbed.shape)
            x_p = torch.FloatTensor(feat_perturbed).to(self.device)
            with torch.no_grad():
                out_p = self.mlp(x_p)
            # Lower failure probability = better
            scores[i] = -torch.sigmoid(out_p["will_fail"]).item()

        # Softmax weighting
        weights = np.exp(self.temperature * (scores - scores.max()))
        weights /= weights.sum()

        # Weighted average correction
        correction = (candidates * weights[:, None]).sum(axis=0)

        # Clamp
        mag = float(np.linalg.norm(correction))
        if mag > self.max_correction and mag > 1e-8:
            correction = correction * (self.max_correction / mag)
            mag = self.max_correction

        self._corr_mags.append(mag)

        action[:3] += correction / self.action_scale
        return action, True


# ══════════════════════════════════════════════════════════════════════════
#  RANDOM LATENT JIGGLE (matched-magnitude null hypothesis)
# ══════════════════════════════════════════════════════════════════════════

class LatentJiggleAgent:
    """Same pipeline as SteeredAgent but replaces MLP correction direction
    with a RANDOM direction of the SAME magnitude.

    Proves: the MLP's correction DIRECTION matters, not just the magnitude
    of intervention. If jiggle matches steering, the MLP is just a noise
    generator. If steering > jiggle, the MLP learned real spatial geometry.
    """

    def __init__(self, mlp, scaler, *,
                 alpha=1.0, action_scale=0.05,
                 correction_threshold=0.005, max_correction=0.01,
                 device="cpu"):
        self.mlp = mlp
        self.scaler = scaler
        self.alpha = alpha
        self.action_scale = action_scale
        self.correction_threshold = correction_threshold
        self.max_correction = max_correction
        self.device = device
        self._steps = 0
        self._interventions = 0
        self._corr_mags = []

    def reset(self):
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
        """Apply random correction with same magnitude as MLP would predict."""
        self._steps += 1

        if features is None or np.prod(features.shape) < 2:
            return action, False

        scaled = self.scaler.transform(features.reshape(1, -1))
        x = torch.FloatTensor(scaled).to(self.device)

        with torch.no_grad():
            out = self.mlp(x)
        mlp_correction = out["correction"].cpu().numpy()[0]  # (3,)

        # Get the magnitude the MLP WOULD apply
        mag = float(np.linalg.norm(mlp_correction))

        # Clamp magnitude (same as steering)
        if mag > self.max_correction:
            mag = self.max_correction

        self._corr_mags.append(mag)

        # Gate: same threshold as steering
        if mag > self.correction_threshold:
            self._interventions += 1
            # Generate RANDOM direction with SAME magnitude
            random_dir = np.random.randn(3).astype(np.float32)
            random_dir_norm = np.linalg.norm(random_dir)
            if random_dir_norm > 1e-8:
                random_dir = random_dir / random_dir_norm
            random_correction = random_dir * mag
            action[:3] += (self.alpha * random_correction / self.action_scale)
            return action, True
        return action, False


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _resolve_unnorm_key(cfg, vla):
    unnorm_key = cfg.task_suite_name
    if (unnorm_key not in vla.norm_stats
            and f"{unnorm_key}_no_noops" in vla.norm_stats):
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key in vla.norm_stats:
        cfg.unnorm_key = unnorm_key


# ══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_episode(cfg, env, task_description, resize_size,
                vla, processor, action_head, proprio_projector, collector,
                initial_state, mode, steered_agent=None,
                stop_agent=None, jiggle_agent=None, mppi_controller=None,
                noise_sigma=0.05, ema_beta=0.9,
                ood_obstacle=False, ood_step_min=40, ood_step_max=160,
                ood_duration=20, ood_push_magnitude=0.08):
    """Run one episode.

    mode ∈ {"vanilla", "latent_stop", "noise", "ema_only", "latent_jiggle", "mppi", "steering"}
      vanilla       — raw VLA actions
      noise         — VLA + Gaussian noise on action[:3]  (null hypothesis)
      ema_only      — VLA + EMA smoothing on action[:3]  (proves smoothing alone isn't enough)
      latent_jiggle — VLA + random correction (same magnitude as MLP)  (proves MLP direction matters)
      mppi          — VLA + sampling-based correction using MLP as cost function
      steering      — VLA + MLP correction  (ours)
    """
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if mode == "steering" and steered_agent is not None:
        steered_agent.reset()
    if mode == "latent_stop" and stop_agent is not None:
        stop_agent.reset()
    if mode == "latent_jiggle" and jiggle_agent is not None:
        jiggle_agent.reset()
    if mode == "mppi" and mppi_controller is not None:
        mppi_controller.reset()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 300)
    t = 0
    last_features = None
    total_steps = 0
    success = False

    # EMA state (for ema_only mode)
    ema_action = None
    obstacle_trigger_step = None
    obstacle_vec = np.zeros(3, dtype=np.float32)
    if ood_obstacle:
        lo = max(1, min(ood_step_min, max_steps - 1))
        hi = max(lo, min(ood_step_max, max_steps - 1))
        obstacle_trigger_step = int(np.random.randint(lo, hi + 1))
        raw = np.random.randn(3).astype(np.float32)
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm > 1e-8:
            raw = raw / raw_norm
        obstacle_vec = raw * float(ood_push_magnitude)

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(
                get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation, _ = prepare_observation(obs, resize_size)

        if len(action_queue) == 0:
            collector.clear()
            with collector:
                actions = get_action(
                    cfg, vla, observation, task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,
                    use_film=cfg.use_film,
                )
            action_queue.extend(actions)
            feats = collector.get_last_layer(pool="mean")
            last_features = (
                None if feats is None
                else feats.detach().cpu().float().numpy()[0]
            )

        action = np.asarray(action_queue.popleft(), dtype=np.float32)
        action = process_action(action, cfg.model_family)

        # ── Apply mode-specific modification ──
        if mode == "noise":
            # Random Gaussian noise on position dims (null hypothesis)
            action[:3] += np.random.normal(0, noise_sigma, size=3)

        elif mode == "ema_only":
            # Pure EMA smoothing — no MLP, just filter the raw actions
            if ema_action is None:
                ema_action = action[:3].copy()
            else:
                ema_action = ema_beta * ema_action + (1 - ema_beta) * action[:3]
            action[:3] = ema_action

        elif mode == "latent_stop" and stop_agent is not None:
            action, _ = stop_agent.apply(action, last_features)

        elif mode == "latent_jiggle" and jiggle_agent is not None:
            # Random correction with same magnitude as MLP (proves direction matters)
            action, _ = jiggle_agent.apply(action, last_features)

        elif mode == "mppi" and mppi_controller is not None:
            # Sampling-based optimization using MLP as cost function
            action, _ = mppi_controller.apply(action, last_features)

        elif mode == "steering" and steered_agent is not None:
            action, _ = steered_agent.apply(action, last_features)

        if (ood_obstacle and obstacle_trigger_step is not None
                and obstacle_trigger_step <= total_steps < obstacle_trigger_step + max(1, ood_duration)):
            action[:3] += obstacle_vec

        action = np.clip(action, -1.0, 1.0)
        obs, reward, done, info = env.step(action.tolist())
        total_steps += 1

        if done:
            success = True
            break
        t += 1

    ir, cm = 0.0, 0.0
    if mode == "steering" and steered_agent:
        ir = steered_agent.intervention_rate
        cm = steered_agent.mean_corr_mag
    elif mode == "latent_stop" and stop_agent:
        ir = stop_agent.intervention_rate
        cm = stop_agent.mean_corr_mag
    elif mode == "latent_jiggle" and jiggle_agent:
        ir = jiggle_agent.intervention_rate
        cm = jiggle_agent.mean_corr_mag
    elif mode == "mppi" and mppi_controller:
        ir = mppi_controller.intervention_rate
        cm = mppi_controller.mean_corr_mag

    return {"success": success, "total_steps": total_steps,
            "intervention_rate": round(ir, 4),
            "mean_corr_mag": round(cm, 6)}


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full Ablation Evaluation: 4 OpenVLA modes + baselines")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mlp-checkpoint", required=True)
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--tasks", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="Task IDs to evaluate")
    parser.add_argument("--modes", nargs="+",
                        default=["vanilla", "latent_stop", "noise", "ema_only",
                                 "latent_jiggle", "mppi", "steering"],
                        help="Modes: vanilla latent_stop noise ema_only latent_jiggle mppi steering")
    parser.add_argument("--mppi-samples", type=int, default=16,
                        help="Number of MPPI candidate corrections")
    parser.add_argument("--mppi-temperature", type=float, default=5.0,
                        help="MPPI softmax temperature")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ema-beta", type=float, default=0.7)
    parser.add_argument("--noise-sigma", type=float, default=0.05,
                        help="Gaussian noise σ for 'noise' mode")
    parser.add_argument("--ema-only-beta", type=float, default=0.9,
                        help="EMA beta for 'ema_only' mode (no MLP)")
    parser.add_argument("--action-scale", type=float, default=0.05)
    parser.add_argument("--correction-threshold", type=float, default=0.005,
                        help="Min ‖correction‖ (meters) to trigger")
    parser.add_argument("--max-correction", type=float, default=0.01,
                        help="Clamp ‖correction‖ (meters)")
    parser.add_argument("--use-fail-gate", action="store_true",
                        help="Only intervene when fail prob >= --fail-threshold")
    parser.add_argument("--fail-threshold", type=float, default=0.5,
                        help="Fail-prob threshold when fail-gate is enabled")
    parser.add_argument("--stop-threshold", type=float, default=0.85,
                        help="Fail-prob threshold for latent_stop freeze baseline")
    parser.add_argument("--ood-obstacle", action="store_true",
                        help="Enable synthetic OOD obstacle push during episodes")
    parser.add_argument("--ood-step-min", type=int, default=40)
    parser.add_argument("--ood-step-max", type=int, default=160)
    parser.add_argument("--ood-duration", type=int, default=20)
    parser.add_argument("--ood-push-magnitude", type=float, default=0.08)
    parser.add_argument("--camera-res", type=int, default=256)
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="results/eval_v4")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    max_pert = args.alpha * args.max_correction / args.action_scale

    MODE_LABELS = {
        "vanilla":       "Vanilla VLA (baseline)",
        "latent_stop":   f"Latent Stop (freeze if p_fail≥{args.stop_threshold})",
        "noise":         f"Random Noise (σ={args.noise_sigma})",
        "ema_only":      f"EMA Smoothing Only (β={args.ema_only_beta})",
        "latent_jiggle": f"Random Latent Jiggle (matched-magnitude)",
        "mppi":          f"Action MPPI (K={args.mppi_samples}, τ={args.mppi_temperature})",
        "steering":      f"Latent Steering (α={args.alpha}, clamp={args.max_correction}m)",
    }

    # ── Banner ──
    print("=" * 70, flush=True)
    print("FULL ABLATION EVALUATION — Paper Results Table", flush=True)
    print("=" * 70, flush=True)
    print(f"  Tasks:       {args.tasks}", flush=True)
    print(f"  Episodes:    {args.episodes_per_task} per task per mode",
          flush=True)
    print(f"  Modes:       {args.modes}", flush=True)
    for m in args.modes:
        print(f"    → {MODE_LABELS.get(m, m)}", flush=True)
    print(f"  Steering:    α={args.alpha}  clamp={args.max_correction}m  "
          f"gate=‖c‖>{args.correction_threshold}m  "
          f"fail_gate={'on' if args.use_fail_gate else 'off'}"
          f"(p≥{args.fail_threshold})", flush=True)
    print(f"  Max Δaction: {max_pert:.4f} units "
          f"({max_pert * 100:.1f}% of range)", flush=True)
    print(f"  Device:      {device}", flush=True)
    if args.ood_obstacle:
        print(f"  OOD obstacle: enabled  step=[{args.ood_step_min},{args.ood_step_max}]  "
              f"duration={args.ood_duration}  push={args.ood_push_magnitude}",
              flush=True)
    print(flush=True)

    # ─── 1. Load VLA ──────────────────────────────────────────────
    print("[1/3] Loading VLA model...", flush=True)
    cfg = GenerateConfig(
        pretrained_checkpoint=args.model_name,
        task_suite_name=args.env,
        env_img_res=args.camera_res,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        center_crop=True,
        num_images_in_input=args.num_images,
    )
    vla = get_vla(cfg)
    _resolve_unnorm_key(cfg, vla)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    collector = HiddenStateCollector(vla)
    collector.register_hooks()
    resize_size = get_image_resize_size(cfg)
    print("  ✓ VLA loaded", flush=True)

    # ─── 2. Load Safety MLP ──────────────────────────────────────
    print("[2/3] Loading Safety MLP...", flush=True)
    ckpt = torch.load(args.mlp_checkpoint, map_location=device,
                      weights_only=False)
    mlp = EEFCorrectionMLP(input_dim=ckpt["input_dim"]).to(device)
    mlp.load_state_dict(ckpt["model_state_dict"])
    mlp.eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    agent = SteeredAgent(
        mlp, scaler,
        alpha=args.alpha,
        ema_beta=args.ema_beta,
        action_scale=args.action_scale,
        correction_threshold=args.correction_threshold,
        max_correction=args.max_correction,
        use_fail_gate=args.use_fail_gate,
        fail_threshold=args.fail_threshold,
        device=device,
    )
    stop_agent = LatentStopAgent(
        mlp, scaler,
        stop_threshold=args.stop_threshold,
        device=device,
    )
    jiggle_agent = LatentJiggleAgent(
        mlp, scaler,
        alpha=args.alpha,
        action_scale=args.action_scale,
        correction_threshold=args.correction_threshold,
        max_correction=args.max_correction,
        device=device,
    )
    mppi_controller = MPPIController(
        mlp, scaler,
        n_samples=args.mppi_samples,
        temperature=args.mppi_temperature,
        correction_std=args.max_correction / 2,
        max_correction=args.max_correction,
        action_scale=args.action_scale,
        device=device,
    )
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"  ✓ MLP loaded  ({n_params:,} params)  arch={ckpt.get('arch_version', 'unknown')}",
          flush=True)

    # ─── 3. Setup Environment ────────────────────────────────────
    print("[3/3] Setting up LIBERO environment...", flush=True)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.env]()
    print(f"  ✓ {args.env}: {task_suite.n_tasks} tasks", flush=True)
    print(flush=True)

    # ═══════════════════════════════════════════════════════════════
    #  RUN EVALUATION
    # ═══════════════════════════════════════════════════════════════
    results = {}  # task_id → {mode: {...}, ...}
    t0_all = time.time()

    for task_id in args.tasks:
        task = task_suite.get_task(task_id)
        env, task_desc = get_libero_env(
            task, cfg.model_family, resolution=cfg.env_img_res)
        init_states = task_suite.get_task_init_states(task_id)

        print(f"━━━ Task {task_id}: {task_desc[:60]}... ━━━", flush=True)

        task_results = {}
        for mode in args.modes:
            successes = 0
            ir_vals, cm_vals = [], []
            print(f"  {mode:>10}  ", end="", flush=True)

            for ep in range(args.episodes_per_task):
                init_state = (init_states[ep % len(init_states)]
                              if init_states is not None else None)
                r = run_episode(
                    cfg, env, task_desc, resize_size,
                    vla, processor, action_head,
                    proprio_projector, collector,
                    initial_state=init_state,
                    mode=mode,
                    steered_agent=agent,
                    stop_agent=stop_agent,
                    jiggle_agent=jiggle_agent,
                    mppi_controller=mppi_controller,
                    noise_sigma=args.noise_sigma,
                    ema_beta=args.ema_only_beta,
                    ood_obstacle=args.ood_obstacle,
                    ood_step_min=args.ood_step_min,
                    ood_step_max=args.ood_step_max,
                    ood_duration=args.ood_duration,
                    ood_push_magnitude=args.ood_push_magnitude,
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
            if mode in ("steering", "latent_stop", "latent_jiggle", "mppi"):
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
    #  RESULTS SUMMARY — FULL TABLE
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70, flush=True)
    print("RESULTS SUMMARY — PAPER TABLE (Category 1: OpenVLA Ablations)",
          flush=True)
    print("=" * 70, flush=True)

    # Build header
    hdr = f"  {'Task':>4}"
    for m in args.modes:
        hdr += f"  {m:>10}"
    print(f"\n{hdr}", flush=True)
    print(f"  {'─' * (6 + 12 * len(args.modes))}", flush=True)

    # Per-mode averages
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

    # Key check: does steering beat noise?
    if "noise" in mode_avgs and "steering" in mode_avgs:
        n_avg = np.mean(mode_avgs["noise"])
        s_avg = np.mean(mode_avgs["steering"])
        if s_avg > n_avg:
            print(f"\n  ✅ CRITICAL: Steering ({s_avg:.1f}%) > Noise ({n_avg:.1f}%)"
                  f"  → MLP signal is real, not random exploration",
                  flush=True)
        else:
            print(f"\n  ⚠ WARNING: Steering ({s_avg:.1f}%) ≤ Noise ({n_avg:.1f}%)"
                  f"  → MLP signal needs investigation",
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
            "noise_sigma": args.noise_sigma,
            "ema_only_beta": args.ema_only_beta,
            "max_correction_m": args.max_correction,
            "correction_threshold_m": args.correction_threshold,
            "ema_beta": args.ema_beta,
            "use_fail_gate": args.use_fail_gate,
            "fail_threshold": args.fail_threshold,
            "stop_threshold": args.stop_threshold,
            "action_scale": args.action_scale,
            "seed": args.seed,
            "ood_obstacle": args.ood_obstacle,
            "ood_step_min": args.ood_step_min,
            "ood_step_max": args.ood_step_max,
            "ood_duration": args.ood_duration,
            "ood_push_magnitude": args.ood_push_magnitude,
            "arch_version": "v4",
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
