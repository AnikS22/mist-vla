#!/usr/bin/env python3
"""
Closed-Loop Evaluation Study: 4-Way Baseline Comparison
========================================================

Proves that Latent Safety Steering is mechanistically superior
to random exploration by comparing against strict baselines.

Modes
-----
  A  Vanilla VLA          — Lower bound (pure VLA, no modification)
  B  Random Noise σ=0.05  — Null hypothesis (Gaussian XYZ noise)
  C  Latent Steering α=1  — OURS (MLP correction with EMA smoothing)
  D  Oracle Expert         — Upper bound (replay success-rollout actions)

Safety Metrics (per episode)
----------------------------
  - success:                bool
  - constraint_violations:  int   (robot→table/obstacle collisions)
  - trajectory_deviation:   float (DTW distance from nearest expert path)
  - total_steps:            int

Output
------
  results_table.json     — per-mode aggregated comparison
  episode_details.json   — per-episode breakdown
  combined_video.mp4     — side-by-side comparison (optional)

Usage
-----
  python scripts/eval_closed_loop_study.py \\
      --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \\
      --mlp-checkpoint checkpoints/eef_correction_mlp/best_model.pt \\
      --env libero_spatial \\
      --n-episodes 50 \\
      --alpha 1.0 \\
      --save-dir results/closed_loop
"""

import argparse
import gc
import json
import os
import pickle
import time
from collections import defaultdict, deque
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

from src.data_collection.collision_detection import CollisionDetector
from src.data_collection.hooks import HiddenStateCollector
from sklearn.preprocessing import StandardScaler

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ══════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION  (must match train_eef_correction_mlp.py)
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
#  STEERED AGENT  (Hybrid Controller)
# ══════════════════════════════════════════════════════════════════════════

class SteeredAgent:
    """Gated steering controller with correction clamping.

    Safety hierarchy (applied in order):
      1. **Correction clamping** — ‖correction‖ is clamped to ``max_correction``
         meters *before* anything else.  This is the primary safety mechanism:
         even if the MLP predicts a 13 cm correction, it gets clipped to e.g.
         2 cm.  This prevents catastrophic over-steering.
      2. **Gating** — Only intervene when the gate condition is satisfied.
         ``"magnitude"``  — ‖correction‖ > ``correction_threshold``
         ``"p_fail"``     — P(fail) > ``fail_threshold``
         ``"combined"``   — both must hold
      3. **EMA smoothing** — exponential moving average prevents jitter.
      4. **Unit conversion** — divide by ``action_scale`` (meters → action units).
      5. **Action clamping** — final action clipped to [-1, 1] (caller does this).
    """

    GATE_MODES = ("magnitude", "p_fail", "combined")

    def __init__(self, mlp, scaler, alpha=1.0, ema_beta=0.7,
                 fail_threshold=0.85, action_scale=0.05,
                 gate_mode="magnitude", correction_threshold=0.01,
                 max_correction=0.02, device="cpu"):
        assert gate_mode in self.GATE_MODES, \
            f"gate_mode must be one of {self.GATE_MODES}, got '{gate_mode}'"
        self.mlp = mlp
        self.scaler = scaler
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.fail_threshold = fail_threshold
        self.action_scale = action_scale
        self.gate_mode = gate_mode
        self.correction_threshold = correction_threshold
        self.max_correction = max_correction
        self.device = device
        self.prev_correction = None
        # ── Intervention monitoring ──
        self._ep_total_steps = 0
        self._ep_interventions = 0
        # ── Diagnostics (per-episode) ──
        self._ep_corr_magnitudes = []
        self._ep_p_fails = []

    def reset(self):
        """Call at the start of each episode."""
        self.prev_correction = None
        self._ep_total_steps = 0
        self._ep_interventions = 0
        self._ep_corr_magnitudes = []
        self._ep_p_fails = []

    @property
    def intervention_rate(self):
        """Fraction of steps where a correction was actually applied."""
        if self._ep_total_steps == 0:
            return 0.0
        return self._ep_interventions / self._ep_total_steps

    @property
    def mean_correction_magnitude(self):
        """Mean ‖correction‖ (meters) across the episode."""
        if not self._ep_corr_magnitudes:
            return 0.0
        return float(np.mean(self._ep_corr_magnitudes))

    @property
    def mean_p_fail(self):
        """Mean predicted failure probability across the episode."""
        if not self._ep_p_fails:
            return 0.0
        return float(np.mean(self._ep_p_fails))

    def get_correction(self, features):
        """Return ``(p_fail, smoothed_correction)`` for the current step.

        Returns
        -------
        p_fail : float
            Predicted probability of failure (0-1).
        correction : np.ndarray, shape (3,)
            EMA-smoothed correction in **meters** (caller handles scaling).
        """
        if features is None or np.prod(features.shape) < 2:
            return 0.0, np.zeros(3, dtype=np.float32)

        scaled = self.scaler.transform(features.reshape(1, -1))
        x = torch.FloatTensor(scaled).to(self.device)

        with torch.no_grad():
            out = self.mlp(x)
        raw = out["correction"].cpu().numpy()[0]           # (3,) meters
        p_fail = float(torch.sigmoid(out["will_fail"]).cpu().item())

        if self.prev_correction is not None:
            smoothed = (self.ema_beta * self.prev_correction
                        + (1.0 - self.ema_beta) * raw)
        else:
            smoothed = raw.copy()

        self.prev_correction = smoothed.copy()
        return p_fail, smoothed

    def _should_intervene(self, p_fail, correction_mag):
        """Decide whether to apply the correction based on gate_mode."""
        if self.gate_mode == "p_fail":
            return p_fail > self.fail_threshold
        elif self.gate_mode == "magnitude":
            return correction_mag > self.correction_threshold
        elif self.gate_mode == "combined":
            return (p_fail > self.fail_threshold
                    and correction_mag > self.correction_threshold)
        return False

    def apply(self, action, features):
        """Apply gated steering correction to *action* **in-place**.

        Returns ``(action, applied, p_fail, corr_mag)`` where *applied*
        indicates whether the correction was actually added.
        """
        self._ep_total_steps += 1
        p_fail, correction = self.get_correction(features)

        # ── Clamp correction magnitude (primary safety mechanism) ──
        corr_mag = float(np.linalg.norm(correction))
        if corr_mag > self.max_correction and corr_mag > 1e-8:
            correction = correction * (self.max_correction / corr_mag)
            corr_mag = self.max_correction

        # ── Record diagnostics ──
        self._ep_corr_magnitudes.append(corr_mag)
        self._ep_p_fails.append(p_fail)

        applied = self._should_intervene(p_fail, corr_mag)
        if applied:
            self._ep_interventions += 1
            action[:3] += (self.alpha * correction / self.action_scale)
        return action, applied, p_fail, corr_mag


# ══════════════════════════════════════════════════════════════════════════
#  TRAJECTORY METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_dtw(traj_a, traj_b):
    """Dynamic Time Warping distance between (T1,3) and (T2,3) EEF paths.

    Returns normalised distance: total_cost / max(n, m).
    """
    n, m = len(traj_a), len(traj_b)
    if n == 0 or m == 0:
        return float("nan")
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(traj_a[i - 1] - traj_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])
    return dtw[n, m] / max(n, m)


def min_expert_dtw(eef, expert_trajs):
    """Minimum DTW to any expert trajectory for the same task."""
    if not expert_trajs or len(eef) == 0:
        return float("nan")
    return min(compute_dtw(eef, e) for e in expert_trajs)


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _get_robot_state(env):
    """Extract robot state including EEF position."""
    sim = None
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        sim = env.env.sim
    elif (hasattr(env, "env") and hasattr(env.env, "env")
          and hasattr(env.env.env, "sim")):
        sim = env.env.env.sim
    elif hasattr(env, "sim"):
        sim = env.sim
    if sim is None:
        return {}

    state = {"qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy()}
    for name in ("gripper0_grip_site", "robot0_eef", "eef", "ee_site",
                 "right_gripper"):
        try:
            sid = sim.model.site_name2id(name)
            state["eef_pos"] = sim.data.site_xpos[sid].copy()
            break
        except Exception:
            continue
    return state


def _resolve_unnorm_key(cfg, vla):
    unnorm_key = cfg.task_suite_name
    if (unnorm_key not in vla.norm_stats
            and f"{unnorm_key}_no_noops" in vla.norm_stats):
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key in vla.norm_stats:
        cfg.unnorm_key = unnorm_key


# ══════════════════════════════════════════════════════════════════════════
#  EXPERT DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def _glob_pkls(data_path):
    """Recursively find all non-partial .pkl files under *data_path*."""
    p = Path(data_path)
    pkls = []
    if p.is_file() and p.suffix == ".pkl":
        pkls = [p]
    elif p.is_dir():
        for sub in sorted(p.iterdir()):
            if sub.is_dir():
                pkls.extend(sorted(sub.glob("*.pkl")))
            elif sub.suffix == ".pkl":
                pkls.append(sub)
    return [pk for pk in pkls if "_partial" not in pk.name]


def load_expert_eef_trajectories(data_path):
    """Load success-rollout EEF trajectories as expert references for DTW."""
    expert = defaultdict(list)
    for pkl in _glob_pkls(data_path):
        try:
            with open(pkl, "rb") as f:
                rols = pickle.load(f)
        except Exception:
            continue
        for r in rols:
            if not r.get("success"):
                continue
            eef = []
            for rs in r.get("robot_states", []):
                if "eef_pos" in rs:
                    eef.append(np.array(rs["eef_pos"], dtype=np.float32))
            if eef:
                expert[r["task_id"]].append(np.array(eef))
    return expert


def load_oracle_actions(data_path):
    """Load success-rollout action sequences for Oracle replay (Mode D)."""
    oracle = defaultdict(list)
    for pkl in _glob_pkls(data_path):
        try:
            with open(pkl, "rb") as f:
                rols = pickle.load(f)
        except Exception:
            continue
        for r in rols:
            if not r.get("success"):
                continue
            actions = r.get("actions", [])
            if actions is not None and len(actions) > 0:
                acts = [np.array(a, dtype=np.float32) for a in actions]
                oracle[r["task_id"]].append(acts)
    return oracle


# ══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_episode(
    cfg, env, task_description, resize_size,
    vla, processor, action_head, proprio_projector, collector,
    initial_state,
    mode,                   # "vanilla" | "noise" | "steering" | "oracle"
    steered_agent=None,
    noise_sigma=0.05,
    oracle_actions=None,    # list[np.ndarray] — one action per timestep
    record_frames=False,
):
    """Run one closed-loop episode under the specified *mode*.

    Parameters
    ----------
    mode : str
        "vanilla"  — pure VLA (Mode A)
        "noise"    — VLA + Gaussian noise on XYZ (Mode B)
        "steering" — VLA + MLP correction with EMA (Mode C)
        "oracle"   — replay stored success actions (Mode D)
    """
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    detector = CollisionDetector(env)
    if steered_agent is not None and mode == "steering":
        steered_agent.reset()

    result = {
        "mode": mode,
        "success": False,
        "total_steps": 0,
        "constraint_violations": 0,
        "eef_trajectory": [],
        "frames": [] if record_frames else None,
    }

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 300)
    t = 0
    last_features = None
    oracle_idx = 0

    while t < max_steps + cfg.num_steps_wait:
        # ── Warm-up (dummy actions, matches official OFT eval) ──
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(
                get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        # ────────────────────────────────────────────────────────
        #  MODE D — Oracle: replay stored success-rollout actions
        # ────────────────────────────────────────────────────────
        if mode == "oracle":
            if oracle_actions is not None and oracle_idx < len(oracle_actions):
                action = np.array(oracle_actions[oracle_idx], dtype=np.float32)
            else:
                # Demo exhausted — issue dummy
                action = np.array(
                    get_libero_dummy_action(cfg.model_family),
                    dtype=np.float32)
            oracle_idx += 1
        else:
            # ────────────────────────────────────────────────────
            #  MODES A/B/C — VLA inference (shared)
            # ────────────────────────────────────────────────────
            observation, raw_img = prepare_observation(obs, resize_size)

            if record_frames and raw_img is not None:
                result["frames"].append(raw_img.copy())

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

            # ── Mode-specific modifications ──
            if mode == "noise":
                noise = np.random.normal(
                    0, noise_sigma, size=3).astype(np.float32)
                action[:3] += noise

            elif mode == "steering" and steered_agent is not None:
                action, _applied, _pf, _cm = steered_agent.apply(
                    action, last_features)

            # ── Safety clamp ──
            action = np.clip(action, -1.0, 1.0)

        # ── Execute action ──
        obs, reward, done, info = env.step(action.tolist())
        result["total_steps"] += 1

        # ── Constraint violation check ──
        has_collision, _, _, _, _ = detector.check_collision_details()
        if has_collision:
            result["constraint_violations"] += 1

        # ── Log EEF position ──
        rs = _get_robot_state(env)
        if "eef_pos" in rs:
            result["eef_trajectory"].append(rs["eef_pos"].tolist())

        # ── Success check (same as official OFT eval) ──
        if done:
            result["success"] = True
            break
        t += 1

    result["eef_trajectory"] = (
        np.array(result["eef_trajectory"], dtype=np.float32)
        if result["eef_trajectory"]
        else np.empty((0, 3))
    )

    # ── Intervention monitoring (steering mode only) ──
    if mode == "steering" and steered_agent is not None:
        result["intervention_rate"] = round(steered_agent.intervention_rate, 4)
        result["interventions"] = steered_agent._ep_interventions
        result["mean_corr_mag"] = round(steered_agent.mean_correction_magnitude, 6)
        result["mean_p_fail"] = round(steered_agent.mean_p_fail, 4)
    else:
        result["intervention_rate"] = 0.0
        result["interventions"] = 0
        result["mean_corr_mag"] = 0.0
        result["mean_p_fail"] = 0.0

    return result


# ══════════════════════════════════════════════════════════════════════════
#  SIDE-BY-SIDE VIDEO
# ══════════════════════════════════════════════════════════════════════════

MODE_LABELS = {
    "vanilla":  "A: Vanilla VLA",
    "noise":    "B: Random Noise",
    "steering": "C: Steering (Ours)",
    "oracle":   "D: Oracle Expert",
}


def create_comparison_video(frames_dict, save_path, fps=10):
    """Create a 2×2 grid video from ``{mode: [frames]}``."""
    if not HAS_IMAGEIO or not HAS_PIL:
        print("  [video] imageio/PIL not available — skipping", flush=True)
        return

    modes = [m for m in frames_dict if len(frames_dict[m]) > 0]
    if len(modes) < 2:
        print("  [video] fewer than 2 modes recorded — skipping", flush=True)
        return

    max_len = max(len(frames_dict[m]) for m in modes)
    if max_len == 0:
        return

    # Pad shorter frame lists with the last frame
    padded = {}
    for m in modes:
        fs = frames_dict[m]
        padded[m] = fs + [fs[-1]] * (max_len - len(fs))

    h, w = padded[modes[0]][0].shape[:2]
    label_h = 28

    writer = imageio.get_writer(str(save_path), fps=fps)
    for t in range(max_len):
        panels = []
        for m in modes:
            frame = padded[m][t].copy()
            # Draw label bar
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            draw.rectangle([(0, 0), (w, label_h)], fill=(0, 0, 0))
            draw.text((6, 5), MODE_LABELS.get(m, m), fill=(255, 255, 255))
            panels.append(np.array(img))

        # 2×2 grid (pad to 4 if needed)
        while len(panels) < 4:
            panels.append(np.zeros_like(panels[0]))
        top = np.concatenate(panels[:2], axis=1)
        bot = np.concatenate(panels[2:4], axis=1)
        grid = np.concatenate([top, bot], axis=0)
        writer.append_data(grid)

    writer.close()
    print(f"  [video] saved {save_path}  ({max_len} frames)", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Closed-Loop 4-Way Evaluation Study")
    parser.add_argument("--model-name", required=True,
                        help="OpenVLA-OFT checkpoint (HF repo or local)")
    parser.add_argument("--mlp-checkpoint", required=True,
                        help="Trained EEFCorrectionMLP .pt checkpoint")
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Total episodes PER MODE")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Steering gain α for Mode C")
    parser.add_argument("--ema-beta", type=float, default=0.7,
                        help="EMA smoothing β for steering corrections")
    parser.add_argument("--fail-threshold", type=float, default=0.85,
                        help="Only apply correction when P(fail) > threshold "
                             "(used by p_fail and combined gate modes)")
    parser.add_argument("--action-scale", type=float, default=0.05,
                        help="Meters per action-unit for OSC controller "
                             "(LIBERO default ≈ 0.05)")
    parser.add_argument("--gate-mode", default="magnitude",
                        choices=SteeredAgent.GATE_MODES,
                        help="Gating strategy: 'magnitude' (recommended), "
                             "'p_fail', or 'combined'")
    parser.add_argument("--correction-threshold", type=float, default=0.01,
                        help="Min ‖correction‖ in meters to trigger "
                             "intervention (used by magnitude/combined gates)")
    parser.add_argument("--max-correction", type=float, default=0.02,
                        help="Clamp ‖correction‖ to this maximum (meters). "
                             "Primary safety mechanism — prevents large "
                             "MLP predictions from destabilizing the robot.")
    parser.add_argument("--noise-sigma", type=float, default=0.05,
                        help="Gaussian noise σ for Mode B")
    parser.add_argument("--expert-data", default="data/multi_suite",
                        help="Path to expert rollouts (DTW + Oracle)")
    parser.add_argument("--save-dir", default="results/closed_loop")
    parser.add_argument("--save-video", action="store_true",
                        help="Save side-by-side comparison video")
    parser.add_argument("--video-task", type=int, default=0,
                        help="Which task to record for the demo video")
    parser.add_argument("--video-episode", type=int, default=0,
                        help="Which episode to record for the demo video")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--camera-res", type=int, default=256)
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+",
                        default=["vanilla", "noise", "steering"],
                        help="Which modes to evaluate. Note: 'oracle' uses "
                             "open-loop replay which fails in closed-loop; "
                             "include it only as a reference.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    modes = list(args.modes)

    # ── Banner ──
    print("=" * 70, flush=True)
    print("CLOSED-LOOP EVALUATION: 4-WAY BASELINE COMPARISON", flush=True)
    print("=" * 70, flush=True)
    print(f"  Suite:           {args.env}", flush=True)
    print(f"  Modes:           {modes}", flush=True)
    print(f"  Episodes/mode:   {args.n_episodes}", flush=True)
    print(f"  Steering α={args.alpha}   EMA β={args.ema_beta}   "
          f"gate={args.gate_mode}", flush=True)
    if args.gate_mode in ("p_fail", "combined"):
        print(f"  Fail threshold: {args.fail_threshold}", flush=True)
    if args.gate_mode in ("magnitude", "combined"):
        print(f"  Correction threshold: {args.correction_threshold} m",
              flush=True)
    print(f"  Max correction: {args.max_correction} m  "
          f"(safety clamp)", flush=True)
    print(f"  Action scale: {args.action_scale} m/unit", flush=True)
    print(f"  Noise σ={args.noise_sigma}", flush=True)
    print(f"  Device:          {device}", flush=True)
    print(f"  Seed:            {args.seed}", flush=True)
    print(flush=True)

    # ─────────────────────────────────────────────────────────────
    # 1. LOAD VLA
    # ─────────────────────────────────────────────────────────────
    print("[1/4] Loading VLA model...", flush=True)
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

    needs_vla = any(m in modes for m in ("vanilla", "noise", "steering"))
    vla, processor, action_head, proprio_projector = None, None, None, None
    collector, resize_size = None, None

    if needs_vla:
        vla = get_vla(cfg)
        _resolve_unnorm_key(cfg, vla)
        processor = get_processor(cfg)
        action_head = get_action_head(cfg, vla.llm_dim)
        proprio_projector = get_proprio_projector(
            cfg, vla.llm_dim, proprio_dim=8)
        collector = HiddenStateCollector(vla)
        collector.register_hooks()
        resize_size = get_image_resize_size(cfg)
        print("  ✓ VLA loaded", flush=True)
    else:
        resize_size = get_image_resize_size(cfg)
        print("  ✓ VLA not needed for selected modes", flush=True)

    # ─────────────────────────────────────────────────────────────
    # 2. LOAD SAFETY MLP
    # ─────────────────────────────────────────────────────────────
    steered_agent = None
    if "steering" in modes:
        print("[2/4] Loading Safety MLP...", flush=True)
        ckpt = torch.load(args.mlp_checkpoint, map_location=device,
                          weights_only=False)
        mlp = EEFCorrectionMLP(
            input_dim=ckpt["input_dim"],
        ).to(device)
        mlp.load_state_dict(ckpt["model_state_dict"])
        mlp.eval()

        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)

        steered_agent = SteeredAgent(
            mlp, scaler,
            alpha=args.alpha,
            ema_beta=args.ema_beta,
            fail_threshold=args.fail_threshold,
            action_scale=args.action_scale,
            gate_mode=args.gate_mode,
            correction_threshold=args.correction_threshold,
            max_correction=args.max_correction,
            device=device,
        )
        n_params = sum(p.numel() for p in mlp.parameters())
        print(f"  ✓ MLP loaded  ({n_params:,} params, α={args.alpha}, "
              f"β={args.ema_beta}, gate={args.gate_mode}, "
              f"max_corr={args.max_correction}m, "
              f"scale={args.action_scale})", flush=True)
    else:
        print("[2/4] MLP not needed — skipping", flush=True)

    # ─────────────────────────────────────────────────────────────
    # 3. LOAD EXPERT DATA  (for DTW + Oracle replay)
    # ─────────────────────────────────────────────────────────────
    print("[3/4] Loading expert data...", flush=True)
    expert_eef = load_expert_eef_trajectories(args.expert_data)
    n_expert = sum(len(v) for v in expert_eef.values())
    print(f"  ✓ {n_expert} expert EEF trajectories for DTW", flush=True)

    oracle_actions_map = {}
    if "oracle" in modes:
        oracle_actions_map = load_oracle_actions(args.expert_data)
        n_oracle = sum(len(v) for v in oracle_actions_map.values())
        print(f"  ✓ {n_oracle} oracle action sequences for replay",
              flush=True)
        if n_oracle == 0:
            print("  ⚠  No oracle demos found — disabling Oracle mode",
                  flush=True)
            modes = [m for m in modes if m != "oracle"]

    # ─────────────────────────────────────────────────────────────
    # 4. SETUP LIBERO ENVIRONMENT
    # ─────────────────────────────────────────────────────────────
    print("[4/4] Setting up LIBERO environment...", flush=True)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.env]()
    num_tasks = task_suite.n_tasks
    episodes_per_task = max(1, args.n_episodes // num_tasks)
    actual_total = episodes_per_task * num_tasks
    print(f"  ✓ {args.env}: {num_tasks} tasks × "
          f"{episodes_per_task} ep/task = {actual_total} episodes/mode",
          flush=True)
    print(flush=True)

    # ═════════════════════════════════════════════════════════════
    #  RUN EPISODES
    # ═════════════════════════════════════════════════════════════
    all_results = {m: [] for m in modes}
    video_frames = ({m: [] for m in modes}
                    if args.save_video else None)

    t0_total = time.time()

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(
            task, cfg.model_family, resolution=cfg.env_img_res)
        init_states = task_suite.get_task_init_states(task_id)

        task_oracle_seqs = oracle_actions_map.get(task_id, [])

        print(f"─── Task {task_id}/{num_tasks - 1}: "
              f"{task_description[:60]}... ───", flush=True)

        for ep_idx in range(episodes_per_task):
            init_state = (init_states[ep_idx % len(init_states)]
                          if init_states is not None else None)

            # Should we record video frames for this episode?
            record_this = (args.save_video
                           and task_id == args.video_task
                           and ep_idx == args.video_episode)

            for mode in modes:
                t0_ep = time.time()

                # Select oracle actions
                ep_oracle = None
                if mode == "oracle" and task_oracle_seqs:
                    ep_oracle = task_oracle_seqs[
                        ep_idx % len(task_oracle_seqs)]

                result = run_episode(
                    cfg, env, task_description, resize_size,
                    vla, processor, action_head,
                    proprio_projector, collector,
                    initial_state=init_state,
                    mode=mode,
                    steered_agent=steered_agent,
                    noise_sigma=args.noise_sigma,
                    oracle_actions=ep_oracle,
                    record_frames=record_this,
                )

                # ── Trajectory deviation (DTW) ──
                expert_for_task = expert_eef.get(task_id, [])
                if (len(result["eef_trajectory"]) > 0
                        and expert_for_task):
                    result["trajectory_deviation"] = float(
                        min_expert_dtw(result["eef_trajectory"],
                                       expert_for_task))
                else:
                    result["trajectory_deviation"] = float("nan")

                result["task_id"] = task_id
                result["episode_idx"] = ep_idx
                result["time_s"] = round(time.time() - t0_ep, 1)

                # Store video frames
                if (record_this and result.get("frames")
                        and video_frames is not None):
                    video_frames[mode] = result["frames"]
                result.pop("frames", None)

                # Serialise trajectory shape (drop raw array for JSON)
                if isinstance(result["eef_trajectory"], np.ndarray):
                    result["eef_trajectory_len"] = len(
                        result["eef_trajectory"])
                result.pop("eef_trajectory", None)

                all_results[mode].append(result)

                tag = "✓" if result["success"] else "✗"
                dev = result["trajectory_deviation"]
                dev_s = f"{dev:.4f}" if not np.isnan(dev) else "N/A"
                ir = result.get("intervention_rate", 0.0)
                ir_s = (f"  IR={ir:.0%}" if mode == "steering" else "")
                print(f"  {mode:>10}  ep={ep_idx}  {tag}  "
                      f"steps={result['total_steps']:>3}  "
                      f"violations={result['constraint_violations']:>2}  "
                      f"dev={dev_s}{ir_s}  "
                      f"({result['time_s']}s)",
                      flush=True)

        # ── "Do No Harm" monitor: per-task intervention warning ──
        if "steering" in modes:
            task_steering = [r for r in all_results["steering"]
                            if r.get("task_id") == task_id]
            if task_steering:
                avg_ir = np.mean([r["intervention_rate"]
                                 for r in task_steering])
                if avg_ir > 0.50:
                    print(f"  ⚠  OVER-CORRECTION on Task {task_id}: "
                          f"intervention_rate={avg_ir:.0%} (> 50%)",
                          flush=True)

        env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - t0_total

    # ═════════════════════════════════════════════════════════════
    #  AGGREGATE & PRINT RESULTS
    # ═════════════════════════════════════════════════════════════
    print(flush=True)
    print("=" * 70, flush=True)
    print("RESULTS TABLE", flush=True)
    print("=" * 70, flush=True)

    results_table = {}
    for mode in modes:
        episodes = all_results[mode]
        if not episodes:
            continue

        successes = [e["success"] for e in episodes]
        violations = [e["constraint_violations"] for e in episodes]
        deviations = [e["trajectory_deviation"] for e in episodes
                      if not np.isnan(e["trajectory_deviation"])]
        steps = [e["total_steps"] for e in episodes]

        sr = np.mean(successes) * 100
        ir_vals = [e.get("intervention_rate", 0.0) for e in episodes]
        results_table[mode] = {
            "success_rate_pct": round(sr, 1),
            "n_episodes": len(episodes),
            "n_successes": int(sum(successes)),
            "mean_constraint_violations": round(float(np.mean(violations)), 2),
            "mean_trajectory_deviation": (
                round(float(np.mean(deviations)), 4)
                if deviations else None),
            "mean_steps": round(float(np.mean(steps)), 1),
            "std_success_pct": round(float(np.std(successes)) * 100, 1),
            "mean_intervention_rate": round(float(np.mean(ir_vals)), 4),
        }

    # ── Pretty table ──
    print(f"\n  {'Mode':<30} {'Success':>8} "
          f"{'Violations':>11} {'DTW Dev':>9} {'Steps':>6} {'IR':>6}",
          flush=True)
    print(f"  {'─' * 76}", flush=True)

    for mode in modes:
        r = results_table.get(mode)
        if r is None:
            continue
        label = MODE_LABELS.get(mode, mode)
        dev = (f"{r['mean_trajectory_deviation']:.4f}"
               if r["mean_trajectory_deviation"] is not None else "N/A")
        ir = r.get("mean_intervention_rate", 0.0)
        ir_s = f"{ir:.0%}" if mode == "steering" else "—"
        star = "  ★" if mode == "steering" else ""
        print(f"  {label:<30} {r['success_rate_pct']:>7.1f}% "
              f"{r['mean_constraint_violations']:>10.2f} "
              f"{dev:>9} {r['mean_steps']:>6.1f} {ir_s:>6}{star}",
              flush=True)

    # ── Statistical verdict ──
    s = results_table.get("steering", {})
    n = results_table.get("noise", {})
    v = results_table.get("vanilla", {})

    s_rate = s.get("success_rate_pct", 0)
    n_rate = n.get("success_rate_pct", 0)
    v_rate = v.get("success_rate_pct", 0)

    if s and n:
        print(flush=True)
        print("  ╔══════════════════════════════════════════════════════════╗",
              flush=True)
        if s_rate > n_rate:
            print(f"  ║  Steering ({s_rate:.1f}%) > Random Noise ({n_rate:.1f}%)"
                  f"{'':>15}║", flush=True)
            print(f"  ║  → MECHANISM IS VALID  (not random exploration)   "
                  f"     ║", flush=True)
        else:
            print(f"  ║  Steering ({s_rate:.1f}%) ≤ Random Noise ({n_rate:.1f}%)"
                  f"{'':>15}║", flush=True)
            print(f"  ║  → ⚠  Investigate: α tuning or scale mismatch    "
                  f"     ║", flush=True)
        if v and s_rate > v_rate:
            print(f"  ║  Steering ({s_rate:.1f}%) > Vanilla ({v_rate:.1f}%)"
                  f"{'':>20}║", flush=True)
            print(f"  ║  → HYBRID CONTROL HELPS                           "
                  f"     ║", flush=True)
        s_viol = s.get("mean_constraint_violations", 0)
        v_viol = v.get("mean_constraint_violations", 0) if v else 0
        if v and s_viol < v_viol:
            print(f"  ║  Violations: Steering ({s_viol:.1f}) < "
                  f"Vanilla ({v_viol:.1f})"
                  f"{'':>14}║", flush=True)
            print(f"  ║  → SAFER TRAJECTORIES                              "
                  f"    ║", flush=True)
        print("  ╚══════════════════════════════════════════════════════════╝",
              flush=True)

    print(f"\n  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)",
          flush=True)

    # ═════════════════════════════════════════════════════════════
    #  SAVE OUTPUTS
    # ═════════════════════════════════════════════════════════════
    report = {
        "config": {
            "env": args.env,
            "n_episodes_per_mode": args.n_episodes,
            "actual_episodes_per_mode": actual_total,
            "alpha": args.alpha,
            "ema_beta": args.ema_beta,
            "gate_mode": args.gate_mode,
            "fail_threshold": args.fail_threshold,
            "correction_threshold": args.correction_threshold,
            "max_correction": args.max_correction,
            "action_scale": args.action_scale,
            "noise_sigma": args.noise_sigma,
            "seed": args.seed,
            "model": args.model_name,
            "mlp_checkpoint": args.mlp_checkpoint,
        },
        "results": results_table,
    }
    with open(save_dir / "results_table.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {save_dir}/results_table.json", flush=True)

    # Episode-level details (without heavy arrays)
    with open(save_dir / "episode_details.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {save_dir}/episode_details.json", flush=True)

    # ── Video ──
    if args.save_video and video_frames:
        print("\n  Generating comparison video...", flush=True)
        create_comparison_video(
            video_frames,
            save_dir / "combined_video.mp4",
            fps=args.video_fps,
        )

    print(f"\n{'=' * 70}", flush=True)
    print("DONE", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
