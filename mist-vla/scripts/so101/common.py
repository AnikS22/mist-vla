"""Shared utilities for the SO-101 physical robot pipeline.

Conventions:
    Rollouts are written in the SAME pkl schema as the sim data so the existing
    `train_eef_correction_mlp.py` and `paper/scripts/run_stat_tests.py` work
    unchanged. The schema (per-rollout dict):

        {
            'actions':    [np.ndarray(action_dim,)]_T,
            'features':   [np.ndarray(hidden_dim,)]_T,
            'rewards':    [float]_T,
            'robot_states': [{'eef_pos': np.ndarray(3,), 'qpos': np.ndarray, 'qvel': np.ndarray}]_T,
            'steps':      [{'action', 'hidden_state', 'robot_state', 'collision', 'done', ...}]_T,
            'success':    bool,
            'collision_occurred': bool,
            'collision_step': int | None,
            'instruction': str,
            'task_id':    str,
            'model_tag':  str,
            'suite_tag':  str,
        }
"""
from __future__ import annotations

import contextlib
import dataclasses
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
ROLLOUTS_DIR = REPO / "research_data" / "rollouts" / "so101"
CHECKPOINTS_DIR = REPO / "research_data" / "checkpoints" / "so101"


# ─── Config ────────────────────────────────────────────────────────────────


@dataclass
class SO101RunConfig:
    """Top-level config for a data-collection or eval run."""

    # ── Hardware ──
    # Empirically on this rig: ACM0 = leader, ACM1 = follower. LeRobot's port
    # ordering depends on USB enumeration, so verify after any re-plug with
    # `lerobot-find-port` if connection fails.
    follower_port: str = "/dev/ttyACM1"
    leader_port: str = "/dev/ttyACM0"
    # On this rig: icspring (wrist) = /dev/video0 (cv idx 0); USB 2.0 Camera
    # (scene, external) = /dev/video2 (cv idx 2). video1/3 are metadata nodes
    # of the same physical cameras. Swap with --wrist-cam / --scene-cam if reversed.
    wrist_cam_index: int = 0
    scene_cam_index: int = 2
    device: str = "cuda:0"  # 2080 Ti #0 by default
    # SmolVLA was trained against SO-100 datasets where joint positions are
    # stored in degrees; the policy's normalizer mean/std are degree-valued
    # (e.g., shoulder_lift mean ~120°). Running the follower with
    # use_degrees=True means both state input AND action output stay in
    # degrees end-to-end and the trained stats apply correctly.
    use_degrees: bool = True
    calibration_dir: str | None = None
    # Saved calibration files live under
    # ~/.cache/huggingface/lerobot/calibration/{robots,teleoperators}/<kind>/<id>.json
    follower_id: str = "my_follower"
    leader_id: str = "my_leader"

    # ── Task ──
    task_id: str = "so101_pick_bowl_place_plate"
    instruction: str = "pick up the black bowl and place it on the plate"

    # ── Policy ──
    policy_name: str = "pi0"  # "pi0" | "pi05" | "act" | "groot"
    policy_repo: str = "lerobot/pi0"
    dtype: str = "bfloat16"  # bf16 fits Pi0-3B on a 2080 Ti

    # ── Loop ──
    fps: int = 30
    max_steps: int = 300
    n_episodes: int = 20

    # ── Safety interlocks ──
    workspace_bounds_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = field(
        # (x_min, x_max), (y_min, y_max), (z_min, z_max) in meters; tune for your bench.
        default_factory=lambda: ((0.05, 0.45), (-0.30, 0.30), (0.00, 0.40))
    )
    max_velocity_pct: float = 0.30  # cap commanded delta per step at 30% of servo range
    # Tightened from 15 -> 5 after the gripper servo over-extended during a SmolVLA
    # autonomous run. 5 units/step at 30 fps = ~150 units/s, slow enough to e-stop
    # before the arm hits a hard stop.
    max_relative_target_deg: float | None = 5.0
    # Per-joint absolute position clamp in degrees (use_degrees=True). SO-101
    # joints have calibrated ranges of roughly ±100°; 90° leaves a ~10° margin
    # from each mechanical end stop.
    joint_abs_limit: float = 90.0
    # The gripper repeatedly stalled against its close-stop during autonomous
    # SmolVLA runs (no object to grip → over-current overload). Tight clamp
    # of ±20° keeps the commanded gripper position safely between extremes.
    # Set both bounds to 0 to effectively freeze the gripper.
    gripper_abs_limit: float = 20.0
    # Even with tight position clamps the gripper servo overloads (the user's
    # SO-101 calibration appears to have been performed with the gripper jaws
    # self-contacting, so "0°" already produces static load). When True, we
    # write torque_enable=0 to motor 6 on connect and skip gripper commands —
    # the arm reaches and articulates but the gripper hangs limp.
    disable_gripper_torque: bool = True

    # ── PULSE (used by eval/replay scripts) ──
    probe_ckpt: str | None = None
    failure_threshold: float = 0.60
    intervention_alpha: float = 0.20
    correction_clamp_m: float = 0.004
    ema_beta: float = 0.30

    # ── Output ──
    output_dir: Path = field(default_factory=lambda: ROLLOUTS_DIR)
    run_tag: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))


# ─── Hidden-state hook ─────────────────────────────────────────────────────


class HiddenStateHook:
    """Capture the most recent forward-pass output of a target submodule.

    Pi0's per-step latent of interest is the PaliGemma language_model output
    on the (image + language + state) prefix — computed once per env step.
    For other policies we look up by attribute path or `--module-path`.

    Usage:
        hook = HiddenStateHook(policy.model, "paligemma_with_expert.paligemma.language_model")
        hook.attach()
        action = policy.select_action(batch)
        h = hook.latest()   # np.ndarray(hidden_dim,)
        hook.reset()
        ...
        hook.detach()
    """

    DEFAULT_PATHS = {
        "pi0":     "paligemma_with_expert.paligemma.language_model",
        "pi05":    "paligemma_with_expert.paligemma.language_model",
        "smolvla":    "vlm_with_expert.vlm.model.text_model.norm",
        "smolvla_pp": "vlm_with_expert.vlm.model.text_model.norm",
        "groot":   "model.transformer",
        "act":     "model.encoder",
    }

    def __init__(self, root_module: torch.nn.Module, module_path: str, pool: str = "last_token"):
        self.root = root_module
        self.path = module_path
        self.pool = pool   # "last_token" | "mean" | "first_token"
        self._target: torch.nn.Module | None = None
        self._handle = None
        self._latest: np.ndarray | None = None

    def _resolve(self) -> torch.nn.Module:
        m: torch.nn.Module = self.root
        for part in self.path.split("."):
            if not hasattr(m, part):
                raise AttributeError(
                    f"HiddenStateHook: cannot resolve '{self.path}' on "
                    f"{type(self.root).__name__}; missing attribute '{part}'. "
                    f"Inspect with `dir(policy.model)` and pass --module-path."
                )
            m = getattr(m, part)
        return m

    def _on_forward(self, _module, _inputs, output):
        # output may be a tuple (hidden_states, ...) or a ModelOutput.
        h = output[0] if isinstance(output, tuple) else getattr(output, "last_hidden_state", output)
        if not torch.is_tensor(h):
            return
        # h: (batch, seq, dim) → pool to (dim,)
        with torch.no_grad():
            if h.ndim == 3:
                if self.pool == "last_token":
                    v = h[0, -1, :]
                elif self.pool == "first_token":
                    v = h[0, 0, :]
                else:
                    v = h[0].mean(dim=0)
            elif h.ndim == 2:
                v = h[0]
            else:
                v = h.flatten()[: h.shape[-1] if h.ndim > 0 else 1]
            self._latest = v.detach().to("cpu", torch.float32).numpy()

    def attach(self):
        self._target = self._resolve()
        self._handle = self._target.register_forward_hook(self._on_forward)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def reset(self):
        self._latest = None

    def latest(self) -> np.ndarray | None:
        return self._latest


# ─── Safety interlocks ─────────────────────────────────────────────────────


class WorkspaceClamp:
    """Clamp end-effector targets to a safe workspace box."""

    def __init__(self, bounds_m):
        (self.x_lo, self.x_hi), (self.y_lo, self.y_hi), (self.z_lo, self.z_hi) = bounds_m
        self.last_violation = None

    def clamp(self, eef_xyz: np.ndarray) -> tuple[np.ndarray, bool]:
        clamped = np.array([
            np.clip(eef_xyz[0], self.x_lo, self.x_hi),
            np.clip(eef_xyz[1], self.y_lo, self.y_hi),
            np.clip(eef_xyz[2], self.z_lo, self.z_hi),
        ], dtype=np.float32)
        violated = bool(np.any(np.abs(clamped - eef_xyz) > 1e-6))
        if violated:
            self.last_violation = eef_xyz.copy()
        return clamped, violated


@contextlib.contextmanager
def estop_handler(robot, label: str = "robot"):
    """Install a SIGINT handler that disconnects the robot cleanly."""
    original = signal.getsignal(signal.SIGINT)

    def handler(_sig, _frm):
        print(f"\n[E-STOP] Ctrl+C received; disconnecting {label}.")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"[E-STOP] disconnect raised: {e}")
        sys.exit(130)

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original)


# ─── Rollout writer (matches sim pkl schema) ───────────────────────────────


@dataclass
class RolloutRecord:
    """Streaming buffer for one episode. Call .step() each control tick."""

    instruction: str
    task_id: str
    model_tag: str
    suite_tag: str = "so101_real"
    actions: list = field(default_factory=list)
    features: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    robot_states: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    success: bool = False
    collision_occurred: bool = False
    collision_step: int | None = None

    def step(
        self,
        action: np.ndarray,
        hidden_state: np.ndarray | None,
        eef_pos: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray | None = None,
        reward: float = 0.0,
        collision: bool = False,
        done: bool = False,
        extra: dict | None = None,
    ):
        if hidden_state is None:
            # Skip silently for the first call before the policy fires (caller can probe).
            return
        robot_state = {
            "eef_pos": np.asarray(eef_pos, dtype=np.float32),
            "qpos":    np.asarray(qpos,    dtype=np.float32),
            "qvel":    np.asarray(qvel if qvel is not None else np.zeros_like(qpos), dtype=np.float32),
        }
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.features.append(np.asarray(hidden_state, dtype=np.float32))
        self.rewards.append(float(reward))
        self.robot_states.append(robot_state)
        step = {
            "action":       np.asarray(action, dtype=np.float32),
            "hidden_state": np.asarray(hidden_state, dtype=np.float32),
            "robot_state":  robot_state,
            "collision":    bool(collision),
            "collision_geoms":  [],
            "collision_pos":    None,
            "collision_normal": None,
            "done":         bool(done),
        }
        if extra:
            step.update(extra)
        self.steps.append(step)
        if collision and self.collision_step is None:
            self.collision_step = len(self.steps) - 1
            self.collision_occurred = True

    def finalize(self, success: bool) -> dict:
        self.success = bool(success)
        return {
            "actions":      self.actions,
            "features":     self.features,
            "rewards":      self.rewards,
            "robot_states": self.robot_states,
            "steps":        self.steps,
            "success":      self.success,
            "collision_occurred": self.collision_occurred,
            "collision_step":     self.collision_step,
            "collision_steps":    self.collision_step if self.collision_step is not None else 0,
            "instruction":  self.instruction,
            "task_id":      self.task_id,
            "model_tag":    self.model_tag,
            "suite_tag":    self.suite_tag,
            "original_task_id": self.task_id,
            "source_dir":   "so101_real",
        }
