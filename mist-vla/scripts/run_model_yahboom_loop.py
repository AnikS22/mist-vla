#!/usr/bin/env python3
"""Run a safe closed-loop model->Yahboom control cycle via Jetson API.

Pipeline:
  Jetson /snapshot + /status -> policy action (OpenVLA/OpenVLA-OFT/random) ->
  bounded Cartesian delta -> Jetson /action move_to (+ optional gripper).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from typing import Dict, Optional, Tuple
from urllib import request

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def api_get(base: str, path: str, timeout: float = 8.0) -> bytes:
    with request.urlopen(f"{base}{path}", timeout=timeout) as resp:
        return resp.read()


def api_post(base: str, payload: Dict, timeout: float = 30.0) -> Dict:
    data = json.dumps(payload).encode()
    req = request.Request(
        f"{base}/action",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_coords(coords):
    x, y, z, rx, ry, rz = coords
    return [
        clamp(float(x), -220.0, 220.0),
        clamp(float(y), -220.0, 220.0),
        clamp(float(z), 40.0, 450.0),
        clamp(float(rx), -180.0, 180.0),
        clamp(float(ry), -180.0, 180.0),
        clamp(float(rz), -180.0, 180.0),
    ]


def clamp_coords_with_bounds(coords, x_min, x_max, y_min, y_max, z_min, z_max):
    x, y, z, rx, ry, rz = coords
    return [
        clamp(float(x), float(x_min), float(x_max)),
        clamp(float(y), float(y_min), float(y_max)),
        clamp(float(z), float(z_min), float(z_max)),
        clamp(float(rx), -180.0, 180.0),
        clamp(float(ry), -180.0, 180.0),
        clamp(float(rz), -180.0, 180.0),
    ]


def get_obs(base: str) -> Tuple[Image.Image, Dict]:
    img_bytes = api_get(base, "/snapshot")
    st = json.loads(api_get(base, "/status").decode())
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img, st


def move_to_pose_staged(
    base: str,
    target: list[float],
    speed: int,
    wait_s: float,
    max_step_mm: float = 15.0,
    max_step_deg: float = 5.0,
    tol_mm: float = 2.0,
    tol_deg: float = 2.0,
    max_iters: int = 30,
) -> bool:
    for _ in range(max_iters):
        st = json.loads(api_get(base, "/status").decode())
        cur = st.get("coords")
        if not isinstance(cur, list) or len(cur) != 6:
            return False
        dxyz = np.asarray(target[:3], dtype=np.float32) - np.asarray(cur[:3], dtype=np.float32)
        drpy = np.asarray(target[3:], dtype=np.float32) - np.asarray(cur[3:], dtype=np.float32)
        if float(np.linalg.norm(dxyz)) <= tol_mm and float(np.max(np.abs(drpy))) <= tol_deg:
            return True
        step = [
            float(cur[0]) + clamp(float(dxyz[0]), -max_step_mm, max_step_mm),
            float(cur[1]) + clamp(float(dxyz[1]), -max_step_mm, max_step_mm),
            float(cur[2]) + clamp(float(dxyz[2]), -max_step_mm, max_step_mm),
            float(cur[3]) + clamp(float(drpy[0]), -max_step_deg, max_step_deg),
            float(cur[4]) + clamp(float(drpy[1]), -max_step_deg, max_step_deg),
            float(cur[5]) + clamp(float(drpy[2]), -max_step_deg, max_step_deg),
        ]
        step = clamp_coords(step)
        r = api_post(base, {"action": "move_to", "coords": step, "speed": int(speed), "wait": float(wait_s)})
        if not r.get("ok"):
            return False
    return False


def standardize_libero_image(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to square then resize, matching common LIBERO policy inputs."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = arr[y0 : y0 + side, x0 : x0 + side]
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(out)


class EEFCorrectionMLP(nn.Module):
    """Runtime model for cartesian EEF correction checkpoints."""

    HIDDEN_DIM = 256

    def __init__(self, input_dim: int = 4096):
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

    def forward(self, x: torch.Tensor):
        z = self.encoder(self.input_norm(x))
        return {
            "will_fail": self.fail_head(z).squeeze(-1),
            "ttf": self.ttf_head(z).squeeze(-1),
            "correction": self.correction_head(z),
        }


class SteeringRuntime:
    def __init__(
        self,
        checkpoint_path: str,
        model_device: str,
        alpha: float,
        fail_threshold: float,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.input_dim = int(ckpt.get("input_dim", 4096))
        self.model = EEFCorrectionMLP(input_dim=self.input_dim)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()
        self.device = torch.device(model_device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.alpha = float(alpha)
        self.fail_threshold = float(fail_threshold)
        self.scaler_mean = ckpt.get("scaler_mean")
        self.scaler_scale = ckpt.get("scaler_scale")

    def _prepare_feat(self, feat: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if feat is None:
            return None
        x = np.asarray(feat, dtype=np.float32).reshape(-1)
        if x.size != self.input_dim:
            return None
        if self.scaler_mean is not None and self.scaler_scale is not None:
            mean = np.asarray(self.scaler_mean, dtype=np.float32).reshape(-1)
            scale = np.asarray(self.scaler_scale, dtype=np.float32).reshape(-1)
            if mean.size == x.size and scale.size == x.size:
                x = (x - mean) / np.clip(scale, 1e-8, None)
        return x

    def _forward_scaled_feat(self, scaled_feat: np.ndarray):
        x = torch.from_numpy(scaled_feat.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        correction = out["correction"].detach().cpu().numpy().reshape(-1)[:3]
        raw_logit = float(out["will_fail"].detach().cpu().item())
        fail_prob = float(torch.sigmoid(out["will_fail"]).detach().cpu().item())
        return fail_prob, correction, raw_logit

    def maybe_steer(self, action: np.ndarray, feat: Optional[np.ndarray]):
        x = self._prepare_feat(feat)
        if x is None:
            return action, {"applied": False, "reason": "feature_unavailable_or_dim_mismatch"}
        fail_prob, correction, raw_logit = self._forward_scaled_feat(x)
        steered = action.copy()
        # Gate on raw logit — sigmoid saturates at 1.0 for logits > 10.
        # Calibrated: success logits ~16.09, failure ~17.03, threshold = 16.56
        applied = raw_logit > self.fail_threshold
        if applied:
            steered[:3] = steered[:3] + self.alpha * correction
        return steered, {
            "applied": applied,
            "fail_prob": fail_prob,
            "raw_logit": raw_logit,
            "threshold": self.fail_threshold,
            "correction_m": correction.tolist(),
            "alpha": self.alpha,
        }

    def maybe_mppi(
        self,
        action: np.ndarray,
        feat: Optional[np.ndarray],
        n_samples: int,
        temperature: float,
        correction_std: float,
        max_correction: float,
        fail_gate: bool = True,
    ):
        x = self._prepare_feat(feat)
        if x is None:
            return action, {"applied": False, "reason": "feature_unavailable_or_dim_mismatch"}

        fail_prob, _, raw_logit = self._forward_scaled_feat(x)
        if fail_gate and raw_logit < self.fail_threshold:
            return action, {
                "applied": False,
                "reason": "below_fail_threshold",
                "fail_prob": fail_prob,
                "threshold": self.fail_threshold,
            }

        # MPPI baseline style: sample candidate corrections and score via fail head.
        candidates = np.random.normal(0.0, correction_std, size=(n_samples, 3)).astype(np.float32)
        scores = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            feat_pert = x.copy()
            feat_pert += np.random.normal(0.0, 0.01, size=feat_pert.shape).astype(np.float32)
            fp_i, _, _ = self._forward_scaled_feat(feat_pert)
            # Lower fail probability is better; small penalty for excessive correction magnitude.
            scores[i] = -fp_i - 0.1 * float(np.linalg.norm(candidates[i]))

        w = np.exp(temperature * (scores - np.max(scores)))
        w = w / np.clip(np.sum(w), 1e-8, None)
        correction = np.sum(candidates * w[:, None], axis=0)
        mag = float(np.linalg.norm(correction))
        if mag > max_correction and mag > 1e-8:
            correction = correction * (max_correction / mag)
            mag = max_correction

        out_action = action.copy()
        out_action[:3] = out_action[:3] + correction
        return out_action, {
            "applied": True,
            "method": "mppi",
            "fail_prob": fail_prob,
            "threshold": self.fail_threshold,
            "correction_m": correction.tolist(),
            "correction_mag_m": mag,
            "n_samples": n_samples,
            "temperature": temperature,
        }


def load_policy(
    policy_kind: str,
    model_name: str,
    device: str,
    device_map: str | None,
    enable_hidden_state_hooks: bool = False,
    force_token_fallback: bool = False,
):
    if policy_kind == "random":
        return None
    if policy_kind == "openvla":
        from src.models.vla_wrapper import create_vla_wrapper

        return create_vla_wrapper(
            model_type="openvla",
            model_name=model_name,
            device=device,
            device_map=device_map,
            force_token_fallback=force_token_fallback,
            enable_hidden_state_hooks=enable_hidden_state_hooks,
        )
    if policy_kind == "openvla_oft":
        from src.models.vla_wrapper import create_vla_wrapper

        return create_vla_wrapper(
            model_type="openvla_oft",
            model_name=model_name,
            device=device,
            enable_hidden_state_hooks=enable_hidden_state_hooks,
        )
    if policy_kind == "smolvla":
        from src.models.vla_wrapper import create_vla_wrapper

        return create_vla_wrapper(model_type="smolvla", model_name=model_name, device=device)
    raise ValueError(f"unsupported policy_kind: {policy_kind}")


def predict_action(policy, policy_kind: str, image: Image.Image, instruction: str, rng: np.random.Generator):
    if policy_kind == "random":
        a = rng.uniform(-0.3, 0.3, size=(7,)).astype(np.float32)
        a[-1] = rng.uniform(-1.0, 1.0)
        return a, None
    action, feat = policy.get_action_with_features(image, instruction, obs=None)
    if hasattr(action, "detach"):
        a = action.detach().cpu().numpy().astype(np.float32)
    else:
        a = np.asarray(action, dtype=np.float32)
    if a.ndim > 1:
        a = a.reshape(-1)
    if a.shape[0] < 7:
        a = np.pad(a, (0, 7 - a.shape[0]))
    feat_np = None
    if feat is not None:
        if hasattr(feat, "detach"):
            feat_np = feat.detach().float().cpu().numpy()
        else:
            feat_np = np.asarray(feat)
    return a[:7], feat_np


def main() -> int:
    def _split_list(raw: str):
        return [x.strip() for x in str(raw).split(",") if x.strip()]

    def _parse_float_list(raw: str, n: int, default: float):
        if not raw:
            return [float(default)] * n
        vals = [float(x) for x in _split_list(raw)]
        if len(vals) == 1 and n > 1:
            vals = vals * n
        if len(vals) != n:
            raise ValueError(f"expected {n} float values, got {len(vals)}")
        return vals

    def _parse_int_list(raw: str, n: int, default: int):
        if not raw:
            return [int(default)] * n
        vals = [int(float(x)) for x in _split_list(raw)]
        if len(vals) == 1 and n > 1:
            vals = vals * n
        if len(vals) != n:
            raise ValueError(f"expected {n} int values, got {len(vals)}")
        return vals

    ap = argparse.ArgumentParser(description="Safe model->Yahboom loop")
    ap.add_argument("--jetson-host", default="192.168.55.1")
    ap.add_argument("--jetson-port", type=int, default=5000)
    ap.add_argument("--policy", choices=["openvla", "openvla_oft", "smolvla", "random"], default="random")
    ap.add_argument("--model-name", default="openvla/openvla-7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto", help="HF device_map for OpenVLA (e.g., auto, none)")
    ap.add_argument("--force-token-fallback", action="store_true", help="force token-decoding action path instead of model.predict_action")
    ap.add_argument("--instruction", default="pick up the block and place it carefully")
    ap.add_argument("--task-profile", choices=["none", "pick_yellow_cube"], default="none", help="optional staged task profile")
    ap.add_argument("--unsafe-no-constraints", action="store_true", help="disable all motion safety constraints/guards and use raw VLA commands")
    ap.add_argument("--instruction-sequence", default="", help="phase instructions separated by '||'")
    ap.add_argument("--phase-steps", default="", help="comma-separated steps per phase")
    ap.add_argument("--phase-xyz-gains", default="", help="comma-separated xyz gains per phase")
    ap.add_argument("--phase-rot-gains", default="", help="comma-separated rot gains per phase")
    ap.add_argument("--phase-max-xy-step-mm", default="", help="comma-separated per-phase max XY step clamp (mm)")
    ap.add_argument("--phase-max-z-step-mm", default="", help="comma-separated per-phase max Z step clamp (mm)")
    ap.add_argument("--phase-force-gripper", default="", help="comma-separated per-phase gripper overrides 0..100, use -1 for none")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--model-only-strict", action="store_true", help="disable heuristic motion/gripper overrides (safety clamps remain)")
    ap.add_argument("--libero-image-size", type=int, default=224, help="square size for LIBERO-like image standardization")
    ap.add_argument("--no-libero-standardize", action="store_true", help="disable center-crop+resize preprocessing")
    ap.add_argument("--dt", type=float, default=0.8)
    ap.add_argument("--wait-for-motion", action="store_true", help="block on move_to completion (slower, deterministic)")
    ap.add_argument("--prestart-coords", default="", help="optional staged pre-start pose x,y,z,rx,ry,rz")
    ap.add_argument("--prestart-max-step-mm", type=float, default=15.0)
    ap.add_argument("--prestart-tol-mm", type=float, default=2.0)
    ap.add_argument("--action-space", choices=["normalized", "meters"], default="meters", help="interpretation of policy xyz/rz outputs")
    ap.add_argument("--xyz-gain", type=float, default=1.0, help="gain on xyz deltas before mapping")
    ap.add_argument("--rot-gain", type=float, default=1.0, help="gain on rz delta before mapping")
    ap.add_argument("--xy-scale-mm", type=float, default=18.0)
    ap.add_argument("--z-scale-mm", type=float, default=12.0)
    ap.add_argument("--rz-scale-deg", type=float, default=8.0)
    ap.add_argument("--force-visible-motion", action="store_true", help="enforce minimum nonzero deltas for visibility")
    ap.add_argument("--min-xy-mm", type=float, default=6.0)
    ap.add_argument("--min-z-mm", type=float, default=4.0)
    ap.add_argument("--min-rz-deg", type=float, default=3.0)
    ap.add_argument("--speed", type=int, default=20)
    ap.add_argument("--workspace-x-min", type=float, default=-220.0)
    ap.add_argument("--workspace-x-max", type=float, default=220.0)
    ap.add_argument("--workspace-y-min", type=float, default=-220.0)
    ap.add_argument("--workspace-y-max", type=float, default=220.0)
    ap.add_argument("--workspace-z-min", type=float, default=40.0)
    ap.add_argument("--workspace-z-max", type=float, default=450.0)
    ap.add_argument("--min-motion-mm", type=float, default=1.5, help="warn if achieved xyz motion is below this threshold")
    ap.add_argument("--max-xy-step-mm", type=float, default=0.0, help="optional absolute clamp for per-step dx/dy in mm (0 disables)")
    ap.add_argument("--max-z-step-mm", type=float, default=0.0, help="optional absolute clamp for per-step dz in mm (0 disables)")
    ap.add_argument("--lock-orientation", action="store_true", help="hold rx/ry/rz fixed to the first observed pose")
    ap.add_argument("--gripper-threshold", type=float, default=0.2)
    ap.add_argument("--gripper-continuous", action="store_true", help="map model gripper action continuously to 0..100")
    ap.add_argument("--skip-redundant-gripper", action="store_true", help="skip set_gripper calls when target gripper value is unchanged")
    ap.add_argument("--execute", action="store_true", help="actually send move_to/set_gripper")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat-delta-threshold", type=float, default=0.02, help="L2 action delta threshold for repeated-action detection")
    ap.add_argument("--max-repeat-streak", type=int, default=4, help="terminate if too many near-identical actions are produced consecutively")
    ap.add_argument("--steering-checkpoint", default="", help="optional eef correction MLP checkpoint for live steering")
    ap.add_argument("--steering-alpha", type=float, default=1.0, help="gain for safety correction when fail_prob > threshold")
    ap.add_argument("--steering-fail-threshold", type=float, default=16.56, help="Raw logit threshold (not sigmoid). Calibrated: success=16.09, failure=17.03")
    ap.add_argument("--control-mode", choices=["vanilla", "steering", "mppi"], default="vanilla", help="action control mode when steering checkpoint is provided")
    ap.add_argument("--mppi-samples", type=int, default=16, help="MPPI candidate correction count")
    ap.add_argument("--mppi-temperature", type=float, default=5.0, help="MPPI softmax temperature")
    ap.add_argument("--mppi-correction-std", type=float, default=0.005, help="MPPI candidate correction std in meters")
    ap.add_argument("--mppi-max-correction", type=float, default=0.01, help="MPPI correction magnitude clamp in meters")
    ap.add_argument("--mppi-no-fail-gate", action="store_true", help="disable fail-threshold gate for MPPI")
    ap.add_argument("--disable-steering", action="store_true", help="disable steering even when checkpoint is provided")
    ap.add_argument("--log-json", default="", help="optional output JSON path for per-step trial log")
    args = ap.parse_args()

    if args.unsafe_no_constraints:
        # User-requested raw policy mode: disable clamps/guards/profiles to mirror open-loop behavior.
        args.task_profile = "none"
        args.instruction_sequence = ""
        args.phase_steps = ""
        args.phase_xyz_gains = ""
        args.phase_rot_gains = ""
        args.phase_max_xy_step_mm = ""
        args.phase_max_z_step_mm = ""
        args.phase_force_gripper = ""
        args.lock_orientation = False
        args.force_visible_motion = False
        args.max_xy_step_mm = 0.0
        args.max_z_step_mm = 0.0
        args.workspace_x_min = -1e9
        args.workspace_x_max = 1e9
        args.workspace_y_min = -1e9
        args.workspace_y_max = 1e9
        args.workspace_z_min = -1e9
        args.workspace_z_max = 1e9
        args.max_repeat_streak = 10**9
        args.min_motion_mm = -1.0
        args.model_only_strict = True
        args.gripper_continuous = True

    base = f"http://{args.jetson_host}:{args.jetson_port}"
    rng = np.random.default_rng(args.seed)
    print(f"[loop] base={base} policy={args.policy} model={args.model_name} execute={args.execute}")
    if args.prestart_coords:
        vals = [float(x.strip()) for x in str(args.prestart_coords).split(",") if x.strip()]
        if len(vals) != 6:
            raise ValueError("prestart-coords must contain 6 comma-separated values: x,y,z,rx,ry,rz")
        ok = move_to_pose_staged(
            base=base,
            target=vals,
            speed=args.speed,
            wait_s=max(0.3, float(args.dt)),
            max_step_mm=float(args.prestart_max_step_mm),
            tol_mm=float(args.prestart_tol_mm),
        )
        print(f"[loop] prestart move {'ok' if ok else 'failed'} target={np.round(np.asarray(vals),2).tolist()}")

    use_steering = bool(args.steering_checkpoint) and not args.disable_steering
    policy = load_policy(
        args.policy,
        args.model_name,
        args.device,
        args.device_map,
        enable_hidden_state_hooks=use_steering,
        force_token_fallback=args.force_token_fallback,
    )
    steering_runtime = None
    if use_steering:
        steering_runtime = SteeringRuntime(
            checkpoint_path=args.steering_checkpoint,
            model_device=args.device,
            alpha=args.steering_alpha,
            fail_threshold=args.steering_fail_threshold,
        )
        print(
            f"[loop] steering enabled checkpoint={args.steering_checkpoint} "
            f"alpha={args.steering_alpha} fail_threshold={args.steering_fail_threshold} "
            f"control_mode={args.control_mode}"
        )
    run_log = {
        "policy": args.policy,
        "model": args.model_name,
        "instruction": args.instruction,
        "execute": bool(args.execute),
        "steering_checkpoint": args.steering_checkpoint if use_steering else "",
        "control_mode": args.control_mode,
        "steps": [],
        "termination_reason": "max_steps",
    }
    last_action = None
    last_grip_sent = None
    repeat_streak = 0
    locked_rpy = None
    # Build phase schedule (single-phase default).
    phases = []
    if args.task_profile == "pick_yellow_cube" and not args.instruction_sequence:
        args.instruction_sequence = "move directly above the yellow cube and center over it||lower straight down to grasp the yellow cube||close gripper firmly on the yellow cube||lift the yellow cube straight up"
        args.phase_steps = "10,8,3,6"
        args.phase_xyz_gains = "0.16,0.10,0.02,0.12"
        args.phase_rot_gains = "0,0,0,0"
        args.phase_force_gripper = "-1,-1,95,95"
        args.phase_max_xy_step_mm = "5,2,0,2"
        args.phase_max_z_step_mm = "6,4,1,6"
        args.lock_orientation = True
        args.gripper_continuous = True
        args.force_visible_motion = True
        args.min_xy_mm = 0.0
        args.min_z_mm = 4.0
        # Keep task constrained around a typical tabletop yellow-cube workspace.
        args.workspace_x_min = 95.0
        args.workspace_x_max = 170.0
        args.workspace_y_min = -70.0
        args.workspace_y_max = 35.0
        args.workspace_z_min = 250.0
        args.workspace_z_max = 390.0

    if args.instruction_sequence:
        instrs = [s.strip() for s in args.instruction_sequence.split("||") if s.strip()]
        if not instrs:
            raise ValueError("instruction-sequence provided but empty after parsing")
        nph = len(instrs)
        p_steps = _parse_int_list(args.phase_steps, nph, args.steps)
        p_xyz = _parse_float_list(args.phase_xyz_gains, nph, args.xyz_gain)
        p_rot = _parse_float_list(args.phase_rot_gains, nph, args.rot_gain)
        p_max_xy = _parse_float_list(args.phase_max_xy_step_mm, nph, args.max_xy_step_mm)
        p_max_z = _parse_float_list(args.phase_max_z_step_mm, nph, args.max_z_step_mm)
        p_grip = _parse_int_list(args.phase_force_gripper, nph, -1)
        for i in range(nph):
            phases.append(
                {
                    "instruction": instrs[i],
                    "steps": int(p_steps[i]),
                    "xyz_gain": float(p_xyz[i]),
                    "rot_gain": float(p_rot[i]),
                    "max_xy_step_mm": float(p_max_xy[i]),
                    "max_z_step_mm": float(p_max_z[i]),
                    "force_gripper": None if int(p_grip[i]) < 0 else int(p_grip[i]),
                }
            )
    else:
        phases.append(
            {
                "instruction": args.instruction,
                "steps": int(args.steps),
                "xyz_gain": float(args.xyz_gain),
                "rot_gain": float(args.rot_gain),
                "max_xy_step_mm": float(args.max_xy_step_mm),
                "max_z_step_mm": float(args.max_z_step_mm),
                "force_gripper": None,
            }
        )

    print(f"[loop] phase_count={len(phases)} profile={args.task_profile}")
    for i, ph in enumerate(phases):
        print(
            f"[loop] phase {i}: steps={ph['steps']} xyz_gain={ph['xyz_gain']} "
            f"rot_gain={ph['rot_gain']} max_xy={ph['max_xy_step_mm']} max_z={ph['max_z_step_mm']} "
            f"force_gripper={ph['force_gripper']} instruction={ph['instruction']}"
        )
    try:
        global_step = 0
        for phase_idx, phase in enumerate(phases):
            for phase_step in range(int(phase["steps"])):
                step = global_step
                img, st = get_obs(base)
                raw_size = img.size
                if not args.no_libero_standardize:
                    img = standardize_libero_image(img, args.libero_image_size)
                if not st.get("ok"):
                    print(f"[step {step}] status failed:", st)
                    return 2
                cur = st.get("coords")
                if not isinstance(cur, list) or len(cur) != 6:
                    print(f"[step {step}] invalid coords from status:", cur)
                    return 3
                if locked_rpy is None:
                    locked_rpy = [float(cur[3]), float(cur[4]), float(cur[5])]

                a, feat = predict_action(policy, args.policy, img, phase["instruction"], rng)
                steering_info = {"applied": False, "reason": "disabled"}
                if steering_runtime is not None:
                    if args.control_mode == "steering":
                        a, steering_info = steering_runtime.maybe_steer(a, feat)
                    elif args.control_mode == "mppi":
                        a, steering_info = steering_runtime.maybe_mppi(
                            a,
                            feat,
                            n_samples=args.mppi_samples,
                            temperature=args.mppi_temperature,
                            correction_std=args.mppi_correction_std,
                            max_correction=args.mppi_max_correction,
                            fail_gate=not args.mppi_no_fail_gate,
                        )
                action_delta = None if last_action is None else float(np.linalg.norm(a - last_action))
                if (not args.unsafe_no_constraints) and action_delta is not None and action_delta < args.repeat_delta_threshold:
                    repeat_streak += 1
                else:
                    repeat_streak = 0
                if (not args.unsafe_no_constraints) and repeat_streak >= args.max_repeat_streak:
                    print(f"[step {step}] terminated: stuck repeated actions (delta={action_delta:.6f}, streak={repeat_streak})")
                    run_log["termination_reason"] = "stuck_action"
                    break
                sat_ratio = float(np.mean(np.abs(a[:6]) >= 0.95))
                if (not args.unsafe_no_constraints) and (not args.model_only_strict) and sat_ratio >= 0.8:
                    # Guardrail: token-decoding mismatches can produce saturated actions.
                    print(f"[step {step}] warning: saturated action ratio={sat_ratio:.2f}; zeroing motion delta for safety")
                    a[:6] = 0.0
                if args.action_space == "meters":
                    # OpenVLA-style task-space deltas: xyz in meters, rz in radians.
                    dx = float(a[0]) * 1000.0 * float(phase["xyz_gain"])
                    dy = float(a[1]) * 1000.0 * float(phase["xyz_gain"])
                    dz = float(a[2]) * 1000.0 * float(phase["xyz_gain"])
                    drz = float(np.degrees(a[5])) * float(phase["rot_gain"])
                else:
                    # Legacy normalized mapping.
                    dx = float(a[0]) * args.xy_scale_mm
                    dy = float(a[1]) * args.xy_scale_mm
                    dz = float(a[2]) * args.z_scale_mm
                    drz = float(a[5]) * args.rz_scale_deg
                if float(phase["max_xy_step_mm"]) > 0.0:
                    dx = clamp(dx, -float(phase["max_xy_step_mm"]), float(phase["max_xy_step_mm"]))
                    dy = clamp(dy, -float(phase["max_xy_step_mm"]), float(phase["max_xy_step_mm"]))
                if float(phase["max_z_step_mm"]) > 0.0:
                    dz = clamp(dz, -float(phase["max_z_step_mm"]), float(phase["max_z_step_mm"]))
                if args.force_visible_motion and not args.model_only_strict:
                    if abs(dx) > 1e-5:
                        dx = float(np.sign(dx)) * max(abs(dx), args.min_xy_mm)
                    if abs(dy) > 1e-5:
                        dy = float(np.sign(dy)) * max(abs(dy), args.min_xy_mm)
                    if abs(dz) > 1e-5:
                        dz = float(np.sign(dz)) * max(abs(dz), args.min_z_mm)
                    if abs(drz) > 1e-5:
                        drz = float(np.sign(drz)) * max(abs(drz), args.min_rz_deg)

                target = [
                    float(cur[0]) + dx,
                    float(cur[1]) + dy,
                    float(cur[2]) + dz,
                    float(locked_rpy[0]) if args.lock_orientation else float(cur[3]),
                    float(locked_rpy[1]) if args.lock_orientation else float(cur[4]),
                    float(locked_rpy[2]) if args.lock_orientation else float(cur[5]) + drz,
                ]
                if not args.unsafe_no_constraints:
                    target = clamp_coords_with_bounds(
                        target,
                        args.workspace_x_min,
                        args.workspace_x_max,
                        args.workspace_y_min,
                        args.workspace_y_max,
                        args.workspace_z_min,
                        args.workspace_z_max,
                    )
                if phase["force_gripper"] is not None:
                    grip_val = int(clamp(float(phase["force_gripper"]), 0.0, 100.0))
                elif args.gripper_continuous or args.model_only_strict:
                    grip_val = int(clamp(((float(a[6]) + 1.0) * 50.0), 0.0, 100.0))
                else:
                    grip_val = 80 if float(a[6]) > args.gripper_threshold else 20
                pretty_target = [round(float(v), 2) for v in target]
                if step == 0:
                    print(
                        f"[step {step}] camera_raw={raw_size} policy_input={img.size} "
                        f"libero_standardize={not args.no_libero_standardize}"
                    )
                print(
                    f"[step {step}] phase={phase_idx}:{phase_step} action={np.round(a,3).tolist()} -> "
                    f"target={pretty_target} gripper={grip_val}"
                )
                if steering_runtime is not None:
                    print(
                        f"  steering: applied={steering_info.get('applied')} "
                        f"fail_prob={steering_info.get('fail_prob')} "
                        f"corr_m={np.round(np.asarray(steering_info.get('correction_m', [0,0,0])), 5).tolist()}"
                    )
                run_log["steps"].append(
                    {
                        "step": int(step),
                        "phase_idx": int(phase_idx),
                        "phase_step": int(phase_step),
                        "instruction": str(phase["instruction"]),
                        "action": np.round(a, 6).tolist(),
                        "action_delta": action_delta,
                        "target": [float(x) for x in target],
                        "gripper": int(grip_val),
                        "repeat_streak": int(repeat_streak),
                        "steering": steering_info,
                    }
                )

                if args.execute:
                    r = api_post(
                        base,
                        {
                            "action": "move_to",
                            "coords": target,
                            "speed": args.speed,
                            "wait": bool(args.wait_for_motion),
                        },
                    )
                    print("  move_to:", r.get("ok"), "elapsed_ms:", r.get("elapsed_ms"), "err:", r.get("error"))
                    final_coords = r.get("final_coords")
                    if isinstance(final_coords, list) and len(final_coords) >= 3:
                        achieved = float(
                            np.linalg.norm(
                                np.asarray(final_coords[:3], dtype=np.float32) - np.asarray(cur[:3], dtype=np.float32)
                            )
                        )
                        print(
                            f"  move_to final_xyz={np.round(np.asarray(final_coords[:3], dtype=np.float32), 2).tolist()} "
                            f"achieved_mm={achieved:.2f}"
                        )
                        if (not args.unsafe_no_constraints) and achieved < args.min_motion_mm:
                            print(
                                f"  warning: achieved motion {achieved:.2f}mm < min-motion-mm={args.min_motion_mm:.2f}; "
                                "target may be unreachable or filtered by controller"
                            )
                    if not r.get("ok"):
                        print(f"[step {step}] terminated: robot command failure")
                        run_log["termination_reason"] = "command_error"
                        return 4
                    if args.skip_redundant_gripper and last_grip_sent is not None and int(last_grip_sent) == int(grip_val):
                        print("  gripper: skipped (unchanged)")
                    else:
                        rg = api_post(base, {"action": "set_gripper", "value": grip_val, "speed": 40})
                        print("  gripper:", rg.get("ok"), "elapsed_ms:", rg.get("elapsed_ms"), "err:", rg.get("error"))
                        if not rg.get("ok"):
                            print(f"[step {step}] terminated: gripper command failure")
                            run_log["termination_reason"] = "command_error"
                            return 5
                        last_grip_sent = int(grip_val)

                last_action = a.copy()
                global_step += 1
                time.sleep(max(0.0, args.dt))
            if run_log["termination_reason"] != "max_steps":
                break
        if run_log["termination_reason"] == "max_steps":
            total_steps = int(sum(int(p["steps"]) for p in phases))
            print(f"[loop] finished normally after max steps={total_steps}")
    finally:
        if args.log_json:
            out_path = Path(args.log_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(run_log, indent=2))
            print(f"[loop] wrote log: {out_path}")
        if policy is not None and hasattr(policy, "close"):
            policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
