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
from typing import Dict, Tuple
from urllib import request

import cv2
import numpy as np
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
        clamp(float(z), 40.0, 260.0),
        clamp(float(rx), -180.0, 180.0),
        clamp(float(ry), -180.0, 180.0),
        clamp(float(rz), -180.0, 180.0),
    ]


def get_obs(base: str) -> Tuple[Image.Image, Dict]:
    img_bytes = api_get(base, "/snapshot")
    st = json.loads(api_get(base, "/status").decode())
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img, st


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


def load_policy(policy_kind: str, model_name: str, device: str, device_map: str | None):
    if policy_kind == "random":
        return None
    if policy_kind == "openvla":
        from src.models.vla_wrapper import create_vla_wrapper

        return create_vla_wrapper(
            model_type="openvla",
            model_name=model_name,
            device=device,
            device_map=device_map,
        )
    if policy_kind == "openvla_oft":
        from src.models.vla_wrapper import create_vla_wrapper

        return create_vla_wrapper(model_type="openvla_oft", model_name=model_name, device=device)
    raise ValueError(f"unsupported policy_kind: {policy_kind}")


def predict_action(policy, policy_kind: str, image: Image.Image, instruction: str, rng: np.random.Generator):
    if policy_kind == "random":
        a = rng.uniform(-0.3, 0.3, size=(7,)).astype(np.float32)
        a[-1] = rng.uniform(-1.0, 1.0)
        return a
    action, _feat = policy.get_action_with_features(image, instruction, obs=None)
    if hasattr(action, "detach"):
        a = action.detach().cpu().numpy().astype(np.float32)
    else:
        a = np.asarray(action, dtype=np.float32)
    if a.ndim > 1:
        a = a.reshape(-1)
    if a.shape[0] < 7:
        a = np.pad(a, (0, 7 - a.shape[0]))
    return a[:7]


def main() -> int:
    ap = argparse.ArgumentParser(description="Safe model->Yahboom loop")
    ap.add_argument("--jetson-host", default="192.168.55.1")
    ap.add_argument("--jetson-port", type=int, default=5000)
    ap.add_argument("--policy", choices=["openvla", "openvla_oft", "random"], default="random")
    ap.add_argument("--model-name", default="openvla/openvla-7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto", help="HF device_map for OpenVLA (e.g., auto, none)")
    ap.add_argument("--instruction", default="pick up the block and place it carefully")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--model-only-strict", action="store_true", help="disable heuristic motion/gripper overrides (safety clamps remain)")
    ap.add_argument("--libero-image-size", type=int, default=224, help="square size for LIBERO-like image standardization")
    ap.add_argument("--no-libero-standardize", action="store_true", help="disable center-crop+resize preprocessing")
    ap.add_argument("--dt", type=float, default=0.8)
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
    ap.add_argument("--gripper-threshold", type=float, default=0.2)
    ap.add_argument("--gripper-continuous", action="store_true", help="map model gripper action continuously to 0..100")
    ap.add_argument("--execute", action="store_true", help="actually send move_to/set_gripper")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = f"http://{args.jetson_host}:{args.jetson_port}"
    rng = np.random.default_rng(args.seed)
    print(f"[loop] base={base} policy={args.policy} model={args.model_name} execute={args.execute}")

    policy = load_policy(args.policy, args.model_name, args.device, args.device_map)
    try:
        for step in range(args.steps):
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

            a = predict_action(policy, args.policy, img, args.instruction, rng)
            sat_ratio = float(np.mean(np.abs(a[:6]) >= 0.95))
            if (not args.model_only_strict) and sat_ratio >= 0.8:
                # Guardrail: token-decoding mismatches can produce saturated actions.
                print(f"[step {step}] warning: saturated action ratio={sat_ratio:.2f}; zeroing motion delta for safety")
                a[:6] = 0.0
            if args.action_space == "meters":
                # OpenVLA-style task-space deltas: xyz in meters, rz in radians.
                dx = float(a[0]) * 1000.0 * args.xyz_gain
                dy = float(a[1]) * 1000.0 * args.xyz_gain
                dz = float(a[2]) * 1000.0 * args.xyz_gain
                drz = float(np.degrees(a[5])) * args.rot_gain
            else:
                # Legacy normalized mapping.
                dx = float(a[0]) * args.xy_scale_mm
                dy = float(a[1]) * args.xy_scale_mm
                dz = float(a[2]) * args.z_scale_mm
                drz = float(a[5]) * args.rz_scale_deg
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
                float(cur[3]),
                float(cur[4]),
                float(cur[5]) + drz,
            ]
            target = clamp_coords(target)
            if args.gripper_continuous or args.model_only_strict:
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
                f"[step {step}] action={np.round(a,3).tolist()} -> "
                f"target={pretty_target} gripper={grip_val}"
            )

            if args.execute:
                r = api_post(base, {"action": "move_to", "coords": target, "speed": args.speed, "wait": args.dt})
                print("  move_to:", r.get("ok"), "elapsed_ms:", r.get("elapsed_ms"), "err:", r.get("error"))
                if not r.get("ok"):
                    print(f"[step {step}] terminated: robot command failure")
                    return 4
                rg = api_post(base, {"action": "set_gripper", "value": grip_val, "speed": 40})
                print("  gripper:", rg.get("ok"), "elapsed_ms:", rg.get("elapsed_ms"), "err:", rg.get("error"))
                if not rg.get("ok"):
                    print(f"[step {step}] terminated: gripper command failure")
                    return 5

            time.sleep(max(0.05, args.dt))
        print(f"[loop] finished normally after max steps={args.steps}")
    finally:
        if policy is not None and hasattr(policy, "close"):
            policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
