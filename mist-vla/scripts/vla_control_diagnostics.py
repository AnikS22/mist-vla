#!/usr/bin/env python3
"""Diagnose VLA control sensitivity on the live Jetson camera feed."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from urllib import request

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.vla_wrapper import create_vla_wrapper


def api_get(base: str, path: str, timeout: float = 8.0) -> bytes:
    with request.urlopen(f"{base}{path}", timeout=timeout) as resp:
        return resp.read()


def get_frame(base: str, size: int) -> Image.Image:
    b = api_get(base, "/snapshot")
    img = Image.open(io.BytesIO(b)).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = arr[y0 : y0 + side, x0 : x0 + side]
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(out)


def to_np_action(a) -> np.ndarray:
    if hasattr(a, "detach"):
        x = a.detach().cpu().numpy().astype(np.float32)
    else:
        x = np.asarray(a, dtype=np.float32)
    x = x.reshape(-1)
    if x.shape[0] < 7:
        x = np.pad(x, (0, 7 - x.shape[0]))
    return x[:7]


def main() -> int:
    ap = argparse.ArgumentParser(description="Live VLA control diagnostics")
    ap.add_argument("--host", default="192.168.55.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--model-name", default="openvla/openvla-7b")
    ap.add_argument("--policy", choices=["openvla", "openvla_oft", "smolvla"], default="openvla")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--min-global-std", type=float, default=0.005)
    ap.add_argument("--min-prompt-delta", type=float, default=0.02)
    ap.add_argument("--min-frame-delta", type=float, default=0.01)
    ap.add_argument("--min-gripper-std", type=float, default=0.01)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    prompts = [
        "pick up the block",
        "move left toward the block",
        "move right toward the target",
        "lower gripper to grasp",
    ]

    model = create_vla_wrapper(
        args.policy,
        args.model_name,
        device=args.device,
        device_map=args.device_map,
    )

    actions = []
    frames = []
    try:
        for i in range(args.frames):
            img = get_frame(base, args.image_size)
            frames.append(img)
            for p in prompts:
                a, _ = model.get_action_with_features(img, p, obs=None)
                actions.append({"frame": i, "prompt": p, "action": to_np_action(a)})
            time.sleep(args.sleep)
    finally:
        model.close()

    # Aggregate diagnostics
    arr = np.stack([x["action"] for x in actions], axis=0)
    prompt_means = {}
    for p in prompts:
        apm = np.stack([x["action"] for x in actions if x["prompt"] == p], axis=0)
        prompt_means[p] = apm.mean(axis=0).tolist()
    frame_means = {}
    for i in range(args.frames):
        afm = np.stack([x["action"] for x in actions if x["frame"] == i], axis=0)
        frame_means[str(i)] = afm.mean(axis=0).tolist()

    out = {
        "policy": args.policy,
        "model": args.model_name,
        "prompts": prompts,
        "n_samples": int(arr.shape[0]),
        "global_action_mean": arr.mean(axis=0).tolist(),
        "global_action_std": arr.std(axis=0).tolist(),
        "prompt_means": prompt_means,
        "frame_means": frame_means,
        "xyz_norm_mean": float(np.mean(np.linalg.norm(arr[:, :3], axis=1))),
        "xyz_norm_std": float(np.std(np.linalg.norm(arr[:, :3], axis=1))),
    }

    # Acceptance gates for research-ready non-degenerate control.
    prompt_mean_actions = np.stack([np.asarray(prompt_means[p], dtype=np.float32) for p in prompts], axis=0)
    frame_mean_actions = np.stack([np.asarray(frame_means[str(i)], dtype=np.float32) for i in range(args.frames)], axis=0)
    prompt_deltas = []
    for i in range(len(prompt_mean_actions)):
        for j in range(i + 1, len(prompt_mean_actions)):
            prompt_deltas.append(float(np.linalg.norm(prompt_mean_actions[i] - prompt_mean_actions[j])))
    frame_deltas = []
    for i in range(len(frame_mean_actions)):
        for j in range(i + 1, len(frame_mean_actions)):
            frame_deltas.append(float(np.linalg.norm(frame_mean_actions[i] - frame_mean_actions[j])))
    max_prompt_delta = max(prompt_deltas) if prompt_deltas else 0.0
    max_frame_delta = max(frame_deltas) if frame_deltas else 0.0
    global_std_mean = float(arr.std(axis=0).mean())
    gripper_std = float(arr[:, 6].std())
    passed = (
        global_std_mean >= args.min_global_std
        and max_prompt_delta >= args.min_prompt_delta
        and max_frame_delta >= args.min_frame_delta
        and gripper_std >= args.min_gripper_std
    )
    out["acceptance"] = {
        "passed": bool(passed),
        "global_std_mean": global_std_mean,
        "max_prompt_delta_l2": max_prompt_delta,
        "max_frame_delta_l2": max_frame_delta,
        "gripper_std": gripper_std,
        "thresholds": {
            "min_global_std": args.min_global_std,
            "min_prompt_delta": args.min_prompt_delta,
            "min_frame_delta": args.min_frame_delta,
            "min_gripper_std": args.min_gripper_std,
        },
    }

    out_path = REPO_ROOT / "research_data" / "results" / "vla_control_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved {out_path}")
    if not passed:
        print("[diagnostics] FAILED acceptance gates; model output appears degenerate.")
        return 2
    print("[diagnostics] PASSED acceptance gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
