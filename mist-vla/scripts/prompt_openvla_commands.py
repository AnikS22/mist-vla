#!/usr/bin/env python3
"""Interactive prompt -> OpenVLA action -> robot command preview (optional execute)."""

from __future__ import annotations

import argparse
import io
import json
import sys
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


def api_post(base: str, payload: dict, timeout: float = 25.0) -> dict:
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


def standardize(img: Image.Image, size: int) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = arr[y0 : y0 + side, x0 : x0 + side]
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(out)


def to_action_np(a) -> np.ndarray:
    if hasattr(a, "detach"):
        x = a.detach().cpu().numpy().astype(np.float32)
    else:
        x = np.asarray(a, dtype=np.float32)
    x = x.reshape(-1)
    if x.shape[0] < 7:
        x = np.pad(x, (0, 7 - x.shape[0]))
    return x[:7]


def main() -> int:
    ap = argparse.ArgumentParser(description="Prompt OpenVLA and print robot commands")
    ap.add_argument("--host", default="192.168.55.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--model-name", default="openvla/openvla-7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--execute", action="store_true", help="send move_to/set_gripper to robot")
    ap.add_argument("--speed", type=int, default=16)
    ap.add_argument("--wait", type=float, default=0.8)
    ap.add_argument("--xyz-gain", type=float, default=1.0, help="gain for xyz when action-space is meters")
    ap.add_argument("--rot-gain", type=float, default=1.0, help="gain for rz (radians->deg) when action-space is meters")
    ap.add_argument(
        "--prompt-mode",
        choices=["openvla", "raw"],
        default="raw",
        help="openvla: wraps your text in OpenVLA template; raw: sends your text verbatim",
    )
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"[prompt] base={base} model={args.model_name} execute={args.execute}")
    print(f"[prompt] mode={args.prompt_mode} (type anything, or 'quit')")

    policy = create_vla_wrapper(
        "openvla",
        args.model_name,
        device=args.device,
        device_map=args.device_map,
    )
    try:
        while True:
            try:
                instruction = input("\nInstruction> ").strip()
            except EOFError:
                break
            if not instruction or instruction.lower() in {"q", "quit", "exit"}:
                break

            st = json.loads(api_get(base, "/status").decode())
            if not st.get("ok"):
                print("[prompt] status failed:", st)
                continue
            cur = st.get("coords")
            if not isinstance(cur, list) or len(cur) != 6:
                print("[prompt] invalid status coords:", cur)
                continue

            img_bytes = api_get(base, "/snapshot")
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = standardize(img, args.image_size)

            if args.prompt_mode == "openvla":
                prompt = instruction
            else:
                # True passthrough: no template edits, no lower-casing.
                prompt = "RAW_PROMPT::" + instruction
            action, _feat = policy.get_action_with_features(img, prompt, obs=None)
            a = to_action_np(action)

            # OpenVLA metric interpretation (xyz in meters, rz in radians)
            dx = float(a[0]) * 1000.0 * args.xyz_gain
            dy = float(a[1]) * 1000.0 * args.xyz_gain
            dz = float(a[2]) * 1000.0 * args.xyz_gain
            drz = float(np.degrees(a[5])) * args.rot_gain
            target = clamp_coords(
                [
                    float(cur[0]) + dx,
                    float(cur[1]) + dy,
                    float(cur[2]) + dz,
                    float(cur[3]),
                    float(cur[4]),
                    float(cur[5]) + drz,
                ]
            )
            grip = int(clamp(((float(a[6]) + 1.0) * 50.0), 0.0, 100.0))

            print("[prompt] action:", np.round(a, 4).tolist())
            print("[prompt] command target:", [round(float(v), 2) for v in target], "gripper:", grip)

            if args.execute:
                rm = api_post(base, {"action": "move_to", "coords": target, "speed": args.speed, "wait": args.wait})
                rg = api_post(base, {"action": "set_gripper", "value": grip, "speed": 40})
                print("[prompt] move_to:", rm.get("ok"), "err:", rm.get("error"))
                print("[prompt] gripper:", rg.get("ok"), "err:", rg.get("error"))
    finally:
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
