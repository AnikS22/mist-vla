#!/usr/bin/env python3
"""Quick camera quality and occlusion diagnostics for Jetson arm feed."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from urllib import request

import cv2
import numpy as np
from PIL import Image


def api_get(base: str, path: str, timeout: float = 8.0) -> bytes:
    with request.urlopen(f"{base}{path}", timeout=timeout) as resp:
        return resp.read()


def api_post(base: str, payload: dict, timeout: float = 20.0) -> dict:
    data = json.dumps(payload).encode()
    req = request.Request(
        f"{base}/action",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def snapshot(base: str) -> np.ndarray:
    b = api_get(base, "/snapshot")
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return np.array(img)


def metrics(arr: np.ndarray) -> dict:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    center = gray[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    return {
        "shape": [int(h), int(w), 3],
        "mean_luma": float(gray.mean()),
        "std_luma": float(gray.std()),
        "lap_var_full": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "center_mean_luma": float(center.mean()),
        "center_lap_var": float(cv2.Laplacian(center, cv2.CV_64F).var()),
        "center_dark_ratio": float((center < 40).mean()),
    }


def check(m: dict) -> dict:
    # Conservative pass bands; tune after collecting more on-site data.
    return {
        "brightness_ok": 45.0 <= m["mean_luma"] <= 210.0,
        "contrast_ok": m["std_luma"] >= 18.0,
        "focus_ok": m["lap_var_full"] >= 120.0,
        "center_not_heavily_occluded": m["center_dark_ratio"] <= 0.25,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Jetson camera diagnostics")
    ap.add_argument("--host", default="192.168.55.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--save-dir", default="/home/mpcr/Desktop/SalusV5/mist-vla/research_data/results")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture in two canonical poses to check arm-induced occlusion.
    api_post(base, {"action": "home"})
    arr_home = snapshot(base)
    api_post(base, {"action": "watch"})
    arr_watch = snapshot(base)

    m_home = metrics(arr_home)
    m_watch = metrics(arr_watch)
    c_home = check(m_home)
    c_watch = check(m_watch)

    Image.fromarray(arr_home).save(out_dir / "jetson_cam_home.jpg")
    Image.fromarray(arr_watch).save(out_dir / "jetson_cam_watch.jpg")

    summary = {
        "home": {"metrics": m_home, "checks": c_home},
        "watch": {"metrics": m_watch, "checks": c_watch},
        "all_checks_pass": all(c_home.values()) and all(c_watch.values()),
    }
    (out_dir / "jetson_camera_diagnostics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
