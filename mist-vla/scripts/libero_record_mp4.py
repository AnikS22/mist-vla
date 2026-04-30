#!/usr/bin/env python3
"""Record a short LIBERO (Franka) rollout to MP4 — headless, no VLA.

Same env path as HPC evals (OffScreenRenderEnv + libero_spatial task 0).

Movement styles (``--movement``):
  box       — uniform sample in [low, high] each dim (default)
  extremes  — each dim independently at low[i] or high[i] (max command per axis, no soft cap)
  gaussian  — N(μ=mid, σ=span/2) clipped to [low, high] (wild motion)

``--uncapped`` sends Gaussian-distributed actions **without** clipping to action_spec first
(robosuite may still clamp internally).

Usage:
  cd mist-vla && python3 scripts/libero_record_mp4.py -o figures/my_rollout.mp4 --movement extremes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def sample_action(
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
    movement: str,
    uncapped: bool,
) -> np.ndarray:
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    mid = 0.5 * (low + high)
    span = high - low

    if uncapped:
        z = rng.standard_normal(len(low))
        return (mid + z * span).astype(np.float32)

    if movement == "box":
        return rng.uniform(low, high).astype(np.float32)
    if movement == "extremes":
        pick_low = rng.random(len(low)) < 0.5
        return np.where(pick_low, low, high).astype(np.float32)
    if movement == "gaussian":
        z = rng.standard_normal(len(low))
        a = mid + 0.5 * span * z
        return np.clip(a, low, high).astype(np.float32)
    raise ValueError(movement)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", type=str, default=str(REPO / "figures" / "libero_recorded.mp4"))
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--res", type=int, default=256)
    p.add_argument(
        "--movement",
        choices=("box", "extremes", "gaussian"),
        default="extremes",
        help="How to sample actions (default: extremes = full per-axis range, no dampening)",
    )
    p.add_argument(
        "--uncapped",
        action="store_true",
        help="Do not clip actions to [low,high] before env.step (Gaussian × span)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    bddl = task_suite.get_task_bddl_file_path(args.task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        render_camera="agentview",
        camera_heights=args.res,
        camera_widths=args.res,
    )
    obs = env.reset()
    low, high = env.env.action_spec
    frames = []
    for _ in range(args.frames):
        action = sample_action(low, high, rng, args.movement, args.uncapped)
        if not args.uncapped:
            action = np.clip(action, low, high)
        obs, _r, done, _info = env.step(action)
        img = obs["agentview_image"]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(np.ascontiguousarray(img))
        if done:
            obs = env.reset()
    env.close()

    imageio.mimsave(str(out), frames, fps=args.fps, macro_block_size=1)
    print(
        f"Wrote {out.resolve()} ({len(frames)} frames @ {args.fps} fps) "
        f"movement={args.movement} uncapped={args.uncapped}"
    )


if __name__ == "__main__":
    main()
