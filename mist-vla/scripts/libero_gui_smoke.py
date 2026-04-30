#!/usr/bin/env python3
"""Open LIBERO in a **native MuJoCo/robosuite window** (not EGL offscreen).

Uses ``ControlEnv`` with ``has_renderer=True`` — same stack as ``LIBERO/scripts/collect_demonstration.py``,
but only random actions for a smoke test.

Requires:
  - A display (``$DISPLAY`` on Linux, or Wayland with XWayland).
  - ``MUJOCO_GL=glfw`` (set by ``local_run_from_hpc.sh libero-gui``).

Close the viewer window or press Ctrl+C to exit.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv


def _sample_action(low, high, rng, movement: str, uncapped: bool) -> np.ndarray:
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
    z = rng.standard_normal(len(low))
    a = mid + 0.5 * span * z
    return np.clip(a, low, high).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--steps", type=int, default=600, help="Max steps (or until window closed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--movement",
        choices=("box", "extremes", "gaussian"),
        default="extremes",
        help="Action sampling: extremes = per-axis low/high (max swing); box = uniform",
    )
    p.add_argument(
        "--uncapped",
        action="store_true",
        help="Gaussian actions without clipping to action_spec before step",
    )
    args = p.parse_args()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    bddl = task_suite.get_task_bddl_file_path(args.task_id)
    instruction = task_suite.get_task(args.task_id).language
    print(f"Task {args.task_id}: {instruction[:100]}…", flush=True)
    print(
        "Loading simulation + GLFW viewer (first launch can take 1–2 minutes)…",
        flush=True,
    )

    # Offscreen + on-screen together is very slow; for live GUI use viewer only (like
    # LIBERO/scripts/collect_demonstration.py when keyboard teleop).
    env = ControlEnv(
        bddl_file_name=bddl,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="agentview",
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    obs = env.reset()
    low, high = env.env.action_spec
    print(
        "Viewer should be open. Random exploration for smoke test.\n"
        "Close the window or Ctrl+C to stop.",
        flush=True,
    )
    try:
        for step in range(args.steps):
            action = _sample_action(low, high, rng, args.movement, args.uncapped)
            if not args.uncapped:
                action = np.clip(action, low, high)
            out = env.step(action)
            obs = out[0]
            done = bool(out[2])
            if done:
                obs = env.reset()
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
