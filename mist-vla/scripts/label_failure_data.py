#!/usr/bin/env python3
"""
Generate SAFE-style labels and per-dimension risk labels from rollouts.

Labels:
  - time_to_failure (steps)
  - time_to_collision (steps or -1)
  - fail_within_k (binary per step)
  - collision_within_k (binary per step)
  - per_dim_risk (7-dim binary vector)
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_rollouts(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "trajectories" in data:
        return data["trajectories"], data.get("metadata", {})
    return data, {}


def _save_rollouts(path: Path, trajectories, metadata: Dict):
    payload = {"trajectories": trajectories, "metadata": metadata}
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _label_rollout(
    traj: Dict,
    k_fail: int,
    k_collision: int,
    action_thresh: float,
):
    steps = traj.get("steps", [])
    if not steps:
        return

    # Determine failure step (end of episode if not success)
    success = bool(traj.get("success"))
    failure_step = None if success else len(steps) - 1

    # Determine collision step if available
    collision_step = traj.get("collision_step")
    if collision_step is None:
        # Try to infer from per-step collision flags
        for i, step in enumerate(steps):
            if step.get("collision"):
                collision_step = i
                break

    for i, step in enumerate(steps):
        # time-to labels
        time_to_failure = -1 if failure_step is None else max(failure_step - i, 0)
        time_to_collision = -1 if collision_step is None else max(collision_step - i, 0)

        fail_within_k = (
            1 if (failure_step is not None and time_to_failure <= k_fail) else 0
        )
        collision_within_k = (
            1
            if (collision_step is not None and time_to_collision <= k_collision)
            else 0
        )

        # Per-dimension risk (heuristic until predictor is trained):
        # Mark dimensions with large action magnitude if failure/collision is imminent.
        action = np.array(step.get("action", np.zeros(7)), dtype=np.float32)
        per_dim_risk = (np.abs(action) >= action_thresh).astype(np.int32)
        if (collision_within_k == 0) and (fail_within_k == 0):
            per_dim_risk = np.zeros_like(per_dim_risk)

        step["labels"] = {
            "time_to_failure": int(time_to_failure),
            "time_to_collision": int(time_to_collision),
            "fail_within_k": int(fail_within_k),
            "collision_within_k": int(collision_within_k),
            "per_dim_risk": per_dim_risk.tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Label rollouts with failure timing and per-dim risk")
    parser.add_argument("--input", required=True, help="Input rollouts .pkl")
    parser.add_argument("--output", required=True, help="Output labeled .pkl")
    parser.add_argument("--k-fail", type=int, default=10)
    parser.add_argument("--k-collision", type=int, default=10)
    parser.add_argument("--action-thresh", type=float, default=0.2)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    trajectories, metadata = _load_rollouts(in_path)
    for traj in trajectories:
        _label_rollout(traj, args.k_fail, args.k_collision, args.action_thresh)

    metadata = dict(metadata)
    metadata["labeling"] = {
        "k_fail": args.k_fail,
        "k_collision": args.k_collision,
        "action_thresh": args.action_thresh,
    }
    _save_rollouts(out_path, trajectories, metadata)

    print(f"Labeled {len(trajectories)} trajectories -> {out_path}")


if __name__ == "__main__":
    main()
