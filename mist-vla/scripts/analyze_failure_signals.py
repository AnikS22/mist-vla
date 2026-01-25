#!/usr/bin/env python3
"""
Analyze hidden-state and action signals for success vs failure rollouts.
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
        return data["trajectories"]
    return data


def _collect_step_arrays(trajectory: Dict) -> Dict[str, np.ndarray]:
    steps = trajectory.get("steps", [])
    if not steps:
        return {
            "hidden": np.empty((0,)),
            "actions": np.empty((0,)),
        }
    hidden = np.stack([s["hidden_state"] for s in steps])
    actions = np.stack([s["action"] for s in steps])
    return {"hidden": hidden, "actions": actions}


def _summarize_group(trajs: List[Dict]) -> Dict[str, float]:
    if not trajs:
        return {"count": 0}

    hidden_all = []
    actions_all = []
    for t in trajs:
        arrays = _collect_step_arrays(t)
        if arrays["hidden"].size:
            hidden_all.append(arrays["hidden"])
        if arrays["actions"].size:
            actions_all.append(arrays["actions"])

    if not hidden_all or not actions_all:
        return {"count": len(trajs)}

    hidden = np.concatenate(hidden_all, axis=0)
    actions = np.concatenate(actions_all, axis=0)

    return {
        "count": len(trajs),
        "steps": int(hidden.shape[0]),
        "hidden_mean": float(hidden.mean()),
        "hidden_std": float(hidden.std()),
        "hidden_norm_mean": float(np.linalg.norm(hidden, axis=1).mean()),
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "action_norm_mean": float(np.linalg.norm(actions, axis=1).mean()),
    }


def _mean_vector(trajs: List[Dict]) -> np.ndarray:
    vectors = []
    for t in trajs:
        arrays = _collect_step_arrays(t)
        if arrays["hidden"].size:
            vectors.append(arrays["hidden"].mean(axis=0))
    if not vectors:
        return np.array([])
    return np.stack(vectors, axis=0).mean(axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze success vs failure signals in collected rollouts"
    )
    parser.add_argument("--input", type=str, required=True, help="Rollout pkl path")
    args = parser.parse_args()

    path = Path(args.input)
    trajs = _load_rollouts(path)

    successes = [t for t in trajs if t.get("success") is True]
    failures = [t for t in trajs if t.get("success") is False]
    unknown = [t for t in trajs if t.get("success") is None]

    print(f"Loaded {len(trajs)} trajectories from {path}")
    print(f"  Success: {len(successes)} | Failure: {len(failures)} | Unknown: {len(unknown)}")

    success_stats = _summarize_group(successes)
    failure_stats = _summarize_group(failures)

    print("\nGroup statistics (step-level aggregates)")
    print(f"  Success: {success_stats}")
    print(f"  Failure: {failure_stats}")

    success_mean_vec = _mean_vector(successes)
    failure_mean_vec = _mean_vector(failures)
    print("\nHidden-state mean-vector comparison")
    print(f"  Cosine similarity: {_cosine(success_mean_vec, failure_mean_vec):.4f}")
    if success_mean_vec.size and failure_mean_vec.size:
        diff_norm = float(np.linalg.norm(success_mean_vec - failure_mean_vec))
        print(f"  Mean-vector L2 distance: {diff_norm:.4f}")


if __name__ == "__main__":
    main()
