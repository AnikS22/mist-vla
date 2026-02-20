#!/usr/bin/env python3
"""
Build a per-step dataset for risk prediction from rollout pickles.

This emits a dict with a single key:
  - dataset: list of {hidden_state, risk_label, action}
"""
import argparse
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np


def _load_rollouts(path: Path) -> List[Dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "trajectories" in data:
        return data["trajectories"]
    return data


def _get_steps(traj: Dict):
    if "steps" in traj and traj["steps"]:
        return traj["steps"]
    actions = traj.get("actions", [])
    features = traj.get("features", [])
    steps = []
    for i in range(min(len(actions), len(features))):
        steps.append(
            {
                "action": np.asarray(actions[i], dtype=np.float32),
                "hidden_state": np.asarray(features[i], dtype=np.float32),
                "collision": False,
            }
        )
    return steps


def build_dataset(
    success_path: Path,
    failure_path: Path,
    output_path: Path,
    k_fail: int,
    action_thresh: float,
):
    success_rollouts = _load_rollouts(success_path)
    failure_rollouts = _load_rollouts(failure_path)
    rollouts = list(success_rollouts) + list(failure_rollouts)

    dataset = []
    for traj in rollouts:
        steps = _get_steps(traj)
        if not steps:
            continue

        for i, step in enumerate(steps):
            action = np.asarray(step.get("action", np.zeros(7)), dtype=np.float32)
            hidden = np.asarray(step.get("hidden_state", np.zeros(1)), dtype=np.float32)
            labels = step.get("labels", {})
            if "per_dim_risk" in labels:
                risk = np.asarray(labels["per_dim_risk"], dtype=np.float32)
                time_to_failure = labels.get("time_to_failure", -1)
                fail_within_k = labels.get("fail_within_k", 0)
            else:
                success = bool(traj.get("success", False))
                failure_step = None if success else len(steps) - 1
                if failure_step is not None and (failure_step - i) <= k_fail:
                    risk = (np.abs(action) >= action_thresh).astype(np.float32)
                else:
                    risk = np.zeros_like(action, dtype=np.float32)
                time_to_failure = -1 if failure_step is None else max(failure_step - i, 0)
                fail_within_k = 1 if (failure_step is not None and time_to_failure <= k_fail) else 0

            dataset.append(
                {
                    "hidden_state": hidden,
                    "risk_label": risk,
                    "action": action,
                    "time_to_failure": int(time_to_failure),
                    "fail_within_k": int(fail_within_k),
                }
            )

    payload = {
        "dataset": dataset,
        "metadata": {
            "k_fail": k_fail,
            "action_thresh": action_thresh,
            "num_rollouts": len(rollouts),
            "num_samples": len(dataset),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved dataset with {len(dataset)} samples -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", required=True, help="Path to success_rollouts.pkl")
    parser.add_argument("--failure", required=True, help="Path to failure_rollouts.pkl")
    parser.add_argument("--output", required=True, help="Output dataset .pkl")
    parser.add_argument("--k-fail", type=int, default=10)
    parser.add_argument("--action-thresh", type=float, default=0.2)
    args = parser.parse_args()

    build_dataset(
        Path(args.success),
        Path(args.failure),
        Path(args.output),
        args.k_fail,
        args.action_thresh,
    )


if __name__ == "__main__":
    main()
