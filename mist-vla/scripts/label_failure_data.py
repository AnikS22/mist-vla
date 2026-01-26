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


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _rotate_world_to_eef(world_vec: np.ndarray, eef_quat: np.ndarray) -> np.ndarray:
    # Assumes eef_quat in (x, y, z, w) format.
    q = eef_quat.astype(np.float32)
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)
    v = np.array([world_vec[0], world_vec[1], world_vec[2], 0.0], dtype=np.float32)
    rotated = _quat_mul(_quat_mul(q_conj, v), q)
    return rotated[:3]


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
    normal_thresh: float,
    action_thresh_trans: float,
    action_thresh_rot: float,
    action_thresh_grip: float,
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

        # Per-dimension risk (geometry-guided heuristic):
        # Use collision normal (world -> EEF frame) to assign translational axis.
        # Use action magnitudes for rotation + gripper.
        action = np.array(step.get("action", np.zeros(7)), dtype=np.float32)
        per_dim_risk = np.zeros_like(action, dtype=np.int32)

        collision_normal = step.get("collision_normal")
        robot_state = step.get("robot_state")
        if collision_normal is not None and robot_state and "eef_quat" in robot_state:
            world_n = np.array(collision_normal, dtype=np.float32)
            eef_quat = np.array(robot_state["eef_quat"], dtype=np.float32)
            eef_n = _rotate_world_to_eef(world_n, eef_quat)
            for axis in range(3):
                if abs(eef_n[axis]) >= normal_thresh:
                    per_dim_risk[axis] = 1

        # Fall back to action magnitude for translation if no collision normal available
        if per_dim_risk[:3].sum() == 0:
            per_dim_risk[:3] = (np.abs(action[:3]) >= action_thresh_trans).astype(np.int32)

        rot_mask = (np.abs(action[3:6]) >= action_thresh_rot).astype(np.int32)
        grip_mask = 1 if abs(action[6]) >= action_thresh_grip else 0
        per_dim_risk[3:6] = np.maximum(per_dim_risk[3:6], rot_mask)
        per_dim_risk[6] = max(per_dim_risk[6], grip_mask)

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
    parser.add_argument("--action-thresh-trans", type=float, default=0.05)
    parser.add_argument("--action-thresh-rot", type=float, default=0.2)
    parser.add_argument("--action-thresh-grip", type=float, default=0.2)
    parser.add_argument("--normal-thresh", type=float, default=0.3)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    trajectories, metadata = _load_rollouts(in_path)
    for traj in trajectories:
        _label_rollout(
            traj,
            args.k_fail,
            args.k_collision,
            args.action_thresh,
            args.normal_thresh,
            args.action_thresh_trans,
            args.action_thresh_rot,
            args.action_thresh_grip,
        )

    metadata = dict(metadata)
    metadata["labeling"] = {
        "k_fail": args.k_fail,
        "k_collision": args.k_collision,
        "action_thresh": args.action_thresh,
        "normal_thresh": args.normal_thresh,
        "action_thresh_trans": args.action_thresh_trans,
        "action_thresh_rot": args.action_thresh_rot,
        "action_thresh_grip": args.action_thresh_grip,
    }
    _save_rollouts(out_path, trajectories, metadata)

    print(f"Labeled {len(trajectories)} trajectories -> {out_path}")


if __name__ == "__main__":
    main()
