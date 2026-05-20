#!/usr/bin/env python3
"""Recovery-conditioned EEF correction labels (non-teleporting).

Time-index-matched labels (``compute_eef_corrections`` in ``train_eef_correction_mlp``)
resample a success demo onto the failure timeline by normalized progress
(0→1). That ``teleportation'' pairs each failure step with a success pose at the
same clock fraction, not the nearest recoverable pose ahead on the success
manifold.

Recovery-conditioned labels instead use a **monotonic forward match** on the
success EEF polyline: at failure step ``t``, pick the closest success index
``s_t >= s_{t-1}`` and set
    correction[t] = EEF_succ[s_t] - EEF_fail[t].

This is the label regime needed to re-test spatial--control decoupling
(§5.3): corrections point toward the nearest forward point on a successful
reference trajectory, not a time-synchronized ghost pose.

Usage::

    python3 scripts/recovery_conditioned_labels.py \\
        --success-pkl research_data/rollouts/openvla_spatial_seed0/success_rollouts.pkl \\
        --failure-pkl research_data/rollouts/openvla_spatial_seed0/failure_rollouts.pkl \\
        --compare-teleport --max-rollouts 50
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_eef_correction_mlp import (
    compute_eef_corrections,
    get_eef_trajectory,
    match_failure_to_success,
)


def monotonic_recovery_indices(
    fail_eef: np.ndarray,
    succ_eef: np.ndarray,
) -> np.ndarray:
    """Greedy monotonic forward indices ``s_t`` on the success polyline.

    For each failure timestep, search ``s in [s_{t-1}, T_succ-1]`` for the
    success index minimizing ``||p_fail[t] - p_succ[s]||``. Monotonicity prevents
    rewinding along the reference recovery path.
    """
    fail_eef = np.asarray(fail_eef, dtype=np.float32)
    succ_eef = np.asarray(succ_eef, dtype=np.float32)
    t_fail = len(fail_eef)
    t_succ = len(succ_eef)
    if t_fail == 0 or t_succ == 0:
        return np.zeros(t_fail, dtype=np.int64)

    s_idx = 0
    indices = np.zeros(t_fail, dtype=np.int64)
    for t in range(t_fail):
        window = succ_eef[s_idx:]
        if len(window) == 0:
            indices[t] = t_succ - 1
            continue
        rel = int(np.argmin(np.linalg.norm(window - fail_eef[t], axis=1)))
        s_idx = min(s_idx + rel, t_succ - 1)
        indices[t] = s_idx
    return indices


def compute_recovery_corrections(
    fail_rollout: dict,
    succ_rollout: dict,
) -> Optional[np.ndarray]:
    """Recovery-conditioned corrections ``(T_fail, 3)`` in meters."""
    fail_eef = get_eef_trajectory(fail_rollout)
    succ_eef = get_eef_trajectory(succ_rollout)
    if fail_eef is None or succ_eef is None:
        return None

    indices = monotonic_recovery_indices(fail_eef, succ_eef)
    target = succ_eef[indices]
    return (target - fail_eef).astype(np.float32)


def label_diagnostics(
    teleport: np.ndarray,
    recovery: np.ndarray,
) -> dict:
    """Compare two label fields on aligned failure steps."""
    n = min(len(teleport), len(recovery))
    if n == 0:
        return {}
    t = teleport[:n]
    r = recovery[:n]
    t_norm = np.linalg.norm(t, axis=1) + 1e-8
    r_norm = np.linalg.norm(r, axis=1) + 1e-8
    cos = np.sum((t / t_norm[:, None]) * (r / r_norm[:, None]), axis=1)
    return {
        "n_steps": int(n),
        "mean_teleport_cm": float(t_norm.mean() * 100),
        "mean_recovery_cm": float(r_norm.mean() * 100),
        "mean_displacement_between_labels_cm": float(np.linalg.norm(t - r, axis=1).mean() * 100),
        "median_cosine_teleport_vs_recovery": float(np.median(cos)),
    }


def build_success_index(success_rollouts: list) -> dict:
    by_task: dict = {}
    for r in success_rollouts:
        by_task.setdefault(r["task_id"], []).append(r)
    return by_task


def compare_rollout_corpus(
    success_rollouts: list,
    failure_rollouts: list,
    max_rollouts: int | None = None,
) -> dict:
    by_task = build_success_index(success_rollouts)
    per_rollout = []
    agg_cos = []
    agg_delta_cm = []

    failures = failure_rollouts[:max_rollouts] if max_rollouts else failure_rollouts
    for r in failures:
        if r.get("success"):
            continue
        match = match_failure_to_success(r, by_task)
        if match is None:
            continue
        tele = compute_eef_corrections(r, match)
        rec = compute_recovery_corrections(r, match)
        if tele is None or rec is None:
            continue
        d = label_diagnostics(tele, rec)
        d["task_id"] = int(r.get("task_id", -1))
        d["rollout_id"] = int(r.get("rollout_id", -1))
        per_rollout.append(d)
        agg_cos.append(d["median_cosine_teleport_vs_recovery"])
        agg_delta_cm.append(d["mean_displacement_between_labels_cm"])

    if not per_rollout:
        return {"n_rollouts": 0}

    return {
        "n_rollouts": len(per_rollout),
        "pooled_median_cosine": float(np.median(agg_cos)),
        "pooled_mean_label_displacement_cm": float(np.mean(agg_delta_cm)),
        "per_rollout": per_rollout,
    }


def main():
    ap = argparse.ArgumentParser(description="Recovery-conditioned label utilities")
    ap.add_argument("--success-pkl", type=Path, required=True)
    ap.add_argument("--failure-pkl", type=Path, required=True)
    ap.add_argument("--compare-teleport", action="store_true")
    ap.add_argument("--max-rollouts", type=int, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    with args.success_pkl.open("rb") as f:
        succ = pickle.load(f)
    with args.failure_pkl.open("rb") as f:
        fail = pickle.load(f)

    if not args.compare_teleport:
        print("Nothing to do; pass --compare-teleport")
        return

    summary = compare_rollout_corpus(succ, fail, max_rollouts=args.max_rollouts)
    print(f"Compared {summary.get('n_rollouts', 0)} failure rollouts")
    if summary.get("n_rollouts", 0) > 0:
        print(f"  pooled median cosine(teleport, recovery): {summary['pooled_median_cosine']:.3f}")
        print(f"  pooled mean ||teleport - recovery||: {summary['pooled_mean_label_displacement_cm']:.2f} cm")

    if args.out_json:
        import json
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
