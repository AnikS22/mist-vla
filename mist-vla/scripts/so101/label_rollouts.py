#!/usr/bin/env python3
"""Interactive blind labeling of SO-101 rollouts.

Loads `unlabeled_rollouts.pkl`, plays back each episode as a wrist-cam video,
asks the user to label success (s) / failure (f) / discard (d), and splits the
labeled set into success_rollouts.pkl + failure_rollouts.pkl (sim-schema match).

The wrist camera frames aren't stored in the rollout (only hidden states + actions),
so the playback shows EEF trajectory + joint motion as a quick proxy. For visual
playback you can pass --frames-pkl pointing to a separately stored frame file
(see collect_rollouts.py extras).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def text_summary(r: dict) -> str:
    n = len(r.get("actions", []))
    eef_traj = np.stack([s["eef_pos"] for s in r.get("robot_states", []) if "eef_pos" in s]) if r.get("robot_states") else None
    rng = None
    if eef_traj is not None and len(eef_traj) > 1:
        rng = eef_traj.max(0) - eef_traj.min(0)
    last_qpos = r["robot_states"][-1]["qpos"] if r.get("robot_states") else None
    return (
        f"  steps={n}  "
        f"eef_range_m={None if rng is None else np.round(rng, 3).tolist()}  "
        f"final_qpos={None if last_qpos is None else np.round(last_qpos, 2).tolist()}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args()
    run_dir: Path = args.run_dir

    src = run_dir / "unlabeled_rollouts.pkl"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    with src.open("rb") as f:
        rollouts = pickle.load(f)
    print(f"Loaded {len(rollouts)} unlabeled rollouts from {src}")

    successes, failures, dropped = [], [], []
    for i, r in enumerate(rollouts):
        print(f"\n--- episode {i+1}/{len(rollouts)} ---")
        print(text_summary(r))
        while True:
            ans = input("Label  [s]uccess / [f]ailure / [d]iscard / [q]uit > ").strip().lower()
            if ans in ("s", "f", "d", "q"):
                break
        if ans == "q":
            print("aborted; partial labels not written.")
            return
        if ans == "s":
            r["success"] = True
            successes.append(r)
        elif ans == "f":
            r["success"] = False
            failures.append(r)
        else:
            dropped.append(r)

    print(f"\nSummary: {len(successes)} success, {len(failures)} failure, {len(dropped)} dropped")

    if successes:
        with (run_dir / "success_rollouts.pkl").open("wb") as f:
            pickle.dump(successes, f)
        print(f"wrote {run_dir / 'success_rollouts.pkl'}")
    if failures:
        with (run_dir / "failure_rollouts.pkl").open("wb") as f:
            pickle.dump(failures, f)
        print(f"wrote {run_dir / 'failure_rollouts.pkl'}")
    print(f"\nNext: python scripts/so101/train_probe.py --data-dir {run_dir}")


if __name__ == "__main__":
    main()
