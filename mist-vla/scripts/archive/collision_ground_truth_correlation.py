#!/usr/bin/env python3
"""
Ground Truth Validation: Do our z-score deviation labels correlate with
actual physical collision data?

For every failure rollout that has collision_pos/collision_normal:
  1. Get the collision normal vector (which physical direction the collision pushes)
  2. Get our deviation labels at/near the collision step
  3. Correlate: are the dimensions we flag as "deviating" the same ones
     that the collision normal is large in?

If YES → we are detecting "Spatial Failure Direction" (strong paper claim)
If NO  → we are detecting "Policy Drift" (still valuable, different framing)
"""

import pickle
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import json

DIM_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]
# Collision normal is 3D (x,y,z), action deviation is 7D (x,y,z,roll,pitch,yaw,gripper)
# We can only correlate the translational dimensions (0:3) with collision normal


def build_success_baselines(succ_rollouts, n_bins=10):
    baselines = {}
    for task_id in range(10):
        task_succ = [r for r in succ_rollouts if r["task_id"] == task_id]
        if not task_succ:
            continue
        for bin_idx in range(n_bins):
            actions_in_bin = []
            for r in task_succ:
                acts = np.array(r["actions"])
                T = len(acts)
                t_start = int(bin_idx / n_bins * T)
                t_end = int((bin_idx + 1) / n_bins * T)
                if t_end > t_start:
                    actions_in_bin.append(acts[t_start:t_end])
            if actions_in_bin:
                all_acts = np.concatenate(actions_in_bin, axis=0)
                baselines[(task_id, bin_idx)] = (all_acts.mean(axis=0), all_acts.std(axis=0) + 1e-6)
    return baselines


def main():
    print("Loading data...")
    with open("data/combined/success_rollouts.pkl", "rb") as f:
        succ = pickle.load(f)
    with open("data/combined/failure_rollouts.pkl", "rb") as f:
        fail = pickle.load(f)
    print(f"  {len(succ)}S + {len(fail)}F")

    baselines = build_success_baselines(succ)

    # ── 1. Inventory of collision data ────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. COLLISION DATA INVENTORY")
    print("=" * 70)

    n_with_collision = sum(1 for r in fail if r.get("collision_occurred", False))
    n_with_steps = sum(1 for r in fail if r.get("collision_step") is not None)
    n_with_normals = 0
    n_with_positions = 0

    for r in fail:
        steps = r.get("steps", [])
        for s in steps:
            if s.get("collision") and s.get("collision_normal") is not None:
                n_with_normals += 1
                break
        for s in steps:
            if s.get("collision") and s.get("collision_pos") is not None:
                n_with_positions += 1
                break

    print(f"  Failure rollouts with collision_occurred=True: {n_with_collision}/{len(fail)}")
    print(f"  Failure rollouts with collision_step != None:  {n_with_steps}/{len(fail)}")
    print(f"  Failure rollouts with collision_normal data:   {n_with_normals}/{len(fail)}")
    print(f"  Failure rollouts with collision_pos data:      {n_with_positions}/{len(fail)}")

    # ── 2. Analyze collision normals vs deviation labels ──────────────────
    print("\n" + "=" * 70)
    print("2. COLLISION NORMAL vs DEVIATION LABEL CORRELATION")
    print("=" * 70)

    correlations = []
    collision_dim_match = defaultdict(int)
    collision_dim_total = defaultdict(int)
    collision_normals_all = []
    deviation_at_collision = []

    for r in fail:
        tid = r["task_id"]
        steps = r.get("steps", [])
        acts = np.array(r["actions"])
        T = len(acts)

        for si, s in enumerate(steps):
            if not s.get("collision"):
                continue
            normal = s.get("collision_normal")
            if normal is None:
                continue
            normal = np.array(normal)
            if np.linalg.norm(normal) < 1e-8:
                continue

            # Get deviation at this step
            progress = si / max(T - 1, 1)
            bin_idx = min(int(progress * 10), 9)
            key = (tid, bin_idx)
            if key not in baselines:
                continue

            mean, std = baselines[key]
            if si < len(acts):
                deviation = (acts[si] - mean) / std
            else:
                continue

            # Collision normal is 3D (x,y,z)
            # Deviation is 7D: first 3 dims are translational (x,y,z)
            norm_abs = np.abs(normal)
            dev_abs_xyz = np.abs(deviation[:3])

            # Which collision dimension has largest normal component?
            collision_dominant_dim = np.argmax(norm_abs)
            # Which translational deviation dimension is largest?
            deviation_dominant_dim = np.argmax(dev_abs_xyz)

            collision_normals_all.append(normal)
            deviation_at_collision.append(deviation[:3])

            # Do they match?
            if collision_dominant_dim == deviation_dominant_dim:
                collision_dim_match["match"] += 1
            else:
                collision_dim_match["mismatch"] += 1

            for dim in range(3):
                dim_name = DIM_NAMES[dim]
                # Is this dim "deviating" (>1.5σ)?
                is_deviating = abs(deviation[dim]) > 1.5
                # Is this dim significant in collision normal?
                is_collision_dim = norm_abs[dim] > 0.3 * norm_abs.max()

                collision_dim_total[dim_name] += 1
                if is_deviating and is_collision_dim:
                    collision_dim_match[f"{dim_name}_true_pos"] += 1
                elif not is_deviating and not is_collision_dim:
                    collision_dim_match[f"{dim_name}_true_neg"] += 1
                elif is_deviating and not is_collision_dim:
                    collision_dim_match[f"{dim_name}_false_pos"] += 1
                else:
                    collision_dim_match[f"{dim_name}_false_neg"] += 1

            # Cosine similarity between collision normal and deviation direction (xyz only)
            dev_xyz = deviation[:3]
            if np.linalg.norm(dev_xyz) > 1e-8:
                cos_sim = np.dot(normal, dev_xyz) / (np.linalg.norm(normal) * np.linalg.norm(dev_xyz))
                correlations.append(cos_sim)

    if correlations:
        correlations = np.array(correlations)
        print(f"\n  Collision steps analyzed: {len(correlations)}")
        print(f"\n  Cosine similarity between collision_normal and deviation_direction (XYZ):")
        print(f"    Mean: {correlations.mean():.4f}")
        print(f"    Std:  {correlations.std():.4f}")
        print(f"    Median: {np.median(correlations):.4f}")
        print(f"    |cos| > 0.5 (aligned): {(np.abs(correlations) > 0.5).mean()*100:.1f}%")
        print(f"    |cos| > 0.7 (strongly aligned): {(np.abs(correlations) > 0.7).mean()*100:.1f}%")
        print(f"    cos > 0 (same direction): {(correlations > 0).mean()*100:.1f}%")

        total = collision_dim_match.get("match", 0) + collision_dim_match.get("mismatch", 0)
        if total > 0:
            match_rate = collision_dim_match["match"] / total
            print(f"\n  Dominant dimension match rate: {match_rate:.1%} ({collision_dim_match['match']}/{total})")
            print(f"    (Random chance = 33.3%)")

        print(f"\n  Per-dimension agreement (deviation label ↔ collision normal):")
        print(f"    {'Dim':>6} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'Prec':>6} {'Recall':>6} {'F1':>6}")
        print(f"    {'-'*48}")
        for dim in ["X", "Y", "Z"]:
            tp = collision_dim_match.get(f"{dim}_true_pos", 0)
            tn = collision_dim_match.get(f"{dim}_true_neg", 0)
            fp = collision_dim_match.get(f"{dim}_false_pos", 0)
            fn = collision_dim_match.get(f"{dim}_false_neg", 0)
            prec = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * prec * recall / max(prec + recall, 1e-8)
            print(f"    {dim:>6} {tp:>6} {tn:>6} {fp:>6} {fn:>6} {prec:>6.3f} {recall:>6.3f} {f1:>6.3f}")
    else:
        print("\n  ⚠ NO collision steps with valid collision_normal found in the data.")
        print("  This means we CANNOT validate against ground-truth collision geometry.")
        print("  The correct framing is 'Directional Policy Drift Prediction' not 'Spatial Failure'.")

    # ── 3. Collision occurrence statistics ─────────────────────────────────
    print("\n" + "=" * 70)
    print("3. COLLISION OCCURRENCE STATISTICS")
    print("=" * 70)

    collision_tasks = Counter()
    collision_step_positions = []
    total_collision_steps = 0

    for r in fail:
        if r.get("collision_occurred"):
            collision_tasks[r["task_id"]] += 1
        cs = r.get("collision_steps", 0)
        total_collision_steps += cs
        if r.get("collision_step") is not None:
            T = len(r["actions"])
            collision_step_positions.append(r["collision_step"] / max(T, 1))

    print(f"  Total collision steps across all failure rollouts: {total_collision_steps}")
    print(f"  Failure rollouts per task with collisions:")
    for t in range(10):
        n_fail_task = sum(1 for r in fail if r["task_id"] == t)
        n_coll_task = collision_tasks.get(t, 0)
        print(f"    Task {t}: {n_coll_task}/{n_fail_task} failures have collisions "
              f"({n_coll_task/max(n_fail_task,1)*100:.0f}%)")

    if collision_step_positions:
        positions = np.array(collision_step_positions)
        print(f"\n  First collision position in episode (as % of total steps):")
        print(f"    Mean: {positions.mean()*100:.1f}%")
        print(f"    Median: {np.median(positions)*100:.1f}%")
        print(f"    Std: {positions.std()*100:.1f}%")

    # ── 4. Success sample contamination check ─────────────────────────────
    print("\n" + "=" * 70)
    print("4. SUCCESS SAMPLE CONTAMINATION CHECK")
    print("=" * 70)
    print("  What fraction of SUCCESS steps exceed the 1.5σ deviation threshold?")
    print("  (High contamination = threshold too loose)")

    succ_dev_rates = defaultdict(int)
    succ_total = 0

    for r in succ:
        tid = r["task_id"]
        acts = np.array(r["actions"])
        T = len(acts)
        for t in range(T):
            progress = t / max(T - 1, 1)
            bin_idx = min(int(progress * 10), 9)
            key = (tid, bin_idx)
            if key not in baselines:
                continue
            mean, std = baselines[key]
            deviation = (acts[t] - mean) / std
            is_dev = np.abs(deviation) > 1.5
            succ_total += 1
            for i in range(7):
                if is_dev[i]:
                    succ_dev_rates[DIM_NAMES[i]] += 1

    print(f"\n  Total success steps analyzed: {succ_total}")
    print(f"  {'Dim':>8} {'% Deviating':>12} {'Count':>8} {'Verdict':>12}")
    print(f"  {'-'*44}")
    for d in DIM_NAMES:
        rate = succ_dev_rates.get(d, 0) / max(succ_total, 1) * 100
        count = succ_dev_rates.get(d, 0)
        verdict = "⚠ HIGH" if rate > 20 else ("OK" if rate < 15 else "BORDERLINE")
        print(f"  {d:>8} {rate:>11.1f}% {count:>8} {verdict:>12}")

    # ── Save report ───────────────────────────────────────────────────────
    save_dir = Path("analysis/collision_correlation")
    save_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "n_failure_rollouts": len(fail),
        "n_with_collision": n_with_collision,
        "n_with_collision_normal": n_with_normals,
        "n_collision_steps_analyzed": len(correlations) if correlations else 0,
        "cosine_similarity": {
            "mean": float(correlations.mean()) if len(correlations) > 0 else None,
            "std": float(correlations.std()) if len(correlations) > 0 else None,
            "aligned_50pct": float((np.abs(correlations) > 0.5).mean()) if len(correlations) > 0 else None,
        } if correlations else None,
        "success_contamination": {d: succ_dev_rates.get(d, 0) / max(succ_total, 1) for d in DIM_NAMES},
        "framing_recommendation": "spatial_failure" if (correlations and len(correlations) > 10 and np.mean(np.abs(correlations)) > 0.4) else "policy_drift",
    }
    with open(save_dir / "collision_correlation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {save_dir}/collision_correlation_report.json")
    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
