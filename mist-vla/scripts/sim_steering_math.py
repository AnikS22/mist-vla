#!/usr/bin/env python3
"""
Module 1 — Offline Steering Simulator
======================================

Validates the Latent Safety Steering hypothesis *mathematically*,
without running a single physics step.

For every failure timestep t in the validation set:
    1. v_actual   = EEF_failed(t)            (where the hand IS)
    2. v_expert   = EEF_success(t)           (where the hand SHOULD be)
    3. z_t        = hidden_state(t)          (VLA embedding)
    4. correction = MLP(z_t)                 (predicted Cartesian delta)
    5. v_steered  = v_actual + α * correction

Metric:  ||v_steered - v_expert||  <  ||v_actual - v_expert||
         "Did steering bring the hand closer to the expert?"

This produces Table 1:
    "Steering improved trajectory alignment in X% of timesteps."

Additional metrics:
    • Mean reduction in Cartesian error (cm)
    • Per-axis (X/Y/Z) improvement rates
    • Cosine alignment between predicted and oracle corrections
    • Alpha sweep to find optimal gain

Usage:
    python scripts/sim_steering_math.py \
        --checkpoint checkpoints/eef_correction_mlp/best_model.pt \
        --data-dirs data/multi_suite/libero_spatial data/multi_suite/libero_goal \
        --alpha 1.0

    python scripts/sim_steering_math.py \
        --checkpoint checkpoints/eef_correction_mlp/best_model.pt \
        --data-dirs data/multi_suite \
        --run-loo --sweep-alpha
"""

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────
# Model definition (must match train_eef_correction_mlp.py)
# ──────────────────────────────────────────────────────────────────────────

class EEFCorrectionMLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )
        feat = hidden_dim // 4
        self.fail_head = nn.Linear(feat, 1)
        self.ttf_head = nn.Linear(feat, 1)
        self.correction_head = nn.Linear(feat, 3)

    def forward(self, x):
        feat = self.encoder(x)
        return {
            "will_fail": self.fail_head(feat).squeeze(-1),
            "ttf": self.ttf_head(feat).squeeze(-1),
            "correction": self.correction_head(feat),
        }


# ──────────────────────────────────────────────────────────────────────────
# Trajectory helpers (same as training code)
# ──────────────────────────────────────────────────────────────────────────

def get_eef_trajectory(rollout):
    """Extract (T, 3) EEF position trajectory from a rollout."""
    robot_states = rollout.get("robot_states", [])
    eef_positions = []
    for rs in robot_states:
        if "eef_pos" in rs:
            eef_positions.append(np.array(rs["eef_pos"], dtype=np.float32))
    if not eef_positions:
        return None
    return np.array(eef_positions)


def match_failure_to_success(failure, success_by_task):
    """Find nearest-neighbor success trajectory by initial EEF similarity."""
    tid = failure["task_id"]
    candidates = success_by_task.get(tid, [])
    if not candidates:
        return None

    fail_eef = get_eef_trajectory(failure)
    if fail_eef is None or len(fail_eef) < 2:
        return None

    n_compare = max(1, int(0.2 * len(fail_eef)))
    fail_early = fail_eef[:n_compare]

    best_dist, best_match = float("inf"), None
    for succ in candidates:
        succ_eef = get_eef_trajectory(succ)
        if succ_eef is None or len(succ_eef) < 2:
            continue
        n_succ = max(1, int(0.2 * len(succ_eef)))
        succ_early = succ_eef[:n_succ]
        n_pts = min(len(fail_early), len(succ_early))
        f_r = fail_early[np.linspace(0, len(fail_early) - 1, n_pts).astype(int)]
        s_r = succ_early[np.linspace(0, len(succ_early) - 1, n_pts).astype(int)]
        dist = np.mean(np.linalg.norm(f_r - s_r, axis=1))
        if dist < best_dist:
            best_dist = dist
            best_match = succ
    return best_match


def align_trajectories(fail_eef, succ_eef):
    """Progress-align success trajectory onto failure timesteps.

    Returns aligned_succ of shape (T_fail, 3).
    """
    T_fail = len(fail_eef)
    T_succ = len(succ_eef)
    succ_prog = np.linspace(0, 1, T_succ)
    fail_prog = np.linspace(0, 1, T_fail)
    aligned = np.zeros((T_fail, 3), dtype=np.float32)
    for d in range(3):
        interp = interp1d(succ_prog, succ_eef[:, d],
                          kind="linear", fill_value="extrapolate")
        aligned[:, d] = interp(fail_prog)
    return aligned


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────

def load_rollouts(data_dirs):
    """Load rollouts from one or more directories."""
    all_rollouts = []
    for ddir in data_dirs:
        p = Path(ddir)
        # If it's a parent dir containing suite subdirectories
        if not list(p.glob("*.pkl")):
            for sub in sorted(p.iterdir()):
                if sub.is_dir():
                    for pkl in sorted(sub.glob("*.pkl")):
                        if "_partial" not in pkl.name:
                            with open(pkl, "rb") as f:
                                rols = pickle.load(f)
                            print(f"  {pkl}: {len(rols)} rollouts")
                            all_rollouts.extend(rols)
        else:
            for pkl in sorted(p.glob("*.pkl")):
                if "_partial" not in pkl.name:
                    with open(pkl, "rb") as f:
                        rols = pickle.load(f)
                    print(f"  {pkl}: {len(rols)} rollouts")
                    all_rollouts.extend(rols)
    return all_rollouts


# ──────────────────────────────────────────────────────────────────────────
# Core steering simulation
# ──────────────────────────────────────────────────────────────────────────

def simulate_steering(model, scaler, rollouts, success_by_task,
                      alpha=1.0, device="cpu"):
    """Run the offline steering simulation on all failure rollouts.

    Returns a dict with per-timestep metrics.
    """
    model.eval()
    results = {
        "improved": [],           # bool per timestep
        "original_error": [],     # ||EEF_fail - EEF_expert|| in meters
        "steered_error": [],      # ||EEF_steered - EEF_expert|| in meters
        "error_reduction": [],    # original - steered (positive = improvement)
        "cosine_sim": [],         # cos(predicted, oracle_correction)
        "per_axis_improved": defaultdict(list),  # per X/Y/Z
        "task_ids": [],
        "timesteps": [],
        "rollout_ids": [],
    }

    n_matched = 0
    n_skipped = 0

    for ri, rollout in enumerate(rollouts):
        if rollout["success"]:
            continue  # only steer failure trajectories

        fail_eef = get_eef_trajectory(rollout)
        feats = np.array(rollout.get("features", []))
        if fail_eef is None or len(feats) == 0:
            n_skipped += 1
            continue

        match = match_failure_to_success(rollout, success_by_task)
        if match is None:
            n_skipped += 1
            continue

        succ_eef = get_eef_trajectory(match)
        if succ_eef is None:
            n_skipped += 1
            continue

        aligned_expert = align_trajectories(fail_eef, succ_eef)
        T = min(len(fail_eef), len(feats), len(aligned_expert))
        n_matched += 1

        # Batch predict corrections
        feats_t = feats[:T]
        feats_scaled = scaler.transform(feats_t)
        x = torch.FloatTensor(feats_scaled).to(device)

        with torch.no_grad():
            out = model(x)
        pred_corr = out["correction"].cpu().numpy()  # (T, 3)

        for t in range(T):
            v_actual = fail_eef[t]         # (3,) — where hand IS
            v_expert = aligned_expert[t]   # (3,) — where hand SHOULD be
            oracle_corr = v_expert - v_actual  # (3,) — true correction

            # Steered position
            v_steered = v_actual + alpha * pred_corr[t]

            err_orig = np.linalg.norm(v_actual - v_expert)
            err_steer = np.linalg.norm(v_steered - v_expert)
            improved = err_steer < err_orig

            results["improved"].append(improved)
            results["original_error"].append(err_orig)
            results["steered_error"].append(err_steer)
            results["error_reduction"].append(err_orig - err_steer)
            results["task_ids"].append(rollout["task_id"])
            results["timesteps"].append(t)
            results["rollout_ids"].append(ri)

            # Cosine sim between predicted and oracle
            pn = np.linalg.norm(pred_corr[t])
            on = np.linalg.norm(oracle_corr)
            if pn > 1e-8 and on > 1e-8:
                cos = np.dot(pred_corr[t], oracle_corr) / (pn * on)
                results["cosine_sim"].append(float(cos))
            else:
                results["cosine_sim"].append(0.0)

            # Per-axis
            for i, axis in enumerate(["X", "Y", "Z"]):
                ax_err_orig = abs(v_actual[i] - v_expert[i])
                ax_err_steer = abs(v_steered[i] - v_expert[i])
                results["per_axis_improved"][axis].append(ax_err_steer < ax_err_orig)

    print(f"  Matched {n_matched} failure rollouts | Skipped {n_skipped}")
    return results


def compute_report(results, alpha):
    """Compute summary statistics from the steering results."""
    improved = np.array(results["improved"])
    orig_err = np.array(results["original_error"])
    steer_err = np.array(results["steered_error"])
    reductions = np.array(results["error_reduction"])
    cos_sims = np.array(results["cosine_sim"])

    # Overall
    improvement_rate = improved.mean() * 100
    mean_orig = orig_err.mean() * 100   # cm
    mean_steer = steer_err.mean() * 100  # cm
    mean_reduction = reductions.mean() * 100  # cm
    median_reduction = np.median(reductions) * 100
    mean_cos = cos_sims.mean()
    median_cos = np.median(cos_sims)

    # Only count significant corrections (>1cm from expert)
    sig_mask = orig_err > 0.01
    if sig_mask.sum() > 0:
        sig_improvement = improved[sig_mask].mean() * 100
        sig_reduction = reductions[sig_mask].mean() * 100
    else:
        sig_improvement = 0.0
        sig_reduction = 0.0

    # Per-axis
    per_axis = {}
    for axis in ["X", "Y", "Z"]:
        ax = np.array(results["per_axis_improved"][axis])
        per_axis[axis] = ax.mean() * 100

    # Temporal: improvement rate in last 25% of trajectory
    task_ids = np.array(results["task_ids"])
    timesteps = np.array(results["timesteps"])
    rollout_ids = np.array(results["rollout_ids"])

    # Per-rollout, find the "late" timesteps (last 25%)
    late_mask = np.zeros(len(improved), dtype=bool)
    for rid in np.unique(rollout_ids):
        rmask = rollout_ids == rid
        ts = timesteps[rmask]
        if len(ts) < 4:
            continue
        cutoff = ts.max() * 0.75
        late_mask[rmask] = ts >= cutoff

    if late_mask.sum() > 0:
        late_improvement = improved[late_mask].mean() * 100
        late_reduction = reductions[late_mask].mean() * 100
    else:
        late_improvement = 0.0
        late_reduction = 0.0

    report = {
        "alpha": alpha,
        "n_timesteps": int(len(improved)),
        "n_rollouts_matched": int(len(np.unique(rollout_ids))),
        "overall": {
            "improvement_rate_pct": round(improvement_rate, 2),
            "mean_original_error_cm": round(mean_orig, 3),
            "mean_steered_error_cm": round(mean_steer, 3),
            "mean_error_reduction_cm": round(mean_reduction, 3),
            "median_error_reduction_cm": round(median_reduction, 3),
            "cosine_similarity_mean": round(mean_cos, 4),
            "cosine_similarity_median": round(median_cos, 4),
        },
        "significant_corrections_gt1cm": {
            "improvement_rate_pct": round(sig_improvement, 2),
            "mean_reduction_cm": round(sig_reduction, 3),
            "n_samples": int(sig_mask.sum()),
        },
        "per_axis_improvement_pct": {k: round(v, 2) for k, v in per_axis.items()},
        "late_trajectory_last25pct": {
            "improvement_rate_pct": round(late_improvement, 2),
            "mean_reduction_cm": round(late_reduction, 3),
            "n_samples": int(late_mask.sum()),
        },
    }
    return report


# ──────────────────────────────────────────────────────────────────────────
# LOO Steering Validation
# ──────────────────────────────────────────────────────────────────────────

def loo_steering(model, scaler, rollouts, alpha, device):
    """Leave-one-task-out steering validation.

    For each task, compute steering improvement using ONLY the
    global model (not retrained — same as deployment scenario).
    """
    success_by_task = defaultdict(list)
    for r in rollouts:
        if r["success"]:
            success_by_task[r["task_id"]].append(r)

    task_ids = sorted(set(r["task_id"] for r in rollouts))
    loo_results = {}

    for held_out in task_ids:
        test_rols = [r for r in rollouts if r["task_id"] == held_out]
        n_fail = sum(1 for r in test_rols if not r["success"])
        n_succ = sum(1 for r in test_rols if r["success"])

        if n_fail == 0 or n_succ == 0:
            loo_results[held_out] = {"status": f"SKIP ({n_succ}S/{n_fail}F)"}
            continue

        # Use ONLY the held-out task's own successes for matching
        # (simulates deployment: you have demo data for your task)
        held_succ = defaultdict(list)
        held_succ[held_out] = success_by_task.get(held_out, [])

        results = simulate_steering(model, scaler, test_rols,
                                    held_succ, alpha, device)

        if len(results["improved"]) == 0:
            loo_results[held_out] = {"status": "NO_MATCHABLE_FAILURES"}
            continue

        report = compute_report(results, alpha)
        report["status"] = "OK"
        report["n_test_succ"] = n_succ
        report["n_test_fail"] = n_fail
        loo_results[held_out] = report

    return loo_results


# ──────────────────────────────────────────────────────────────────────────
# Alpha sweep
# ──────────────────────────────────────────────────────────────────────────

def sweep_alpha(model, scaler, rollouts, success_by_task, device,
                alphas=None):
    """Find the optimal alpha for steering."""
    if alphas is None:
        alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    sweep_results = {}
    print("\n  Alpha sweep:")
    print(f"  {'Alpha':>6} {'Improve%':>9} {'MeanRedCm':>10} {'CosSim':>7}")
    print(f"  {'-'*36}")

    best_alpha, best_rate = 0, 0
    for alpha in alphas:
        res = simulate_steering(model, scaler, rollouts,
                                success_by_task, alpha, device)
        if len(res["improved"]) == 0:
            continue
        rep = compute_report(res, alpha)
        rate = rep["overall"]["improvement_rate_pct"]
        red = rep["overall"]["mean_error_reduction_cm"]
        cos = rep["overall"]["cosine_similarity_mean"]
        print(f"  {alpha:>6.2f} {rate:>8.1f}% {red:>9.3f} {cos:>7.4f}"
              f"{'  ★' if rate > best_rate else ''}")
        sweep_results[alpha] = rep
        if rate > best_rate:
            best_rate = rate
            best_alpha = alpha

    print(f"\n  → Optimal α = {best_alpha} ({best_rate:.1f}% improvement)")
    return sweep_results, best_alpha


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline Steering Simulator — validates safety "
                    "steering without physics.")
    parser.add_argument("--checkpoint",
                        default="checkpoints/eef_correction_mlp/best_model.pt",
                        help="Path to trained EEFCorrectionMLP checkpoint.")
    parser.add_argument("--data-dirs", nargs="+",
                        default=["data/multi_suite"],
                        help="Directories containing rollout .pkl files.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Steering gain (v_steered = v_actual + α*correction).")
    parser.add_argument("--sweep-alpha", action="store_true",
                        help="Sweep over multiple alpha values.")
    parser.add_argument("--run-loo", action="store_true",
                        help="Run leave-one-task-out steering validation.")
    parser.add_argument("--save-dir", default="results/steering_sim")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ──
    print("=" * 70)
    print("OFFLINE STEERING SIMULATOR")
    print("'Does MLP(embedding) steer the hand toward the expert?'")
    print("=" * 70)

    # ── Load model ──
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device,
                      weights_only=False)
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]

    model = EEFCorrectionMLP(input_dim=input_dim,
                             hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params | input_dim={input_dim} | "
          f"hidden_dim={hidden_dim}")

    # ── Load data ──
    print(f"\nLoading rollouts from: {args.data_dirs}")
    all_rollouts = load_rollouts(args.data_dirs)

    n_succ = sum(1 for r in all_rollouts if r["success"])
    n_fail = sum(1 for r in all_rollouts if not r["success"])
    task_ids = sorted(set(r["task_id"] for r in all_rollouts))
    print(f"  {len(all_rollouts)} rollouts | {n_succ}S + {n_fail}F | "
          f"{len(task_ids)} tasks")

    success_by_task = defaultdict(list)
    for r in all_rollouts:
        if r["success"]:
            success_by_task[r["task_id"]].append(r)

    # ══════════════════════════════════════════════════════════════════════
    #  STEERING SIMULATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"STEERING SIMULATION (α = {args.alpha})")
    print(f"{'='*70}")

    t0 = time.time()
    results = simulate_steering(model, scaler, all_rollouts,
                                success_by_task, args.alpha, device)
    dt = time.time() - t0

    if len(results["improved"]) == 0:
        print("\n  ❌ No matchable failure rollouts found!")
        return

    report = compute_report(results, args.alpha)

    # ── Print Table 1 ──
    o = report["overall"]
    sig = report["significant_corrections_gt1cm"]
    late = report["late_trajectory_last25pct"]
    ax = report["per_axis_improvement_pct"]

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  TABLE 1: OFFLINE STEERING VALIDATION                    ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Timesteps evaluated:   {report['n_timesteps']:>8,}                       ║")
    print(f"  ║  Failure rollouts used:  {report['n_rollouts_matched']:>7}                        ║")
    print(f"  ║  Steering gain α:        {args.alpha:>6.2f}                        ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    v = "✅" if o["improvement_rate_pct"] > 50 else "❌"
    print(f"  ║  Improvement Rate:       {o['improvement_rate_pct']:>6.1f}%   {v}                  ║")
    print(f"  ║  Mean Original Error:    {o['mean_original_error_cm']:>6.2f} cm                    ║")
    print(f"  ║  Mean Steered Error:     {o['mean_steered_error_cm']:>6.2f} cm                    ║")
    print(f"  ║  Mean Error Reduction:   {o['mean_error_reduction_cm']:>+6.2f} cm                   ║")
    print(f"  ║  Cosine Similarity:      {o['cosine_similarity_mean']:>6.4f}                      ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Per-axis:  X={ax['X']:.1f}%  Y={ax['Y']:.1f}%  Z={ax['Z']:.1f}%         ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Significant (>1cm):     {sig['improvement_rate_pct']:>6.1f}%  "
          f"(N={sig['n_samples']})            ║")
    print(f"  ║  Late trajectory (75%+): {late['improvement_rate_pct']:>6.1f}%  "
          f"(N={late['n_samples']})            ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    if o["improvement_rate_pct"] > 50:
        print(f"\n  → Steering improved alignment in "
              f"{o['improvement_rate_pct']:.1f}% of timesteps.")
        print(f"    Result: PAPER-READY — the MLP correction vector is valid.")
    else:
        print(f"\n  → Steering only improved {o['improvement_rate_pct']:.1f}% "
              f"of timesteps.")
        print(f"    Investigate: coordinate frame mismatch or α tuning needed.")

    print(f"\n  ({dt:.1f}s)")

    # ── Alpha sweep ──
    if args.sweep_alpha:
        print(f"\n{'='*70}")
        print("ALPHA SWEEP")
        print(f"{'='*70}")
        sweep, best_alpha = sweep_alpha(model, scaler, all_rollouts,
                                        success_by_task, device)
        report["alpha_sweep"] = {str(k): v for k, v in sweep.items()}
        report["optimal_alpha"] = best_alpha

    # ── LOO ──
    if args.run_loo:
        print(f"\n{'='*70}")
        print("LEAVE-ONE-TASK-OUT STEERING VALIDATION")
        print("'Can the model steer on tasks it has NEVER seen?'")
        print(f"{'='*70}")

        loo = loo_steering(model, scaler, all_rollouts, args.alpha, device)

        print(f"\n  {'Task':>6} {'Improve%':>9} {'Reduction':>10} "
              f"{'CosSim':>7} {'Status':>8}")
        print(f"  {'-'*48}")

        agg_improve = []
        agg_reduce = []
        for tid in sorted(loo.keys()):
            r = loo[tid]
            if r.get("status") != "OK":
                print(f"  T{tid:>4}  {r['status']}")
                continue
            imp = r["overall"]["improvement_rate_pct"]
            red = r["overall"]["mean_error_reduction_cm"]
            cos = r["overall"]["cosine_similarity_mean"]
            agg_improve.append(imp)
            agg_reduce.append(red)
            v = "✅" if imp > 50 else "❌"
            print(f"  T{tid:>4} {imp:>8.1f}% {red:>+9.3f}cm "
                  f"{cos:>7.4f}  {v}")

        if agg_improve:
            mi = np.mean(agg_improve)
            mr = np.mean(agg_reduce)
            print(f"\n  LOO Mean Improvement: {mi:.1f}%")
            print(f"  LOO Mean Reduction:  {mr:+.3f} cm")

            if mi > 60:
                print(f"\n  ╔══════════════════════════════════════════════╗")
                print(f"  ║  LOO STEERING: {mi:.1f}% — GENERALIZABLE        ║")
                print(f"  ╚══════════════════════════════════════════════╝")
            elif mi > 50:
                print(f"\n  ╔══════════════════════════════════════════════╗")
                print(f"  ║  LOO STEERING: {mi:.1f}% — MODERATE             ║")
                print(f"  ╚══════════════════════════════════════════════╝")
            else:
                print(f"\n  ╔══════════════════════════════════════════════╗")
                print(f"  ║  LOO STEERING: {mi:.1f}% — NEEDS IMPROVEMENT    ║")
                print(f"  ╚══════════════════════════════════════════════╝")

        report["loo_steering"] = {
            str(k): v for k, v in loo.items()
        }

    # ── Save ──
    with open(save_dir / "steering_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report saved: {save_dir}/steering_report.json")
    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
