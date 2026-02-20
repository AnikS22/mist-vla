#!/usr/bin/env python3
"""
Correction Vector MLP — Predicts the EXACT correction needed to fix a failing trajectory.

Instead of comparing to statistical distributions (z-score), we pair each failure
trajectory with its nearest-neighbor success trajectory (same task) and compute:

    correction[t] = action_success[t] - action_failed[t]

This vector literally says: "You went left, you should have gone right."

Input:  hidden_state (4096,)  — internal VLA vector at step t
Output: {
    "will_fail":    scalar sigmoid  (0=safe, 1=failing),
    "ttf":          scalar          (normalized steps until failure, 0-1),
    "correction":   (7,)            (the correction vector to apply per action dim),
}

For success steps, correction is zero (no fix needed).
For failure steps, correction tells you exactly what to change.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from collections import defaultdict
from pathlib import Path
from scipy.interpolate import interp1d
import argparse
import json
import time

DIM_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]
CORRECTION_THRESHOLD = 0.02  # If |correction| > this → "needs correction"


# ─── Trajectory Matching & Alignment ──────────────────────────────────────

def match_failure_to_success(failure_rollout, success_rollouts_by_task):
    """Find the nearest-neighbor success trajectory for a failure trajectory.

    Matching criterion: smallest L2 distance between the first 20% of actions.
    This ensures we match trajectories that started similarly but diverged.
    """
    tid = failure_rollout["task_id"]
    candidates = success_rollouts_by_task.get(tid, [])
    if not candidates:
        return None

    fail_acts = np.array(failure_rollout["actions"])
    n_compare = max(1, int(0.2 * len(fail_acts)))  # First 20% of steps
    fail_early = fail_acts[:n_compare]

    best_dist = float("inf")
    best_match = None

    for succ in candidates:
        succ_acts = np.array(succ["actions"])
        n_succ_compare = max(1, int(0.2 * len(succ_acts)))
        succ_early = succ_acts[:n_succ_compare]

        # Resample to same length for comparison
        n_pts = min(len(fail_early), len(succ_early))
        if n_pts < 1:
            continue
        f_resamp = fail_early[np.linspace(0, len(fail_early) - 1, n_pts).astype(int)]
        s_resamp = succ_early[np.linspace(0, len(succ_early) - 1, n_pts).astype(int)]

        dist = np.mean(np.linalg.norm(f_resamp - s_resamp, axis=1))
        if dist < best_dist:
            best_dist = dist
            best_match = succ

    return best_match


def align_trajectories(fail_actions, succ_actions):
    """Align two trajectories by progress (0→1) using linear interpolation.

    Returns correction vectors at each failure timestep:
        correction[t] = interpolated_success_action[t] - failure_action[t]
    """
    T_fail = len(fail_actions)
    T_succ = len(succ_actions)

    fail_acts = np.array(fail_actions)
    succ_acts = np.array(succ_actions)

    # Progress grid for success trajectory
    succ_progress = np.linspace(0, 1, T_succ)
    # Progress grid for failure trajectory
    fail_progress = np.linspace(0, 1, T_fail)

    # Interpolate success trajectory to failure's time grid
    corrections = np.zeros_like(fail_acts)
    for dim in range(fail_acts.shape[1]):
        interpolator = interp1d(succ_progress, succ_acts[:, dim],
                                kind="linear", fill_value="extrapolate")
        succ_at_fail_time = interpolator(fail_progress)
        corrections[:, dim] = succ_at_fail_time - fail_acts[:, dim]

    return corrections


# ─── Sample Preparation ──────────────────────────────────────────────────

def prepare_samples(rollouts, success_rollouts_by_task):
    """Convert rollouts to per-step samples with correction vector labels.

    For failure steps: correction = matched_success_action - failure_action
    For success steps: correction = zeros (no fix needed)
    """
    samples = {
        "hidden_states": [],
        "labels": [],           # 1=failure, 0=success
        "corrections": [],      # (7,) correction vector
        "ttf": [],
        "task_ids": [],
        "rollout_ids": [],
    }

    n_matched = 0
    n_unmatched = 0

    for ri, r in enumerate(rollouts):
        feats = np.array(r["features"])
        acts = np.array(r["actions"])
        tid = r["task_id"]
        is_fail = not r["success"]
        T = min(len(feats), len(acts))

        if is_fail:
            # Find matched success trajectory
            match = match_failure_to_success(r, success_rollouts_by_task)
            if match is not None:
                corrections = align_trajectories(acts[:T], np.array(match["actions"]))
                n_matched += 1
            else:
                # No match found — skip this rollout entirely
                n_unmatched += 1
                continue
        else:
            # Success: correction is zero
            corrections = np.zeros((T, 7), dtype=np.float32)

        for t in range(T):
            ttf = (T - 1 - t) / max(T - 1, 1) if is_fail else 1.0

            samples["hidden_states"].append(feats[t])
            samples["labels"].append(1.0 if is_fail else 0.0)
            samples["corrections"].append(corrections[t].astype(np.float32))
            samples["ttf"].append(ttf)
            samples["task_ids"].append(tid)
            samples["rollout_ids"].append(ri)

    print(f"    Matched: {n_matched} failure rollouts, Unmatched (skipped): {n_unmatched}")
    return {k: np.array(v) for k, v in samples.items()}


# ─── Model ────────────────────────────────────────────────────────────────

class CorrectionMLP(nn.Module):
    """
    Predicts correction vectors from VLA hidden states.
    Single encoder → three heads:
      1. will_fail: binary logit
      2. ttf: time-to-failure (0-1)
      3. correction: (7,) continuous correction vector
    """
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
        feat = hidden_dim // 4  # 128

        self.fail_head = nn.Linear(feat, 1)
        self.ttf_head = nn.Linear(feat, 1)
        self.correction_head = nn.Sequential(
            nn.Linear(feat, 7),
            # No activation — correction can be positive or negative
        )

    def forward(self, x):
        feat = self.encoder(x)
        return {
            "will_fail": self.fail_head(feat).squeeze(-1),
            "ttf": self.ttf_head(feat).squeeze(-1),
            "correction": self.correction_head(feat),  # (B, 7)
        }


# ─── Dataset ─────────────────────────────────────────────────────────────

class CorrectionDataset(Dataset):
    def __init__(self, hidden_states, labels, corrections, ttf_targets):
        self.x = torch.FloatTensor(hidden_states)
        self.labels = torch.FloatTensor(labels)
        self.corrections = torch.FloatTensor(corrections)
        self.ttf_targets = torch.FloatTensor(ttf_targets)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.labels[idx],
                self.corrections[idx], self.ttf_targets[idx])


# ─── Training ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, fail_pos_weight):
    model.train()
    total_loss = 0
    fail_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([fail_pos_weight]).to(device)
    )

    for x, labels, corrections, ttf_targets in loader:
        x = x.to(device)
        labels = labels.to(device)
        corrections = corrections.to(device)
        ttf_targets = ttf_targets.to(device)

        out = model(x)

        # Loss 1: Binary failure detection
        loss_fail = fail_criterion(out["will_fail"], labels)

        # Loss 2: TTF regression (only on failure samples)
        fail_mask = labels > 0.5
        if fail_mask.sum() > 0:
            loss_ttf = F.mse_loss(out["ttf"][fail_mask], ttf_targets[fail_mask])
        else:
            loss_ttf = torch.tensor(0.0, device=device)

        # Loss 3: Correction vector (Smooth L1 / Huber loss)
        # On failure samples: predict the actual correction needed
        # On success samples: predict zero (no correction needed)
        loss_correction = F.smooth_l1_loss(out["correction"], corrections)

        # Combined loss
        loss = loss_fail + 0.5 * loss_ttf + 2.0 * loss_correction

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, correction_threshold=CORRECTION_THRESHOLD):
    model.eval()
    all_preds = defaultdict(list)
    all_targets = defaultdict(list)

    for x, labels, corrections, ttf_targets in loader:
        x = x.to(device)
        out = model(x)

        all_preds["fail"].extend(torch.sigmoid(out["will_fail"]).cpu().numpy())
        all_targets["fail"].extend(labels.numpy())
        all_preds["ttf"].extend(out["ttf"].cpu().numpy())
        all_targets["ttf"].extend(ttf_targets.numpy())
        all_preds["correction"].extend(out["correction"].cpu().numpy())
        all_targets["correction"].extend(corrections.numpy())

    results = {}

    # ── Failure AUC ──
    fail_preds = np.array(all_preds["fail"])
    fail_labels = np.array(all_targets["fail"])
    if len(np.unique(fail_labels)) > 1:
        results["fail_auc"] = roc_auc_score(fail_labels, fail_preds)
        results["fail_acc"] = accuracy_score(fail_labels, (fail_preds > 0.5).astype(int))
    else:
        results["fail_auc"] = 0.5
        results["fail_acc"] = 0.5

    # ── TTF ──
    ttf_preds = np.array(all_preds["ttf"])
    ttf_labels = np.array(all_targets["ttf"])
    fail_mask = fail_labels > 0.5
    if fail_mask.sum() > 10:
        results["ttf_r2"] = r2_score(ttf_labels[fail_mask], ttf_preds[fail_mask])
        results["ttf_corr"] = float(np.corrcoef(ttf_labels[fail_mask], ttf_preds[fail_mask])[0, 1])
    else:
        results["ttf_r2"] = 0.0
        results["ttf_corr"] = 0.0

    # ── Correction Vector ──
    pred_corr = np.array(all_preds["correction"])   # (N, 7)
    true_corr = np.array(all_targets["correction"])  # (N, 7)

    # Overall correction magnitude correlation (on failure samples only)
    if fail_mask.sum() > 10:
        pred_mag = np.linalg.norm(pred_corr[fail_mask], axis=1)
        true_mag = np.linalg.norm(true_corr[fail_mask], axis=1)
        results["correction_mag_corr"] = float(np.corrcoef(pred_mag, true_mag)[0, 1])
        results["correction_mag_r2"] = r2_score(true_mag, pred_mag)

        # Cosine similarity between predicted and true correction vectors (failure only)
        norms_pred = np.linalg.norm(pred_corr[fail_mask], axis=1, keepdims=True) + 1e-8
        norms_true = np.linalg.norm(true_corr[fail_mask], axis=1, keepdims=True) + 1e-8
        cos_sim = np.sum(
            (pred_corr[fail_mask] / norms_pred) * (true_corr[fail_mask] / norms_true), axis=1
        )
        results["correction_cosine_sim"] = float(np.mean(cos_sim))
        results["correction_cosine_sim_median"] = float(np.median(cos_sim))
    else:
        results["correction_mag_corr"] = 0.0
        results["correction_mag_r2"] = 0.0
        results["correction_cosine_sim"] = 0.0
        results["correction_cosine_sim_median"] = 0.0

    # Per-dimension metrics
    for i in range(7):
        d = DIM_NAMES[i]
        p = pred_corr[:, i]
        t = true_corr[:, i]

        # Regression: R² per dimension
        if fail_mask.sum() > 10:
            results[f"dim_{d}_r2"] = r2_score(t[fail_mask], p[fail_mask])
            corr_val = np.corrcoef(t[fail_mask], p[fail_mask])[0, 1]
            results[f"dim_{d}_corr"] = float(corr_val) if not np.isnan(corr_val) else 0.0

        # Binary: "needs correction in this dim?" AUC
        needs_corr = (np.abs(t) > correction_threshold).astype(int)
        pred_needs = np.abs(p)  # Use magnitude as confidence
        if len(np.unique(needs_corr)) > 1:
            results[f"dim_{d}_needs_corr_auc"] = roc_auc_score(needs_corr, pred_needs)
        else:
            results[f"dim_{d}_needs_corr_auc"] = 0.5

        # Direction: among samples needing correction, is the sign right?
        corr_mask = np.abs(t) > correction_threshold
        if corr_mask.sum() > 10:
            true_dir = (t[corr_mask] > 0).astype(int)
            pred_dir = (p[corr_mask] > 0).astype(int)
            if len(np.unique(true_dir)) > 1:
                results[f"dim_{d}_dir_acc"] = accuracy_score(true_dir, pred_dir)
                results[f"dim_{d}_dir_auc"] = roc_auc_score(
                    true_dir, p[corr_mask]  # Use raw value as confidence (positive → positive dir)
                )
            else:
                results[f"dim_{d}_dir_acc"] = 0.5
                results[f"dim_{d}_dir_auc"] = 0.5
        else:
            results[f"dim_{d}_dir_acc"] = 0.5
            results[f"dim_{d}_dir_auc"] = 0.5

    return results


# ─── Cross-task evaluation ────────────────────────────────────────────────

def cross_task_evaluate(model, all_samples, device, scaler):
    results = {}
    task_ids = all_samples["task_ids"]
    for tid in range(10):
        mask = task_ids == tid
        if mask.sum() < 50:
            continue
        x = scaler.transform(all_samples["hidden_states"][mask])
        ds = CorrectionDataset(x, all_samples["labels"][mask],
                                all_samples["corrections"][mask],
                                all_samples["ttf"][mask])
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        results[tid] = evaluate(model, loader, device)
    return results


# ─── LOO fold ─────────────────────────────────────────────────────────────

def run_loo_fold(held_out_task, all_rollouts, args, device):
    """True LOO: train on 9 tasks, test on held-out task."""
    train_rollouts = [r for r in all_rollouts if r["task_id"] != held_out_task]
    test_rollouts = [r for r in all_rollouts if r["task_id"] == held_out_task]

    n_test_succ = sum(1 for r in test_rollouts if r["success"])
    n_test_fail = sum(1 for r in test_rollouts if not r["success"])

    if n_test_succ == 0 or n_test_fail == 0:
        return None, f"SKIP ({n_test_succ}S/{n_test_fail}F)"

    # Build success lookup from TRAINING data only
    train_succ_by_task = defaultdict(list)
    for r in train_rollouts:
        if r["success"]:
            train_succ_by_task[r["task_id"]].append(r)

    # For test rollouts: also need success trajectories for matching
    # Use global success pool (from training) as the matching pool
    # This simulates deployment: we only have success data from other tasks
    test_succ_by_task = train_succ_by_task  # Held-out task uses training tasks' successes? No.

    # Actually: for the held-out task, there ARE success rollouts in test_rollouts.
    # We use THOSE for matching (they're not used for training, just for labeling).
    # This is fair: in deployment, you'd have example demonstrations.
    test_succ_by_task_actual = defaultdict(list)
    for r in test_rollouts:
        if r["success"]:
            test_succ_by_task_actual[r["task_id"]].append(r)

    # Merge: train successes for training tasks, test successes for held-out task labeling
    all_succ_for_labeling = defaultdict(list)
    for tid, rols in train_succ_by_task.items():
        all_succ_for_labeling[tid].extend(rols)
    for tid, rols in test_succ_by_task_actual.items():
        all_succ_for_labeling[tid].extend(rols)

    train_samples = prepare_samples(train_rollouts, train_succ_by_task)
    test_samples = prepare_samples(test_rollouts, all_succ_for_labeling)

    if len(np.unique(test_samples["labels"])) < 2:
        return None, "SKIP (single class in test)"

    # Rollout-level split for train/val
    rids = train_samples["rollout_ids"]
    unique_rids = np.unique(rids)
    np.random.shuffle(unique_rids)
    n_tr = int(0.85 * len(unique_rids))
    tr_rids = set(unique_rids[:n_tr])
    val_rids = set(unique_rids[n_tr:])
    tr_mask = np.array([rid in tr_rids for rid in rids])
    val_mask = np.array([rid in val_rids for rid in rids])

    scaler = StandardScaler()
    tr_x = scaler.fit_transform(train_samples["hidden_states"][tr_mask])
    val_x = scaler.transform(train_samples["hidden_states"][val_mask])
    te_x = scaler.transform(test_samples["hidden_states"])

    tr_ds = CorrectionDataset(tr_x, train_samples["labels"][tr_mask],
                               train_samples["corrections"][tr_mask],
                               train_samples["ttf"][tr_mask])
    val_ds = CorrectionDataset(val_x, train_samples["labels"][val_mask],
                                train_samples["corrections"][val_mask],
                                train_samples["ttf"][val_mask])
    te_ds = CorrectionDataset(te_x, test_samples["labels"],
                               test_samples["corrections"],
                               test_samples["ttf"])

    tr_labels = train_samples["labels"][tr_mask]
    n_pos = tr_labels.sum()
    n_neg = len(tr_labels) - n_pos
    pw = n_neg / max(n_pos, 1)
    sw = np.where(tr_labels > 0.5, n_neg / max(n_pos, 1), 1.0)
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False)

    model = CorrectionMLP(input_dim=tr_x.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = 0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, tr_loader, optimizer, device, pw)
        scheduler.step()
        if epoch % 3 == 0 or epoch == 1:
            val_res = evaluate(model, val_loader, device)
            score = 0.4 * val_res["fail_auc"] + 0.6 * val_res.get("correction_cosine_sim", 0)
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    result = evaluate(model, te_loader, device)
    result["n_test_succ"] = n_test_succ
    result["n_test_fail"] = n_test_fail
    return result, "OK"


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Correction Vector MLP Training")
    parser.add_argument("--success-data", default="data/combined/success_rollouts.pkl")
    parser.add_argument("--failure-data", default="data/combined/failure_rollouts.pkl")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--save-dir", default="checkpoints/correction_mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-loo", action="store_true", help="Run true LOO evaluation")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CORRECTION VECTOR MLP")
    print("Label = Action_success(t) - Action_failed(t)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    with open(args.success_data, "rb") as f:
        succ = pickle.load(f)
    with open(args.failure_data, "rb") as f:
        fail = pickle.load(f)
    all_rollouts = succ + fail
    print(f"  {len(succ)}S + {len(fail)}F = {len(all_rollouts)} rollouts")

    # Build success lookup by task
    succ_by_task = defaultdict(list)
    for r in succ:
        succ_by_task[r["task_id"]].append(r)

    # Prepare samples with correction vector labels
    print("\nPreparing samples (matching + aligning trajectories)...")
    all_samples = prepare_samples(all_rollouts, succ_by_task)
    print(f"  Total samples: {len(all_samples['labels'])}")
    print(f"  Failure samples: {all_samples['labels'].sum():.0f} ({all_samples['labels'].mean()*100:.1f}%)")

    # Correction vector statistics
    corr = all_samples["corrections"]
    fail_mask = all_samples["labels"] > 0.5
    print(f"\n  Correction vector stats (failure samples only):")
    for i, d in enumerate(DIM_NAMES):
        c = corr[fail_mask, i]
        pct_nonzero = (np.abs(c) > CORRECTION_THRESHOLD).mean() * 100
        print(f"    {d:>8}: mean={c.mean():+.4f} std={c.std():.4f} "
              f"|needs correction|={pct_nonzero:.1f}%")

    # ── Rollout-level split ───────────────────────────────────────────────
    print("\nSplitting by rollout...")
    rollout_ids = all_samples["rollout_ids"]
    unique_rids = np.unique(rollout_ids)
    np.random.shuffle(unique_rids)

    n_total = len(unique_rids)
    n_train = int(0.75 * n_total)
    n_val = int(0.15 * n_total)

    train_rids = set(unique_rids[:n_train])
    val_rids = set(unique_rids[n_train:n_train + n_val])
    test_rids = set(unique_rids[n_train + n_val:])

    train_mask = np.array([rid in train_rids for rid in rollout_ids])
    val_mask = np.array([rid in val_rids for rid in rollout_ids])
    test_mask = np.array([rid in test_rids for rid in rollout_ids])

    print(f"  Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    # Standardize
    scaler = StandardScaler()
    train_x = scaler.fit_transform(all_samples["hidden_states"][train_mask])
    val_x = scaler.transform(all_samples["hidden_states"][val_mask])
    test_x = scaler.transform(all_samples["hidden_states"][test_mask])

    # Normalize correction targets (helps training stability)
    corr_scale = np.abs(all_samples["corrections"][train_mask]).mean(axis=0) + 1e-6
    print(f"\n  Correction scale per dim: {[f'{s:.4f}' for s in corr_scale]}")

    train_corr = all_samples["corrections"][train_mask] / corr_scale
    val_corr = all_samples["corrections"][val_mask] / corr_scale
    test_corr = all_samples["corrections"][test_mask] / corr_scale

    train_ds = CorrectionDataset(train_x, all_samples["labels"][train_mask],
                                  train_corr, all_samples["ttf"][train_mask])
    val_ds = CorrectionDataset(val_x, all_samples["labels"][val_mask],
                                val_corr, all_samples["ttf"][val_mask])
    test_ds = CorrectionDataset(test_x, all_samples["labels"][test_mask],
                                 test_corr, all_samples["ttf"][test_mask])

    # Weighted sampler
    train_labels = all_samples["labels"][train_mask]
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    sample_weights = np.where(train_labels > 0.5, n_neg / max(n_pos, 1), 1.0)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    pw = n_neg / max(n_pos, 1)

    # Model
    input_dim = train_x.shape[1]
    model = CorrectionMLP(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    # ── Training ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_score = 0
    best_epoch = 0
    patience = 20
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, pw)
        val_res = evaluate(model, val_loader, device)
        scheduler.step()

        cos_sim = val_res.get("correction_cosine_sim", 0)
        score = 0.4 * val_res["fail_auc"] + 0.6 * max(cos_sim, 0)

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "corr_scale": corr_scale,
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "epoch": epoch,
            }, save_dir / "best_model.pt")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            dt = time.time() - t0
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                  f"fail_AUC={val_res['fail_auc']:.4f} | TTF_r²={val_res['ttf_r2']:.3f} | "
                  f"cos_sim={cos_sim:.4f} | {dt:.1f}s"
                  f"{'  ★' if epoch == best_epoch else ''}")

        if no_improve >= patience:
            print(f"\n  Early stop at epoch {epoch} (best={best_epoch})")
            break

    # ── Load best model ───────────────────────────────────────────────────
    ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Test results ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"TEST RESULTS (best epoch {best_epoch})")
    print("=" * 70)

    test_res = evaluate(model, test_loader, device)

    print(f"\n  FAILURE DETECTION:")
    print(f"    AUC:      {test_res['fail_auc']:.4f}")
    print(f"    Accuracy: {test_res['fail_acc']:.4f}")

    print(f"\n  TIME-TO-FAILURE:")
    print(f"    R²:   {test_res['ttf_r2']:.4f}")
    print(f"    Corr: {test_res['ttf_corr']:.4f}")

    print(f"\n  CORRECTION VECTOR (Global):")
    print(f"    Cosine Similarity:  {test_res['correction_cosine_sim']:.4f}")
    print(f"    Magnitude R²:      {test_res['correction_mag_r2']:.4f}")
    print(f"    Magnitude Corr:    {test_res['correction_mag_corr']:.4f}")

    print(f"\n  PER-DIMENSION CORRECTION:")
    print(f"    {'Dim':>8} {'R²':>8} {'Corr':>8} {'NeedAUC':>8} {'DirAcc':>8} {'DirAUC':>8}")
    print(f"    {'-'*48}")
    for d in DIM_NAMES:
        r2 = test_res.get(f"dim_{d}_r2", 0)
        corr = test_res.get(f"dim_{d}_corr", 0)
        need = test_res.get(f"dim_{d}_needs_corr_auc", 0.5)
        da = test_res.get(f"dim_{d}_dir_acc", 0.5)
        dauc = test_res.get(f"dim_{d}_dir_auc", 0.5)
        v = "✅" if need > 0.7 else ("🟡" if need > 0.6 else "❌")
        print(f"    {d:>8} {r2:>8.4f} {corr:>8.4f} {need:>8.4f} {da:>8.4f} {dauc:>8.4f} {v}")

    # ── Cross-task (pooled model, per-task eval) ──────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-TASK EVALUATION (single model, per-task breakdown)")
    print("=" * 70)

    ct_res = cross_task_evaluate(model, all_samples, device, scaler)
    header = f"  {'Task':>6} {'FailAUC':>8} {'CosSim':>7}"
    for d in DIM_NAMES:
        header += f" {d[:3]:>5}"
    print(header)
    print("  " + "-" * len(header))

    for tid in sorted(ct_res.keys()):
        tr = ct_res[tid]
        line = f"  T{tid:>4} {tr['fail_auc']:>8.4f} {tr.get('correction_cosine_sim', 0):>7.4f}"
        for d in DIM_NAMES:
            line += f" {tr.get(f'dim_{d}_needs_corr_auc', 0.5):>5.3f}"
        print(line)

    # ── True LOO ──────────────────────────────────────────────────────────
    if args.run_loo:
        print("\n" + "=" * 70)
        print("TRUE LEAVE-ONE-TASK-OUT (train on 9, test on 1)")
        print("=" * 70)

        loo_results = {}
        for held_out in range(10):
            t0 = time.time()
            print(f"\n  ── Fold {held_out}: held-out Task {held_out} ──")
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            result, status = run_loo_fold(held_out, all_rollouts, args, device)
            dt = time.time() - t0

            if result is None:
                print(f"    {status}")
                continue

            loo_results[held_out] = result
            print(f"    Fail AUC: {result['fail_auc']:.4f} | "
                  f"Cos Sim: {result.get('correction_cosine_sim', 0):.4f} | "
                  f"({result['n_test_succ']}S/{result['n_test_fail']}F) | {dt:.1f}s")
            for d in DIM_NAMES:
                need = result.get(f"dim_{d}_needs_corr_auc", 0.5)
                da = result.get(f"dim_{d}_dir_acc", 0.5)
                print(f"      {d:>8}: need_corr_AUC={need:.3f}  dir_acc={da:.3f}")

        # Summary
        print("\n" + "=" * 70)
        print("LOO SUMMARY")
        print("=" * 70)

        if loo_results:
            agg_fail = [r["fail_auc"] for r in loo_results.values()]
            agg_cos = [r.get("correction_cosine_sim", 0) for r in loo_results.values()]

            header = f"  {'Task':>6} {'FailAUC':>8} {'CosSim':>7} {'MagR²':>6}"
            for d in DIM_NAMES:
                header += f" {d[:3]:>5}"
            header += f" {'DirAcc':>7}"
            print(header)
            print("  " + "-" * len(header))

            agg_dim_need = defaultdict(list)
            agg_dim_dir = defaultdict(list)

            for tid in sorted(loo_results.keys()):
                r = loo_results[tid]
                line = (f"  T{tid:>4} {r['fail_auc']:>8.4f} "
                        f"{r.get('correction_cosine_sim', 0):>7.4f} "
                        f"{r.get('correction_mag_r2', 0):>6.3f}")
                dirs = []
                for d in DIM_NAMES:
                    n = r.get(f"dim_{d}_needs_corr_auc", 0.5)
                    agg_dim_need[d].append(n)
                    da = r.get(f"dim_{d}_dir_acc", 0.5)
                    agg_dim_dir[d].append(da)
                    dirs.append(da)
                    line += f" {n:>5.3f}"
                line += f" {np.mean(dirs):>7.3f}"
                print(line)

            mean_fail = np.mean(agg_fail)
            mean_cos = np.mean(agg_cos)
            line = f"  {'AVG':>5} {mean_fail:>8.4f} {mean_cos:>7.4f} {'':>6}"
            for d in DIM_NAMES:
                line += f" {np.mean(agg_dim_need[d]):>5.3f}"
            avg_dir = np.mean([np.mean(v) for v in agg_dim_dir.values()])
            line += f" {avg_dir:>7.3f}"
            print(line)

            print(f"\n  ╔════════════════════════════════════════════════════╗")
            if mean_fail >= 0.80:
                print(f"  ║  MEAN FAIL AUC: {mean_fail:.4f}  →  GENERALIZABLE     ║")
            elif mean_fail >= 0.65:
                print(f"  ║  MEAN FAIL AUC: {mean_fail:.4f}  →  MODERATE           ║")
            else:
                print(f"  ║  MEAN FAIL AUC: {mean_fail:.4f}  →  TASK-SPECIFIC      ║")
            print(f"  ║  MEAN COSINE SIM: {mean_cos:.4f}                       ║")
            print(f"  ╚════════════════════════════════════════════════════╝")

            # Save
            loo_report = {
                "per_task": {str(k): {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                      for kk, vv in v.items()} for k, v in loo_results.items()},
                "mean_fail_auc": float(mean_fail),
                "mean_cosine_sim": float(mean_cos),
                "mean_dim_needs_corr_auc": {d: float(np.mean(agg_dim_need[d])) for d in DIM_NAMES},
                "mean_dim_dir_acc": {d: float(np.mean(agg_dim_dir[d])) for d in DIM_NAMES},
            }
        else:
            loo_report = {}

    # ── Save report ───────────────────────────────────────────────────────
    report = {
        "test_results": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in test_res.items()},
        "per_task_results": {str(k): {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                       for kk, vv in v.items()} for k, v in ct_res.items()},
        "model_params": n_params,
        "best_epoch": best_epoch,
        "labeling": "correction_vector",
    }
    if args.run_loo:
        report["loo_results"] = loo_report

    with open(save_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results: {save_dir}/results.json")
    print(f"  Model:   {save_dir}/best_model.pt")
    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
