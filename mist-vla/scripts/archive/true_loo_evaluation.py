#!/usr/bin/env python3
"""
TRUE Leave-One-Task-Out Evaluation.

For each of the 10 LIBERO-Spatial tasks:
  1. EXCLUDE all rollouts from that task
  2. Build success baselines from the remaining 9 tasks ONLY
  3. Train a fresh MLP from scratch on those 9 tasks
  4. Evaluate on the held-out task

This is the ONLY valid test of cross-task generalization.
If AUC > 0.80 → breakthrough ("Generalizable VLA Safety")
If AUC < 0.60 → task-specific monitor (still publishable, different story)
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
import argparse
import json
import time

DIM_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]


# ─── Model (identical to train_directional_mlp.py) ────────────────────────
class DirectionalFailureMLP(nn.Module):
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
        self.dim_dev_head = nn.Linear(feat, 7)
        self.dim_dir_head = nn.Linear(feat, 7)

    def forward(self, x):
        feat = self.encoder(x)
        return {
            "will_fail": self.fail_head(feat).squeeze(-1),
            "ttf": self.ttf_head(feat).squeeze(-1),
            "dim_deviating": self.dim_dev_head(feat),
            "dim_direction": self.dim_dir_head(feat),
        }


class FailureDataset(Dataset):
    def __init__(self, hidden_states, labels, dim_is_dev, dim_dir, ttf_targets):
        self.x = torch.FloatTensor(hidden_states)
        self.labels = torch.FloatTensor(labels)
        self.dim_is_dev = torch.FloatTensor(dim_is_dev)
        self.dim_dir = torch.FloatTensor(dim_dir)
        self.ttf_targets = torch.FloatTensor(ttf_targets)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.labels[idx], self.dim_is_dev[idx],
                self.dim_dir[idx], self.ttf_targets[idx])


# ─── Labeling ──────────────────────────────────────────────────────────────
DEVIATION_THRESHOLD = 1.5

def build_success_baselines(succ_rollouts, n_bins=10):
    """Build per-task per-stage AND global baselines from success rollouts."""
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

    # Global baselines (for unseen tasks during LOO)
    for bin_idx in range(n_bins):
        actions_in_bin = []
        for r in succ_rollouts:
            acts = np.array(r["actions"])
            T = len(acts)
            t_start = int(bin_idx / n_bins * T)
            t_end = int((bin_idx + 1) / n_bins * T)
            if t_end > t_start:
                actions_in_bin.append(acts[t_start:t_end])
        if actions_in_bin:
            all_acts = np.concatenate(actions_in_bin, axis=0)
            baselines[("global", bin_idx)] = (all_acts.mean(axis=0), all_acts.std(axis=0) + 1e-6)

    return baselines


def prepare_samples(rollouts, baselines, n_bins=10):
    """Convert rollouts to per-step samples."""
    samples = {
        "hidden_states": [], "labels": [], "dim_is_dev": [],
        "dim_dir": [], "ttf": [], "task_ids": [], "rollout_ids": [],
    }
    for ri, r in enumerate(rollouts):
        feats = np.array(r["features"])
        acts = np.array(r["actions"])
        tid = r["task_id"]
        is_fail = 0 if r["success"] else 1
        T = min(len(feats), len(acts))
        for t in range(T):
            progress = t / max(T - 1, 1)
            bin_idx = min(int(progress * n_bins), n_bins - 1)
            key = (tid, bin_idx)
            # For LOO: the held-out task won't have task-specific baselines
            # Fall back to global baselines (built from training tasks only)
            if key in baselines:
                mean, std = baselines[key]
            elif ("global", bin_idx) in baselines:
                mean, std = baselines[("global", bin_idx)]
            else:
                mean, std = np.zeros(7), np.ones(7)
            deviation = (acts[t] - mean) / std
            is_dev = (np.abs(deviation) > DEVIATION_THRESHOLD).astype(np.float32)
            direction = (deviation > 0).astype(np.float32)
            ttf = (T - 1 - t) / max(T - 1, 1) if is_fail else 1.0
            samples["hidden_states"].append(feats[t])
            samples["labels"].append(is_fail)
            samples["dim_is_dev"].append(is_dev)
            samples["dim_dir"].append(direction)
            samples["ttf"].append(ttf)
            samples["task_ids"].append(tid)
            samples["rollout_ids"].append(ri)
    return {k: np.array(v) for k, v in samples.items()}


# ─── Training utilities ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, pos_weight, dim_pos_weights):
    model.train()
    total_loss = 0
    fail_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    dim_dev_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(dim_pos_weights).to(device))
    dim_dir_criterion = nn.BCEWithLogitsLoss()

    for x, labels, dim_is_dev, dim_dir, ttf_targets in loader:
        x, labels = x.to(device), labels.to(device)
        dim_is_dev, dim_dir = dim_is_dev.to(device), dim_dir.to(device)
        ttf_targets = ttf_targets.to(device)
        out = model(x)

        loss_fail = fail_criterion(out["will_fail"], labels)
        fail_mask = labels > 0.5
        loss_ttf = F.mse_loss(out["ttf"][fail_mask], ttf_targets[fail_mask]) if fail_mask.sum() > 0 else torch.tensor(0.0, device=device)
        loss_dim_dev = dim_dev_criterion(out["dim_deviating"], dim_is_dev)
        dev_mask = dim_is_dev > 0.5
        loss_dim_dir = dim_dir_criterion(out["dim_direction"][dev_mask], dim_dir[dev_mask]) if dev_mask.any() else torch.tensor(0.0, device=device)
        loss = loss_fail + 0.5 * loss_ttf + 1.0 * loss_dim_dev + 0.5 * loss_dim_dir

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = defaultdict(list)
    all_targets = defaultdict(list)

    for x, labels, dim_is_dev, dim_dir, ttf_targets in loader:
        x = x.to(device)
        out = model(x)
        all_preds["fail"].extend(torch.sigmoid(out["will_fail"]).cpu().numpy())
        all_targets["fail"].extend(labels.numpy())
        all_preds["ttf"].extend(out["ttf"].cpu().numpy())
        all_targets["ttf"].extend(ttf_targets.numpy())
        all_preds["dim_dev"].extend(torch.sigmoid(out["dim_deviating"]).cpu().numpy())
        all_targets["dim_dev"].extend(dim_is_dev.numpy())
        all_preds["dim_dir"].extend(torch.sigmoid(out["dim_direction"]).cpu().numpy())
        all_targets["dim_dir"].extend(dim_dir.numpy())

    results = {}
    fail_preds = np.array(all_preds["fail"])
    fail_labels = np.array(all_targets["fail"])
    results["fail_auc"] = roc_auc_score(fail_labels, fail_preds) if len(np.unique(fail_labels)) > 1 else 0.5
    results["fail_acc"] = accuracy_score(fail_labels, (fail_preds > 0.5).astype(int)) if len(np.unique(fail_labels)) > 1 else 0.5

    ttf_preds = np.array(all_preds["ttf"])
    ttf_labels = np.array(all_targets["ttf"])
    fail_mask = fail_labels > 0.5
    if fail_mask.sum() > 10:
        results["ttf_r2"] = r2_score(ttf_labels[fail_mask], ttf_preds[fail_mask])
        results["ttf_corr"] = float(np.corrcoef(ttf_labels[fail_mask], ttf_preds[fail_mask])[0, 1])
    else:
        results["ttf_r2"] = 0.0
        results["ttf_corr"] = 0.0

    dim_dev_preds = np.array(all_preds["dim_dev"])
    dim_dev_labels = np.array(all_targets["dim_dev"])
    dim_dir_preds = np.array(all_preds["dim_dir"])
    dim_dir_labels = np.array(all_targets["dim_dir"])

    for i in range(7):
        d = DIM_NAMES[i]
        results[f"dim_{d}_dev_auc"] = roc_auc_score(dim_dev_labels[:, i], dim_dev_preds[:, i]) if len(np.unique(dim_dev_labels[:, i])) > 1 else 0.5
        dev_mask = dim_dev_labels[:, i] > 0.5
        if dev_mask.sum() > 10:
            true_dir = dim_dir_labels[dev_mask, i]
            results[f"dim_{d}_dir_acc"] = accuracy_score(true_dir, (dim_dir_preds[dev_mask, i] > 0.5).astype(float))
            results[f"dim_{d}_dir_auc"] = roc_auc_score(true_dir, dim_dir_preds[dev_mask, i]) if len(np.unique(true_dir)) > 1 else 0.5
        else:
            results[f"dim_{d}_dir_acc"] = 0.5
            results[f"dim_{d}_dir_auc"] = 0.5

    return results


# ─── Single LOO fold ──────────────────────────────────────────────────────
def run_loo_fold(held_out_task, all_rollouts, args, device):
    """Train on 9 tasks, evaluate on held-out task. Returns metrics dict."""
    train_rollouts = [r for r in all_rollouts if r["task_id"] != held_out_task]
    test_rollouts = [r for r in all_rollouts if r["task_id"] == held_out_task]

    n_test_succ = sum(1 for r in test_rollouts if r["success"])
    n_test_fail = sum(1 for r in test_rollouts if not r["success"])

    if not test_rollouts or n_test_succ == 0 or n_test_fail == 0:
        return None, f"SKIP (test has {n_test_succ}S/{n_test_fail}F)"

    # Build baselines from TRAINING success rollouts only
    train_succ = [r for r in train_rollouts if r["success"]]
    baselines = build_success_baselines(train_succ)

    # Prepare samples — held-out task uses GLOBAL baselines (no task-specific)
    train_samples = prepare_samples(train_rollouts, baselines)
    test_samples = prepare_samples(test_rollouts, baselines)

    # Check test labels aren't degenerate
    if len(np.unique(test_samples["labels"])) < 2:
        return None, "SKIP (single class in test)"

    # Rollout-level split within training data for train/val
    rollout_ids = train_samples["rollout_ids"]
    unique_rids = np.unique(rollout_ids)
    np.random.shuffle(unique_rids)
    n_train = int(0.85 * len(unique_rids))
    train_rids = set(unique_rids[:n_train])
    val_rids = set(unique_rids[n_train:])
    train_mask = np.array([rid in train_rids for rid in rollout_ids])
    val_mask = np.array([rid in val_rids for rid in rollout_ids])

    # Scale features
    scaler = StandardScaler()
    tr_x = scaler.fit_transform(train_samples["hidden_states"][train_mask])
    val_x = scaler.transform(train_samples["hidden_states"][val_mask])
    te_x = scaler.transform(test_samples["hidden_states"])

    # Datasets
    tr_ds = FailureDataset(tr_x, train_samples["labels"][train_mask],
                           train_samples["dim_is_dev"][train_mask],
                           train_samples["dim_dir"][train_mask],
                           train_samples["ttf"][train_mask])
    val_ds = FailureDataset(val_x, train_samples["labels"][val_mask],
                            train_samples["dim_is_dev"][val_mask],
                            train_samples["dim_dir"][val_mask],
                            train_samples["ttf"][val_mask])
    te_ds = FailureDataset(te_x, test_samples["labels"],
                           test_samples["dim_is_dev"],
                           test_samples["dim_dir"],
                           test_samples["ttf"])

    # Class weights
    tr_labels = train_samples["labels"][train_mask]
    n_pos = tr_labels.sum()
    n_neg = len(tr_labels) - n_pos
    pw = n_neg / max(n_pos, 1)
    sample_weights = np.where(tr_labels > 0.5, n_neg / max(n_pos, 1), 1.0)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    dim_dev = train_samples["dim_is_dev"][train_mask]
    dim_pw = []
    for i in range(7):
        dp = dim_dev[:, i].sum()
        dn = len(dim_dev) - dp
        dim_pw.append(dn / max(dp, 1))

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False)

    # Fresh model
    input_dim = tr_x.shape[1]
    model = DirectionalFailureMLP(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train with early stopping on val
    best_score = 0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, tr_loader, optimizer, device, pw, dim_pw)
        scheduler.step()

        if epoch % 3 == 0 or epoch == 1:
            val_res = evaluate(model, val_loader, device)
            dim_aucs = [val_res.get(f"dim_{d}_dev_auc", 0.5) for d in DIM_NAMES]
            score = 0.4 * val_res["fail_auc"] + 0.6 * np.mean(dim_aucs)
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    # Load best and evaluate on held-out task
    if best_state is not None:
        model.load_state_dict(best_state)
    test_results = evaluate(model, te_loader, device)
    test_results["n_test_samples"] = len(test_samples["labels"])
    test_results["n_test_succ"] = n_test_succ
    test_results["n_test_fail"] = n_test_fail
    test_results["n_train_rollouts"] = len(train_rollouts)

    return test_results, "OK"


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="True Leave-One-Task-Out Evaluation")
    parser.add_argument("--success-data", default="data/combined/success_rollouts.pkl")
    parser.add_argument("--failure-data", default="data/combined/failure_rollouts.pkl")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--save-dir", default="checkpoints/true_loo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("Loading data...")
    with open(args.success_data, "rb") as f:
        succ = pickle.load(f)
    with open(args.failure_data, "rb") as f:
        fail = pickle.load(f)
    all_rollouts = succ + fail
    print(f"  {len(succ)}S + {len(fail)}F = {len(all_rollouts)} rollouts")

    # Per-task summary
    from collections import Counter
    ts = Counter(r["task_id"] for r in succ)
    tf = Counter(r["task_id"] for r in fail)
    print("\n  Per-task breakdown:")
    for t in range(10):
        print(f"    Task {t}: {ts.get(t,0)}S / {tf.get(t,0)}F")

    # ── Run 10 LOO folds ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRUE LEAVE-ONE-TASK-OUT EVALUATION")
    print("Train on 9 tasks → Test on 1 held-out task → ×10 folds")
    print("=" * 70)

    all_results = {}
    for held_out in range(10):
        t0 = time.time()
        print(f"\n  ── Fold {held_out}: Held-out Task {held_out} ──")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        result, status = run_loo_fold(held_out, all_rollouts, args, device)
        dt = time.time() - t0

        if result is None:
            print(f"    {status}")
            continue

        all_results[held_out] = result
        print(f"    Fail AUC: {result['fail_auc']:.4f}  |  TTF R²: {result['ttf_r2']:.4f}  |  "
              f"({result['n_test_succ']}S/{result['n_test_fail']}F test)  |  {dt:.1f}s")

        # Per-dim summary
        dim_devs = [result.get(f"dim_{d}_dev_auc", 0.5) for d in DIM_NAMES]
        dim_dirs = [result.get(f"dim_{d}_dir_acc", 0.5) for d in DIM_NAMES]
        print(f"    Dim Dev AUC: {' '.join(f'{d[:3]}={v:.3f}' for d, v in zip(DIM_NAMES, dim_devs))}")
        print(f"    Dir Acc:     {' '.join(f'{d[:3]}={v:.3f}' for d, v in zip(DIM_NAMES, dim_dirs))}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: TRUE LEAVE-ONE-TASK-OUT RESULTS")
    print("=" * 70)

    header = f"  {'Task':>6} {'Fail AUC':>9} {'TTF R²':>8} {'TTF Cor':>8}"
    for d in DIM_NAMES:
        header += f" {d[:3]+'Dv':>6}"
    header += f" {'MnDvAUC':>8} {'MnDirAc':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    agg_fail = []
    agg_ttf_r2 = []
    agg_dim_dev = defaultdict(list)
    agg_dim_dir = defaultdict(list)

    for task_id in sorted(all_results.keys()):
        r = all_results[task_id]
        agg_fail.append(r["fail_auc"])
        agg_ttf_r2.append(r["ttf_r2"])

        line = f"  T{task_id:>4} {r['fail_auc']:>9.4f} {r['ttf_r2']:>8.4f} {r['ttf_corr']:>8.4f}"
        dim_devs = []
        dim_dirs = []
        for d in DIM_NAMES:
            dv = r.get(f"dim_{d}_dev_auc", 0.5)
            da = r.get(f"dim_{d}_dir_acc", 0.5)
            dim_devs.append(dv)
            dim_dirs.append(da)
            agg_dim_dev[d].append(dv)
            agg_dim_dir[d].append(da)
            line += f" {dv:>6.3f}"
        line += f" {np.mean(dim_devs):>8.4f} {np.mean(dim_dirs):>8.4f}"
        print(line)

    # Averages
    if agg_fail:
        line = f"  {'AVG':>5} {np.mean(agg_fail):>9.4f} {np.mean(agg_ttf_r2):>8.4f} {'':>8}"
        all_dim_devs = []
        all_dim_dirs = []
        for d in DIM_NAMES:
            mv = np.mean(agg_dim_dev[d])
            all_dim_devs.append(mv)
            all_dim_dirs.append(np.mean(agg_dim_dir[d]))
            line += f" {mv:>6.3f}"
        line += f" {np.mean(all_dim_devs):>8.4f} {np.mean(all_dim_dirs):>8.4f}"
        print(line)

        print(f"\n  ╔══════════════════════════════════════════════╗")
        mean_fail = np.mean(agg_fail)
        mean_dev = np.mean(all_dim_devs)
        if mean_fail >= 0.80:
            print(f"  ║  MEAN FAIL AUC: {mean_fail:.4f}  →  GENERALIZABLE  ║")
        elif mean_fail >= 0.65:
            print(f"  ║  MEAN FAIL AUC: {mean_fail:.4f}  →  MODERATE        ║")
        else:
            print(f"  ║  MEAN FAIL AUC: {mean_fail:.4f}  →  TASK-SPECIFIC   ║")
        print(f"  ║  MEAN DIM DEV AUC: {mean_dev:.4f}                  ║")
        print(f"  ╚══════════════════════════════════════════════╝")

    # ── Save results ──────────────────────────────────────────────────────
    report = {
        "experiment": "true_leave_one_task_out",
        "description": "Train on 9 tasks, test on 1 held-out task, x10 folds",
        "per_task": {str(k): {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                              for kk, vv in v.items()} for k, v in all_results.items()},
        "mean_fail_auc": float(np.mean(agg_fail)) if agg_fail else 0,
        "mean_ttf_r2": float(np.mean(agg_ttf_r2)) if agg_ttf_r2 else 0,
        "mean_dim_dev_auc": {d: float(np.mean(agg_dim_dev[d])) for d in DIM_NAMES if agg_dim_dev[d]},
        "mean_dim_dir_acc": {d: float(np.mean(agg_dim_dir[d])) for d in DIM_NAMES if agg_dim_dir[d]},
        "hyperparams": {"epochs": args.epochs, "lr": args.lr, "hidden_dim": args.hidden_dim,
                        "batch_size": args.batch_size, "seed": args.seed},
    }
    with open(save_dir / "true_loo_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved to: {save_dir}/true_loo_results.json")
    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
