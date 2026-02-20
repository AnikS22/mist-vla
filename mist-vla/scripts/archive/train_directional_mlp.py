#!/usr/bin/env python3
"""
MVP: Train an MLP that takes ONLY internal VLA hidden states and predicts:
  1. Will this trajectory fail? (binary)
  2. WHEN will it fail? (time-to-failure)
  3. WHERE will it fail? (per-dimension deviation direction — 7 dims × signed)

Input:  hidden_state (4096,)
Output: {
    "will_fail":    scalar sigmoid  (0=safe, 1=failing),
    "time_to_fail": scalar          (normalized steps until failure),
    "dim_risk":     (7,) signed     (per-dim deviation z-score, sign = direction),
}

Training uses rollout-level splits to prevent data leakage.
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

# ─── Config ───────────────────────────────────────────────────────────────
DIM_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]


# ─── Model ────────────────────────────────────────────────────────────────
class DirectionalFailureMLP(nn.Module):
    """
    MLP that predicts failure direction from internal VLA vectors.
    Single encoder → four heads:
      1. will_fail: binary logit
      2. ttf: time-to-failure (0-1)
      3. dim_deviating: per-dim binary logit (is this dim deviating? ×7)
      4. dim_direction: per-dim signed prediction (which direction? ×7)
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

        # Head 1: Will it fail?
        self.fail_head = nn.Linear(feat, 1)

        # Head 2: Time-to-failure (normalized 0-1)
        self.ttf_head = nn.Linear(feat, 1)

        # Head 3: Per-dim "is deviating?" (7 binary logits)
        self.dim_dev_head = nn.Linear(feat, 7)

        # Head 4: Per-dim direction (7 logits: >0 = positive deviation)
        self.dim_dir_head = nn.Linear(feat, 7)

    def forward(self, x):
        feat = self.encoder(x)
        return {
            "will_fail": self.fail_head(feat).squeeze(-1),
            "ttf": self.ttf_head(feat).squeeze(-1),
            "dim_deviating": self.dim_dev_head(feat),    # (B, 7) logits
            "dim_direction": self.dim_dir_head(feat),    # (B, 7) logits
        }


# ─── Dataset ──────────────────────────────────────────────────────────────
class FailureDataset(Dataset):
    def __init__(self, hidden_states, labels, dim_is_dev, dim_dir, ttf_targets):
        self.x = torch.FloatTensor(hidden_states)
        self.labels = torch.FloatTensor(labels)
        self.dim_is_dev = torch.FloatTensor(dim_is_dev)    # (N, 7) binary
        self.dim_dir = torch.FloatTensor(dim_dir)          # (N, 7) binary (1=positive)
        self.ttf_targets = torch.FloatTensor(ttf_targets)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.labels[idx], self.dim_is_dev[idx],
                self.dim_dir[idx], self.ttf_targets[idx])


# ─── Label generation ─────────────────────────────────────────────────────
def build_success_baselines(succ_rollouts, n_bins=10):
    """Build per-task, per-stage AND global action baselines from success rollouts.
    
    Returns dict with:
      - (task_id, bin_idx) → (mean, std)  for task-specific baselines
      - ("global", bin_idx) → (mean, std)  for global fallback
    """
    baselines = {}

    # Per-task baselines
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

    # Global baselines (fallback for unseen tasks)
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


DEVIATION_THRESHOLD = 1.5  # z-scores beyond this = "deviating"


def prepare_samples(rollouts, baselines, n_bins=10):
    """Convert rollouts to per-step samples with binary directional labels."""
    samples = {
        "hidden_states": [],
        "labels": [],        # 1=failure trajectory, 0=success
        "dim_is_dev": [],    # (7,) binary: is this dim deviating?
        "dim_dir": [],       # (7,) binary: 1=positive deviation, 0=negative
        "ttf": [],           # normalized time-to-failure
        "task_ids": [],
        "rollout_ids": [],
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

            # Try task-specific baseline, fall back to global
            if key in baselines:
                mean, std = baselines[key]
            elif ("global", bin_idx) in baselines:
                mean, std = baselines[("global", bin_idx)]
            else:
                mean = np.zeros(7)
                std = np.ones(7)

            deviation = (acts[t] - mean) / std
            is_dev = (np.abs(deviation) > DEVIATION_THRESHOLD).astype(np.float32)
            direction = (deviation > 0).astype(np.float32)

            if is_fail:
                ttf = (T - 1 - t) / max(T - 1, 1)
            else:
                ttf = 1.0

            samples["hidden_states"].append(feats[t])
            samples["labels"].append(is_fail)
            samples["dim_is_dev"].append(is_dev)
            samples["dim_dir"].append(direction)
            samples["ttf"].append(ttf)
            samples["task_ids"].append(tid)
            samples["rollout_ids"].append(ri)

    return {k: np.array(v) for k, v in samples.items()}


# ─── Training ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, pos_weight, dim_pos_weights):
    model.train()
    total_loss = 0
    fail_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    # Per-dim deviation detection loss with class weights
    dim_dev_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(dim_pos_weights).to(device)
    )
    dim_dir_criterion = nn.BCEWithLogitsLoss()

    for x, labels, dim_is_dev, dim_dir, ttf_targets in loader:
        x = x.to(device)
        labels = labels.to(device)
        dim_is_dev = dim_is_dev.to(device)
        dim_dir = dim_dir.to(device)
        ttf_targets = ttf_targets.to(device)

        out = model(x)

        # Loss 1: Binary failure
        loss_fail = fail_criterion(out["will_fail"], labels)

        # Loss 2: TTF regression (only on failure samples)
        fail_mask = labels > 0.5
        if fail_mask.sum() > 0:
            loss_ttf = F.mse_loss(out["ttf"][fail_mask], ttf_targets[fail_mask])
        else:
            loss_ttf = torch.tensor(0.0, device=device)

        # Loss 3: Per-dim "is deviating?" (binary classification ×7)
        loss_dim_dev = dim_dev_criterion(out["dim_deviating"], dim_is_dev)

        # Loss 4: Per-dim direction (only where actually deviating)
        dev_mask = dim_is_dev > 0.5  # (B, 7) bool
        if dev_mask.any():
            loss_dim_dir = dim_dir_criterion(
                out["dim_direction"][dev_mask], dim_dir[dev_mask]
            )
        else:
            loss_dim_dir = torch.tensor(0.0, device=device)

        # Combined loss
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

        probs = torch.sigmoid(out["will_fail"]).cpu().numpy()
        all_preds["fail"].extend(probs)
        all_targets["fail"].extend(labels.numpy())

        all_preds["ttf"].extend(out["ttf"].cpu().numpy())
        all_targets["ttf"].extend(ttf_targets.numpy())

        dim_dev_probs = torch.sigmoid(out["dim_deviating"]).cpu().numpy()
        dim_dir_probs = torch.sigmoid(out["dim_direction"]).cpu().numpy()
        all_preds["dim_dev"].extend(dim_dev_probs)
        all_targets["dim_dev"].extend(dim_is_dev.numpy())
        all_preds["dim_dir"].extend(dim_dir_probs)
        all_targets["dim_dir"].extend(dim_dir.numpy())

    results = {}

    # Failure AUC
    fail_preds = np.array(all_preds["fail"])
    fail_labels = np.array(all_targets["fail"])
    if len(np.unique(fail_labels)) > 1:
        results["fail_auc"] = roc_auc_score(fail_labels, fail_preds)
        results["fail_acc"] = accuracy_score(fail_labels, (fail_preds > 0.5).astype(int))
    else:
        results["fail_auc"] = 0.5
        results["fail_acc"] = 0.5

    # TTF R² (only on failure samples)
    ttf_preds = np.array(all_preds["ttf"])
    ttf_labels = np.array(all_targets["ttf"])
    fail_mask = fail_labels > 0.5
    if fail_mask.sum() > 10:
        results["ttf_r2"] = r2_score(ttf_labels[fail_mask], ttf_preds[fail_mask])
        results["ttf_corr"] = np.corrcoef(ttf_labels[fail_mask], ttf_preds[fail_mask])[0, 1]
    else:
        results["ttf_r2"] = 0.0
        results["ttf_corr"] = 0.0

    # Per-dim deviation detection AUC + direction accuracy
    dim_dev_preds = np.array(all_preds["dim_dev"])
    dim_dev_labels = np.array(all_targets["dim_dev"])
    dim_dir_preds = np.array(all_preds["dim_dir"])
    dim_dir_labels = np.array(all_targets["dim_dir"])

    for i in range(7):
        # Deviation detection AUC
        if len(np.unique(dim_dev_labels[:, i])) > 1:
            results[f"dim_{DIM_NAMES[i]}_dev_auc"] = roc_auc_score(
                dim_dev_labels[:, i], dim_dev_preds[:, i]
            )
        else:
            results[f"dim_{DIM_NAMES[i]}_dev_auc"] = 0.5

        # Direction accuracy (among truly deviating samples)
        dev_mask = dim_dev_labels[:, i] > 0.5
        if dev_mask.sum() > 10:
            true_dir = dim_dir_labels[dev_mask, i]
            pred_dir = (dim_dir_preds[dev_mask, i] > 0.5).astype(float)
            results[f"dim_{DIM_NAMES[i]}_dir_acc"] = accuracy_score(true_dir, pred_dir)
            # Direction AUC
            if len(np.unique(true_dir)) > 1:
                results[f"dim_{DIM_NAMES[i]}_dir_auc"] = roc_auc_score(
                    true_dir, dim_dir_preds[dev_mask, i]
                )
            else:
                results[f"dim_{DIM_NAMES[i]}_dir_auc"] = 0.5
        else:
            results[f"dim_{DIM_NAMES[i]}_dir_acc"] = 0.5
            results[f"dim_{DIM_NAMES[i]}_dir_auc"] = 0.5

    return results


def cross_task_evaluate(model, all_samples, baselines, device, scaler):
    """Per-task evaluation with the trained model."""
    results = {}
    task_ids = all_samples["task_ids"]

    for held_out in range(10):
        test_mask = task_ids == held_out
        if test_mask.sum() < 50:
            continue

        test_x = scaler.transform(all_samples["hidden_states"][test_mask])
        test_labels = all_samples["labels"][test_mask]
        test_dim_dev = all_samples["dim_is_dev"][test_mask]
        test_dim_dir = all_samples["dim_dir"][test_mask]
        test_ttf = all_samples["ttf"][test_mask]

        ds = FailureDataset(test_x, test_labels, test_dim_dev, test_dim_dir, test_ttf)
        loader = DataLoader(ds, batch_size=512, shuffle=False)

        task_results = evaluate(model, loader, device)
        results[held_out] = task_results

    return results


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-data", default="data/combined/success_rollouts.pkl")
    parser.add_argument("--failure-data", default="data/combined/failure_rollouts.pkl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--save-dir", default="checkpoints/directional_mlp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    with open(args.success_data, "rb") as f:
        succ_rollouts = pickle.load(f)
    with open(args.failure_data, "rb") as f:
        fail_rollouts = pickle.load(f)
    print(f"  {len(succ_rollouts)}S + {len(fail_rollouts)}F = {len(succ_rollouts)+len(fail_rollouts)} rollouts")

    # Build baselines from ALL success rollouts (they define "normal")
    print("Building success baselines...")
    baselines = build_success_baselines(succ_rollouts)

    # Prepare all samples
    print("Preparing samples...")
    all_rollouts = succ_rollouts + fail_rollouts
    all_samples = prepare_samples(all_rollouts, baselines)
    print(f"  Total samples: {len(all_samples['labels'])}")
    print(f"  Failure samples: {all_samples['labels'].sum():.0f} ({all_samples['labels'].mean()*100:.1f}%)")

    # ── Rollout-level split ──────────────────────────────────────────────
    print("\nSplitting by rollout (no data leakage)...")
    rollout_ids = all_samples["rollout_ids"]
    unique_rollouts = np.unique(rollout_ids)
    np.random.shuffle(unique_rollouts)

    n_total = len(unique_rollouts)
    n_train = int(0.75 * n_total)
    n_val = int(0.15 * n_total)

    train_rollouts = set(unique_rollouts[:n_train])
    val_rollouts = set(unique_rollouts[n_train:n_train + n_val])
    test_rollouts = set(unique_rollouts[n_train + n_val:])

    train_mask = np.array([rid in train_rollouts for rid in rollout_ids])
    val_mask = np.array([rid in val_rollouts for rid in rollout_ids])
    test_mask = np.array([rid in test_rollouts for rid in rollout_ids])

    print(f"  Train: {train_mask.sum()} samples from {len(train_rollouts)} rollouts")
    print(f"  Val:   {val_mask.sum()} samples from {len(val_rollouts)} rollouts")
    print(f"  Test:  {test_mask.sum()} samples from {len(test_rollouts)} rollouts")

    # Standardize features
    scaler = StandardScaler()
    train_x = scaler.fit_transform(all_samples["hidden_states"][train_mask])
    val_x = scaler.transform(all_samples["hidden_states"][val_mask])
    test_x = scaler.transform(all_samples["hidden_states"][test_mask])

    # Datasets
    train_ds = FailureDataset(train_x, all_samples["labels"][train_mask],
                               all_samples["dim_is_dev"][train_mask],
                               all_samples["dim_dir"][train_mask],
                               all_samples["ttf"][train_mask])
    val_ds = FailureDataset(val_x, all_samples["labels"][val_mask],
                             all_samples["dim_is_dev"][val_mask],
                             all_samples["dim_dir"][val_mask],
                             all_samples["ttf"][val_mask])
    test_ds = FailureDataset(test_x, all_samples["labels"][test_mask],
                              all_samples["dim_is_dev"][test_mask],
                              all_samples["dim_dir"][test_mask],
                              all_samples["ttf"][test_mask])

    # Weighted sampler for class balance
    train_labels = all_samples["labels"][train_mask]
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    weights = np.where(train_labels > 0.5, n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    pos_weight = n_neg / max(n_pos, 1)

    # Per-dim class weights
    train_dim_dev = all_samples["dim_is_dev"][train_mask]
    dim_pos_weights = []
    for i in range(7):
        n_dim_pos = train_dim_dev[:, i].sum()
        n_dim_neg = len(train_dim_dev) - n_dim_pos
        dim_pos_weights.append(n_dim_neg / max(n_dim_pos, 1))
    print(f"\n  Class balance — fail pos_weight: {pos_weight:.2f}")
    print(f"  Per-dim pos_weights: {[f'{w:.1f}' for w in dim_pos_weights]}")

    # Model
    input_dim = train_x.shape[1]
    model = DirectionalFailureMLP(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    # ── Training loop ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    best_val_auc = 0
    best_epoch = 0
    patience = 20
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight, dim_pos_weights)
        val_results = evaluate(model, val_loader, device)
        scheduler.step()

        val_auc = val_results["fail_auc"]
        dim_dev_aucs = [val_results.get(f"dim_{d}_dev_auc", 0.5) for d in DIM_NAMES]
        mean_dim_dev_auc = np.mean(dim_dev_aucs)

        # Track best based on combined metric (fail AUC + dim dev AUC)
        combined_score = 0.4 * val_auc + 0.6 * mean_dim_dev_auc
        if combined_score > best_val_auc:
            best_val_auc = combined_score
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "epoch": epoch,
                "val_auc": val_auc,
                "val_dim_dev_auc": mean_dim_dev_auc,
            }, save_dir / "best_model.pt")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            dt = time.time() - t0
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                  f"fail_AUC={val_auc:.4f} | TTF_r²={val_results['ttf_r2']:.3f} | "
                  f"dim_dev_AUC={mean_dim_dev_auc:.3f} | {dt:.1f}s"
                  f"{' ★' if epoch == best_epoch else ''}")

        if no_improve >= patience:
            print(f"\n  Early stop at epoch {epoch} (best={best_epoch})")
            break

    # ── Load best model ──────────────────────────────────────────────────
    checkpoint = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── Test evaluation ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"TEST RESULTS (best epoch {best_epoch})")
    print("="*70)

    test_results = evaluate(model, test_loader, device)

    print(f"\n  FAILURE DETECTION:")
    print(f"    AUC:      {test_results['fail_auc']:.4f}")
    print(f"    Accuracy: {test_results['fail_acc']:.4f}")

    print(f"\n  TIME-TO-FAILURE:")
    print(f"    R²:   {test_results['ttf_r2']:.4f}")
    print(f"    Corr: {test_results['ttf_corr']:.4f}")

    print(f"\n  PER-DIMENSION DIRECTIONAL PREDICTION:")
    print(f"    {'Dim':>8} {'Dev AUC':>8} {'Dir AUC':>8} {'Dir Acc':>8}")
    print(f"    {'-'*36}")
    for d in DIM_NAMES:
        dev_auc = test_results[f"dim_{d}_dev_auc"]
        dir_auc = test_results.get(f"dim_{d}_dir_auc", 0.5)
        dir_acc = test_results[f"dim_{d}_dir_acc"]
        verdict = "✅" if dev_auc > 0.7 else ("🟡" if dev_auc > 0.6 else "❌")
        print(f"    {d:>8} {dev_auc:>8.4f} {dir_auc:>8.4f} {dir_acc:>8.4f} {verdict}")

    # ── Cross-task generalization ────────────────────────────────────────
    print("\n" + "="*70)
    print("CROSS-TASK GENERALIZATION (model trained on mixed tasks, tested per-task)")
    print("="*70)

    ct_results = cross_task_evaluate(model, all_samples, baselines, device, scaler)

    print(f"\n  {'Task':>6} {'Fail AUC':>9} {'TTF R²':>8}", end="")
    for d in DIM_NAMES:
        print(f" {d[:3]+' dv':>7}", end="")
    print(f" {'DirAcc':>8}")
    print(f"  {'-'*6} {'-'*9} {'-'*8}", end="")
    for _ in range(7):
        print(f" {'-'*7}", end="")
    print(f" {'-'*8}")

    for task_id in sorted(ct_results.keys()):
        tr = ct_results[task_id]
        print(f"  T{task_id:>4} {tr['fail_auc']:>9.4f} {tr['ttf_r2']:>8.4f}", end="")
        dim_devs = []
        dir_accs = []
        for d in DIM_NAMES:
            dev = tr.get(f"dim_{d}_dev_auc", 0.5)
            dim_devs.append(dev)
            dir_accs.append(tr.get(f"dim_{d}_dir_acc", 0.5))
            print(f" {dev:>7.3f}", end="")
        print(f" {np.mean(dir_accs):>8.3f}")

    # ── Leave-one-task-out (single model, no retraining) ────────────────
    # The SAME pooled model is evaluated on each task's data separately.
    # This tests whether a single model generalizes across tasks.
    print("\n" + "="*70)
    print("LEAVE-ONE-TASK-OUT GENERALIZATION (single pooled model, no retraining)")
    print("="*70)

    print(f"\n  {'Task':>6} {'Fail AUC':>9} {'TTF R²':>8}", end="")
    for d in DIM_NAMES:
        print(f" {d[:3]+' dv':>7}", end="")
    print(f" {'DirAcc':>8}")
    print(f"  {'-'*6} {'-'*9} {'-'*8}", end="")
    for _ in range(7):
        print(f" {'-'*7}", end="")
    print(f" {'-'*8}")

    all_fail_aucs = []
    all_dim_dev_aucs = defaultdict(list)
    all_dir_accs = defaultdict(list)

    for task_id in sorted(ct_results.keys()):
        tr = ct_results[task_id]
        all_fail_aucs.append(tr["fail_auc"])
        print(f"  T{task_id:>4} {tr['fail_auc']:>9.4f} {tr['ttf_r2']:>8.4f}", end="")
        for d in DIM_NAMES:
            dev_auc = tr.get(f"dim_{d}_dev_auc", 0.5)
            all_dim_dev_aucs[d].append(dev_auc)
            print(f" {dev_auc:>7.3f}", end="")
        dir_accs = [tr.get(f"dim_{d}_dir_acc", 0.5) for d in DIM_NAMES]
        for d in DIM_NAMES:
            all_dir_accs[d].append(tr.get(f"dim_{d}_dir_acc", 0.5))
        print(f" {np.mean(dir_accs):>8.3f}")

    if all_fail_aucs:
        print(f"\n  {'AVG':>6} {np.mean(all_fail_aucs):>9.4f} {'':>8}", end="")
        for d in DIM_NAMES:
            print(f" {np.mean(all_dim_dev_aucs[d]):>7.3f}", end="")
        avg_dir = np.mean([np.mean(v) for v in all_dir_accs.values()])
        print(f" {avg_dir:>8.3f}")
    else:
        print("\n  No tasks had both success and failure samples for evaluation.")

    # ── Save final report ────────────────────────────────────────────────
    report = {
        "test_results": {k: float(v) for k, v in test_results.items()},
        "per_task_results": {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in ct_results.items()},
        "mean_per_task_fail_auc": float(np.mean(all_fail_aucs)) if all_fail_aucs else 0,
        "mean_per_task_dim_dev_auc": {d: float(np.mean(all_dim_dev_aucs[d])) for d in DIM_NAMES if all_dim_dev_aucs[d]},
        "mean_per_task_dir_acc": {d: float(np.mean(all_dir_accs[d])) for d in DIM_NAMES if all_dir_accs[d]},
        "model_params": n_params,
        "best_epoch": best_epoch,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Model saved to: {save_dir}/best_model.pt")
    print(f"  Results saved to: {save_dir}/results.json")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
