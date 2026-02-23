#!/usr/bin/env python3
"""
Cartesian EEF Correction MLP — Cross-embodiment generalizable safety.

Predicts corrections in END-EFFECTOR CARTESIAN SPACE (dx, dy, dz):
    correction[t] = EEF_pos_success[t] - EEF_pos_failed[t]

This is UNIVERSAL:
  - "Move hand 3cm left" applies to any robot arm in 3D space
  - Only the IK solver changes between robots; the MLP brain stays the same
  - Z-score is distribution-dependent; Cartesian error is physics-dependent

Input:  hidden_state (4096,) — internal VLA vector
Output: {
    "will_fail":    scalar  (0=safe, 1=failing),
    "ttf":          scalar  (normalized time-to-failure),
    "correction":   (3,)    (dx, dy, dz in meters — the Cartesian correction),
}
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

DIM_NAMES_EEF = ["X", "Y", "Z"]
CORRECTION_THRESHOLD_M = 0.01  # 1cm — meaningful correction threshold


# ─── EEF Trajectory Extraction ───────────────────────────────────────────

def get_eef_trajectory(rollout):
    """Extract (T, 3) EEF position trajectory from a rollout."""
    robot_states = rollout.get("robot_states", [])
    eef_positions = []
    for rs in robot_states:
        if "eef_pos" in rs:
            eef_positions.append(np.array(rs["eef_pos"], dtype=np.float32))
    if not eef_positions:
        return None
    return np.array(eef_positions)  # (T, 3)


# ─── Trajectory Matching & Alignment ─────────────────────────────────────

def match_failure_to_success(failure_rollout, success_rollouts_by_task):
    """Find nearest-neighbor success trajectory by initial EEF position similarity."""
    tid = failure_rollout["task_id"]
    candidates = success_rollouts_by_task.get(tid, [])
    if not candidates:
        return None

    fail_eef = get_eef_trajectory(failure_rollout)
    if fail_eef is None or len(fail_eef) < 2:
        return None

    # Match by early trajectory similarity (first 20% of EEF positions)
    n_compare = max(1, int(0.2 * len(fail_eef)))
    fail_early = fail_eef[:n_compare]

    best_dist = float("inf")
    best_match = None

    for succ in candidates:
        succ_eef = get_eef_trajectory(succ)
        if succ_eef is None or len(succ_eef) < 2:
            continue

        n_succ = max(1, int(0.2 * len(succ_eef)))
        succ_early = succ_eef[:n_succ]

        # Resample to same length
        n_pts = min(len(fail_early), len(succ_early))
        f_resamp = fail_early[np.linspace(0, len(fail_early) - 1, n_pts).astype(int)]
        s_resamp = succ_early[np.linspace(0, len(succ_early) - 1, n_pts).astype(int)]

        dist = np.mean(np.linalg.norm(f_resamp - s_resamp, axis=1))
        if dist < best_dist:
            best_dist = dist
            best_match = succ

    return best_match


def compute_eef_corrections(fail_rollout, succ_rollout):
    """Compute Cartesian correction at each failure timestep.

    correction[t] = EEF_success(t) - EEF_failed(t)
    Aligned by progress (0→1) using interpolation.

    Returns (T_fail, 3) correction vectors in meters.
    """
    fail_eef = get_eef_trajectory(fail_rollout)
    succ_eef = get_eef_trajectory(succ_rollout)

    if fail_eef is None or succ_eef is None:
        return None

    T_fail = len(fail_eef)
    T_succ = len(succ_eef)

    succ_progress = np.linspace(0, 1, T_succ)
    fail_progress = np.linspace(0, 1, T_fail)

    corrections = np.zeros((T_fail, 3), dtype=np.float32)
    for dim in range(3):
        interpolator = interp1d(succ_progress, succ_eef[:, dim],
                                kind="linear", fill_value="extrapolate")
        succ_at_fail_time = interpolator(fail_progress)
        corrections[:, dim] = succ_at_fail_time - fail_eef[:, dim]

    return corrections


# ─── Sample Preparation ──────────────────────────────────────────────────

def prepare_samples(rollouts, success_by_task, subsample_chunks=False):
    """Build per-step samples with Cartesian EEF correction labels.

    Args:
        rollouts: list of rollout dicts
        success_by_task: dict mapping task_id → list of success rollouts
        subsample_chunks: If True, only keep the FIRST timestep of each
            action chunk (i.e., where the feature vector changes).
            This removes duplicated features from chunked policies
            (e.g., ACT with 8-step chunks) and prevents inflated metrics.
            The label at the chunk boundary is the label for the moment
            the new latent state was generated — the only honest sample.
    """
    samples = {
        "hidden_states": [],
        "labels": [],           # 1=failure, 0=success
        "corrections": [],      # (3,) Cartesian correction in meters
        "ttf": [],
        "task_ids": [],
        "rollout_ids": [],
    }

    n_matched = 0
    n_unmatched = 0
    n_total_steps = 0
    n_kept_steps = 0

    for ri, r in enumerate(rollouts):
        feats = np.array(r["features"])
        is_fail = not r["success"]
        eef = get_eef_trajectory(r)

        if eef is None:
            continue

        T = min(len(feats), len(eef))

        if is_fail:
            match = match_failure_to_success(r, success_by_task)
            if match is not None:
                corrections = compute_eef_corrections(r, match)
                if corrections is None:
                    n_unmatched += 1
                    continue
                corrections = corrections[:T]
                n_matched += 1
            else:
                n_unmatched += 1
                continue
        else:
            corrections = np.zeros((T, 3), dtype=np.float32)

        prev_feat = None
        for t in range(T):
            n_total_steps += 1
            cur_feat = feats[t]

            # Chunk subsampling: skip if feature is identical to previous
            if subsample_chunks and prev_feat is not None:
                if np.array_equal(cur_feat, prev_feat):
                    continue
            prev_feat = cur_feat

            n_kept_steps += 1
            ttf = (T - 1 - t) / max(T - 1, 1) if is_fail else 1.0

            samples["hidden_states"].append(cur_feat)
            samples["labels"].append(1.0 if is_fail else 0.0)
            samples["corrections"].append(corrections[t])
            samples["ttf"].append(ttf)
            samples["task_ids"].append(r["task_id"])
            samples["rollout_ids"].append(ri)

    print(f"    Matched: {n_matched} failure rollouts | Skipped: {n_unmatched}")
    if subsample_chunks:
        print(f"    Chunk subsampling: {n_total_steps} → {n_kept_steps} steps "
              f"({n_kept_steps/max(n_total_steps,1)*100:.1f}% kept, "
              f"{n_total_steps - n_kept_steps} duplicates removed)")
    return {k: np.array(v) for k, v in samples.items()}


# ─── Model ────────────────────────────────────────────────────────────────

class EEFCorrectionMLP(nn.Module):
    """Predicts Cartesian EEF corrections from VLA hidden states.

    Output is (dx, dy, dz) in meters — universal across embodiments.

    Architecture (v4 — shortcut-learning fix):
      1. LayerNorm on raw embeddings  → tames extreme activations
      2. 4096 → 256 → 128 → 64 bottleneck  (hardcoded)
      3. GELU + Dropout(0.3) throughout
      4. Three heads: fail (1), ttf (1), correction (3)
    """
    HIDDEN_DIM = 256  # hardcoded — prevents capacity overshoot

    def __init__(self, input_dim=4096):
        super().__init__()
        h = self.HIDDEN_DIM
        # ── Input normalisation (tame frozen VLA activations) ──
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(h // 2, h // 4),
            nn.LayerNorm(h // 4),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        feat = h // 4  # 64

        self.fail_head = nn.Linear(feat, 1)
        self.ttf_head = nn.Linear(feat, 1)
        self.correction_head = nn.Linear(feat, 3)  # dx, dy, dz in meters

    def forward(self, x):
        x = self.input_norm(x)
        feat = self.encoder(x)
        return {
            "will_fail": self.fail_head(feat).squeeze(-1),
            "ttf": self.ttf_head(feat).squeeze(-1),
            "correction": self.correction_head(feat),  # (B, 3)
        }


# ─── Dataset ─────────────────────────────────────────────────────────────

class EEFCorrectionDataset(Dataset):
    """Dataset with optional Gaussian noise augmentation on inputs."""

    def __init__(self, hidden_states, labels, corrections, ttf,
                 noise_std=0.0, training=False):
        self.x = torch.FloatTensor(hidden_states)
        self.labels = torch.FloatTensor(labels)
        self.corrections = torch.FloatTensor(corrections)
        self.ttf = torch.FloatTensor(ttf)
        self.noise_std = noise_std
        self.training = training

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        # Input noise augmentation — prevents memorising exact hidden states
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.labels[idx], self.corrections[idx], self.ttf[idx]


# ─── Training ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device,
                corr_mag_penalty=0.1):
    """One training epoch (v4 — shortcut-learning fixes).

    Key changes vs v3:
      • **Dynamic per-batch pos_weight** for BCEWithLogitsLoss — prevents
        the classifier from memorising the dataset-wide failure prior.
      • **HuberLoss(δ=0.1)** for corrections — prevents gradient
        explosions on OOD states with slightly different embeddings.
      • **Correction magnitude L2 penalty** on success samples — pushes
        safe-trajectory predictions toward zero.
    """
    model.train()
    total_loss = 0
    total_items = 0
    loss_parts = {"fail": 0.0, "ttf": 0.0, "corr": 0.0, "mag_pen": 0.0}
    huber = nn.HuberLoss(delta=0.1)

    for x, labels, corrections, ttf in loader:
        x, labels = x.to(device), labels.to(device)
        corrections, ttf = corrections.to(device), ttf.to(device)
        out = model(x)

        # ── Classification: dynamic per-batch pos_weight ──
        n_pos = (labels > 0.5).sum().float()
        n_neg = (labels <= 0.5).sum().float()
        pw = (n_neg / n_pos.clamp(min=1)).clamp(max=10.0)
        fail_crit = nn.BCEWithLogitsLoss(
            pos_weight=pw.unsqueeze(0))
        loss_fail = fail_crit(out["will_fail"], labels)

        fail_mask = labels > 0.5
        loss_ttf = (F.mse_loss(out["ttf"][fail_mask], ttf[fail_mask])
                    if fail_mask.sum() > 0
                    else torch.tensor(0.0, device=device))

        # ── Regression: Huber(δ=0.1) — robust to outlier corrections ──
        loss_corr = huber(out["correction"], corrections)

        # ── Magnitude penalty on success samples (correction → 0) ──
        safe_mask = ~fail_mask
        if safe_mask.sum() > 0 and corr_mag_penalty > 0:
            safe_pred = out["correction"][safe_mask]  # (N_safe, 3)
            loss_mag = torch.mean(safe_pred.pow(2))   # L2 magnitude
        else:
            loss_mag = torch.tensor(0.0, device=device)

        loss = (loss_fail
                + 0.5 * loss_ttf
                + 2.0 * loss_corr
                + corr_mag_penalty * loss_mag)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = len(x)
        total_loss += loss.item() * bs
        total_items += bs
        loss_parts["fail"] += loss_fail.item() * bs
        loss_parts["ttf"] += loss_ttf.item() * bs
        loss_parts["corr"] += loss_corr.item() * bs
        loss_parts["mag_pen"] += loss_mag.item() * bs

    n = max(total_items, 1)
    return (total_loss / n,
            {k: v / n for k, v in loss_parts.items()})


@torch.no_grad()
def evaluate(model, loader, device, threshold=CORRECTION_THRESHOLD_M):
    model.eval()
    preds = defaultdict(list)
    targets = defaultdict(list)

    for x, labels, corrections, ttf in loader:
        x = x.to(device)
        out = model(x)
        preds["fail"].extend(torch.sigmoid(out["will_fail"]).cpu().numpy())
        targets["fail"].extend(labels.numpy())
        preds["ttf"].extend(out["ttf"].cpu().numpy())
        targets["ttf"].extend(ttf.numpy())
        preds["corr"].extend(out["correction"].cpu().numpy())
        targets["corr"].extend(corrections.numpy())

    res = {}

    # Failure
    fp, fl = np.array(preds["fail"]), np.array(targets["fail"])
    res["fail_auc"] = roc_auc_score(fl, fp) if len(np.unique(fl)) > 1 else 0.5
    res["fail_acc"] = accuracy_score(fl, (fp > 0.5).astype(int)) if len(np.unique(fl)) > 1 else 0.5

    # TTF
    tp, tl = np.array(preds["ttf"]), np.array(targets["ttf"])
    fm = fl > 0.5
    if fm.sum() > 10:
        res["ttf_r2"] = r2_score(tl[fm], tp[fm])
        c = np.corrcoef(tl[fm], tp[fm])[0, 1]
        res["ttf_corr"] = float(c) if not np.isnan(c) else 0.0
    else:
        res["ttf_r2"] = 0.0
        res["ttf_corr"] = 0.0

    # Correction
    pc, tc = np.array(preds["corr"]), np.array(targets["corr"])  # (N, 3)

    if fm.sum() > 10:
        # Cosine similarity (failure only)
        pn = np.linalg.norm(pc[fm], axis=1, keepdims=True) + 1e-8
        tn = np.linalg.norm(tc[fm], axis=1, keepdims=True) + 1e-8
        cos = np.sum((pc[fm] / pn) * (tc[fm] / tn), axis=1)
        res["cosine_sim"] = float(np.mean(cos))
        res["cosine_sim_median"] = float(np.median(cos))

        # Magnitude
        pm, tm = np.linalg.norm(pc[fm], axis=1), np.linalg.norm(tc[fm], axis=1)
        res["mag_r2"] = r2_score(tm, pm)
        c = np.corrcoef(pm, tm)[0, 1]
        res["mag_corr"] = float(c) if not np.isnan(c) else 0.0

        # Euclidean error (in meters)
        errors = np.linalg.norm(pc[fm] - tc[fm], axis=1)
        res["mean_error_m"] = float(np.mean(errors))
        res["median_error_m"] = float(np.median(errors))
    else:
        for k in ["cosine_sim", "cosine_sim_median", "mag_r2", "mag_corr",
                   "mean_error_m", "median_error_m"]:
            res[k] = 0.0

    # Per-axis metrics
    for i, d in enumerate(DIM_NAMES_EEF):
        p, t = pc[:, i], tc[:, i]

        if fm.sum() > 10:
            res[f"{d}_r2"] = r2_score(t[fm], p[fm])
            c = np.corrcoef(t[fm], p[fm])[0, 1]
            res[f"{d}_corr"] = float(c) if not np.isnan(c) else 0.0

        # "Needs correction?" binary AUC
        needs = (np.abs(t) > threshold).astype(int)
        if len(np.unique(needs)) > 1:
            res[f"{d}_needs_auc"] = roc_auc_score(needs, np.abs(p))
        else:
            res[f"{d}_needs_auc"] = 0.5

        # Direction accuracy (among significant corrections)
        sig_mask = np.abs(t) > threshold
        if sig_mask.sum() > 10:
            true_dir = (t[sig_mask] > 0).astype(int)
            pred_dir = (p[sig_mask] > 0).astype(int)
            if len(np.unique(true_dir)) > 1:
                res[f"{d}_dir_acc"] = accuracy_score(true_dir, pred_dir)
                res[f"{d}_dir_auc"] = roc_auc_score(true_dir, p[sig_mask])
            else:
                res[f"{d}_dir_acc"] = 0.5
                res[f"{d}_dir_auc"] = 0.5
        else:
            res[f"{d}_dir_acc"] = 0.5
            res[f"{d}_dir_auc"] = 0.5

    return res


# ─── LOO fold ─────────────────────────────────────────────────────────────

def run_loo_fold(held_out, all_rollouts, args, device):
    train_rols = [r for r in all_rollouts if r["task_id"] != held_out]
    test_rols = [r for r in all_rollouts if r["task_id"] == held_out]

    n_ts = sum(1 for r in test_rols if r["success"])
    n_tf = sum(1 for r in test_rols if not r["success"])
    if n_ts == 0 or n_tf == 0:
        return None, f"SKIP ({n_ts}S/{n_tf}F)"

    # Success pools
    train_succ = defaultdict(list)
    for r in train_rols:
        if r["success"]:
            train_succ[r["task_id"]].append(r)
    # Test task successes (for matching only, not training)
    test_succ = defaultdict(list)
    for r in test_rols:
        if r["success"]:
            test_succ[r["task_id"]].append(r)

    # Merge for labeling: train tasks use train successes, held-out uses its own
    all_succ = defaultdict(list)
    for tid, rols in train_succ.items():
        all_succ[tid].extend(rols)
    for tid, rols in test_succ.items():
        all_succ[tid].extend(rols)

    print(f"    Preparing train samples...")
    train_samples = prepare_samples(train_rols, train_succ,
                                     subsample_chunks=args.subsample_chunks)
    print(f"    Preparing test samples...")
    test_samples = prepare_samples(test_rols, all_succ,
                                    subsample_chunks=args.subsample_chunks)

    if len(np.unique(test_samples["labels"])) < 2:
        return None, "SKIP (single class)"

    # Train/val split by rollout
    rids = train_samples["rollout_ids"]
    urids = np.unique(rids)
    np.random.shuffle(urids)
    n_tr = int(0.85 * len(urids))
    tr_set = set(urids[:n_tr])
    val_set = set(urids[n_tr:])
    tr_m = np.array([rid in tr_set for rid in rids])
    val_m = np.array([rid in val_set for rid in rids])

    sc = StandardScaler()
    tr_x = sc.fit_transform(train_samples["hidden_states"][tr_m])
    val_x = sc.transform(train_samples["hidden_states"][val_m])
    te_x = sc.transform(test_samples["hidden_states"])

    tr_ds = EEFCorrectionDataset(tr_x, train_samples["labels"][tr_m],
                                  train_samples["corrections"][tr_m],
                                  train_samples["ttf"][tr_m],
                                  noise_std=args.input_noise, training=True)
    val_ds = EEFCorrectionDataset(val_x, train_samples["labels"][val_m],
                                   train_samples["corrections"][val_m],
                                   train_samples["ttf"][val_m])
    te_ds = EEFCorrectionDataset(te_x, test_samples["labels"],
                                  test_samples["corrections"],
                                  test_samples["ttf"])

    tr_labels = train_samples["labels"][tr_m]
    n_pos = tr_labels.sum()
    n_neg = len(tr_labels) - n_pos
    pw = n_neg / max(n_pos, 1)
    sw = np.where(tr_labels > 0.5, pw, 1.0)
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False)

    model = EEFCorrectionMLP(input_dim=tr_x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score, best_state, no_imp = 0, None, 0
    for ep in range(1, args.epochs + 1):
        train_epoch(model, tr_loader, opt, device,
                    corr_mag_penalty=args.corr_mag_penalty)
        sched.step()
        if ep % 3 == 0 or ep == 1:
            vr = evaluate(model, val_loader, device)
            score = 0.4 * vr["fail_auc"] + 0.6 * max(vr.get("cosine_sim", 0), 0)
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 15:
                break

    if best_state:
        model.load_state_dict(best_state)
    result = evaluate(model, te_loader, device)
    result["n_test_succ"] = n_ts
    result["n_test_fail"] = n_tf
    return result, "OK"


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cartesian EEF Correction MLP")
    parser.add_argument("--success-data", default="data/combined/success_rollouts.pkl")
    parser.add_argument("--failure-data", default="data/combined/failure_rollouts.pkl")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-dir", default="checkpoints/eef_correction_mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-loo", action="store_true")
    parser.add_argument("--input-noise", type=float, default=0.01,
                        help="Gaussian noise σ added to hidden states during "
                             "training (regularisation). 0 = disabled.")
    parser.add_argument("--corr-mag-penalty", type=float, default=0.1,
                        help="L2 penalty weight on correction magnitude for "
                             "success samples (pushes safe-trajectory "
                             "predictions toward zero).")
    parser.add_argument("--subsample-chunks", action="store_true",
                        help="Only keep the first timestep of each action "
                             "chunk (where features change). Essential for "
                             "chunked policies like ACT to prevent inflated "
                             "metrics from duplicated features.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CARTESIAN EEF CORRECTION MLP")
    print("correction[t] = EEF_success(t) - EEF_failed(t)  [meters]")
    print("Cross-embodiment generalizable — 'move hand Xcm left'")
    print("=" * 70)
    print(f"  Architecture v4 (shortcut-learning fix):")
    print(f"    hidden_dim = {EEFCorrectionMLP.HIDDEN_DIM} (hardcoded)")
    print(f"    input LayerNorm = True")
    print(f"    dropout = 0.3 (all layers)")
    print(f"    activation = GELU")
    print(f"  Loss:")
    print(f"    classification = BCEWithLogitsLoss(dynamic per-batch pos_weight)")
    print(f"    regression = HuberLoss(δ=0.1)")
    print(f"    corr_mag_penalty = {args.corr_mag_penalty}")
    print(f"  Optimiser:")
    print(f"    AdamW  lr={args.lr}  weight_decay=1e-3")
    print(f"    input_noise σ = {args.input_noise}")
    print(f"    early stopping patience = 20")
    if args.subsample_chunks:
        print(f"  Chunk Subsampling: ENABLED")
        print(f"    Only chunk-boundary timesteps (unique features) will be used.")
        print(f"    This prevents inflated metrics from action-chunked policies.")

    print("\nLoading data...")
    with open(args.success_data, "rb") as f:
        succ = pickle.load(f)
    with open(args.failure_data, "rb") as f:
        fail = pickle.load(f)
    all_rollouts = succ + fail
    print(f"  {len(succ)}S + {len(fail)}F = {len(all_rollouts)} rollouts")

    succ_by_task = defaultdict(list)
    for r in succ:
        succ_by_task[r["task_id"]].append(r)

    # Prepare samples
    if args.subsample_chunks:
        print("\nPreparing samples (chunk-subsampled — unique features only)...")
    else:
        print("\nPreparing samples (EEF trajectory matching + Cartesian alignment)...")
    all_samples = prepare_samples(all_rollouts, succ_by_task,
                                  subsample_chunks=args.subsample_chunks)
    n = len(all_samples["labels"])
    nf = all_samples["labels"].sum()
    print(f"  Total: {n} samples ({nf:.0f} failure, {n-nf:.0f} success)")

    # Correction stats
    corr = all_samples["corrections"]
    fm = all_samples["labels"] > 0.5
    print(f"\n  Cartesian correction stats (failure samples):")
    for i, d in enumerate(DIM_NAMES_EEF):
        c = corr[fm, i]
        print(f"    {d}: mean={c.mean()*100:+.2f}cm  std={c.std()*100:.2f}cm  "
              f"|>1cm|={((np.abs(c) > 0.01).mean())*100:.1f}%")
    mag = np.linalg.norm(corr[fm], axis=1)
    print(f"    Magnitude: mean={mag.mean()*100:.2f}cm  median={np.median(mag)*100:.2f}cm  "
          f"max={mag.max()*100:.2f}cm")

    # ── Split ─────────────────────────────────────────────────────────────
    print("\nRollout-level split...")
    rids = all_samples["rollout_ids"]
    urids = np.unique(rids)
    np.random.shuffle(urids)
    n_tr = int(0.75 * len(urids))
    n_val = int(0.15 * len(urids))
    tr_set = set(urids[:n_tr])
    val_set = set(urids[n_tr:n_tr + n_val])
    te_set = set(urids[n_tr + n_val:])
    tr_m = np.array([r in tr_set for r in rids])
    val_m = np.array([r in val_set for r in rids])
    te_m = np.array([r in te_set for r in rids])
    print(f"  Train: {tr_m.sum()} | Val: {val_m.sum()} | Test: {te_m.sum()}")

    scaler = StandardScaler()
    tr_x = scaler.fit_transform(all_samples["hidden_states"][tr_m])
    val_x = scaler.transform(all_samples["hidden_states"][val_m])
    te_x = scaler.transform(all_samples["hidden_states"][te_m])

    tr_ds = EEFCorrectionDataset(tr_x, all_samples["labels"][tr_m],
                                  all_samples["corrections"][tr_m],
                                  all_samples["ttf"][tr_m],
                                  noise_std=args.input_noise, training=True)
    val_ds = EEFCorrectionDataset(val_x, all_samples["labels"][val_m],
                                   all_samples["corrections"][val_m],
                                   all_samples["ttf"][val_m])
    te_ds = EEFCorrectionDataset(te_x, all_samples["labels"][te_m],
                                  all_samples["corrections"][te_m],
                                  all_samples["ttf"][te_m])

    tr_labels = all_samples["labels"][tr_m]
    n_pos = tr_labels.sum()
    n_neg = len(tr_labels) - n_pos
    pw = n_neg / max(n_pos, 1)
    sw = np.where(tr_labels > 0.5, pw, 1.0)
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False)

    # Model (v4 architecture — hardcoded capacity)
    input_dim = tr_x.shape[1]
    model = EEFCorrectionMLP(input_dim=input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters (hidden={EEFCorrectionMLP.HIDDEN_DIM})")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING")
    print(f"  Anti-overfit: input_noise={args.input_noise}  "
          f"corr_mag_penalty={args.corr_mag_penalty}  "
          f"weight_decay=1e-3  dropout=0.3  "
          f"HuberLoss(δ=0.1)  dynamic_pos_weight")
    print("=" * 70)

    # Also evaluate on train set (without noise) to monitor overfitting
    tr_eval_ds = EEFCorrectionDataset(tr_x, all_samples["labels"][tr_m],
                                       all_samples["corrections"][tr_m],
                                       all_samples["ttf"][tr_m])
    tr_eval_loader = DataLoader(tr_eval_ds, batch_size=512, shuffle=False)

    best_score, best_epoch = 0, 0
    patience, no_imp = 20, 0
    training_curves = {"epoch": [], "train_loss": [], "val_loss": [],
                       "train_fail_auc": [], "val_fail_auc": [],
                       "train_cos_sim": [], "val_cos_sim": [],
                       "train_error_cm": [], "val_error_cm": [],
                       "loss_parts": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, parts = train_epoch(model, tr_loader, opt, device,
                                  corr_mag_penalty=args.corr_mag_penalty)
        vr = evaluate(model, val_loader, device)
        tr_eval = evaluate(model, tr_eval_loader, device)
        sched.step()

        cos = vr.get("cosine_sim", 0)
        tr_cos = tr_eval.get("cosine_sim", 0)
        score = 0.4 * vr["fail_auc"] + 0.6 * max(cos, 0)

        # Track curves
        training_curves["epoch"].append(epoch)
        training_curves["train_loss"].append(round(loss, 5))
        training_curves["train_fail_auc"].append(round(tr_eval["fail_auc"], 4))
        training_curves["val_fail_auc"].append(round(vr["fail_auc"], 4))
        training_curves["train_cos_sim"].append(round(tr_cos, 4))
        training_curves["val_cos_sim"].append(round(cos, 4))
        training_curves["train_error_cm"].append(
            round(tr_eval.get("mean_error_m", 0) * 100, 3))
        training_curves["val_error_cm"].append(
            round(vr.get("mean_error_m", 0) * 100, 3))
        training_curves["loss_parts"].append(
            {k: round(v, 5) for k, v in parts.items()})

        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_imp = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "input_dim": input_dim,
                "arch_version": "v4",
            }, save_dir / "best_model.pt")
        else:
            no_imp += 1

        if epoch % 5 == 0 or epoch == 1:
            dt = time.time() - t0
            # Overfitting gap detection
            gap_auc = tr_eval["fail_auc"] - vr["fail_auc"]
            gap_cos = tr_cos - cos
            overfit_warn = ""
            if gap_auc > 0.05 or gap_cos > 0.10:
                overfit_warn = "  ⚠ OVERFIT"
            elif gap_auc > 0.03 or gap_cos > 0.05:
                overfit_warn = "  ⚡ gap↑"

            print(f"  Ep {epoch:3d} | loss={loss:.4f} "
                  f"(fail={parts['fail']:.3f} corr={parts['corr']:.3f} "
                  f"mag={parts['mag_pen']:.4f}) | "
                  f"AUC tr={tr_eval['fail_auc']:.3f}/val={vr['fail_auc']:.3f} | "
                  f"cos tr={tr_cos:.3f}/val={cos:.3f} | "
                  f"err={vr.get('mean_error_m', 0)*100:.2f}cm | {dt:.1f}s"
                  f"{'  ★' if epoch == best_epoch else ''}"
                  f"{overfit_warn}")

        if no_imp >= patience:
            print(f"\n  Early stop at epoch {epoch} — no val improvement "
                  f"for {patience} epochs (best={best_epoch})")
            break

        # Cosine-gap early stop: halt if correction head is overfitting hard
        if epoch >= 15 and (tr_cos - cos) > 0.20:
            print(f"\n  ⚠ Cosine-gap early stop at epoch {epoch} — "
                  f"train cos={tr_cos:.3f} vs val cos={cos:.3f} "
                  f"(gap={tr_cos - cos:.3f} > 0.20)")
            break

    # Save training curves for post-hoc inspection
    curves_path = save_dir / "training_curves.json"
    with open(curves_path, "w") as f:
        json.dump(training_curves, f, indent=2)
    print(f"\n  Training curves saved: {curves_path}")

    # Final overfitting report
    if len(training_curves["train_fail_auc"]) > 5:
        final_tr_auc = training_curves["train_fail_auc"][-1]
        final_val_auc = training_curves["val_fail_auc"][-1]
        final_tr_cos = training_curves["train_cos_sim"][-1]
        final_val_cos = training_curves["val_cos_sim"][-1]
        print(f"\n  Overfitting check (final epoch):")
        print(f"    AUC gap:  train={final_tr_auc:.4f}  val={final_val_auc:.4f}  "
              f"Δ={final_tr_auc - final_val_auc:+.4f}")
        print(f"    Cos gap:  train={final_tr_cos:.4f}  val={final_val_cos:.4f}  "
              f"Δ={final_tr_cos - final_val_cos:+.4f}")
        if (final_tr_auc - final_val_auc) > 0.05:
            print(f"    ⚠  FAILURE-DETECTION HEAD may be overfitting "
                  f"(AUC gap > 0.05)")
        if (final_tr_cos - final_val_cos) > 0.10:
            print(f"    ⚠  CORRECTION HEAD may be overfitting "
                  f"(cosine gap > 0.10)")
        if ((final_tr_auc - final_val_auc) <= 0.05
                and (final_tr_cos - final_val_cos) <= 0.10):
            print(f"    ✅ No significant overfitting detected")

    # ── Best model test ───────────────────────────────────────────────────
    ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n" + "=" * 70)
    print(f"TEST RESULTS (best epoch {best_epoch})")
    print("=" * 70)

    tr = evaluate(model, te_loader, device)

    print(f"\n  FAILURE DETECTION:  AUC={tr['fail_auc']:.4f}  Acc={tr['fail_acc']:.4f}")
    print(f"  TIME-TO-FAILURE:   R²={tr['ttf_r2']:.4f}  Corr={tr['ttf_corr']:.4f}")
    print(f"\n  CARTESIAN CORRECTION (failure samples):")
    print(f"    Cosine Similarity:   {tr['cosine_sim']:.4f} (1.0=perfect direction)")
    print(f"    Mean Error:          {tr['mean_error_m']*100:.2f} cm")
    print(f"    Median Error:        {tr['median_error_m']*100:.2f} cm")
    print(f"    Magnitude R²:        {tr['mag_r2']:.4f}")
    print(f"\n  PER-AXIS:")
    print(f"    {'Axis':>6} {'R²':>8} {'Corr':>8} {'NeedAUC':>8} {'DirAcc':>8} {'DirAUC':>8}")
    print(f"    {'-'*48}")
    for d in DIM_NAMES_EEF:
        r2 = tr.get(f"{d}_r2", 0)
        corr = tr.get(f"{d}_corr", 0)
        need = tr.get(f"{d}_needs_auc", 0.5)
        da = tr.get(f"{d}_dir_acc", 0.5)
        dauc = tr.get(f"{d}_dir_auc", 0.5)
        v = "✅" if need > 0.7 else ("🟡" if need > 0.6 else "❌")
        print(f"    {d:>6} {r2:>8.4f} {corr:>8.4f} {need:>8.4f} {da:>8.4f} {dauc:>8.4f} {v}")

    # ── Per-task breakdown (TEST SPLIT ONLY — avoids inflated metrics) ──
    print("\n" + "=" * 70)
    print("PER-TASK BREAKDOWN (test split only — no train data leakage)")
    print("=" * 70)

    # Only evaluate on the held-out test rollouts
    te_tids = all_samples["task_ids"][te_m]
    te_hidden = all_samples["hidden_states"][te_m]
    te_labels = all_samples["labels"][te_m]
    te_corr = all_samples["corrections"][te_m]
    te_ttf = all_samples["ttf"][te_m]

    for tid in range(10):
        mask = te_tids == tid
        if mask.sum() < 10:
            continue
        tx = scaler.transform(te_hidden[mask])
        tds = EEFCorrectionDataset(tx, te_labels[mask],
                                    te_corr[mask],
                                    te_ttf[mask])
        tl = DataLoader(tds, batch_size=512, shuffle=False)
        tr = evaluate(model, tl, device)
        n_fail = (te_labels[mask] > 0.5).sum()
        n_safe = mask.sum() - n_fail
        print(f"  Task {tid} ({n_safe}S/{n_fail}F): "
              f"fail_AUC={tr['fail_auc']:.4f} | "
              f"cos_sim={tr.get('cosine_sim',0):.4f} | "
              f"err={tr.get('mean_error_m',0)*100:.2f}cm | "
              f"X_dir={tr.get('X_dir_acc',0.5):.3f} "
              f"Y_dir={tr.get('Y_dir_acc',0.5):.3f} "
              f"Z_dir={tr.get('Z_dir_acc',0.5):.3f}")

    # ── True LOO ──────────────────────────────────────────────────────────
    if args.run_loo:
        print("\n" + "=" * 70)
        print("TRUE LEAVE-ONE-TASK-OUT (train on 9, test on 1)")
        print("This is the ONLY valid generalization test.")
        print("=" * 70)

        loo = {}
        for held in range(10):
            t0 = time.time()
            print(f"\n  ── Fold {held}: held-out Task {held} ──")
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            result, status = run_loo_fold(held, all_rollouts, args, device)
            dt = time.time() - t0

            if result is None:
                print(f"    {status}")
                continue

            loo[held] = result
            print(f"    Fail AUC: {result['fail_auc']:.4f} | "
                  f"Cos Sim: {result.get('cosine_sim', 0):.4f} | "
                  f"Error: {result.get('mean_error_m', 0)*100:.2f}cm | "
                  f"({result['n_test_succ']}S/{result['n_test_fail']}F) | {dt:.1f}s")
            for d in DIM_NAMES_EEF:
                print(f"      {d}: need_AUC={result.get(f'{d}_needs_auc', 0.5):.3f}  "
                      f"dir_acc={result.get(f'{d}_dir_acc', 0.5):.3f}")

        # LOO Summary
        if loo:
            print("\n" + "=" * 70)
            print("LOO SUMMARY")
            print("=" * 70)

            print(f"  {'Task':>6} {'FailAUC':>8} {'CosSim':>7} {'ErrCm':>6} "
                  f"{'X_dir':>6} {'Y_dir':>6} {'Z_dir':>6}")
            print("  " + "-" * 52)

            agg = defaultdict(list)
            for tid in sorted(loo.keys()):
                r = loo[tid]
                agg["fail"].append(r["fail_auc"])
                agg["cos"].append(r.get("cosine_sim", 0))
                agg["err"].append(r.get("mean_error_m", 0))
                line = (f"  T{tid:>4} {r['fail_auc']:>8.4f} "
                        f"{r.get('cosine_sim',0):>7.4f} "
                        f"{r.get('mean_error_m',0)*100:>6.2f}")
                for d in DIM_NAMES_EEF:
                    da = r.get(f"{d}_dir_acc", 0.5)
                    agg[f"{d}_dir"].append(da)
                    line += f" {da:>6.3f}"
                print(line)

            mf = np.mean(agg["fail"])
            mc = np.mean(agg["cos"])
            me = np.mean(agg["err"])
            line = f"  {'AVG':>5} {mf:>8.4f} {mc:>7.4f} {me*100:>6.2f}"
            for d in DIM_NAMES_EEF:
                line += f" {np.mean(agg[f'{d}_dir']):>6.3f}"
            print(line)

            print(f"\n  ╔══════════════════════════════════════════════════════════╗")
            if mf >= 0.80:
                print(f"  ║  MEAN FAIL AUC: {mf:.4f}  →  GENERALIZABLE              ║")
            elif mf >= 0.65:
                print(f"  ║  MEAN FAIL AUC: {mf:.4f}  →  MODERATE                   ║")
            else:
                print(f"  ║  MEAN FAIL AUC: {mf:.4f}  →  TASK-SPECIFIC              ║")
            print(f"  ║  MEAN COSINE SIM: {mc:.4f}  (direction accuracy)          ║")
            print(f"  ║  MEAN ERROR: {me*100:.2f}cm  (Cartesian precision)           ║")
            print(f"  ╚══════════════════════════════════════════════════════════╝")

            loo_report = {
                "per_task": {str(k): {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                      for kk, vv in v.items()} for k, v in loo.items()},
                "mean_fail_auc": float(mf),
                "mean_cosine_sim": float(mc),
                "mean_error_m": float(me),
            }
        else:
            loo_report = {}

    # Save
    # Final overfitting gap (for inclusion in report)
    final_overfit = {}
    if len(training_curves["train_fail_auc"]) > 2:
        final_overfit = {
            "auc_gap": round(training_curves["train_fail_auc"][-1]
                             - training_curves["val_fail_auc"][-1], 4),
            "cos_gap": round(training_curves["train_cos_sim"][-1]
                             - training_curves["val_cos_sim"][-1], 4),
        }

    report = {
        "method": "cartesian_eef_correction",
        "description": "correction = EEF_success(t) - EEF_failed(t) in meters",
        "arch_version": "v4",
        "test_results": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in evaluate(model, te_loader, device).items()},
        "model_params": n_params,
        "best_epoch": best_epoch,
        "architecture": {
            "hidden_dim": EEFCorrectionMLP.HIDDEN_DIM,
            "input_norm": "LayerNorm(input_dim)",
            "dropout": 0.3,
            "activation": "GELU",
            "classification_loss": "BCEWithLogitsLoss(dynamic_pos_weight)",
            "regression_loss": "HuberLoss(delta=0.1)",
            "weight_decay": 1e-3,
            "input_noise_std": args.input_noise,
            "corr_mag_penalty": args.corr_mag_penalty,
            "early_stopping_patience": patience,
            "cosine_gap_stop": 0.20,
        },
        "overfitting_check": final_overfit,
    }
    if args.run_loo:
        report["loo"] = loo_report
    with open(save_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Model: {save_dir}/best_model.pt")
    print(f"  Results: {save_dir}/results.json")
    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
