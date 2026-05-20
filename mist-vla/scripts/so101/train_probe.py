#!/usr/bin/env python3
"""Train the PULSE probe on real SO-101 rollouts.

Reuses `EEFCorrectionMLP` from the sim training script — same architecture,
same loss — and auto-detects the input feature dimension from the data (Pi0
PaliGemma trunk produces a different dim than OpenVLA, so we don't pin it).

Output: research_data/checkpoints/so101/<run_tag>/best_model.pt + results.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Use the canonical model class from the sim training script.
from scripts.train_eef_correction_mlp import EEFCorrectionMLP

THRESHOLDS = [0.50, 0.80, 0.99]


def features_per_traj(rollouts, label: int):
    out = []
    for r in rollouts:
        feats = [
            np.asarray(s["hidden_state"], dtype=np.float32)
            for s in r.get("steps", [])
            if "hidden_state" in s and s["hidden_state"] is not None
        ]
        if feats:
            out.append((np.stack(feats), label))
    return out


def split_trajs(succ, fail, seed=42):
    rng = np.random.default_rng(seed)
    traj_s = features_per_traj(succ, 0)
    traj_f = features_per_traj(fail, 1)
    all_t = traj_s + traj_f
    rng.shuffle(all_t)
    n = len(all_t)
    n_tr = max(1, int(0.75 * n))
    n_val = max(1, int(0.15 * n))
    tr, val, te = all_t[:n_tr], all_t[n_tr : n_tr + n_val], all_t[n_tr + n_val :]
    return tr, val, te


def cat(trajs):
    X = np.concatenate([x for x, _ in trajs])
    y = np.concatenate([np.full(len(x), l, dtype=np.float32) for x, l in trajs])
    return X, y


def threshold_metrics(y, probs):
    out = {"auc": float(roc_auc_score(y, probs))}
    for thr in THRESHOLDS:
        yp = (probs >= thr).astype(int)
        if yp.sum() == 0:
            d = {"precision": float("nan"), "recall": 0.0, "f1": 0.0, "accuracy": float(accuracy_score(y, yp)), "fire": 0.0}
        else:
            d = {
                "precision": float(precision_score(y, yp, zero_division=0)),
                "recall":    float(recall_score(y, yp, zero_division=0)),
                "f1":        float(f1_score(y, yp, zero_division=0)),
                "accuracy":  float(accuracy_score(y, yp)),
                "fire":      float((probs >= thr).mean()),
            }
        out[f"thr_{thr:.2f}"] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--save-dir", default=None, type=Path)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    with (args.data_dir / "success_rollouts.pkl").open("rb") as f:
        succ = pickle.load(f)
    with (args.data_dir / "failure_rollouts.pkl").open("rb") as f:
        fail = pickle.load(f)
    print(f"loaded {len(succ)} success + {len(fail)} failure trajectories from {args.data_dir}")
    if not succ or not fail:
        raise SystemExit("need ≥1 success and ≥1 failure to train")

    tr, val, te = split_trajs(succ, fail, seed=args.seed)
    Xtr, ytr = cat(tr); Xval, yval = cat(val); Xte, yte = cat(te)
    print(f"  train steps: {len(Xtr):,}  val: {len(Xval):,}  test: {len(Xte):,}")
    print(f"  pos rate (train/val/test): {ytr.mean():.3f} / {yval.mean():.3f} / {yte.mean():.3f}")
    print(f"  hidden_dim: {Xtr.shape[1]}  (auto-detected from data)")

    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr); Xval = sc.transform(Xval); Xte = sc.transform(Xte)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EEFCorrectionMLP(input_dim=Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    pw = (1 - ytr.mean()) / max(ytr.mean(), 1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    Xtr_t = torch.from_numpy(Xtr.astype(np.float32)).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xval_t = torch.from_numpy(Xval.astype(np.float32)).to(device)
    Xte_t = torch.from_numpy(Xte.astype(np.float32)).to(device)

    best = {"auc": 0.0, "epoch": 0, "state_dict": None}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(len(Xtr_t))
        for i in range(0, len(idx), args.bs):
            b = idx[i : i + args.bs]
            opt.zero_grad()
            out = model(Xtr_t[b])
            loss = bce(out["will_fail"], ytr_t[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v_logits = model(Xval_t)["will_fail"].cpu().numpy()
        v_probs = 1.0 / (1.0 + np.exp(-v_logits))
        v_auc = roc_auc_score(yval, v_probs) if len(np.unique(yval)) > 1 else 0.5
        if v_auc > best["auc"]:
            best.update(auc=v_auc, epoch=ep, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}: val AUC = {v_auc:.3f}  (best {best['auc']:.3f} @ ep {best['epoch']+1})")
    print(f"\ntraining done in {time.time()-t0:.1f}s; best val AUC {best['auc']:.3f}")

    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        te_logits = model(Xte_t)["will_fail"].cpu().numpy()
    te_probs = 1.0 / (1.0 + np.exp(-te_logits))
    te_metrics = threshold_metrics(yte, te_probs)
    print(f"\nTEST  AUC = {te_metrics['auc']:.3f}")
    for thr in THRESHOLDS:
        d = te_metrics[f"thr_{thr:.2f}"]
        print(f"  thr={thr:.2f}: prec={d['precision']:.3f}  rec={d['recall']:.3f}  F1={d['f1']:.3f}  acc={d['accuracy']:.3f}")

    save_dir = args.save_dir or (REPO / "research_data" / "checkpoints" / "so101" / args.data_dir.name)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": best["state_dict"],
        "input_dim": int(Xtr.shape[1]),
        "scaler_mean":  sc.mean_.astype(np.float32),
        "scaler_scale": sc.scale_.astype(np.float32),
        "val_auc": float(best["auc"]),
        "test_metrics": te_metrics,
    }
    torch.save(ckpt, save_dir / "best_model.pt")
    (save_dir / "results.json").write_text(json.dumps({
        "input_dim": int(Xtr.shape[1]),
        "val_auc":   float(best["auc"]),
        "test":      te_metrics,
        "n_train":   int(len(Xtr)),
        "n_val":     int(len(Xval)),
        "n_test":    int(len(Xte)),
    }, indent=2))
    print(f"\nwrote {save_dir / 'best_model.pt'} + results.json")
    print(f"[next] python scripts/so101/eval_realtime.py --probe {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
