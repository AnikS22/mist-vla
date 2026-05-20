#!/usr/bin/env python3
"""Evaluate the trained EEF-correction MLP at multiple decision thresholds.

Loads the existing PULSE probe (`research_data/checkpoints/eef_correction_mlp/best_model.pt`
for OpenVLA, `research_data/checkpoints/eef_correction_mlp_act_honest/best_model.pt` for ACT),
runs inference on a trajectory-disjoint test split, and writes precision / recall / F1 /
accuracy at sigmoid thresholds {0.50, 0.80, 0.99} into
`paper/data/safety_head_thresholds.json`.

Usage:
    cd mist-vla && python3 scripts/eval_safety_head_thresholds.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Reuse the canonical model class from the training script.
from scripts.train_eef_correction_mlp import EEFCorrectionMLP, prepare_samples  # noqa: E402

THRESHOLDS = [0.50, 0.80, 0.99]
SEED = 42
OUT_PATH = REPO / "paper" / "data" / "safety_head_thresholds.json"


def split_rollout_ids(n_unique_ids: int, seed: int = SEED):
    """Match the trainer's split protocol exactly so the threshold-replay
    AUCs reproduce the trainer's headline AUCs in tab_safety_head_metrics.

    The trainer (scripts/train_eef_correction_mlp.py:711) does
    ``np.random.seed(args.seed)`` and then ``np.random.shuffle(urids)``
    where ``urids`` is the sorted unique rollout-id array. We replicate
    that protocol here instead of the previous ``np.random.default_rng``
    split, which evaluated the same checkpoint on a different fold and
    produced spuriously higher AUCs.
    """
    np.random.seed(seed)
    perm = np.arange(n_unique_ids)
    np.random.shuffle(perm)
    n_tr = int(0.75 * n_unique_ids)
    n_val = int(0.15 * n_unique_ids)
    return perm[:n_tr], perm[n_tr : n_tr + n_val], perm[n_tr + n_val :]


def eval_threshold_metrics(probs: np.ndarray, y: np.ndarray):
    metrics = {"auc": float(roc_auc_score(y, probs))}
    for thr in THRESHOLDS:
        ypred = (probs >= thr).astype(int)
        fire = float((probs >= thr).mean())
        if ypred.sum() == 0:
            prec, rec, f1, acc = float("nan"), 0.0, 0.0, float(accuracy_score(y, ypred))
        else:
            prec = float(precision_score(y, ypred, zero_division=0))
            rec = float(recall_score(y, ypred, zero_division=0))
            f1 = float(f1_score(y, ypred, zero_division=0))
            acc = float(accuracy_score(y, ypred))
        metrics[f"thr_{thr:.2f}"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "fire_rate": fire,
        }
    return metrics


def evaluate_checkpoint(ckpt_path: Path, success_pkl: Path, failure_pkl: Path, label: str):
    print(f"\n=== {label} ===")
    print(f"  checkpoint: {ckpt_path}")
    with success_pkl.open("rb") as f:
        succ = pickle.load(f)
    with failure_pkl.open("rb") as f:
        fail = pickle.load(f)
    all_rollouts = succ + fail
    print(f"  rollouts: {len(succ)} succ + {len(fail)} fail = {len(all_rollouts)}")

    succ_by_task = {}
    for r in succ:
        succ_by_task.setdefault(r.get("task_id"), []).append(r)
    samples = prepare_samples(all_rollouts, succ_by_task, subsample_chunks=False)

    rids = samples["rollout_ids"]
    urids = np.unique(rids)
    tr_ids, val_ids, te_ids = split_rollout_ids(len(urids))
    te_set = {urids[i] for i in te_ids}
    te_m = np.array([r in te_set for r in rids])
    print(f"  test steps: {te_m.sum():,} / {len(rids):,}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    input_dim = state["input_norm.weight"].shape[0]
    model = EEFCorrectionMLP(input_dim=input_dim)
    model.load_state_dict(state)
    model.eval()

    scaler = StandardScaler()
    tr_m = np.array([r in {urids[i] for i in tr_ids} for r in rids])
    scaler.fit(samples["hidden_states"][tr_m])
    te_x = scaler.transform(samples["hidden_states"][te_m])
    y = samples["labels"][te_m]

    with torch.no_grad():
        out = model(torch.from_numpy(te_x.astype(np.float32)))
        probs = torch.sigmoid(out["will_fail"]).cpu().numpy()

    m = eval_threshold_metrics(probs, y)
    print(f"  AUC = {m['auc']:.3f}  (test pos rate {y.mean():.3f})")
    for thr in THRESHOLDS:
        d = m[f"thr_{thr:.2f}"]
        print(
            f"    thr={thr:.2f}: prec={d['precision']:.3f}  rec={d['recall']:.3f}  "
            f"F1={d['f1']:.3f}  acc={d['accuracy']:.3f}  fire={d['fire_rate']*100:.1f}%"
        )
    return m


def main():
    # NOTE: both targets use the per-arch pool the trainer used
    # (openvla_spatial_seed0 and multi_model/act_spatial). Combined with
    # the trainer-matched seed protocol in ``split_rollout_ids`` above
    # this puts the threshold replay on the same trajectory-disjoint
    # test fold the trainer used. The per-arch OpenVLA pool also avoids
    # the saturated-sigmoid / recall=1.000 degeneracy that the merged_all
    # pool used to produce (reviewer W6).
    targets = [
        (
            REPO / "research_data" / "checkpoints" / "eef_correction_mlp" / "best_model.pt",
            REPO / "research_data" / "rollouts" / "openvla_spatial_seed0" / "success_rollouts.pkl",
            REPO / "research_data" / "rollouts" / "openvla_spatial_seed0" / "failure_rollouts.pkl",
            "OpenVLA (4096-d)",
        ),
        (
            REPO / "research_data" / "checkpoints" / "eef_correction_mlp_act_honest" / "best_model.pt",
            REPO / "research_data" / "rollouts" / "multi_model" / "act_spatial" / "success_rollouts.pkl",
            REPO / "research_data" / "rollouts" / "multi_model" / "act_spatial" / "failure_rollouts.pkl",
            "ACT (256-d)",
        ),
    ]
    results = {}
    for ckpt, succ, fail, label in targets:
        if not ckpt.exists():
            print(f"[skip] missing checkpoint: {ckpt}")
            continue
        results[label] = evaluate_checkpoint(ckpt, succ, fail, label)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
