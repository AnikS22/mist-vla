#!/usr/bin/env python3
"""Post-hoc Platt scaling for the PULSE failure logit (PyTorch).

OpenVLA's raw sigmoid outputs are severely miscalibrated (ECE ≈ 0.22 on the
trajectory-disjoint test fold) despite strong ranking (AUC ≈ 0.83). Platt
scaling fits a 2-parameter monotone map on logits:
    p_cal = sigmoid(a * logit + b)
which preserves ROC/AUC and typically collapses ECE toward ≤ 0.05.

Fits on the validation fold, reports ECE/Brier on the test fold (same split
protocol as ``eval_safety_head_thresholds.py``), and writes
``paper/data/platt_calibration.json``.

Usage::

    cd mist-vla && python3 scripts/platt_calibrate_failure_head.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.eval_safety_head_thresholds import (  # noqa: E402
    OUT_PATH as THRESHOLD_JSON,
    split_rollout_ids,
)
from scripts.train_eef_correction_mlp import EEFCorrectionMLP, prepare_samples  # noqa: E402

OUT_JSON = REPO / "paper" / "data" / "platt_calibration.json"


class PlattScaling(nn.Module):
    """Affine map on logits before sigmoid: σ(a·ℓ + b)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * logits + self.b)

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(logits.astype(np.float32))
            return self.forward(x).cpu().numpy()


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Standard ECE with uniform bins in [0, 1]."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += mask.sum() / n * abs(conf - acc)
    return float(ece)


def fit_platt(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int = 2000,
    lr: float = 0.05,
    device: str = "cpu",
) -> PlattScaling:
    """Fit Platt parameters by minimizing BCE on validation logits."""
    model = PlattScaling().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.from_numpy(logits.astype(np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    bce = nn.BCELoss()

    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = bce(model(x), y)
        loss.backward()
        opt.step()
    return model


def load_corpus(ckpt: Path, succ_pkl: Path, fail_pkl: Path):
    with succ_pkl.open("rb") as f:
        succ = pickle.load(f)
    with fail_pkl.open("rb") as f:
        fail = pickle.load(f)
    all_rollouts = succ + fail
    succ_by_task = {}
    for r in succ:
        succ_by_task.setdefault(r["task_id"], []).append(r)
    samples = prepare_samples(all_rollouts, succ_by_task, subsample_chunks=False)

    rids = samples["rollout_ids"]
    urids = np.unique(rids)
    tr_ids, val_ids, te_ids = split_rollout_ids(len(urids))
    tr_set = {urids[i] for i in tr_ids}
    val_set = {urids[i] for i in val_ids}
    te_set = {urids[i] for i in te_ids}

    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ckpt_data.get("model_state_dict", ckpt_data)
    input_dim = state["input_norm.weight"].shape[0]
    model = EEFCorrectionMLP(input_dim=input_dim)
    model.load_state_dict(state)
    model.eval()

    scaler = StandardScaler()
    tr_m = np.array([r in tr_set for r in rids])
    scaler.fit(samples["hidden_states"][tr_m])

    def logits_for(mask):
        x = scaler.transform(samples["hidden_states"][mask])
        with torch.no_grad():
            out = model(torch.from_numpy(x.astype(np.float32)))
        return out["will_fail"].cpu().numpy()

    val_m = np.array([r in val_set for r in rids])
    te_m = np.array([r in te_set for r in rids])
    return {
        "val_logits": logits_for(val_m),
        "val_labels": samples["labels"][val_m],
        "test_logits": logits_for(te_m),
        "test_labels": samples["labels"][te_m],
    }


def evaluate_split(logits: np.ndarray, labels: np.ndarray, platt: PlattScaling | None):
    raw_p = 1.0 / (1.0 + np.exp(-logits))
    if platt is None:
        probs = raw_p
    else:
        probs = platt.calibrate_probs(logits)
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "ece": expected_calibration_error(probs, labels),
        "brier": float(brier_score_loss(labels, probs)),
        "frac_above_0.99": float((probs >= 0.99).mean()),
        "mean_prob": float(probs.mean()),
    }


def run_arch(ckpt: Path, succ_pkl: Path, fail_pkl: Path, label: str) -> dict:
    print(f"\n=== {label} ===")
    data = load_corpus(ckpt, succ_pkl, fail_pkl)
    platt = fit_platt(data["val_logits"], data["val_labels"])
    a = float(platt.a.detach().cpu().item())
    b = float(platt.b.detach().cpu().item())
    print(f"  Platt: a={a:.4f}, b={b:.4f}")

    raw_test = evaluate_split(data["test_logits"], data["test_labels"], None)
    cal_test = evaluate_split(data["test_logits"], data["test_labels"], platt)
    print(f"  test ECE raw={raw_test['ece']:.4f} -> cal={cal_test['ece']:.4f}")
    print(f"  test Brier raw={raw_test['brier']:.4f} -> cal={cal_test['brier']:.4f}")
    print(f"  test AUC (should match): raw={raw_test['auc']:.4f} cal={cal_test['auc']:.4f}")

    return {
        "label": label,
        "platt_a": a,
        "platt_b": b,
        "val_steps": int(len(data["val_labels"])),
        "test_steps": int(len(data["test_labels"])),
        "test_raw": raw_test,
        "test_calibrated": cal_test,
    }


def main():
    targets = [
        (
            REPO / "research_data/checkpoints/eef_correction_mlp/best_model.pt",
            REPO / "research_data/rollouts/openvla_spatial_seed0/success_rollouts.pkl",
            REPO / "research_data/rollouts/openvla_spatial_seed0/failure_rollouts.pkl",
            "openvla",
        ),
        (
            REPO / "research_data/checkpoints/eef_correction_mlp_act_honest/best_model.pt",
            REPO / "research_data/rollouts/multi_model/act_spatial/success_rollouts.pkl",
            REPO / "research_data/rollouts/multi_model/act_spatial/failure_rollouts.pkl",
            "act",
        ),
    ]

    results = {}
    for ckpt, succ, fail, key in targets:
        if not ckpt.exists():
            print(f"Skip {key}: missing {ckpt}")
            continue
        results[key] = run_arch(ckpt, succ, fail, key)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_JSON}")
    if THRESHOLD_JSON.exists():
        print(f"(threshold table source: {THRESHOLD_JSON})")


if __name__ == "__main__":
    main()
