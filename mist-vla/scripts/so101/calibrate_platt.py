#!/usr/bin/env python3
"""Fit a 2-parameter Platt scaler on val-set probe logits + labels.

Platt scaling fits sigmoid(a * logit + b) on a held-out validation set so that
the raw probe logit is mapped to a calibrated probability. Used by
score_table3.py to compute the "post-Platt ECE" cell.

Inputs:
    --rollouts-dir   directory of labeled rollouts (val split or full pool)
    --probe-ckpt     path to PULSE probe (best_model.pt)
Outputs:
    <output>/platt.json   {"a": ..., "b": ..., "ece_pre": ..., "ece_post": ...}
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from scripts.train_eef_correction_mlp import EEFCorrectionMLP


def gather(rolls_dir: Path, probe, sc_mean, sc_scale, device):
    logits_all, labels_all = [], []
    for p in sorted(rolls_dir.glob("ep*.pkl")):
        with p.open("rb") as f:
            r = pickle.load(f)
        ep_label = 0 if r["success"] else 1
        h = np.stack(r["features"], axis=0)
        with torch.no_grad():
            x = torch.from_numpy(h.astype(np.float32)).to(device)
            x = (x - sc_mean) / sc_scale
            z = probe(x)["will_fail"].squeeze(-1).cpu().numpy()
        logits_all.append(z)
        labels_all.append(np.full(len(z), ep_label))
    return np.concatenate(logits_all), np.concatenate(labels_all)


def ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i + 1])
        if not m.any(): continue
        e += (m.sum() / len(probs)) * abs(probs[m].mean() - (labels[m] == 1).mean())
    return float(e)


def fit_platt(logits, labels):
    """Minimize NLL of sigmoid(a*z + b) against labels."""
    y = labels.astype(np.float64)

    def nll(params):
        a, b = params
        s = a * logits + b
        # numerically stable BCE
        log_sig = -np.logaddexp(0, -s)        # log(sigmoid(s))
        log_1ms = -np.logaddexp(0, s)         # log(1 - sigmoid(s))
        return -np.mean(y * log_sig + (1 - y) * log_1ms)

    # Starting guess: a=-1 (so larger logit -> larger prob via 1/(1+exp(a*z+b)))
    # but the convention here is sigmoid(a*z+b) -> prob; want logit -> prob, so a=1 if
    # high logit means fail. Probe was trained with BCE so high logit = high p(fail).
    x0 = np.array([1.0, 0.0])
    res = minimize(nll, x0, method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-dir", type=Path, required=True)
    ap.add_argument("--probe-ckpt", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.probe_ckpt, map_location=device, weights_only=False)
    probe = EEFCorrectionMLP(input_dim=blob["input_dim"]).to(device).eval()
    probe.load_state_dict(blob["model_state_dict"])
    sc_mean = torch.from_numpy(blob["scaler_mean"]).to(device)
    sc_scale = torch.from_numpy(blob["scaler_scale"]).to(device)

    logits, labels = gather(args.rollouts_dir, probe, sc_mean, sc_scale, device)
    print(f"[gather] {len(logits)} step-level samples  pos={int(labels.sum())}  neg={int((labels==0).sum())}")

    probs_pre = 1.0 / (1.0 + np.exp(-logits))
    ece_pre = ece(probs_pre, labels)
    print(f"[pre]  ECE = {ece_pre:.4f}")

    a, b = fit_platt(logits, labels)
    probs_post = 1.0 / (1.0 + np.exp(-(a * logits + b)))  # sigmoid(a*z+b) form
    ece_post = ece(probs_post, labels)
    print(f"[post] a={a:+.4f}  b={b:+.4f}  ECE = {ece_post:.4f}")

    out = args.output or args.probe_ckpt.parent / "platt.json"
    out.write_text(json.dumps({
        "a": a, "b": b, "ece_pre": ece_pre, "ece_post": ece_post,
        "n_samples": int(len(logits)), "form": "p = sigmoid(a*logit + b)",
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
