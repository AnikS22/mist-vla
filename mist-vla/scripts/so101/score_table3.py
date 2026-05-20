#!/usr/bin/env python3
"""Produce all six Table 3 cells from logged SO-101 rollouts.

Inputs:
    --vanilla-dir   directory of vanilla closed-loop episodes (eval_realtime --controller vanilla)
    --gating-dir    directory of single-step gating episodes (eval_realtime --controller gating)
    --probe-ckpt    path to trained PULSE probe (best_model.pt + results.json)
    --platt-json    optional Platt scaler json from calibrate_platt.py

Outputs:
    Prints six numbers ready to drop into paper/tables/tab_so101_results.tex
    and writes a json summary to <output-dir>/table3_cells.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from scripts.train_eef_correction_mlp import EEFCorrectionMLP


def load_rollouts(d: Path) -> list[dict]:
    rolls = []
    for p in sorted(d.glob("ep*.pkl")):
        with p.open("rb") as f:
            rolls.append(pickle.load(f))
    return rolls


def load_probe(ckpt: Path, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    probe = EEFCorrectionMLP(input_dim=blob["input_dim"]).to(device)
    probe.load_state_dict(blob["model_state_dict"])
    probe.eval()
    sc_mean = torch.from_numpy(blob["scaler_mean"]).to(device)
    sc_scale = torch.from_numpy(blob["scaler_scale"]).to(device)
    return probe, sc_mean, sc_scale


def score_episodes(rolls: list[dict], probe, sc_mean, sc_scale, device,
                   platt: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs_per_step, episode_label_per_step) arrays."""
    probs, labels = [], []
    for r in rolls:
        ep_label = 0 if r["success"] else 1  # 1 = failure (positive class for AUC)
        h_arr = np.stack(r["features"], axis=0)
        with torch.no_grad():
            x = torch.from_numpy(h_arr.astype(np.float32)).to(device)
            x = (x - sc_mean) / sc_scale
            logits = probe(x)["will_fail"].squeeze(-1).cpu().numpy()
        if platt is not None:
            a = float(platt["a"]); b = float(platt["b"])
            # Platt: p = 1 / (1 + exp(a * logit + b))
            scaled = 1.0 / (1.0 + np.exp(a * logits + b))
        else:
            scaled = 1.0 / (1.0 + np.exp(-logits))
        probs.extend(scaled.tolist())
        labels.extend([ep_label] * len(scaled))
    return np.array(probs), np.array(labels)


def roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """AUC via rank-based formula (Mann-Whitney U / nP*nN)."""
    pos = probs[labels == 1]; neg = probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i + 1])
        if not m.any():
            continue
        conf = probs[m].mean()
        acc = (labels[m] == 1).mean()  # fraction of failure-positive
        ece += (m.sum() / len(probs)) * abs(conf - acc)
    return float(ece)


def two_proportion_z_test(n1: int, s1: int, n2: int, s2: int) -> tuple[float, float]:
    """Two-proportion z-test. Returns (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = s1 / n1; p2 = s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = (p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2)) ** 0.5
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided p via normal CDF approximation
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return float(z), float(p)


def pooled_latency(rolls: list[dict]) -> dict:
    """Aggregate per-step total_ms across all episodes."""
    totals = []
    for r in rolls:
        for s in r.get("step_timings_ms", []):
            totals.append(s.get("total_ms", 0.0))
    if not totals:
        return {"mean_ms": float("nan"), "p95_ms": float("nan"), "n_steps": 0}
    arr = np.array(totals)
    return {"mean_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95)),
            "n_steps": len(arr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-dir", type=Path, required=True)
    ap.add_argument("--gating-dir", type=Path, required=True)
    ap.add_argument("--probe-ckpt", type=Path, required=True)
    ap.add_argument("--platt-json", type=Path, default=None,
                    help="Optional Platt scaler from calibrate_platt.py")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = args.output_dir or args.vanilla_dir.parent / "table3"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] vanilla={args.vanilla_dir}  gating={args.gating_dir}")
    vanilla = load_rollouts(args.vanilla_dir)
    gating  = load_rollouts(args.gating_dir)
    print(f"        vanilla: {len(vanilla)} ep   gating: {len(gating)} ep")

    probe, sc_mean, sc_scale = load_probe(args.probe_ckpt, device)
    platt = json.loads(args.platt_json.read_text()) if args.platt_json else None

    # AUC + ECE from vanilla (clean detection metric)
    probs, labels = score_episodes(vanilla, probe, sc_mean, sc_scale, device, platt=None)
    auc_raw = roc_auc(probs, labels)
    ece_raw = expected_calibration_error(probs, labels)
    if platt is not None:
        probs_p, labels_p = score_episodes(vanilla, probe, sc_mean, sc_scale, device, platt=platt)
        ece_post = expected_calibration_error(probs_p, labels_p)
        auc_post = roc_auc(probs_p, labels_p)  # should match auc_raw (monotonic)
    else:
        ece_post = float("nan")
        auc_post = float("nan")

    # Success rates
    n_v = len(vanilla); s_v = sum(1 for r in vanilla if r["success"])
    n_g = len(gating);  s_g = sum(1 for r in gating  if r["success"])
    z, p = two_proportion_z_test(n_v, s_v, n_g, s_g)

    # Latency (from gating mode, since that's the deployment configuration)
    lat = pooled_latency(gating)
    if lat["n_steps"] == 0:
        # fall back to vanilla if gating wasn't timed
        lat = pooled_latency(vanilla)

    cells = {
        "Real-hardware failure-detection AUC": auc_raw,
        "ECE (raw)": ece_raw,
        "ECE (post-Platt)": ece_post,
        "Vanilla closed-loop success": s_v / max(n_v, 1),
        "PULSE single-step gating success": s_g / max(n_g, 1),
        "Two-proportion z-test (p)": p,
        "z_statistic": z,
        "Embedded-compute latency mean (ms)": lat["mean_ms"],
        "Embedded-compute latency p95 (ms)": lat["p95_ms"],
        "counts": {
            "vanilla_n": n_v, "vanilla_success": s_v,
            "gating_n": n_g, "gating_success": s_g,
            "latency_n_steps": lat["n_steps"],
        },
    }

    print("\n=== Table 3 cells ===")
    for k, v in cells.items():
        if isinstance(v, dict):
            continue
        print(f"  {k:50s} {v:.4f}" if isinstance(v, float) else f"  {k:50s} {v}")
    print(f"\n  counts: {cells['counts']}")

    (out / "table3_cells.json").write_text(json.dumps(cells, indent=2))
    print(f"\nwrote {out / 'table3_cells.json'}")

    # Also emit a ready-to-paste LaTeX snippet
    latex = f"""% drop into paper/tables/tab_so101_results.tex (replace TODO cells)
% generated by scripts/so101/score_table3.py
Real-hardware failure-detection AUC (target ${{>}}0.70$) & {auc_raw:.3f} \\\\
ECE (post-Platt) & {ece_post:.3f} \\\\
Vanilla closed-loop success ($N{{=}}{n_v}$/task) & {100 * s_v / max(n_v, 1):.1f}\\% \\\\
PULSE single-step gating success & {100 * s_g / max(n_g, 1):.1f}\\% \\\\
Two-proportion $z$-test ($p$) & {p:.3f} \\\\
Embedded-compute latency, total/step (ms) & {lat['mean_ms']:.2f} \\\\
"""
    (out / "table3_latex.tex").write_text(latex)
    print(f"wrote {out / 'table3_latex.tex'}")


if __name__ == "__main__":
    main()
