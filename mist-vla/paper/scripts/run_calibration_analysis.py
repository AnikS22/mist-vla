#!/usr/bin/env python3
"""Per-model calibration and task-stratified analysis from completed eval JSONs."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER = Path("/home/mpcr/Desktop/SalusV5/mist-vla/paper")
DATA = PAPER / "data"
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)

OUT_JSON = DATA / "calibration_task_stratified_summary.json"
OUT_MD = PAPER / "CALIBRATION_TASK_REPORT.md"


@dataclass
class RunRow:
    file: str
    alpha: float
    max_corr: float
    corr_th: float
    fail_th: float
    vanilla: float
    mppi: float
    steering: float

    @property
    def delta_vs_mppi(self) -> float:
        return self.steering - self.mppi

    @property
    def delta_vs_vanilla(self) -> float:
        return self.steering - self.vanilla


def family(name: str) -> str | None:
    if name.startswith("category1_sweep_"):
        return "openvla"
    if name.startswith("eval_act_steering_sweep_"):
        return "act"
    return None


def load_rows(model: str) -> list[RunRow]:
    rows: list[RunRow] = []
    for fp in sorted(DATA.glob("*eval_results*.json")):
        fam = family(fp.name)
        if fam != model:
            continue
        d = json.loads(fp.read_text())
        cfg = d.get("config", {})
        s = d.get("summary", {})
        if not all(k in s for k in ("avg_vanilla_pct", "avg_mppi_pct", "avg_steering_pct")):
            continue
        rows.append(
            RunRow(
                file=fp.name,
                alpha=float(cfg.get("alpha", 0.0)),
                max_corr=float(cfg.get("max_correction_m", 0.0)),
                corr_th=float(cfg.get("correction_threshold_m", 0.0)),
                fail_th=float(cfg.get("fail_threshold", 0.0)),
                vanilla=float(s["avg_vanilla_pct"]),
                mppi=float(s["avg_mppi_pct"]),
                steering=float(s["avg_steering_pct"]),
            )
        )
    return rows


def robust_score(r: RunRow) -> float:
    # Conservative objective: avoid wins that come from sacrificing vanilla parity.
    return min(r.delta_vs_mppi, r.delta_vs_vanilla)


def recommend(rows: list[RunRow]) -> dict:
    if not rows:
        return {}
    ranked = sorted(rows, key=robust_score, reverse=True)
    topk = ranked[: max(1, len(ranked) // 4)]
    return {
        "n_runs": len(rows),
        "best_by_robust": ranked[0].__dict__ | {"delta_vs_mppi": ranked[0].delta_vs_mppi, "delta_vs_vanilla": ranked[0].delta_vs_vanilla},
        "median_params_top_quartile": {
            "alpha": float(np.median([r.alpha for r in topk])),
            "max_correction_m": float(np.median([r.max_corr for r in topk])),
            "correction_threshold_m": float(np.median([r.corr_th for r in topk])),
            "fail_threshold": float(np.median([r.fail_th for r in topk])),
        },
        "mean_top_quartile_deltas": {
            "steering_minus_mppi_pp": float(np.mean([r.delta_vs_mppi for r in topk])),
            "steering_minus_vanilla_pp": float(np.mean([r.delta_vs_vanilla for r in topk])),
        },
        "top5": [
            {
                "file": r.file,
                "alpha": r.alpha,
                "max_correction_m": r.max_corr,
                "correction_threshold_m": r.corr_th,
                "fail_threshold": r.fail_th,
                "steering_minus_mppi_pp": r.delta_vs_mppi,
                "steering_minus_vanilla_pp": r.delta_vs_vanilla,
                "robust_score": robust_score(r),
            }
            for r in ranked[:5]
        ],
    }


def task_stratified(model: str) -> dict:
    task_mode = defaultdict(lambda: defaultdict(lambda: {"succ": 0, "eps": 0}))
    files = 0
    for fp in sorted(DATA.glob("*eval_results*.json")):
        fam = family(fp.name)
        if fam != model:
            continue
        d = json.loads(fp.read_text())
        per = d.get("per_task", {})
        if not isinstance(per, dict):
            continue
        files += 1
        for tid, tdata in per.items():
            if not isinstance(tdata, dict):
                continue
            for mode in ("vanilla", "mppi", "steering"):
                m = tdata.get(mode, {})
                if not isinstance(m, dict):
                    continue
                ns = m.get("n_successes")
                ne = m.get("n_episodes")
                if isinstance(ns, (int, float)) and isinstance(ne, (int, float)) and ne > 0:
                    task_mode[tid][mode]["succ"] += int(ns)
                    task_mode[tid][mode]["eps"] += int(ne)

    out = {"n_files": files, "tasks": {}}
    for tid in sorted(task_mode.keys(), key=lambda x: int(x)):
        rec = {}
        for mode in ("vanilla", "mppi", "steering"):
            s = task_mode[tid][mode]["succ"]
            e = task_mode[tid][mode]["eps"]
            rec[mode] = 100.0 * s / e if e > 0 else 0.0
        rec["steering_minus_mppi_pp"] = rec["steering"] - rec["mppi"]
        rec["steering_minus_vanilla_pp"] = rec["steering"] - rec["vanilla"]
        out["tasks"][tid] = rec
    return out


def make_fig_task_deltas(model: str, strat: dict) -> None:
    tasks = sorted(strat["tasks"].keys(), key=lambda x: int(x))
    if not tasks:
        return
    d_mppi = [strat["tasks"][t]["steering_minus_mppi_pp"] for t in tasks]
    d_van = [strat["tasks"][t]["steering_minus_vanilla_pp"] for t in tasks]
    x = np.arange(len(tasks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(x - w / 2, d_mppi, width=w, label="Steering - MPPI", color="#5b7bb2")
    ax.bar(x + w / 2, d_van, width=w, label="Steering - Vanilla", color="#2e8b57")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t}" for t in tasks])
    ax.set_ylabel("Delta success (pp)")
    ax.set_title(f"{model.upper()} task-stratified deltas across completed sweeps")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    out = FIG / f"12_{model}_task_stratified_deltas.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_fig_param_vs_delta(model: str, rows: list[RunRow]) -> None:
    if not rows:
        return
    alphas = np.array([r.alpha for r in rows], dtype=float)
    d = np.array([r.delta_vs_mppi for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(alphas, d, c="#7a3db8", alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Steering - MPPI (pp)")
    ax.set_title(f"{model.upper()} calibration landscape (alpha vs delta)")
    ax.grid(alpha=0.25)
    out = FIG / f"13_{model}_alpha_vs_delta.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows_ov = load_rows("openvla")
    rows_act = load_rows("act")
    rec_ov = recommend(rows_ov)
    rec_act = recommend(rows_act)
    strat_ov = task_stratified("openvla")
    strat_act = task_stratified("act")

    make_fig_task_deltas("openvla", strat_ov)
    make_fig_task_deltas("act", strat_act)
    make_fig_param_vs_delta("openvla", rows_ov)
    make_fig_param_vs_delta("act", rows_act)

    payload = {
        "openvla_recommendation": rec_ov,
        "act_recommendation": rec_act,
        "openvla_task_stratified": strat_ov,
        "act_task_stratified": strat_act,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    md = [
        "# Calibration + Task-Stratified Report",
        "",
        "## OpenVLA calibration recommendation",
        f"- Runs analyzed: {rec_ov.get('n_runs', 0)}",
        f"- Median top-quartile params: `{rec_ov.get('median_params_top_quartile', {})}`",
        f"- Mean top-quartile deltas: `{rec_ov.get('mean_top_quartile_deltas', {})}`",
        "",
        "## ACT calibration recommendation",
        f"- Runs analyzed: {rec_act.get('n_runs', 0)}",
        f"- Median top-quartile params: `{rec_act.get('median_params_top_quartile', {})}`",
        f"- Mean top-quartile deltas: `{rec_act.get('mean_top_quartile_deltas', {})}`",
        "",
        "## Task-stratified files",
        "- `figures/12_openvla_task_stratified_deltas.png`",
        "- `figures/12_act_task_stratified_deltas.png`",
        "- `figures/13_openvla_alpha_vs_delta.png`",
        "- `figures/13_act_alpha_vs_delta.png`",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
