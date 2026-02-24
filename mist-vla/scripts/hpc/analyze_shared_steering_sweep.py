#!/usr/bin/env python3
"""Analyze paired OpenVLA+ACT sweep runs from real eval JSON files only."""

import argparse
import json
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score shared steering sweep runs.")
    ap.add_argument("--ledger", default="results/hpc/shared_steering_sweep_jobs.jsonl")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    ledger = Path(args.ledger)
    if not ledger.exists():
        raise SystemExit(f"Ledger not found: {ledger}")

    rows = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        paper = _load(Path(rec["paper_results"]))
        act = _load(Path(rec["act_results"]))
        if not paper or not act:
            continue

        ps = paper.get("summary", {})
        ac = act.get("summary", {})
        row = {
            "run_tag": rec["run_tag"],
            "alpha": rec["params"]["ALPHA"],
            "max_corr": rec["params"]["MAX_CORR"],
            "corr_th": rec["params"]["CORR_THRESH"],
            "fail_th": rec["params"]["FAIL_THRESH"],
            "paper_steering": float(ps.get("avg_steering_pct", 0.0)),
            "paper_mppi": float(ps.get("avg_mppi_pct", 0.0)),
            "act_steering": float(ac.get("avg_steering_pct", 0.0)),
            "act_mppi": float(ac.get("avg_mppi_pct", 0.0)),
        }
        row["paper_delta_vs_mppi"] = row["paper_steering"] - row["paper_mppi"]
        row["act_delta_vs_mppi"] = row["act_steering"] - row["act_mppi"]
        row["mean_delta_vs_mppi"] = (
            row["paper_delta_vs_mppi"] + row["act_delta_vs_mppi"]
        ) / 2.0
        rows.append(row)

    if not rows:
        print("No completed paired results yet.")
        return

    rows.sort(key=lambda r: r["mean_delta_vs_mppi"], reverse=True)
    print("Top runs by mean(steering - mppi) across OpenVLA and ACT:")
    for i, r in enumerate(rows[: args.top_k], start=1):
        print(
            f"{i:>2}. {r['run_tag']}  "
            f"meanΔ={r['mean_delta_vs_mppi']:+.2f}pp  "
            f"OpenVLAΔ={r['paper_delta_vs_mppi']:+.2f}pp  "
            f"ACTΔ={r['act_delta_vs_mppi']:+.2f}pp  "
            f"(a={r['alpha']} mc={r['max_corr']} ct={r['corr_th']} ft={r['fail_th']})"
        )


if __name__ == "__main__":
    main()
