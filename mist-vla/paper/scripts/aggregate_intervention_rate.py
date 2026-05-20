#!/usr/bin/env python3
"""Aggregate steering `mean_ir` across paper/data/*.json -> tab_intervention_rate.

Sources the ~40% intervention-rate claim in method.tex (current pooled mean
is 39.6%, median 36.85%, n=558 run-task cells). We walk every
*eval_results.json in paper/data/, pull the steering mode's `mean_ir`
field per task, and report the pooled distribution + the headline mean.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PAPER_DIR / "data"
TABLE_OUT = PAPER_DIR / "tables" / "tab_intervention_rate.tex"


def iter_steering_irs():
    """Yield (path, task_id, mean_ir) for every steering-mode entry found."""
    for p in sorted(DATA_DIR.glob("*eval_results*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        per_task = data.get("per_task")
        if not per_task:
            continue
        for tid, modes in per_task.items():
            steer = modes.get("steering")
            if not steer or "mean_ir" not in steer:
                continue
            yield p.name, tid, float(steer["mean_ir"])


def emit_table(irs):
    if not irs:
        raise SystemExit("no steering mean_ir found")
    values = [ir for _, _, ir in irs]
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    n = len(values)
    p5 = sorted(values)[max(0, int(0.05 * n))]
    p95 = sorted(values)[min(n - 1, int(0.95 * n))]

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{\\textbf{Source of the $\\sim$40\\% intervention-rate claim in \\S\\ref{sec:method_training}.} "
        "Pooled distribution of \\texttt{mean\\_ir} (fraction of timesteps at which "
        "the double gate fires) across every per-task steering-mode entry in "
        "\\texttt{paper/data/*eval\\_results*.json}. $n$ is the number of "
        "(run, task) cells. The mean is the number quoted in the Method "
        "section; the spread covers easy tasks (low intervention) and hard "
        "tasks (high intervention), consistent with the self-calibration "
        "result in Section~\\ref{sec:adaptive_gating}.}\n"
        "\\label{tab:intervention_rate}\n"
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "Statistic & Value \\\\\n"
        "\\midrule\n"
        f"Mean intervention rate & \\textbf{{{mean*100:.1f}\\%}} \\\\\n"
        f"Median intervention rate & {median*100:.1f}\\% \\\\\n"
        f"Standard deviation & {stdev*100:.1f}\\,pp \\\\\n"
        f"5th--95th percentile & [{p5*100:.1f}\\%, {p95*100:.1f}\\%] \\\\\n"
        f"$n$ (run-task cells) & {n} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    TABLE_OUT.write_text(tex)
    print(f"wrote {TABLE_OUT}")
    print(f"mean={mean*100:.2f}% median={median*100:.2f}% n={n}")


def main():
    irs = list(iter_steering_irs())
    emit_table(irs)


if __name__ == "__main__":
    main()
