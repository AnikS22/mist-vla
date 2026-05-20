#!/usr/bin/env python3
"""Regenerate tables/tab_openvla_ood_status.tex from the s42/s43/s44 JSONs.

The placeholder text "PENDING: seeds s42/s43/s44 still running" is stale ---
all three JSONs in paper/data/ contain complete summary blocks. This script
aggregates Vanilla / Latent Stop / MPPI / Steering across the two tasks
(8, 9) for each seed, plus an across-seeds aggregate row.
"""

from __future__ import annotations

import json
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PAPER_DIR / "data"
TABLE_OUT = PAPER_DIR / "tables" / "tab_openvla_ood_status.tex"

SEEDS = [42, 43, 44]
TASKS = ["8", "9"]
MODES_DISPLAY = [
    ("vanilla", "Vanilla"),
    ("latent_stop", "Latent Stop"),
    ("mppi", "MPPI"),
    ("steering", "Steering"),
]


def load_seed(seed):
    path = DATA_DIR / f"category1_ovla_ood_t89_s{seed}_eval_results.json"
    with open(path) as f:
        return json.load(f)


def aggregate(per_task, mode_key):
    succ = 0
    n = 0
    for tid in TASKS:
        block = per_task[tid][mode_key]
        succ += block["n_successes"]
        n += block["n_episodes"]
    return succ, n


def main():
    rows = []
    pooled = {m: [0, 0] for m, _ in MODES_DISPLAY}
    for seed in SEEDS:
        data = load_seed(seed)
        per_task = data["per_task"]
        cells = []
        for mode_key, _label in MODES_DISPLAY:
            s, n = aggregate(per_task, mode_key)
            cells.append((s, n))
            pooled[mode_key][0] += s
            pooled[mode_key][1] += n
        cell_strs = [
            f"{(s/n)*100:.1f} ({s}/{n})" for s, n in cells
        ]
        rows.append((f"s{seed}", cell_strs))

    pooled_strs = []
    for mode_key, _label in MODES_DISPLAY:
        s, n = pooled[mode_key]
        pooled_strs.append(f"{(s/n)*100:.1f} ({s}/{n})")

    header = (
        "Seed & "
        + " & ".join(label for _, label in MODES_DISPLAY)
        + " \\\\"
    )
    body = []
    for tag, cells in rows:
        body.append(
            "\\texttt{" + tag + "} & " + " & ".join(cells) + " \\\\"
        )
    body.append("\\midrule")
    body.append("Aggregate (s42--s44) & " + " & ".join(pooled_strs) + " \\\\")

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{\\textbf{OpenVLA OOD baseline campaign on hard tasks 8--9} "
        "(seeds s42, s43, s44; $20$ episodes/task; OpenVLA-7B + OOD push). "
        "Each cell reports success rate (\\%) and successes/episodes across the two "
        "hard tasks for that seed. The aggregate row pools all three seeds "
        "($n{=}120$ per controller). All three seeds are complete; the previous "
        "placeholder version of this table was stale.}\n"
        "\\label{tab:openvla_ood_status}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        + header + "\n"
        "\\midrule\n"
        + "\n".join(body) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    TABLE_OUT.write_text(tex)
    print(f"wrote {TABLE_OUT}")
    print("Aggregate (across seeds):")
    for mode_key, label in MODES_DISPLAY:
        s, n = pooled[mode_key]
        print(f"  {label}: {(s/n)*100:.2f}%  ({s}/{n})")


if __name__ == "__main__":
    main()
