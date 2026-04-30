#!/usr/bin/env python3
"""Quick consistency checks across stats JSON, key tables, and results text."""

from __future__ import annotations

import json
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]


def parse_table_counts(tex: str, method: str):
    for line in tex.splitlines():
        if not line.strip().startswith(method):
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 3:
            continue
        try:
            pct = float(parts[1])
            succ_str = parts[2].split()[0]
            succ, eps = succ_str.split("/")
            return (pct, int(succ), int(eps))
        except Exception:
            continue
    return None


def main() -> None:
    stat = json.loads((PAPER / "data/stat_tests_summary.json").read_text())
    tab_final = (PAPER / "tables/tab_final_pooled_results.tex").read_text()
    tab_stat = (PAPER / "tables/tab_stat_tests.tex").read_text()
    results = (PAPER / "sections/results.tex").read_text()

    pc = stat["families"]["paper_curated"]["comparisons"]
    raw_s = pc["steering_vs_mppi"]["raw"]["steering"]
    raw_m = pc["steering_vs_mppi"]["raw"]["mppi"]
    raw_v = pc["steering_vs_vanilla"]["raw"]["vanilla"]

    v = parse_table_counts(tab_final, "Vanilla")
    m = parse_table_counts(tab_final, "MPPI")
    s = parse_table_counts(tab_final, "Steering (Ours)")
    assert v and m and s, "Could not parse key final table rows"
    assert v[1:] == (raw_v["succ"], raw_v["eps"]), "Vanilla count mismatch"
    assert m[1:] == (raw_m["succ"], raw_m["eps"]), "MPPI count mismatch"
    assert s[1:] == (raw_s["succ"], raw_s["eps"]), "Steering count mismatch"

    # key p-values appear in stat table
    for token in ["0.5609", "0.65", "0.8984"]:
        assert token in tab_stat, f"Missing expected p-value token {token}"

    # key headline tokens in results section
    for token in ["52.13", "52.22", "51.81", "5192/9960", "5201/9960", "5160/9960"]:
        assert token in results, f"Missing headline token {token} in results.tex"

    print("[ok] Consistency checks passed.")


if __name__ == "__main__":
    main()

