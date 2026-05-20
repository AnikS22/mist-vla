#!/usr/bin/env python3
"""Consistency checks across stat_tests_summary.json, key LaTeX tables, and prose.

Run from repo:  python3 paper/scripts/check_paper_consistency.py
(or cd paper/scripts && python3 check_paper_consistency.py)

Does not rely on hard-coded historic headline tokens — values are verified against
`data/stat_tests_summary.json` (paper_curated pooled counts / Wilson summaries) and
primary z-test *p*-values echoed in `tables/tab_stat_tests.tex`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]


def check_stranded_tables() -> None:
    """Fail if any tables/*.tex is not \\input'd by main.tex or sections/*.tex.

    Archive subfolder is exempt. cut_content.tex is exempt because it is not
    compiled (commented \\input lines only).
    """
    tables_dir = PAPER / "tables"
    on_disk = {p.stem for p in tables_dir.glob("*.tex")}
    referenced: set[str] = set()
    candidates = [PAPER / "main.tex"] + list((PAPER / "sections").glob("*.tex"))
    pat = re.compile(r"^\s*\\input\{tables/([A-Za-z0-9_]+)\}", re.MULTILINE)
    for p in candidates:
        if p.name == "cut_content.tex":
            continue
        text = p.read_text()
        for m in pat.finditer(text):
            referenced.add(m.group(1))
    stranded = on_disk - referenced
    if stranded:
        raise AssertionError(
            "Stranded tables (in tables/ but not \\input'd anywhere): "
            + ", ".join(sorted(stranded))
            + ". Either \\input them, move to tables/archive/, or delete."
        )


def parse_final_pooled(tex: str) -> dict[str, tuple[float, int, int, str]]:
    """Map row label -> (pct, succ, eps, apply_col_raw)."""
    rows: dict[str, tuple[float, int, int, str]] = {}
    for line in tex.splitlines():
        s = line.strip()
        if "\\midrule" in s or "\\toprule" in s or "\\bottomrule" in s:
            continue
        if "&" not in s or "/" not in s:
            continue
        if s.startswith("$\\Delta") or s.startswith("Speedup"):
            continue
        parts = [p.strip() for p in s.split("&")]
        if len(parts) < 4:
            continue
        label = parts[0].strip()
        try:
            pct = float(parts[1])
            succ_str = parts[2].split()[0]
            succ, eps = succ_str.split("/")
            apply_col = parts[3].split(r"\\")[0].strip()
        except (ValueError, IndexError):
            continue
        rows[label] = (pct, int(succ), int(eps), apply_col)
    return rows


def _norm_pct(x: float) -> float:
    return round(float(x), 2)


def main() -> None:
    stat_path = PAPER / "data" / "stat_tests_summary.json"
    stat = json.loads(stat_path.read_text())
    tab_final_tex = (PAPER / "tables" / "tab_final_pooled_results.tex").read_text()
    tab_stat_tex = (PAPER / "tables" / "tab_stat_tests.tex").read_text()

    wm = stat["families"]["paper_curated"]["wilson_per_mode"]
    final_rows = parse_final_pooled(tab_final_tex)

    expected_labels = [
        ("Vanilla", "vanilla"),
        ("MPPI", "mppi"),
        ("Steering (Ours)", "steering"),
        ("Latent Jiggle", "latent_jiggle"),
    ]

    missing = []
    for tex_label, mode in expected_labels:
        if tex_label not in final_rows:
            missing.append(tex_label)
            continue
        pct, succ, eps, _ = final_rows[tex_label]
        wrow = wm.get(mode)
        if not wrow:
            raise AssertionError(f"wilson_per_mode missing mode {mode}")
        assert (succ, eps) == (wrow["succ"], wrow["eps"]), (
            f"{tex_label}: table counts {succ}/{eps} vs stat_summary {wrow['succ']}/{wrow['eps']}"
        )
        assert _norm_pct(pct) == _norm_pct(wrow["p_hat_pp"]), (
            f"{tex_label}: table pct {pct} vs p_hat_pp {wrow['p_hat_pp']}"
        )
    if missing:
        raise AssertionError(f"Missing expected rows in tab_final_pooled_results.tex: {missing}")

    pc = stat["families"]["paper_curated"]["comparisons"]
    for key_a, key_b in [("steering", "mppi"), ("steering", "vanilla"), ("mppi", "vanilla")]:
        k = f"{key_a}_vs_{key_b}"
        raw_a = pc[k]["raw"][key_a]
        raw_b = pc[k]["raw"][key_b]
        w_a, w_b = wm[key_a], wm[key_b]
        assert raw_a["succ"] == w_a["succ"] and raw_a["eps"] == w_a["eps"], f"{k} raw A mismatch wilson"
        assert raw_b["succ"] == w_b["succ"] and raw_b["eps"] == w_b["eps"], f"{k} raw B mismatch wilson"

    # Primary z-test p-values appear in statistical table (4-decimal substring match)
    for k in ["steering_vs_mppi", "steering_vs_vanilla", "mppi_vs_vanilla"]:
        p_z = pc[k]["pooled"]["p_value_z"]
        tok = f"{p_z:.4f}"
        if tok not in tab_stat_tex:
            raise AssertionError(
                f"tab_stat_tests.tex missing p_value_z substring for {k}: expected {tok}"
            )

    abs_path = PAPER / "sections" / "abstract.tex"
    if abs_path.exists():
        abstract = abs_path.read_text()
        st_r = wm["steering"]["p_hat_pp"]
        mj_r = wm["mppi"]["p_hat_pp"]
        j_r = wm["latent_jiggle"]["p_hat_pp"]

        def _pct_one_dec(xx: float) -> str:
            return f"{round(xx, 1):.1f}"

        rs, rm_, rj = _pct_one_dec(st_r), _pct_one_dec(mj_r), _pct_one_dec(j_r)

        # Abstract reports "(steering %) vs.\ (MPPI %)" rounded to one decimal.
        needle_s = rf"{rs}\%"
        needle_m = rf"{rm_}\%"
        needle_j = rf"{rj}\%"
        if needle_s not in abstract:
            raise AssertionError(
                f"Abstract steering ~{needle_s} vs pooled {_norm_pct(st_r)}% — update abstract?"
            )
        if needle_m not in abstract:
            raise AssertionError(
                f"Abstract MPPI ~{needle_m} vs pooled {_norm_pct(mj_r)}% — update abstract?"
            )
        if needle_j not in abstract:
            raise AssertionError(
                f"Abstract latent-jiggle mention ~{needle_j} vs pooled {_norm_pct(j_r)}%"
            )

    zs = stat.get("zero_shot_aggregate")
    res_path = PAPER / "sections" / "results.tex"
    if zs and res_path.exists():
        results_tex = res_path.read_text()
        vr = round(float(zs["vanilla"]["rate"]), 1)
        sr = round(float(zs["steering"]["rate"]), 1)
        pz = round(float(zs["p_value_z"]), 2)
        if f"{vr:.1f}\\%" not in results_tex:
            raise AssertionError(
                f"results.tex zero-shot vanilla rate should be ~{vr:.1f}% (from stat_tests_summary)"
            )
        if f"{sr:.1f}\\%" not in results_tex:
            raise AssertionError(
                f"results.tex zero-shot steering rate should be ~{sr:.1f}% (from stat_tests_summary)"
            )
        if f"{pz:.2f}" not in results_tex:
            raise AssertionError(
                f"results.tex zero-shot should mention p={pz:.2f}"
            )

    check_stranded_tables()

    print("[ok] Consistency checks passed.")


if __name__ == "__main__":
    main()
