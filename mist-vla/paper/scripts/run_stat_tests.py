#!/usr/bin/env python3
"""Statistical significance analysis over completed paper eval JSONs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

PAPER = Path("/home/mpcr/Desktop/SalusV5/mist-vla/paper")
DATA = PAPER / "data"
OUT_JSON = DATA / "stat_tests_summary.json"
OUT_MD = PAPER / "STAT_TESTS_REPORT.md"
OUT_TEX = PAPER / "tables" / "tab_stat_tests.tex"


def family(name: str) -> str | None:
    if name.startswith("category1_sweep_"):
        return "openvla_sweep"
    if name.startswith("category1_ovla_ood_"):
        return "openvla_ood"
    if name.startswith("eval_act_steering_sweep_"):
        return "act_sweep"
    if name.startswith("eval_act_steering_act_ood"):
        return "act_ood_baselines"
    if name.startswith("eval_act_zero_shot_"):
        return "act_zero_shot_ood"
    return None


def rate_from_file(d: dict, mode: str) -> float | None:
    per = d.get("per_task", {})
    if not isinstance(per, dict):
        return None
    succ = 0
    eps = 0
    for task_data in per.values():
        if not isinstance(task_data, dict):
            continue
        m = task_data.get(mode, {})
        if not isinstance(m, dict):
            continue
        ns = m.get("n_successes")
        ne = m.get("n_episodes")
        if isinstance(ns, (int, float)) and isinstance(ne, (int, float)) and ne > 0:
            succ += int(ns)
            eps += int(ne)
    if eps == 0:
        return None
    return 100.0 * succ / eps


def collect() -> tuple[dict, dict]:
    grouped_files: Dict[str, List[Path]] = defaultdict(list)
    pooled: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"succ": 0, "eps": 0, "rates": []})
    )

    for fp in sorted(DATA.glob("*eval_results*.json")):
        fam = family(fp.name)
        if fam is None:
            continue
        grouped_files[fam].append(fp)
        d = json.loads(fp.read_text())
        per = d.get("per_task", {})
        if not isinstance(per, dict):
            continue

        # pooled counts
        for task_data in per.values():
            if not isinstance(task_data, dict):
                continue
            for mode, m in task_data.items():
                if not isinstance(m, dict):
                    continue
                ns = m.get("n_successes")
                ne = m.get("n_episodes")
                if isinstance(ns, (int, float)) and isinstance(ne, (int, float)) and ne > 0:
                    pooled[fam][mode]["succ"] += int(ns)
                    pooled[fam][mode]["eps"] += int(ne)

        # per-run rates
        for mode in ("vanilla", "mppi", "steering"):
            r = rate_from_file(d, mode)
            if r is not None:
                pooled[fam][mode]["rates"].append(float(r))

    # add curated pooled family
    curated = ["openvla_sweep", "openvla_ood", "act_sweep", "act_ood_baselines", "act_zero_shot_ood"]
    pooled["paper_curated"] = defaultdict(lambda: {"succ": 0, "eps": 0, "rates": []})
    for fam in curated:
        for mode, s in pooled.get(fam, {}).items():
            pooled["paper_curated"][mode]["succ"] += s["succ"]
            pooled["paper_curated"][mode]["eps"] += s["eps"]
            # combine run-level arrays for approximate paired family-level analysis
            pooled["paper_curated"][mode]["rates"].extend(s["rates"])

    return grouped_files, pooled


def proportion_tests(s1: int, n1: int, s2: int, n2: int) -> dict:
    if min(n1, n2) == 0:
        return {}
    p1 = s1 / n1
    p2 = s2 / n2
    diff = p1 - p2

    # Two-proportion z-test (pooled SE)
    p_pool = (s1 + s2) / (n1 + n2)
    se_pool = math.sqrt(max(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2), 1e-12))
    z = diff / se_pool
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% CI for difference (unpooled Wald)
    se_unpooled = math.sqrt(max((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2), 1e-12))
    ci_lo = diff - 1.96 * se_unpooled
    ci_hi = diff + 1.96 * se_unpooled

    # Fisher exact
    table = [[s1, n1 - s1], [s2, n2 - s2]]
    _, p_fisher = stats.fisher_exact(table, alternative="two-sided")

    return {
        "p1": p1,
        "p2": p2,
        "diff_pp": 100 * diff,
        "z_stat": z,
        "p_value_z": p_z,
        "p_value_fisher": p_fisher,
        "ci95_diff_pp": [100 * ci_lo, 100 * ci_hi],
    }


def paired_tests(a: List[float], b: List[float], seed: int = 42) -> dict:
    n = min(len(a), len(b))
    if n < 2:
        return {"n_pairs": n}
    x = np.array(a[:n], dtype=float)
    y = np.array(b[:n], dtype=float)
    d = x - y

    t_res = stats.ttest_rel(x, y, alternative="two-sided")
    try:
        w_res = stats.wilcoxon(d)
        w_p = float(w_res.pvalue)
        w_stat = float(w_res.statistic)
    except ValueError:
        w_p = float("nan")
        w_stat = float("nan")

    # Cohen's d for paired samples
    sd = float(np.std(d, ddof=1))
    d_eff = float(np.mean(d) / sd) if sd > 0 else float("nan")

    # Bootstrap CI for mean delta
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        boots.append(float(np.mean(d[idx])))
    ci = np.percentile(boots, [2.5, 97.5]).tolist()

    return {
        "n_pairs": n,
        "mean_delta_pp": float(np.mean(d)),
        "ci95_mean_delta_pp": [float(ci[0]), float(ci[1])],
        "ttest_stat": float(t_res.statistic),
        "p_value_ttest": float(t_res.pvalue),
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": w_p,
        "cohens_d_paired": d_eff,
    }


def main() -> None:
    grouped_files, pooled = collect()
    comparisons = [("steering", "mppi"), ("steering", "vanilla"), ("mppi", "vanilla")]
    out = {"families": {}, "meta": {"files_per_family": {k: len(v) for k, v in grouped_files.items()}}}

    for fam, modes in pooled.items():
        fam_out = {"comparisons": {}}
        for a, b in comparisons:
            sa = modes.get(a, {"succ": 0, "eps": 0, "rates": []})
            sb = modes.get(b, {"succ": 0, "eps": 0, "rates": []})
            key = f"{a}_vs_{b}"
            fam_out["comparisons"][key] = {
                "pooled": proportion_tests(sa["succ"], sa["eps"], sb["succ"], sb["eps"]),
                "paired_runs": paired_tests(sa["rates"], sb["rates"]),
                "raw": {
                    a: {"succ": sa["succ"], "eps": sa["eps"], "n_runs": len(sa["rates"])},
                    b: {"succ": sb["succ"], "eps": sb["eps"], "n_runs": len(sb["rates"])},
                },
            }
        out["families"][fam] = fam_out

    OUT_JSON.write_text(json.dumps(out, indent=2))

    # Markdown report
    lines = ["# Statistical Significance Report", ""]
    for fam, data in out["families"].items():
        lines.append(f"## {fam}")
        for key, comp in data["comparisons"].items():
            p = comp["pooled"]
            r = comp["paired_runs"]
            if not p:
                continue
            lines.append(f"- **{key}**")
            lines.append(
                f"  - pooled diff: {p['diff_pp']:+.2f} pp (95% CI [{p['ci95_diff_pp'][0]:+.2f}, {p['ci95_diff_pp'][1]:+.2f}])"
            )
            lines.append(f"  - p(z-test): {p['p_value_z']:.4g}, p(Fisher): {p['p_value_fisher']:.4g}")
            if r.get("n_pairs", 0) >= 2:
                lines.append(
                    f"  - paired runs (n={r['n_pairs']}): mean delta {r['mean_delta_pp']:+.2f} pp, "
                    f"p(t-test)={r['p_value_ttest']:.4g}, p(Wilcoxon)={r['p_value_wilcoxon']:.4g}"
                )
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n")

    # Compact LaTeX table for main paper (curated pooled family only)
    cur = out["families"].get("paper_curated", {}).get("comparisons", {})
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Final statistical tests on pooled completed results (paper-curated set).}",
        r"\label{tab:stat_tests}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Comparison & $\Delta$ Success (pp) & 95\% CI (pp) & $p$-value (z-test) \\",
        r"\midrule",
    ]
    for key, label in [
        ("steering_vs_mppi", "Steering vs MPPI"),
        ("steering_vs_vanilla", "Steering vs Vanilla"),
        ("mppi_vs_vanilla", "MPPI vs Vanilla"),
    ]:
        p = cur.get(key, {}).get("pooled", {})
        if not p:
            continue
        tex.append(
            f"{label} & {p['diff_pp']:+.2f} & "
            f"[{p['ci95_diff_pp'][0]:+.2f}, {p['ci95_diff_pp'][1]:+.2f}] & {p['p_value_z']:.4g} {EOL}"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_TEX.write_text("\n".join(tex) + "\n")

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_TEX}")


EOL = r"\\"

if __name__ == "__main__":
    main()
