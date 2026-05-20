#!/usr/bin/env python3
"""Statistical significance analysis over completed paper eval JSONs.

Reads:  paper/data/*eval_results*.json
        hpc_mirror/results/tuning/*.json
Writes: paper/data/stat_tests_summary.json
        paper/STAT_TESTS_REPORT.md
        paper/tables/tab_stat_tests.tex
        paper/tables/tab_equivalence_tost.tex
        paper/tables/tab_detection_vs_correction.tex
        paper/tables/tab_cross_arch.tex
        paper/tables/tab_clamping_ablation.tex
        paper/tables/tab_gating_ablation.tex
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from stats_utils import (
    holm_adjusted_pvalues,
    posthoc_power_two_proportion_z,
    required_n_per_group_two_proportion,
    tost_two_proportion,
    wilson_ci_pp,
)

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data"
TUNING = PAPER.parent / "hpc_mirror" / "results" / "tuning"
OUT_JSON = DATA / "stat_tests_summary.json"
OUT_MD = PAPER / "STAT_TESTS_REPORT.md"
OUT_TEX = PAPER / "tables" / "tab_stat_tests.tex"
TABLES = PAPER / "tables"
EOL = r"\\"

PRIMARY_CONTRASTS = [
    ("steering", "mppi"),
    ("steering", "vanilla"),
    ("mppi", "vanilla"),
    ("steering", "latent_jiggle"),
    ("mppi", "latent_jiggle"),
    ("vanilla", "latent_jiggle"),
]
PRIMARY_LABELS = {
    "steering_vs_mppi": "Steering vs MPPI",
    "steering_vs_vanilla": "Steering vs Vanilla",
    "mppi_vs_vanilla": "MPPI vs Vanilla",
    "steering_vs_latent_jiggle": "Steering vs Latent Jiggle",
    "mppi_vs_latent_jiggle": "MPPI vs Latent Jiggle",
    "vanilla_vs_latent_jiggle": "Vanilla vs Latent Jiggle",
}


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


def per_task_counts_from_files(paths: List[Path]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """task_id -> mode -> {succ, eps} aggregated across all JSON files in a family."""
    acc: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"succ": 0, "eps": 0})
    )
    for fp in paths:
        d = json.loads(fp.read_text())
        per = d.get("per_task", {})
        if not isinstance(per, dict):
            continue
        for tid, task_data in per.items():
            if not isinstance(task_data, dict):
                continue
            for mode, m in task_data.items():
                if not isinstance(mode, str) or mode.startswith("delta"):
                    continue
                if not isinstance(m, dict):
                    continue
                ns = m.get("n_successes")
                ne = m.get("n_episodes")
                if isinstance(ns, (int, float)) and isinstance(ne, (int, float)) and ne > 0:
                    acc[str(tid)][mode]["succ"] += int(ns)
                    acc[str(tid)][mode]["eps"] += int(ne)
    return acc


def collect() -> tuple[dict, dict, dict]:
    grouped_files: Dict[str, List[Path]] = defaultdict(list)
    pooled: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"succ": 0, "eps": 0, "rates": []})
    )
    per_family_tasks: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}

    for fp in sorted(DATA.glob("*eval_results*.json")):
        fam = family(fp.name)
        if fam is None:
            continue
        grouped_files[fam].append(fp)
        d = json.loads(fp.read_text())
        per = d.get("per_task", {})
        if not isinstance(per, dict):
            continue

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

        for mode in ("vanilla", "mppi", "steering", "latent_stop", "latent_jiggle", "noise"):
            r = rate_from_file(d, mode)
            if r is not None:
                pooled[fam][mode]["rates"].append(float(r))

    curated = [
        "openvla_sweep",
        "openvla_ood",
        "act_sweep",
        "act_ood_baselines",
        "act_zero_shot_ood",
    ]
    pooled["paper_curated"] = defaultdict(lambda: {"succ": 0, "eps": 0, "rates": []})
    for fam in curated:
        for mode, s in pooled.get(fam, {}).items():
            pooled["paper_curated"][mode]["succ"] += s["succ"]
            pooled["paper_curated"][mode]["eps"] += s["eps"]
            pooled["paper_curated"][mode]["rates"].extend(s["rates"])

    for fam, paths in grouped_files.items():
        per_family_tasks[fam] = per_task_counts_from_files(paths)
    per_family_tasks["paper_curated"] = per_task_counts_from_files(
        [p for fam in curated for p in grouped_files.get(fam, [])]
    )

    return grouped_files, pooled, per_family_tasks


def proportion_tests(s1: int, n1: int, s2: int, n2: int) -> dict:
    if min(n1, n2) == 0:
        return {}
    p1 = s1 / n1
    p2 = s2 / n2
    diff = p1 - p2

    p_pool = (s1 + s2) / (n1 + n2)
    se_pool = math.sqrt(max(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2), 1e-12))
    z = diff / se_pool
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))

    se_unpooled = math.sqrt(max((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2), 1e-12))
    ci_lo = diff - 1.96 * se_unpooled
    ci_hi = diff + 1.96 * se_unpooled

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
        "wilson_ci_p1_pp": list(wilson_ci_pp(s1, n1)),
        "wilson_ci_p2_pp": list(wilson_ci_pp(s2, n2)),
        "posthoc_power_z": posthoc_power_two_proportion_z(s1, n1, s2, n2),
        "required_n_per_arm_delta_pp": _required_n_grid(p1, p2),
    }


def _required_n_grid(p1: float, p2: float) -> Dict[str, int]:
    out: Dict[str, int] = {}
    delta = abs(p1 - p2)
    if delta > 1e-9:
        out["detect_observed_effect_80pct"] = required_n_per_group_two_proportion(p1, p2)
    for dpp in (1.0, 2.0, 5.0, 10.0):
        d = dpp / 100.0
        p_a, p_b = p1, p2
        if p1 >= p2:
            p_b = max(0.0, p1 - d)
        else:
            p_a = max(0.0, p2 - d)
        n = required_n_per_group_two_proportion(p_a, p_b)
        if n > 0:
            out[f"delta_{int(dpp)}pp_symmetric"] = n
    return out


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

    sd = float(np.std(d, ddof=1))
    d_eff = float(np.mean(d) / sd) if sd > 0 else float("nan")

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


def paired_task_tests(
    task_acc: Dict[str, Dict[str, Dict[str, int]]], mode_a: str, mode_b: str, seed: int = 42
) -> dict:
    deltas: List[float] = []
    for _tid, modes in sorted(task_acc.items()):
        ma = modes.get(mode_a)
        mb = modes.get(mode_b)
        if not ma or not mb:
            continue
        if ma["eps"] <= 0 or mb["eps"] <= 0:
            continue
        ra = ma["succ"] / ma["eps"]
        rb = mb["succ"] / mb["eps"]
        deltas.append(100.0 * (ra - rb))

    n = len(deltas)
    if n < 2:
        return {"n_tasks": n, "note": "insufficient tasks for paired test"}

    d = np.array(deltas, dtype=float)
    t_res = stats.ttest_1samp(d, 0.0, alternative="two-sided")
    try:
        w_res = stats.wilcoxon(d)
        w_p = float(w_res.pvalue)
        w_stat = float(w_res.statistic)
    except ValueError:
        w_p = float("nan")
        w_stat = float("nan")

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        boots.append(float(np.mean(d[idx])))
    ci = np.percentile(boots, [2.5, 97.5]).tolist()

    return {
        "n_tasks": n,
        "mean_delta_pp": float(np.mean(d)),
        "ci95_mean_delta_pp": [float(ci[0]), float(ci[1])],
        "ttest_stat": float(t_res.statistic),
        "p_value_ttest": float(t_res.pvalue),
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": w_p,
        "task_deltas_pp_sample": [round(x, 4) for x in deltas[:15]],
    }


def wilson_per_mode(modes: Dict[str, dict]) -> Dict[str, Any]:
    out = {}
    for mode, s in modes.items():
        if not isinstance(s, dict):
            continue
        succ, eps = int(s.get("succ", 0)), int(s.get("eps", 0))
        if eps <= 0:
            continue
        lo, hi = wilson_ci_pp(succ, eps)
        out[mode] = {
            "succ": succ,
            "eps": eps,
            "p_hat_pp": 100.0 * succ / eps,
            "wilson95_pp": [lo, hi],
        }
    return out


def render_markdown(
    out: dict,
    holm_row: Dict[str, float],
) -> str:
    lines = [
        "# Statistical Significance Report",
        "",
        "Generated by `paper/scripts/run_stat_tests.py` from `paper/data/*eval_results*.json`.",
        "",
        "## Conventions",
        "",
        "- **Pooled**: two-proportion z-test (pooled SE under H0) and Fisher exact on success/failure counts; **95% CI** on difference uses unpooled Wald SE.",
        "- **Wilson**: Wilson score interval for each method's success rate.",
        "- **Paired runs**: paired t-test / Wilcoxon on **per-run aggregate success rates** (one rate per eval JSON file).",
        "- **Paired tasks**: paired t-test / Wilcoxon on **per-task success rates** after aggregating counts across all files in the family (LIBERO tasks as paired units).",
        "- **Holm**: adjusted p-values for the three primary z-tests on **paper_curated** only (pre-specified family).",
        "- **TOST**: Two One-Sided Tests for equivalence within ±2 pp margin.",
        "",
        "## Primary contrasts (paper-curated): Holm-adjusted p (z-test)",
        "",
    ]
    for key, p_h in holm_row.items():
        lines.append(f"- `{key}`: Holm-adjusted p = {p_h:.4g}")
    lines.append("")

    # TOST section
    tost_data = out.get("tost_equivalence", {})
    if tost_data:
        lines.append("## Equivalence Tests (TOST, ±2 pp)")
        lines.append("")
        for key, td in tost_data.items():
            eq_str = "YES" if td.get("equivalent") else "NO"
            lines.append(f"- `{key}`: p_TOST = {td['p_tost']:.4g}, equivalent = {eq_str}")
        lines.append("")

    for fam, data in out["families"].items():
        lines.append(f"## {fam}")
        lines.append("")
        wm = data.get("wilson_per_mode", {})
        if wm:
            lines.append("### Wilson 95% CI on success rate (%)")
            for mode, row in sorted(wm.items()):
                lo, hi = row["wilson95_pp"]
                lines.append(
                    f"- **{mode}**: {row['p_hat_pp']:.2f}% "
                    f"[{lo:.2f}, {hi:.2f}] (n={row['succ']}/{row['eps']})"
                )
            lines.append("")

        for key, comp in data["comparisons"].items():
            p = comp.get("pooled", {})
            r_run = comp.get("paired_runs", {})
            r_task = comp.get("paired_tasks", {})
            if not p:
                continue
            lines.append(f"### {key}")
            lines.append(
                f"- **Pooled diff**: {p['diff_pp']:+.2f} pp "
                f"(95% CI [{p['ci95_diff_pp'][0]:+.2f}, {p['ci95_diff_pp'][1]:+.2f}])"
            )
            lines.append(
                f"- **p (z-test)**: {p['p_value_z']:.4g}; **p (Fisher)**: {p['p_value_fisher']:.4g}; "
                f"**post-hoc power (z, caut.)**: {p.get('posthoc_power_z', float('nan')):.4g}"
            )
            req = p.get("required_n_per_arm_delta_pp", {})
            if req:
                lines.append(
                    "- **Approx. episodes per arm (80% power, α=0.05, equal n)**: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(req.items()) if v > 0)
                )
            if r_run.get("n_pairs", 0) >= 2:
                lines.append(
                    f"- **Paired runs** (n={r_run['n_pairs']}): mean Δ {r_run['mean_delta_pp']:+.2f} pp, "
                    f"p(t)={r_run['p_value_ttest']:.4g}, p(Wilcoxon)={r_run['p_value_wilcoxon']:.4g}"
                )
            if r_task.get("n_tasks", 0) >= 2:
                lines.append(
                    f"- **Paired tasks** (n={r_task['n_tasks']}): mean Δ {r_task['mean_delta_pp']:+.2f} pp, "
                    f"p(t)={r_task['p_value_ttest']:.4g}, p(Wilcoxon)={r_task['p_value_wilcoxon']:.4g}"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


# ─── Additional contrasts and tables ──────────────────────────────

def detection_vs_correction_table(pooled: dict) -> None:
    """Generate tab_detection_vs_correction.tex from paper_curated pool."""
    pc = pooled.get("paper_curated", {})
    modes = ["vanilla", "latent_stop", "steering", "mppi"]
    labels = {
        "vanilla": "Vanilla (no intervention)",
        "latent_stop": "Latent Stop (detect only)",
        "steering": "Steering (detect + correct)",
        "mppi": "MPPI (planning baseline)",
    }
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection vs.\ correction: latent stop destroys task completion while continuous correction preserves it.}",
        r"\label{tab:detection_vs_correction}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Mode & Success (\%) & $n$ (succ/total) \\",
        r"\midrule",
    ]
    for mode in modes:
        s = pc.get(mode, {"succ": 0, "eps": 0})
        if s["eps"] == 0:
            continue
        sr = 100.0 * s["succ"] / s["eps"]
        tex.append(f"{labels[mode]} & {sr:.1f} & {s['succ']}/{s['eps']} {EOL}")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    archive_path = TABLES / "archive" / "tab_detection_vs_correction.tex"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text("\n".join(tex) + "\n")
    print(f"wrote {archive_path} (archived; superseded by tab_final_pooled_results)")


def equivalence_table(pooled: dict) -> dict:
    """Generate tab_equivalence_tost.tex and return TOST results."""
    pc = pooled.get("paper_curated", {})
    tost_results = {}
    contrasts = [
        ("steering", "mppi", "Steering vs MPPI"),
        ("steering", "vanilla", "Steering vs Vanilla"),
        ("mppi", "vanilla", "MPPI vs Vanilla"),
        ("steering", "latent_jiggle", "Steering vs Latent Jiggle"),
        ("mppi", "latent_jiggle", "MPPI vs Latent Jiggle"),
        ("vanilla", "latent_jiggle", "Vanilla vs Latent Jiggle"),
    ]
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{TOST equivalence tests ($\pm$2\,pp margin) on paper-curated pool. Equivalence established if $p_{\mathrm{TOST}} < 0.05$. \textbf{Choice of $\pm$2\,pp:} at the $\sim$52\% base success rate this corresponds to $\le$4\% relative change, which matches the smallest difference our pooled sample size ($n{\approx}10{,}000$ per mode) is powered to detect at $\alpha{=}0.05$ with $\ge$80\% power under a two-proportion test --- it is the tightest \emph{honest} equivalence claim our data support.}",
        r"\label{tab:equivalence_tost}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Comparison & $\Delta$ (pp) & $p_{\mathrm{TOST}}$ & Equivalent? \\",
        r"\midrule",
    ]
    for a, b, label in contrasts:
        sa = pc.get(a, {"succ": 0, "eps": 0})
        sb = pc.get(b, {"succ": 0, "eps": 0})
        if sa["eps"] == 0 or sb["eps"] == 0:
            continue
        t = tost_two_proportion(sa["succ"], sa["eps"], sb["succ"], sb["eps"], delta=0.02)
        key = f"{a}_vs_{b}"
        tost_results[key] = t
        eq = "Yes" if t["equivalent"] else "No"
        diff_pp = 100.0 * t["diff"]
        tex.append(f"{label} & {diff_pp:+.2f} & {t['p_tost']:.4g} & {eq} {EOL}")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "tab_equivalence_tost.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {TABLES / 'tab_equivalence_tost.tex'}")
    return tost_results


def cross_arch_table(pooled: dict) -> None:
    """Generate tab_cross_arch.tex comparing OpenVLA and ACT families."""
    # Aggregate OpenVLA = sweep + ood; ACT = sweep + ood + zs_ood
    ov = defaultdict(lambda: {"succ": 0, "eps": 0})
    for fam in ["openvla_sweep", "openvla_ood"]:
        for mode, s in pooled.get(fam, {}).items():
            ov[mode]["succ"] += s["succ"]
            ov[mode]["eps"] += s["eps"]

    act = defaultdict(lambda: {"succ": 0, "eps": 0})
    for fam in ["act_sweep", "act_ood_baselines", "act_zero_shot_ood"]:
        for mode, s in pooled.get(fam, {}).items():
            act[mode]["succ"] += s["succ"]
            act[mode]["eps"] += s["eps"]

    modes = ["vanilla", "latent_stop", "steering", "mppi", "latent_jiggle"]
    labels = {"vanilla": "Vanilla", "latent_stop": "Latent Stop", "steering": "Steering", "mppi": "MPPI", "latent_jiggle": "Latent Jiggle"}

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-architecture comparison: OpenVLA (4096-d) vs.\ ACT (256-d). Both exhibit the same qualitative pattern --- Vanilla, Steering, and MPPI are within $\sim$1\,pp of each other on each architecture, Latent Stop is materially worse, and Latent Jiggle is numerically slightly higher (consistent with the pooled equivalence in Table~\ref{tab:final_pooled_results} and the jiggle-is-outlier finding in \S\ref{sec:steering_results}).}",
        r"\label{tab:cross_arch}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{OpenVLA} & \multicolumn{2}{c}{ACT} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Mode & Success (\%) & $n$ & Success (\%) & $n$ \\",
        r"\midrule",
    ]
    for mode in modes:
        ov_s = ov.get(mode, {"succ": 0, "eps": 0})
        act_s = act.get(mode, {"succ": 0, "eps": 0})
        ov_sr = f"{100.0 * ov_s['succ'] / ov_s['eps']:.1f}" if ov_s["eps"] > 0 else "--"
        act_sr = f"{100.0 * act_s['succ'] / act_s['eps']:.1f}" if act_s["eps"] > 0 else "--"
        ov_n = f"{ov_s['succ']}/{ov_s['eps']}" if ov_s["eps"] > 0 else "--"
        act_n = f"{act_s['succ']}/{act_s['eps']}" if act_s["eps"] > 0 else "--"
        tex.append(f"{labels[mode]} & {ov_sr} & {ov_n} & {act_sr} & {act_n} {EOL}")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "tab_cross_arch.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {TABLES / 'tab_cross_arch.tex'}")


def clamping_ablation_table() -> None:
    """Generate tab_clamping_ablation.tex from tuning/clamping_sweep.json."""
    path = TUNING / "clamping_sweep.json"
    if not path.exists():
        print(f"skipping clamping ablation: {path} not found")
        return
    d = json.loads(path.read_text())
    results = d.get("results", {})

    # Group by task
    tasks = sorted(set(v["task_id"] for v in results.values()))
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Clamping (trust region) ablation: success rate (\%) by task with $c_{\max}=0.01$\,m vs.\ vanilla. $n=20$ episodes per cell. \textbf{Caveat:} per-cell $n{=}20$ is small; the pooled row averages across tasks with very different base rates ($35\%$--$90\%$) and is \emph{directional only} --- it carries no claim of statistical significance. We report this ablation as a sensitivity probe, not as a hypothesis test.}",
        r"\label{tab:clamping_ablation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Task & Vanilla (\%) & Clamp=0.01m (\%) \\",
        r"\midrule",
    ]
    v_total_s, v_total_e = 0, 0
    c_total_s, c_total_e = 0, 0
    for tid in tasks:
        v_key = f"({tid}, 'vanilla')"
        c_key = f"({tid}, 'clamp=0.010m')"
        v = results.get(v_key, {})
        c = results.get(c_key, {})
        v_sr = v.get("success_rate_pct", 0)
        c_sr = c.get("success_rate_pct", 0)
        v_total_s += v.get("n_successes", 0)
        v_total_e += v.get("n_episodes", 0)
        c_total_s += c.get("n_successes", 0)
        c_total_e += c.get("n_episodes", 0)
        tex.append(f"Task {tid} & {v_sr:.0f} & {c_sr:.0f} {EOL}")
    if v_total_e > 0 and c_total_e > 0:
        tex.append(r"\midrule")
        tex.append(
            f"Pooled & {100.0 * v_total_s / v_total_e:.1f} & "
            f"{100.0 * c_total_s / c_total_e:.1f} {EOL}"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "tab_clamping_ablation.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {TABLES / 'tab_clamping_ablation.tex'}")


def gating_ablation_table() -> None:
    """Generate tab_gating_ablation.tex from tuning/gating_sweep.json."""
    path = TUNING / "gating_sweep.json"
    if not path.exists():
        print(f"skipping gating ablation: {path} not found")
        return
    d = json.loads(path.read_text())
    results = d.get("results", {})

    # Get unique configs
    configs = d.get("sweep_configs", [])
    tasks = sorted(set(v["task_id"] for v in results.values()))

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Gating threshold sensitivity ($n=5$ episodes per cell). Success rate (\%) by task and gating configuration.}",
        r"\label{tab:gating_ablation}",
    ]

    # Collect labels
    config_labels = ["vanilla"] + [c["label"] for c in configs]
    short_labels = ["Van."] + [c["label"].replace("mag δ=", "m").replace("pfail τ=", "p") for c in configs]
    ncols = len(config_labels)

    tex.append(r"\begin{tabular}{l" + "c" * ncols + "}")
    tex.append(r"\toprule")
    header = "Task & " + " & ".join(short_labels) + r" \\"
    tex.append(header)
    tex.append(r"\midrule")

    for tid in tasks:
        row_vals = []
        v_key = f"({tid}, 'vanilla')"
        v = results.get(v_key, {})
        row_vals.append(f"{v.get('success_rate_pct', 0):.0f}")
        for cfg in configs:
            c_key = f"({tid}, '{cfg['label']}')"
            c = results.get(c_key, {})
            row_vals.append(f"{c.get('success_rate_pct', 0):.0f}")
        tex.append(f"T{tid} & " + " & ".join(row_vals) + f" {EOL}")

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "tab_gating_ablation.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {TABLES / 'tab_gating_ablation.tex'}")


def intervention_correlation(per_family_tasks: dict) -> dict:
    """Compute correlation between intervention rate and task difficulty."""
    # Use paper_curated task data
    task_data = per_family_tasks.get("paper_curated", {})
    vanilla_rates = []
    intervention_rates = []

    for tid, modes in task_data.items():
        v = modes.get("vanilla", {"succ": 0, "eps": 0})
        s = modes.get("steering", {"succ": 0, "eps": 0})
        if v["eps"] > 0 and s["eps"] > 0:
            vanilla_rates.append(100.0 * v["succ"] / v["eps"])
            # We don't have intervention rate in the per-task data, so approximate
            # via success rate difference (lower steering success on already-hard tasks
            # = more intervention)

    return {"n_tasks": len(vanilla_rates)}


def safety_head_metrics_table() -> None:
    """Generate tab_safety_head_metrics.tex comparing OpenVLA vs ACT safety heads."""
    ov_path = PAPER.parent / "hpc_mirror" / "checkpoints" / "eef_correction_mlp" / "results.json"
    act_path = PAPER.parent / "hpc_mirror" / "checkpoints" / "eef_correction_mlp_act_honest" / "results.json"
    if not ov_path.exists() or not act_path.exists():
        print(f"skipping safety_head_metrics_table: checkpoint results not found")
        return
    ov = json.loads(ov_path.read_text())
    act = json.loads(act_path.read_text())
    ov_t = ov["test_results"]
    act_t = act["test_results"]

    # Linear-probe and chance baselines (computed once via scripts/eval_safety_head_baselines.py
    # and pinned here so the table reproduces deterministically).
    # OpenVLA linear-probe AUC and ACT linear-probe AUC come from logistic regression on the
    # same trajectory-disjoint split that produced the MLP-probe numbers above.
    lin_ov, lin_act = 0.785, 0.788
    rows = [
        # --- Failure detection ---
        (r"\textit{Failure AUC --- chance}", "0.500", "0.500"),
        (r"\textit{Failure AUC --- linear probe baseline}", f"{lin_ov:.3f}", f"{lin_act:.3f}"),
        (r"\textbf{Failure AUC --- PULSE (ours)}", rf"\textbf{{{ov_t['fail_auc']:.3f}}}", rf"\textbf{{{act_t['fail_auc']:.3f}}}"),
        (r"$\Delta$ AUC over linear baseline", f"$+{ov_t['fail_auc']-lin_ov:.3f}$", f"$+{act_t['fail_auc']-lin_act:.3f}$"),
        (r"\midrule[0.4pt] Failure Accuracy ($\sigma{\geq}0.5$)", f"{ov_t['fail_acc']:.3f}", f"{act_t['fail_acc']:.3f}"),
        # --- TTF ---
        (r"\midrule[0.4pt] \textit{TTF correlation --- chance}", "0.000", "0.000"),
        (r"\textbf{TTF Correlation ($r$) --- PULSE (ours)}", rf"\textbf{{{ov_t['ttf_corr']:.3f}}}", rf"\textbf{{{act_t['ttf_corr']:.3f}}}"),
        (r"TTF $R^2$", f"{ov_t['ttf_r2']:.3f}", f"{act_t['ttf_r2']:.3f}"),
        # --- Correction (kept for reference; ablation analysis in mechanism section) ---
        (r"\midrule[0.4pt] \textit{Correction cosine --- chance (random unit)}", "0.000", "0.000"),
        ("Correction Cosine Sim (median)", f"{ov_t['cosine_sim_median']:.3f}", f"{act_t['cosine_sim_median']:.3f}"),
        (r"\textit{X Direction AUC --- chance}", "0.500", "0.500"),
        ("X Direction AUC", f"{ov_t['X_dir_auc']:.3f}", f"{act_t['X_dir_auc']:.3f}"),
        (r"\textit{Y Direction AUC --- chance}", "0.500", "0.500"),
        ("Y Direction AUC", f"{ov_t['Y_dir_auc']:.3f}", f"{act_t['Y_dir_auc']:.3f}"),
        (r"\textit{Z Direction AUC --- chance}", "0.500", "0.500"),
        ("Z Direction AUC", f"{ov_t['Z_dir_auc']:.3f}", f"{act_t['Z_dir_auc']:.3f}"),
        (r"\midrule[0.4pt] Parameters", f"{ov['model_params']:,}", f"{act['model_params']:,}"),
    ]

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Safety head metrics on the trajectory-disjoint held-out test set. \emph{Chance} is random guessing; \emph{linear probe} is L2-regularized logistic regression on the same hidden states and split, isolating linearly accessible failure signal. \textbf{PULSE} is the trained MLP probe. The correction-head AUCs are kept for reference; the mechanism analysis (Section~\ref{sec:mechanism}) shows the correction direction does not outperform random perturbation despite the apparent gap over chance. \emph{Provenance:} Failure AUCs are read from \texttt{hpc\_mirror/checkpoints/eef\_correction\_mlp/results.json} (and the ACT analog), produced by \texttt{scripts/train\_eef\_correction\_mlp.py} on the OpenVLA-seed0 / ACT-spatial rollout pools with a 75/15/10 rollout-level split (\texttt{np.random.shuffle} under \texttt{np.random.seed}). The post-hoc threshold-replay AUCs reported in Table~\ref{tab:threshold_operating} are slightly higher because they use a larger merged rollout pool and a different RNG (see Appendix~\ref{app:auc_reconciliation}).}",
        r"\label{tab:safety_head_metrics}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & OpenVLA (4096-d) & ACT (256-d) \\",
        r"\midrule",
    ]
    for label, ov_val, act_val in rows:
        tex.append(f"{label} & {ov_val} & {act_val} {EOL}")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "tab_safety_head_metrics.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {TABLES / 'tab_safety_head_metrics.tex'}")


def stratified_difficulty_tests(per_family_tasks: dict) -> None:
    """Stratify tasks by difficulty and run z-tests on hard vs easy subsets.

    Hard: vanilla SR < 30%; Easy: vanilla SR > 70%.
    Uses paper_curated pooled data.
    """
    task_data = per_family_tasks.get("paper_curated", {})
    hard_v_succ, hard_v_eps = 0, 0
    hard_s_succ, hard_s_eps = 0, 0
    easy_v_succ, easy_v_eps = 0, 0
    easy_s_succ, easy_s_eps = 0, 0
    hard_tasks: list[str] = []
    easy_tasks: list[str] = []

    for tid, modes in sorted(task_data.items()):
        v = modes.get("vanilla", {"succ": 0, "eps": 0})
        s = modes.get("steering", {"succ": 0, "eps": 0})
        if v["eps"] == 0 or s["eps"] == 0:
            continue
        vr = 100.0 * v["succ"] / v["eps"]
        if vr < 30.0:
            hard_tasks.append(tid)
            hard_v_succ += v["succ"]
            hard_v_eps += v["eps"]
            hard_s_succ += s["succ"]
            hard_s_eps += s["eps"]
        elif vr > 70.0:
            easy_tasks.append(tid)
            easy_v_succ += v["succ"]
            easy_v_eps += v["eps"]
            easy_s_succ += s["succ"]
            easy_s_eps += s["eps"]

    # Two-proportion z-test helper
    def _z_test(s1: int, n1: int, s2: int, n2: int) -> dict:
        if min(n1, n2) == 0:
            return {"diff_pp": float("nan"), "z": float("nan"), "p": float("nan")}
        p1 = s1 / n1
        p2 = s2 / n2
        diff = p1 - p2
        p_pool = (s1 + s2) / (n1 + n2)
        se = math.sqrt(max(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2), 1e-12))
        z = diff / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        se_unp = math.sqrt(max(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2, 1e-12))
        ci_lo = diff - 1.96 * se_unp
        ci_hi = diff + 1.96 * se_unp
        return {
            "diff_pp": 100.0 * diff,
            "ci95_pp": [100.0 * ci_lo, 100.0 * ci_hi],
            "z": z,
            "p": p_val,
            "n_steering": n1,
            "n_vanilla": n2,
            "sr_steering": 100.0 * p1,
            "sr_vanilla": 100.0 * p2,
        }

    hard_res = _z_test(hard_s_succ, hard_s_eps, hard_v_succ, hard_v_eps)
    easy_res = _z_test(easy_s_succ, easy_s_eps, easy_v_succ, easy_v_eps)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stratified difficulty analysis. Hard tasks: vanilla SR $<$30\%; Easy tasks: vanilla SR $>$70\%. Two-proportion z-test on steering vs.\ vanilla.}",
        r"\label{tab:stratified_difficulty}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Stratum & \#Tasks & Vanilla (\%) & Steering (\%) & $\Delta$ (pp) & 95\% CI & $p$ (z) \\",
        r"\midrule",
    ]

    def _row(label: str, n_tasks: int, res: dict) -> str:
        if math.isnan(res.get("diff_pp", float("nan"))):
            return f"{label} & {n_tasks} & -- & -- & -- & -- & -- {EOL}"
        ci = res["ci95_pp"]
        return (
            f"{label} & {n_tasks} & {res['sr_vanilla']:.1f} & {res['sr_steering']:.1f} & "
            f"{res['diff_pp']:+.1f} & [{ci[0]:+.1f}, {ci[1]:+.1f}] & {res['p']:.4g} {EOL}"
        )

    tex.append(_row("Hard (SR$<$30\\%)", len(hard_tasks), hard_res))
    tex.append(_row("Easy (SR$>$70\\%)", len(easy_tasks), easy_res))
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "tab_stratified_difficulty.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {TABLES / 'tab_stratified_difficulty.tex'}")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    grouped_files, pooled, per_family_tasks = collect()
    comparisons_keys = [f"{a}_vs_{b}" for a, b in PRIMARY_CONTRASTS]
    git_head = "unknown"
    try:
        import subprocess

        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PAPER.parent.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            git_head = p.stdout.strip()
    except Exception:
        pass

    out: Dict[str, Any] = {
        "families": {},
        "meta": {
            "files_per_family": {k: len(v) for k, v in grouped_files.items()},
            "script": str(Path(__file__).resolve()),
            "primary_contrasts": comparisons_keys,
            "git_head": git_head,
        },
    }

    z_ps_for_holm: List[float] = []
    holm_keys_order: List[str] = []

    # Extended contrasts: add latent_stop vs vanilla, steering vs jiggle, steering vs noise
    extended_contrasts = PRIMARY_CONTRASTS + [
        ("latent_stop", "vanilla"),
        ("steering", "latent_jiggle"),
        ("steering", "noise"),
    ]

    for fam, modes in pooled.items():
        fam_out: Dict[str, Any] = {"comparisons": {}}
        fam_out["wilson_per_mode"] = wilson_per_mode(dict(modes))

        task_acc = per_family_tasks.get(fam, {})
        for a, b in extended_contrasts:
            sa = modes.get(a, {"succ": 0, "eps": 0, "rates": []})
            sb = modes.get(b, {"succ": 0, "eps": 0, "rates": []})
            if sa["eps"] == 0 or sb["eps"] == 0:
                continue
            key = f"{a}_vs_{b}"
            pt = proportion_tests(sa["succ"], sa["eps"], sb["succ"], sb["eps"])
            pr = paired_tests(sa.get("rates", []), sb.get("rates", []))
            ptask = paired_task_tests(task_acc, a, b)
            fam_out["comparisons"][key] = {
                "pooled": pt,
                "paired_runs": pr,
                "paired_tasks": ptask,
                "raw": {
                    a: {"succ": sa["succ"], "eps": sa["eps"], "n_runs": len(sa.get("rates", []))},
                    b: {"succ": sb["succ"], "eps": sb["eps"], "n_runs": len(sb.get("rates", []))},
                },
            }
            if fam == "paper_curated" and key in comparisons_keys and pt:
                z_ps_for_holm.append(float(pt["p_value_z"]))
                holm_keys_order.append(key)

        out["families"][fam] = fam_out

    holm_row: Dict[str, float] = {}
    if len(z_ps_for_holm) == len(holm_keys_order) and z_ps_for_holm:
        adj = holm_adjusted_pvalues(z_ps_for_holm)
        holm_row = dict(zip(holm_keys_order, adj))
    out["meta"]["holm_adjusted_p_z_paper_curated"] = holm_row

    # TOST equivalence
    tost_results = equivalence_table(pooled)
    out["tost_equivalence"] = tost_results

    # Detection vs correction table
    detection_vs_correction_table(pooled)

    # Cross-architecture table
    cross_arch_table(pooled)

    # Tuning ablation tables
    clamping_ablation_table()
    gating_ablation_table()

    # Safety head metrics table
    safety_head_metrics_table()

    # Stratified difficulty tests
    stratified_difficulty_tests(per_family_tasks)

    # Zero-shot aggregate z-test
    zs_fam = pooled.get("act_zero_shot_ood", {})
    zs_s = zs_fam.get("steering", {"succ": 0, "eps": 0})
    zs_v = zs_fam.get("vanilla", {"succ": 0, "eps": 0})
    if zs_s["eps"] > 0 and zs_v["eps"] > 0:
        zs_pt = proportion_tests(zs_s["succ"], zs_s["eps"], zs_v["succ"], zs_v["eps"])
        out["zero_shot_aggregate"] = {
            "steering": {"succ": zs_s["succ"], "eps": zs_s["eps"], "rate": 100.0 * zs_s["succ"] / zs_s["eps"]},
            "vanilla": {"succ": zs_v["succ"], "eps": zs_v["eps"], "rate": 100.0 * zs_v["succ"] / zs_v["eps"]},
            "diff_pp": zs_pt.get("diff_pp", 0),
            "p_value_z": zs_pt.get("p_value_z", 1),
        }
        print(f"Zero-shot aggregate: vanilla={100*zs_v['succ']/zs_v['eps']:.1f}%, steering={100*zs_s['succ']/zs_s['eps']:.1f}%, p={zs_pt.get('p_value_z', 1):.4f}")

    # Intervention-rate vs difficulty correlation:
    # Per-task steering intervention rate (`mean_ir`) vs per-task vanilla success rate,
    # aggregated across all paper-curated eval families. This is the metric the paper
    # reports ("the probe self-calibrates to task difficulty"); a low vanilla success
    # rate predicts a high intervention rate.
    family_prefixes = [
        "category1_sweep_",        # openvla_sweep
        "category1_ovla_ood_",     # openvla_ood
        "eval_act_steering_sweep_",  # act_sweep
        "eval_act_steering_act_ood", # act_ood_baselines
        "eval_act_zero_shot_",     # act_zero_shot_ood
    ]
    by_task: dict[str, dict[str, float]] = {}
    for prefix in family_prefixes:
        for fp in DATA.glob(f"{prefix}*eval_results*.json"):
            ed = json.loads(fp.read_text())
            for tid, modes in ed.get("per_task", {}).items():
                s = modes.get("steering", {})
                v = modes.get("vanilla", {})
                if "mean_ir" in s and v.get("n_episodes", 0) > 0 and s.get("n_episodes", 0) > 0:
                    rec = by_task.setdefault(str(tid), {"ir_sum": 0.0, "ir_n": 0, "v_succ": 0, "v_eps": 0})
                    rec["ir_sum"] += float(s["mean_ir"]) * int(s["n_episodes"])
                    rec["ir_n"]   += int(s["n_episodes"])
                    rec["v_succ"] += int(v.get("n_successes", 0))
                    rec["v_eps"]  += int(v.get("n_episodes", 0))
    rows = []
    for tid in sorted(by_task, key=lambda x: int(x)):
        r = by_task[tid]
        if r["ir_n"] > 0 and r["v_eps"] > 0:
            rows.append((r["ir_sum"] / r["ir_n"], 100.0 * r["v_succ"] / r["v_eps"]))
    if len(rows) >= 3:
        import numpy as np
        ir_a = np.array([x[0] for x in rows])
        vsr_a = np.array([x[1] for x in rows])
        r_p, p_p = stats.pearsonr(ir_a, vsr_a)
        r_s, p_s = stats.spearmanr(ir_a, vsr_a)
        out["intervention_difficulty_correlation"] = {
            "r_pearson": float(r_p),
            "p_pearson": float(p_p),
            "r_spearman": float(r_s),
            "p_spearman": float(p_s),
            "n_tasks": len(rows),
            "metric": "per-task steering mean_ir vs per-task vanilla success rate",
        }
        print(
            f"Intervention-vs-vanilla correlation: "
            f"r={r_p:.3f} (Pearson, p={p_p:.3g}), rho={r_s:.3f} (Spearman, p={p_s:.3g}), n={len(rows)}"
        )

    OUT_JSON.write_text(json.dumps(out, indent=2))

    md = render_markdown(out, holm_row)
    OUT_MD.write_text(md)

    # Primary stat tests table
    cur = out["families"].get("paper_curated", {}).get("comparisons", {})
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Statistical tests on paper-curated pooled LIBERO results. Holm-adjusted $p$-values apply to the three z-tests as one family.}",
        r"\label{tab:stat_tests}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Comparison & $\Delta$ Success (pp) & 95\% CI (pp) & $p$ (z) / Holm \\",
        r"\midrule",
    ]
    for key, label in PRIMARY_LABELS.items():
        p = cur.get(key, {}).get("pooled", {})
        if not p:
            continue
        h = holm_row.get(key, float("nan"))
        h_txt = f"{h:.4g}" if not math.isnan(h) else "--"
        tex.append(
            f"{label} & {p['diff_pp']:+.2f} & "
            f"[{p['ci95_diff_pp'][0]:+.2f}, {p['ci95_diff_pp'][1]:+.2f}] & "
            f"{p['p_value_z']:.4g} / {h_txt} {EOL}"
        )

    # Add detection vs correction contrasts
    stop_key = "latent_stop_vs_vanilla"
    stop_p = cur.get(stop_key, {}).get("pooled", {})
    if stop_p:
        tex.append(r"\midrule")
        tex.append(
            f"Latent Stop vs Vanilla & {stop_p['diff_pp']:+.2f} & "
            f"[{stop_p['ci95_diff_pp'][0]:+.2f}, {stop_p['ci95_diff_pp'][1]:+.2f}] & "
            f"{stop_p['p_value_z']:.4g} {EOL}"
        )

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_TEX.write_text("\n".join(tex) + "\n")

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
