#!/usr/bin/env python3
import json
from pathlib import Path

PAPER = Path('/home/mpcr/Desktop/SalusV5/mist-vla/paper')
DATA = PAPER / 'data'
TABLES = PAPER / 'tables'
TABLES.mkdir(parents=True, exist_ok=True)


def load(name):
    p = DATA / name
    if not p.exists():
        return None
    with open(p, 'r') as f:
        return json.load(f)


def fmt(v):
    return f"{v:.1f}"


EOL = r"\\"

# Main ACT summary table (best-known + key baselines)
act_baseline = load('eval_act_steering_eval_results.json')
act_v3 = load('eval_act_steering_shared_profile_v3_eval_results.json')
act_t02 = load('eval_act_steering_sweep_20260224_204844_t02_a0p12_mc0p003_ct0p002_ft0p65_eval_results.json')

rows = []
for label, obj in [
    ('ACT Baseline Sweep (pre-gate fix)', act_baseline),
    ('ACT Shared Profile v3', act_v3),
    ('ACT Clean Sweep t02', act_t02),
]:
    if obj is None:
        continue
    s = obj.get('summary', {})
    rows.append((
        label,
        s.get('avg_vanilla_pct', 0.0),
        s.get('avg_mppi_pct', 0.0),
        s.get('avg_steering_pct', 0.0),
        s.get('avg_steering_pct', 0.0) - s.get('avg_mppi_pct', 0.0),
    ))

act_table = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{ACT results across key controller configurations.}",
    r"\label{tab:act_key_results}",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Run & Vanilla & MPPI & Steering (Ours) & $\Delta$(Ours$-$MPPI) \\",
    r"\midrule",
]
for r in rows:
    act_table.append(f"{r[0]} & {fmt(r[1])} & {fmt(r[2])} & {fmt(r[3])} & {r[4]:+.1f} {EOL}")
act_table += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
(TABLES / 'tab_act_key_results.tex').write_text('\n'.join(act_table) + '\n')

# Paired clean runs aggregate t01-t04
paired_tags = [
    'sweep_20260224_204844_t01_a0p1_mc0p003_ct0p003_ft0p6',
    'sweep_20260224_204844_t02_a0p12_mc0p003_ct0p002_ft0p65',
    'sweep_20260224_204844_t03_a0p1_mc0p004_ct0p003_ft0p65',
    'sweep_20260224_204844_t04_a0p1_mc0p004_ct0p0025_ft0p55',
]
paired_rows = []
act_d, ov_d = [], []
for t in paired_tags:
    a = load(f'eval_act_steering_{t}_eval_results.json')
    o = load(f'category1_{t}_eval_results.json')
    if a is None or o is None:
        continue
    sa = a.get('summary', {})
    so = o.get('summary', {})
    da = sa.get('avg_steering_pct', 0.0) - sa.get('avg_mppi_pct', 0.0)
    do = so.get('avg_steering_pct', 0.0) - so.get('avg_mppi_pct', 0.0)
    paired_rows.append((t, da, do, 0.5 * (da + do)))
    act_d.append(da)
    ov_d.append(do)

paired_table = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Completed paired clean-sweep runs (same parameters on ACT and OpenVLA).}",
    r"\label{tab:paired_clean_runs}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Run tag & ACT $\Delta$ (Ours$-$MPPI) & OpenVLA $\Delta$ & Cross-model mean $\Delta$ \\",
    r"\midrule",
]
for t, da, do, dm in paired_rows:
    short_t = t.replace('sweep_20260224_204844_', '')
    paired_table.append(f"{short_t} & {da:+.1f} & {do:+.1f} & {dm:+.2f} {EOL}")
if paired_rows:
    paired_table.append(r"\midrule")
    paired_table.append(
        f"Aggregate (completed) & {sum(act_d)/len(act_d):+.2f} & {sum(ov_d)/len(ov_d):+.2f} & {(sum(act_d)/len(act_d)+sum(ov_d)/len(ov_d))/2:+.2f} {EOL}"
    )
paired_table += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
(TABLES / 'tab_paired_clean_runs.tex').write_text('\n'.join(paired_table) + '\n')

# Latency table from benchmark job
lat = load('job_4539787_eval_results.json')
lat_table = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Controller apply-time latency on ACT benchmark job (GPU).}",
    r"\label{tab:latency}",
    r"\begin{tabular}{lcc}",
    r"\toprule",
    r"Method & Mean apply time (ms) & P95 apply time (ms) \\",
    r"\midrule",
]
if lat is not None:
    s = lat.get('summary', {})
    mppi_m = s.get('avg_mppi_apply_ms', 0.0)
    mppi_p = s.get('avg_mppi_apply_p95_ms', 0.0)
    st_m = s.get('avg_steering_apply_ms', 0.0)
    st_p = s.get('avg_steering_apply_p95_ms', 0.0)
    lat_table.append(f"MPPI & {mppi_m:.3f} & {mppi_p:.3f} {EOL}")
    lat_table.append(f"Steering (Ours) & {st_m:.3f} & {st_p:.3f} {EOL}")
    if st_m > 1e-9 and st_p > 1e-9:
        lat_table.append(r"\midrule")
        lat_table.append(
            f"Speedup (MPPI / Ours) & {mppi_m/st_m:.2f}$\\times$ & {mppi_p/st_p:.2f}$\\times$ {EOL}"
        )
else:
    lat_table.append(f"MPPI & \\texttt{{TODO}} & \\texttt{{TODO}} {EOL}")
    lat_table.append(f"Steering (Ours) & \\texttt{{TODO}} & \\texttt{{TODO}} {EOL}")
lat_table += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
(TABLES / 'tab_latency.tex').write_text('\n'.join(lat_table) + '\n')

print('Wrote tables:', [p.name for p in TABLES.glob('tab_*.tex')])
