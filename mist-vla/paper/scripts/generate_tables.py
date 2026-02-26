#!/usr/bin/env python3
import json
from pathlib import Path

PAPER = Path('/home/mpcr/Desktop/SalusV5/mist-vla/paper')
DATA = PAPER / 'data'
TABLES = PAPER / 'tables'
TABLES.mkdir(parents=True, exist_ok=True)

EOL = r"\\"


def load(name):
    p = DATA / name
    if not p.exists():
        return None
    with open(p, 'r') as f:
        return json.load(f)


def fmt(v):
    return f"{v:.1f}"


def write(name, lines):
    (TABLES / name).write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Legacy ACT key table
# -----------------------------------------------------------------------------
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
act_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write('tab_act_key_results.tex', act_table)


# -----------------------------------------------------------------------------
# Legacy paired clean runs
# -----------------------------------------------------------------------------
paired_tags = [
    'sweep_20260224_204844_t01_a0p1_mc0p003_ct0p003_ft0p6',
    'sweep_20260224_204844_t02_a0p12_mc0p003_ct0p002_ft0p65',
    'sweep_20260224_204844_t03_a0p1_mc0p004_ct0p003_ft0p65',
    'sweep_20260224_204844_t04_a0p1_mc0p004_ct0p0025_ft0p55',
]
paired_rows, act_d, ov_d = [], [], []
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
    short_t = t.replace('sweep_20260224_204844_', '').replace('_', r'\_')
    paired_table.append(f"{short_t} & {da:+.1f} & {do:+.1f} & {dm:+.2f} {EOL}")
if paired_rows:
    paired_table.append(r"\midrule")
    paired_table.append(
        f"Aggregate (completed) & {sum(act_d)/len(act_d):+.2f} & {sum(ov_d)/len(ov_d):+.2f} & {(sum(act_d)/len(act_d)+sum(ov_d)/len(ov_d))/2:+.2f} {EOL}"
    )
paired_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write('tab_paired_clean_runs.tex', paired_table)


# -----------------------------------------------------------------------------
# Controller latency table
# -----------------------------------------------------------------------------
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
lat_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write('tab_latency.tex', lat_table)


# -----------------------------------------------------------------------------
# NEW: ACT strict zero-shot OOD table
# -----------------------------------------------------------------------------
zero_shot_names = [
    'eval_act_zero_shot_zs_act_ood_s42_tA01234567_tB89_eval_results.json',
    'eval_act_zero_shot_zs_act_ood_s43_tA01234567_tB89_eval_results.json',
    'eval_act_zero_shot_zs_act_ood_s44_tA01234567_tB89_eval_results.json',
    'eval_act_zero_shot_zs_act_ood_s42_tA23456789_tB01_eval_results.json',
    'eval_act_zero_shot_zs_act_ood_s42_tA012345_tB6789_eval_results.json',
    'eval_act_zero_shot_zs_act_ood_stop_tA01234567_tB89_eval_results.json',
]
zs_rows = []
for n in zero_shot_names:
    d = load(n)
    if d is None:
        continue
    s = d.get('summary', {})
    tag = n.replace('eval_act_zero_shot_', '').replace('_eval_results.json', '')
    zs_rows.append((
        tag.replace('_', r'\_'),
        s.get('avg_vanilla_pct', 0.0),
        s.get('avg_latent_stop_pct', 0.0),
        s.get('avg_mppi_pct', 0.0),
        s.get('avg_steering_pct', 0.0),
        s.get('delta_steering_vs_vanilla_pp', 0.0),
    ))

zs_table = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{ACT strict zero-shot OOD runs (train-task/test-task splits with disturbance stress).}",
    r"\label{tab:act_zero_shot_ood}",
    r"\begin{tabular}{lccccc}",
    r"\toprule",
    r"Run & Vanilla & Latent Stop & MPPI & Steering & $\Delta$(Steering$-$Vanilla) \\",
    r"\midrule",
]
for r in zs_rows:
    zs_table.append(f"{r[0]} & {r[1]:.1f} & {r[2]:.1f} & {r[3]:.1f} & {r[4]:.1f} & {r[5]:+.1f} {EOL}")
if zs_rows:
    zs_table.append(r"\midrule")
    zs_table.append(
        f"Aggregate (completed) & {sum(r[1] for r in zs_rows)/len(zs_rows):.1f} & "
        f"{sum(r[2] for r in zs_rows)/len(zs_rows):.1f} & "
        f"{sum(r[3] for r in zs_rows)/len(zs_rows):.1f} & "
        f"{sum(r[4] for r in zs_rows)/len(zs_rows):.1f} & "
        f"{sum(r[5] for r in zs_rows)/len(zs_rows):+.2f} {EOL}"
    )
else:
    zs_table.append(r"\texttt{TODO} & -- & -- & -- & -- & -- \\")
zs_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write('tab_act_zero_shot_ood.tex', zs_table)


# -----------------------------------------------------------------------------
# NEW: ACT OOD baseline runs (non-zero-shot) table
# -----------------------------------------------------------------------------
act_ood_names = [
    'eval_act_steering_act_ood_baselines_s42_t89_eval_results.json',
    'eval_act_steering_act_ood_t89_s42_eval_results.json',
    'eval_act_steering_act_ood_t89_s43_eval_results.json',
    'eval_act_steering_act_ood_t89_s44_eval_results.json',
]
act_ood_rows = []
for n in act_ood_names:
    d = load(n)
    if d is None:
        continue
    s = d.get('summary', {})
    tag = n.replace('eval_act_steering_', '').replace('_eval_results.json', '').replace('_', r'\_')
    act_ood_rows.append((
        tag,
        s.get('avg_vanilla_pct', 0.0),
        s.get('avg_latent_stop_pct', 0.0),
        s.get('avg_mppi_pct', 0.0),
        s.get('avg_steering_pct', 0.0),
        s.get('avg_mppi_apply_ms', 0.0),
        s.get('avg_steering_apply_ms', 0.0),
    ))

act_ood_table = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{ACT OOD baseline campaign on hard tasks (completed seeds).}",
    r"\label{tab:act_ood_baselines}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"Run & Vanilla & Latent Stop & MPPI & Steering & MPPI ms & Steering ms \\",
    r"\midrule",
]
for r in act_ood_rows:
    act_ood_table.append(f"{r[0]} & {r[1]:.1f} & {r[2]:.1f} & {r[3]:.1f} & {r[4]:.1f} & {r[5]:.3f} & {r[6]:.3f} {EOL}")
if act_ood_rows:
    act_ood_table.append(r"\midrule")
    act_ood_table.append(
        f"Aggregate (completed) & {sum(r[1] for r in act_ood_rows)/len(act_ood_rows):.1f} & "
        f"{sum(r[2] for r in act_ood_rows)/len(act_ood_rows):.1f} & "
        f"{sum(r[3] for r in act_ood_rows)/len(act_ood_rows):.1f} & "
        f"{sum(r[4] for r in act_ood_rows)/len(act_ood_rows):.1f} & "
        f"{sum(r[5] for r in act_ood_rows)/len(act_ood_rows):.3f} & "
        f"{sum(r[6] for r in act_ood_rows)/len(act_ood_rows):.3f} {EOL}"
    )
else:
    act_ood_table.append(r"\texttt{TODO} & -- & -- & -- & -- & -- & -- \\")
act_ood_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write('tab_act_ood_baselines.tex', act_ood_table)


# -----------------------------------------------------------------------------
# NEW: OpenVLA OOD progress/status table
# -----------------------------------------------------------------------------
ov_ood_names = [
    'category1_ovla_ood_t89_s42_eval_results.json',
    'category1_ovla_ood_t89_s43_eval_results.json',
    'category1_ovla_ood_t89_s44_eval_results.json',
]
ov_rows = []
for n in ov_ood_names:
    d = load(n)
    if d is None:
        continue
    s = d.get('summary', {})
    tag = n.replace('category1_', '').replace('_eval_results.json', '').replace('_', r'\_')
    ov_rows.append((
        tag,
        s.get('avg_vanilla_pct', 0.0),
        s.get('avg_latent_stop_pct', 0.0),
        s.get('avg_mppi_pct', 0.0),
        s.get('avg_steering_pct', 0.0),
    ))

ov_table = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{OpenVLA OOD baseline campaign status (completed seeds only).}",
    r"\label{tab:openvla_ood_status}",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Run & Vanilla & Latent Stop & MPPI & Steering \\",
    r"\midrule",
]
for r in ov_rows:
    ov_table.append(f"{r[0]} & {r[1]:.1f} & {r[2]:.1f} & {r[3]:.1f} & {r[4]:.1f} {EOL}")
if ov_rows:
    ov_table.append(r"\midrule")
    ov_table.append(
        f"Aggregate (completed) & {sum(r[1] for r in ov_rows)/len(ov_rows):.1f} & "
        f"{sum(r[2] for r in ov_rows)/len(ov_rows):.1f} & "
        f"{sum(r[3] for r in ov_rows)/len(ov_rows):.1f} & "
        f"{sum(r[4] for r in ov_rows)/len(ov_rows):.1f} {EOL}"
    )
else:
    ov_table.append(r"\texttt{PENDING: seeds s42/s43/s44 still running} & -- & -- & -- & -- \\")
ov_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write('tab_openvla_ood_status.tex', ov_table)

print('Wrote tables:', sorted([p.name for p in TABLES.glob('tab_*.tex')]))
