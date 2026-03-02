#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path

PAPER = Path('/home/mpcr/Desktop/SalusV5/mist-vla/paper')
DATA = PAPER / 'data'
TABLES = PAPER / 'tables'
TABLES.mkdir(parents=True, exist_ok=True)
EOL = r"\\"


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write(name, lines):
    (TABLES / name).write_text("\n".join(lines) + "\n")


def family(name):
    if name.startswith('category1_sweep_'):
        return 'openvla_sweep'
    if name.startswith('category1_ovla_ood_'):
        return 'openvla_ood'
    if name.startswith('eval_act_steering_sweep_'):
        return 'act_sweep'
    if name.startswith('eval_act_steering_act_ood'):
        return 'act_ood_baselines'
    if name.startswith('eval_act_zero_shot_'):
        return 'act_zero_shot_ood'
    return None


def pool(files):
    stats = defaultdict(lambda: {'succ': 0, 'eps': 0, 'apply_ms': []})
    for path in files:
        d = load_json(path)
        per = d.get('per_task', {})
        if not isinstance(per, dict):
            continue
        for task_data in per.values():
            if not isinstance(task_data, dict):
                continue
            for mode, m in task_data.items():
                if not isinstance(m, dict):
                    continue
                ns = m.get('n_successes')
                ne = m.get('n_episodes')
                if isinstance(ns, (int, float)) and isinstance(ne, (int, float)) and ne > 0:
                    stats[mode]['succ'] += int(ns)
                    stats[mode]['eps'] += int(ne)
                am = m.get('mean_apply_ms')
                if isinstance(am, (int, float)) and am > 0:
                    stats[mode]['apply_ms'].append(float(am))
    return stats


def pct(s):
    if s['eps'] <= 0:
        return None
    return 100.0 * s['succ'] / s['eps']


def mean_or_none(vals):
    return None if not vals else sum(vals) / len(vals)


all_files = sorted(DATA.glob('*eval_results*.json'))
groups = defaultdict(list)
for f in all_files:
    fam = family(f.name)
    if fam:
        groups[fam].append(f)

pooled_by_family = {k: pool(v) for k, v in groups.items()}
curated_families = [
    'openvla_sweep',
    'openvla_ood',
    'act_sweep',
    'act_ood_baselines',
    'act_zero_shot_ood',
]

# ---------------------------------------------------------------------
# Table 1: Final pooled results (paper-level)
# ---------------------------------------------------------------------
combined = defaultdict(lambda: {'succ': 0, 'eps': 0, 'apply_ms': []})
for fam in curated_families:
    for mode, s in pooled_by_family.get(fam, {}).items():
        combined[mode]['succ'] += s['succ']
        combined[mode]['eps'] += s['eps']
        combined[mode]['apply_ms'].extend(s['apply_ms'])

final_rows = []
for mode in ['vanilla', 'mppi', 'steering', 'latent_jiggle', 'noise', 'latent_stop', 'ema_only']:
    s = combined.get(mode)
    if not s or s['eps'] == 0:
        continue
    sr = pct(s)
    ms = mean_or_none(s['apply_ms'])
    final_rows.append((mode, sr, s['succ'], s['eps'], ms))

table = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Final pooled results across completed runs (OpenVLA sweep+OOD and ACT sweep+OOD+zero-shot OOD).}",
    r"\label{tab:final_pooled_results}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Method & Success (\%) & Successes / Episodes & Mean apply time (ms) \\",
    r"\midrule",
]
for mode, sr, succ, eps, ms in final_rows:
    label = {
        'vanilla': 'Vanilla',
        'mppi': 'MPPI',
        'steering': 'Steering (Ours)',
        'latent_jiggle': 'Latent Jiggle',
        'noise': 'Random Noise',
        'latent_stop': 'Latent Stop',
        'ema_only': 'EMA-only',
    }[mode]
    ms_txt = f"{ms:.3f}" if ms is not None else "--"
    table.append(f"{label} & {sr:.2f} & {succ}/{eps} & {ms_txt} {EOL}")

if combined['steering']['eps'] > 0 and combined['mppi']['eps'] > 0:
    st = pct(combined['steering'])
    mp = pct(combined['mppi'])
    table.append(r"\midrule")
    table.append(f"$\\Delta$(Steering$-$MPPI) & {st-mp:+.2f} pp & -- & -- {EOL}")
    st_ms = mean_or_none(combined['steering']['apply_ms'])
    mp_ms = mean_or_none(combined['mppi']['apply_ms'])
    if st_ms and mp_ms:
        table.append(f"Speedup (MPPI/Ours) & -- & -- & {mp_ms/st_ms:.2f}$\\times$ {EOL}")
table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write('tab_final_pooled_results.tex', table)

# ---------------------------------------------------------------------
# Table 2: OpenVLA final pooled
# ---------------------------------------------------------------------
ov = pooled_by_family.get('openvla_sweep', defaultdict(dict))
for mode, s in pooled_by_family.get('openvla_ood', {}).items():
    ov.setdefault(mode, {'succ': 0, 'eps': 0, 'apply_ms': []})
    ov[mode]['succ'] += s['succ']
    ov[mode]['eps'] += s['eps']
    ov[mode]['apply_ms'].extend(s['apply_ms'])

ov_table = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{OpenVLA pooled final results (sweep + OOD).}",
    r"\label{tab:openvla_final_pooled}",
    r"\begin{tabular}{lcc}",
    r"\toprule",
    r"Method & Success (\%) & Successes / Episodes \\",
    r"\midrule",
]
for mode in ['vanilla', 'mppi', 'steering', 'latent_jiggle', 'noise', 'latent_stop', 'ema_only']:
    s = ov.get(mode)
    if not s or s['eps'] == 0:
        continue
    ov_table.append(f"{mode.replace('_', ' ').title()} & {pct(s):.2f} & {s['succ']}/{s['eps']} {EOL}")
ov_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write('tab_openvla_final_pooled.tex', ov_table)

# ---------------------------------------------------------------------
# Table 3: ACT final pooled
# ---------------------------------------------------------------------
act = pooled_by_family.get('act_sweep', defaultdict(dict))
for fam in ['act_ood_baselines', 'act_zero_shot_ood']:
    for mode, s in pooled_by_family.get(fam, {}).items():
        act.setdefault(mode, {'succ': 0, 'eps': 0, 'apply_ms': []})
        act[mode]['succ'] += s['succ']
        act[mode]['eps'] += s['eps']
        act[mode]['apply_ms'].extend(s['apply_ms'])

act_table = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{ACT pooled final results (sweep + OOD + zero-shot OOD).}",
    r"\label{tab:act_final_pooled}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Method & Success (\%) & Successes / Episodes & Mean apply time (ms) \\",
    r"\midrule",
]
for mode in ['vanilla', 'mppi', 'steering', 'latent_jiggle', 'latent_stop']:
    s = act.get(mode)
    if not s or s['eps'] == 0:
        continue
    ms = mean_or_none(s['apply_ms'])
    ms_txt = f"{ms:.3f}" if ms is not None else "--"
    act_table.append(f"{mode.replace('_', ' ').title()} & {pct(s):.2f} & {s['succ']}/{s['eps']} & {ms_txt} {EOL}")
act_table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write('tab_act_final_pooled.tex', act_table)

print('Wrote tables:', sorted([p.name for p in TABLES.glob('tab_*.tex')]))
