#!/usr/bin/env python3
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

PAPER = Path('/home/mpcr/Desktop/SalusV5/mist-vla/paper')
DATA = PAPER / 'data'
FIG = PAPER / 'figures'
FIG.mkdir(parents=True, exist_ok=True)


def load(name):
    p = DATA / name
    if not p.exists():
        return None
    with open(p, 'r') as f:
        return json.load(f)


def save(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out.name)


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


def pooled_stats():
    groups = defaultdict(list)
    for p in sorted(DATA.glob('*eval_results*.json')):
        fam = family(p.name)
        if fam:
            groups[fam].append(p)
    pooled = defaultdict(lambda: defaultdict(lambda: {'succ': 0, 'eps': 0, 'apply_ms': []}))
    for fam, files in groups.items():
        for fp in files:
            d = json.loads(fp.read_text())
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
                        pooled[fam][mode]['succ'] += int(ns)
                        pooled[fam][mode]['eps'] += int(ne)
                    am = m.get('mean_apply_ms')
                    if isinstance(am, (int, float)) and am > 0:
                        pooled[fam][mode]['apply_ms'].append(float(am))
    return pooled


def pct(s):
    return 100.0 * s['succ'] / s['eps'] if s['eps'] > 0 else np.nan


def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    def box(x, y, w, h, txt, fc='#eaf2ff', ec='#335c99', fs=10):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs)

    box(0.03, 0.35, 0.16, 0.3, 'Observation +\nInstruction', fc='#f5f5f5', ec='#666')
    box(0.23, 0.35, 0.16, 0.3, 'Base Policy\n(VLA / ACT)', fc='#fff3e6', ec='#cc7a00')
    box(0.43, 0.35, 0.16, 0.3, 'Hidden State\n$h_t$', fc='#eefaf0', ec='#2f7d32')
    box(0.62, 0.50, 0.16, 0.15, 'Fail Head\n$\\hat y^{fail}$', fc='#fdecea', ec='#a33')
    box(0.62, 0.35, 0.16, 0.13, 'TTF Head\n$\\hat\\tau$', fc='#f0ecfd', ec='#5a4ea1')
    box(0.62, 0.18, 0.16, 0.13, 'Correction Head\n$\\widehat{\\Delta p}$', fc='#e8f7ff', ec='#1b6a9e')
    box(0.82, 0.35, 0.15, 0.3, 'Gate + Clamp\n+ EMA + Stop', fc='#fff9e6', ec='#997a00')

    for (x1, y1, x2, y2) in [
        (0.19, 0.5, 0.23, 0.5), (0.39, 0.5, 0.43, 0.5),
        (0.59, 0.5, 0.62, 0.57), (0.59, 0.5, 0.62, 0.41), (0.59, 0.5, 0.62, 0.245),
        (0.78, 0.57, 0.82, 0.5), (0.78, 0.41, 0.82, 0.5), (0.78, 0.245, 0.82, 0.5),
    ]:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=1.8))

    ax.text(0.5, 0.04, 'Inference-Time Latent Safety Steering Pipeline', ha='center', fontsize=12, fontweight='bold')
    save(fig, '06_ours_system_architecture.png')


def fig_zero_shot_seed_deltas():
    names = [
        'eval_act_zero_shot_zs_act_ood_s42_tA01234567_tB89_eval_results.json',
        'eval_act_zero_shot_zs_act_ood_s43_tA01234567_tB89_eval_results.json',
        'eval_act_zero_shot_zs_act_ood_s44_tA01234567_tB89_eval_results.json',
    ]
    rows = []
    for n in names:
        d = load(n)
        if not d:
            continue
        s = d.get('summary', {})
        seed = n.split('_s')[1].split('_')[0]
        rows.append((seed, s.get('delta_latent_stop_vs_vanilla_pp', np.nan), s.get('delta_mppi_vs_vanilla_pp', np.nan), s.get('delta_steering_vs_vanilla_pp', np.nan)))
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(rows))
    w = 0.24
    stop = [r[1] for r in rows]
    mppi = [r[2] for r in rows]
    steer = [r[3] for r in rows]

    ax.bar(x - w, stop, width=w, label='Latent Stop - Vanilla', color='#d55e5e')
    ax.bar(x, mppi, width=w, label='MPPI - Vanilla', color='#6c8ebf')
    ax.bar(x + w, steer, width=w, label='Steering - Vanilla', color='#2e8b57')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {r[0]}' for r in rows])
    ax.set_ylabel('Delta Success (pp)')
    ax.set_title('Strict Zero-Shot OOD: Baseline Deltas by Seed (ACT)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', alpha=0.25)
    save(fig, '07_zero_shot_seed_deltas.png')


def fig_task_mode_heatmap():
    d = load('eval_act_zero_shot_zs_act_ood_s42_tA01234567_tB89_eval_results.json')
    if not d:
        return
    per = d.get('per_task', {})
    tasks = sorted(int(k) for k in per.keys())
    modes = ['vanilla', 'latent_stop', 'mppi', 'steering']
    M = np.zeros((len(tasks), len(modes)))
    for i, t in enumerate(tasks):
        row = per[str(t)]
        for j, m in enumerate(modes):
            M[i, j] = row.get(m, {}).get('success_rate_pct', 0.0)

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    im = ax.imshow(M, cmap='viridis', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(modes, rotation=20, ha='right')
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels([f'Task {t}' for t in tasks])
    ax.set_title('Per-Task Success Heatmap (strict zero-shot run)')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i,j]:.0f}', ha='center', va='center', color='white', fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Success (%)')
    save(fig, '08_task_mode_heatmap.png')


def fig_latency_speed():
    pooled = pooled_stats()
    curated = ['openvla_sweep', 'openvla_ood', 'act_sweep', 'act_ood_baselines', 'act_zero_shot_ood']
    mppi_vals, steer_vals = [], []
    for fam in curated:
        mppi_vals.extend(pooled[fam].get('mppi', {}).get('apply_ms', []))
        steer_vals.extend(pooled[fam].get('steering', {}).get('apply_ms', []))
    if not mppi_vals or not steer_vals:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = [np.mean(mppi_vals), np.mean(steer_vals)]
    labels = ['MPPI', 'Steering (Ours)']
    colors = ['#5b7bb2', '#2e8b57']
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel('Mean apply latency (ms)')
    ax.set_title('Controller Step Latency (Final pooled completed runs)')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f'{v:.2f} ms', ha='center', fontsize=9)
    speed = vals[0] / max(vals[1], 1e-9)
    ax.text(0.5, max(vals) * 0.82, f'Speedup = {speed:.1f}x', ha='center', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))
    ax.grid(axis='y', alpha=0.25)
    save(fig, '09_latency_speedup_ood.png')


def fig_final_pooled_success():
    pooled = pooled_stats()
    curated = ['openvla_sweep', 'openvla_ood', 'act_sweep', 'act_ood_baselines', 'act_zero_shot_ood']
    combined = defaultdict(lambda: {'succ': 0, 'eps': 0})
    for fam in curated:
        for mode in ['vanilla', 'mppi', 'steering']:
            s = pooled[fam].get(mode, {'succ': 0, 'eps': 0})
            combined[mode]['succ'] += s['succ']
            combined[mode]['eps'] += s['eps']
    labels = ['Vanilla', 'MPPI', 'Steering']
    vals = [pct(combined['vanilla']), pct(combined['mppi']), pct(combined['steering'])]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(labels, vals, color=['#777777', '#5b7bb2', '#2e8b57'])
    ax.set_ylabel('Success (%)')
    ax.set_title('Final Pooled Success Across Completed Runs')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f'{v:.2f}', ha='center', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    save(fig, '10_final_pooled_success.png')


def fig_family_breakdown():
    pooled = pooled_stats()
    fams = ['openvla_sweep', 'openvla_ood', 'act_sweep', 'act_ood_baselines', 'act_zero_shot_ood']
    labels = ['OV Sweep', 'OV OOD', 'ACT Sweep', 'ACT OOD', 'ACT ZS-OOD']
    vanilla, mppi, steering = [], [], []
    for fam in fams:
        vanilla.append(pct(pooled[fam].get('vanilla', {'succ': 0, 'eps': 0})))
        mppi.append(pct(pooled[fam].get('mppi', {'succ': 0, 'eps': 0})))
        steering.append(pct(pooled[fam].get('steering', {'succ': 0, 'eps': 0})))
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.bar(x - w, vanilla, width=w, label='Vanilla', color='#777777')
    ax.bar(x, mppi, width=w, label='MPPI', color='#5b7bb2')
    ax.bar(x + w, steering, width=w, label='Steering', color='#2e8b57')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Success (%)')
    ax.set_title('Per-Family Success Rates (Completed Runs)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.25)
    save(fig, '11_family_breakdown_success.png')


def main():
    fig_architecture()
    fig_zero_shot_seed_deltas()
    fig_task_mode_heatmap()
    fig_latency_speed()
    fig_final_pooled_success()
    fig_family_breakdown()


if __name__ == '__main__':
    main()
