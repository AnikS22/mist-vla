#!/usr/bin/env python3
"""Generate all paper figures from frozen data artifacts."""

from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
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


def fig_primary_ci():
    st = load('stat_tests_summary.json')
    if not st:
        return
    comp = st.get('families', {}).get('paper_curated', {}).get('comparisons', {})
    order = [
        ('steering_vs_mppi', 'Steering - MPPI'),
        ('steering_vs_vanilla', 'Steering - Vanilla'),
        ('mppi_vs_vanilla', 'MPPI - Vanilla'),
    ]
    labels, means, lo_err, hi_err = [], [], [], []
    for key, lbl in order:
        pooled = comp.get(key, {}).get('pooled', {})
        if not pooled:
            continue
        d = float(pooled['diff_pp'])
        ci = pooled['ci95_diff_pp']
        labels.append(lbl)
        means.append(d)
        lo_err.append(d - float(ci[0]))
        hi_err.append(float(ci[1]) - d)
    if not labels:
        return

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([lo_err, hi_err]),
        fmt='o',
        color='#2e8b57',
        ecolor='#2e8b57',
        capsize=4,
    )
    ax.axvline(0.0, color='black', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Delta success (pp), 95% CI')
    ax.set_title('Primary Contrasts on Paper-Curated Pool')
    ax.grid(axis='x', alpha=0.25)
    save(fig, '12_primary_contrasts_ci.png')


def fig_holm_pvals():
    st = load('stat_tests_summary.json')
    if not st:
        return
    comp = st.get('families', {}).get('paper_curated', {}).get('comparisons', {})
    holm = st.get('meta', {}).get('holm_adjusted_p_z_paper_curated', {})
    order = [
        ('steering_vs_mppi', 'Steering vs MPPI'),
        ('steering_vs_vanilla', 'Steering vs Vanilla'),
        ('mppi_vs_vanilla', 'MPPI vs Vanilla'),
    ]
    labels, zvals, hvals = [], [], []
    for key, lbl in order:
        pooled = comp.get(key, {}).get('pooled', {})
        if not pooled:
            continue
        labels.append(lbl)
        zvals.append(float(pooled.get('p_value_z', np.nan)))
        hvals.append(float(holm.get(key, np.nan)))
    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x - w / 2, zvals, width=w, label='p(z-test)', color='#5b7bb2')
    ax.bar(x + w / 2, hvals, width=w, label='p(Holm-adjusted)', color='#d55e5e')
    ax.axhline(0.05, color='black', linestyle='--', linewidth=1, label='alpha=0.05')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0.0, max(1.02, max(zvals + hvals) * 1.05))
    ax.set_ylabel('p-value')
    ax.set_title('Primary Contrast p-values (Raw and Holm-adjusted)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.25)
    save(fig, '13_primary_pvalues_holm.png')


def fig_required_n():
    st = load('stat_tests_summary.json')
    if not st:
        return
    comp = st.get('families', {}).get('paper_curated', {}).get('comparisons', {})
    order = [
        ('steering_vs_mppi', 'S-M'),
        ('steering_vs_vanilla', 'S-V'),
        ('mppi_vs_vanilla', 'M-V'),
    ]
    deltas = ['delta_1pp_symmetric', 'delta_2pp_symmetric', 'delta_5pp_symmetric', 'delta_10pp_symmetric']
    delta_labels = ['1pp', '2pp', '5pp', '10pp']
    vals = np.full((len(order), len(deltas)), np.nan)
    for i, (key, _) in enumerate(order):
        req = comp.get(key, {}).get('pooled', {}).get('required_n_per_arm_delta_pp', {})
        for j, d in enumerate(deltas):
            if d in req:
                vals[i, j] = float(req[d])
    if np.isnan(vals).all():
        return

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for i, (_, lbl) in enumerate(order):
        ax.plot(delta_labels, vals[i], marker='o', linewidth=2, label=lbl)
    ax.set_yscale('log')
    ax.set_ylabel('Required episodes per arm (log scale)')
    ax.set_title('Required N for Detecting Fixed Effect Sizes (80% power)')
    ax.grid(axis='y', alpha=0.25, which='both')
    ax.legend(title='Contrast', fontsize=8)
    save(fig, '14_required_n_by_effect_size.png')


def fig_cross_arch_comparison():
    """Bar chart comparing OpenVLA vs ACT across modes."""
    pooled = pooled_stats()

    ov = defaultdict(lambda: {'succ': 0, 'eps': 0})
    for fam in ['openvla_sweep', 'openvla_ood']:
        for mode, s in pooled[fam].items():
            ov[mode]['succ'] += s['succ']
            ov[mode]['eps'] += s['eps']

    act = defaultdict(lambda: {'succ': 0, 'eps': 0})
    for fam in ['act_sweep', 'act_ood_baselines', 'act_zero_shot_ood']:
        for mode, s in pooled[fam].items():
            act[mode]['succ'] += s['succ']
            act[mode]['eps'] += s['eps']

    modes = ['vanilla', 'latent_stop', 'mppi', 'steering']
    mode_labels = ['Vanilla', 'Latent Stop', 'MPPI', 'Steering']
    ov_vals = [pct(ov.get(m, {'succ': 0, 'eps': 0})) for m in modes]
    act_vals = [pct(act.get(m, {'succ': 0, 'eps': 0})) for m in modes]

    x = np.arange(len(modes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, ov_vals, width=w, label='OpenVLA (4096-d)', color='#5b7bb2')
    ax.bar(x + w/2, act_vals, width=w, label='ACT (256-d)', color='#2e8b57')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.set_ylabel('Success (%)')
    ax.set_title('Cross-Architecture Comparison: OpenVLA vs ACT')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    for i in range(len(modes)):
        if not np.isnan(ov_vals[i]):
            ax.text(x[i] - w/2, ov_vals[i] + 0.5, f'{ov_vals[i]:.1f}', ha='center', fontsize=7)
        if not np.isnan(act_vals[i]):
            ax.text(x[i] + w/2, act_vals[i] + 0.5, f'{act_vals[i]:.1f}', ha='center', fontsize=7)
    save(fig, '21_cross_arch_comparison.png')


def fig_clamping_ablation():
    """Plot clamping ablation from tuning data."""
    path = PAPER.parent / 'hpc_mirror' / 'results' / 'tuning' / 'clamping_sweep.json'
    if not path.exists():
        return
    d = json.loads(path.read_text())
    results = d.get('results', {})
    tasks = sorted(set(v['task_id'] for v in results.values()))

    vanilla_sr = []
    clamp_sr = []
    task_labels = []
    for tid in tasks:
        v = results.get(f"({tid}, 'vanilla')", {})
        c = results.get(f"({tid}, 'clamp=0.010m')", {})
        vanilla_sr.append(v.get('success_rate_pct', 0))
        clamp_sr.append(c.get('success_rate_pct', 0))
        task_labels.append(f'Task {tid}')

    x = np.arange(len(tasks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, vanilla_sr, width=w, label='Vanilla', color='#777777')
    ax.bar(x + w/2, clamp_sr, width=w, label='Clamp=0.01m', color='#2e8b57')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel('Success (%)')
    ax.set_title('Clamping Ablation: Trust Region Effect')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    save(fig, '22_clamping_ablation.png')


def fig_intervention_vs_difficulty():
    """Scatter: vanilla SR (x) vs actual intervention rate (y) per task, computed from all eval JSONs."""
    from collections import defaultdict

    # Compute actual mean_ir per task from all steering eval data
    task_ir = defaultdict(list)
    task_vanilla = defaultdict(lambda: {'succ': 0, 'eps': 0})

    for fp in sorted(DATA.glob('*eval_results*.json')):
        d_file = json.loads(fp.read_text())
        per = d_file.get('per_task', {})
        if not isinstance(per, dict):
            continue
        for tid, td in per.items():
            if not isinstance(td, dict):
                continue
            v = td.get('vanilla', {})
            if isinstance(v, dict) and v.get('n_episodes', 0) > 0:
                task_vanilla[tid]['succ'] += int(v.get('n_successes', 0))
                task_vanilla[tid]['eps'] += int(v['n_episodes'])
            s = td.get('steering', {})
            if isinstance(s, dict):
                ir = s.get('mean_ir')
                if isinstance(ir, (int, float)) and ir > 0:
                    task_ir[tid].append(float(ir))

    tasks = sorted(set(task_ir.keys()) & set(task_vanilla.keys()), key=int)
    if len(tasks) < 3:
        return

    xs, ys, labels = [], [], []
    for tid in tasks:
        if task_vanilla[tid]['eps'] > 0 and task_ir[tid]:
            vsr = 100.0 * task_vanilla[tid]['succ'] / task_vanilla[tid]['eps']
            mir = np.mean(task_ir[tid])
            xs.append(vsr)
            ys.append(mir * 100)  # as percentage
            labels.append(f'T{tid}')

    xs = np.array(xs)
    ys = np.array(ys)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(xs, ys, s=60, color='#2e8b57', edgecolors='#1a5e37', linewidths=0.6, zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (xs[i], ys[i]), fontsize=7.5, textcoords='offset points',
                    xytext=(5, 4), color='#333')

    # Correlation + fit line
    from scipy import stats as sp_stats
    r_val, p_val = sp_stats.pearsonr(xs, ys)
    m, b = np.polyfit(xs, ys, 1)
    xfit = np.linspace(xs.min() - 5, xs.max() + 5, 50)
    ax.plot(xfit, m * xfit + b, color='#999', linewidth=1, linestyle='--', zorder=1)
    ax.text(0.03, 0.97, f'r = {r_val:.2f}, p = {p_val:.1e}', transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85))

    ax.set_xlabel('Vanilla Success Rate (%)')
    ax.set_ylabel('Mean Intervention Rate (%)')
    ax.set_title('Adaptive Gating: Intervention Rate vs Task Difficulty')
    ax.grid(alpha=0.2)
    save(fig, '23_intervention_vs_difficulty.png')


def fig_safety_head_auc():
    """Bar chart comparing OpenVLA vs ACT safety head metrics."""
    ov_path = PAPER.parent / 'hpc_mirror' / 'checkpoints' / 'eef_correction_mlp' / 'results.json'
    act_path = PAPER.parent / 'hpc_mirror' / 'checkpoints' / 'eef_correction_mlp_act_honest' / 'results.json'
    if not ov_path.exists() or not act_path.exists():
        return
    ov = json.loads(ov_path.read_text())['test_results']
    act = json.loads(act_path.read_text())['test_results']

    metrics = ['fail_auc', 'ttf_corr', 'X_dir_auc', 'Y_dir_auc', 'Z_dir_auc']
    labels = ['Failure\nAUC', 'TTF\nCorr', 'X Dir\nAUC', 'Y Dir\nAUC', 'Z Dir\nAUC']
    ov_vals = [ov[m] for m in metrics]
    act_vals = [act[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, ov_vals, width=w, label='OpenVLA (4096-d, 1.1M)', color='#5b7bb2')
    ax.bar(x + w / 2, act_vals, width=w, label='ACT (256-d, 108K)', color='#2e8b57')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Safety Head: OpenVLA vs ACT')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    for i in range(len(metrics)):
        ax.text(x[i] - w / 2, ov_vals[i] + 0.01, f'{ov_vals[i]:.2f}', ha='center', fontsize=7)
        ax.text(x[i] + w / 2, act_vals[i] + 0.01, f'{act_vals[i]:.2f}', ha='center', fontsize=7)
    save(fig, '24_safety_head_auc.png')


def fig_latent_pca():
    """Check if latent PCA figure already exists; note availability."""
    pca_path = FIG / '17_latent_pca_success_failure.png'
    if pca_path.exists():
        print(f'fig_latent_pca: {pca_path.name} already exists, skipping regeneration')
    else:
        print(f'fig_latent_pca: {pca_path.name} not found; run generate_latent_embeddings.py to create it')


def main():
    fig_architecture()
    fig_zero_shot_seed_deltas()
    fig_task_mode_heatmap()
    fig_latency_speed()
    fig_final_pooled_success()
    fig_family_breakdown()
    fig_primary_ci()
    fig_holm_pvals()
    fig_required_n()
    fig_cross_arch_comparison()
    fig_clamping_ablation()
    fig_intervention_vs_difficulty()
    fig_safety_head_auc()
    fig_latent_pca()


if __name__ == '__main__':
    main()
