#!/usr/bin/env python3
from pathlib import Path
import json
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
    files = [
        'eval_act_zero_shot_zs_act_ood_s42_tA01234567_tB89_eval_results.json',
        'eval_act_zero_shot_zs_act_ood_s43_tA01234567_tB89_eval_results.json',
        'eval_act_zero_shot_zs_act_ood_s44_tA01234567_tB89_eval_results.json',
        'eval_act_zero_shot_zs_act_ood_stop_tA01234567_tB89_eval_results.json',
    ]
    mppi, steer = [], []
    for f in files:
        d = load(f)
        if not d:
            continue
        s = d.get('summary', {})
        if s.get('avg_mppi_apply_ms') is not None and s.get('avg_steering_apply_ms') is not None:
            mppi.append(float(s['avg_mppi_apply_ms']))
            steer.append(float(s['avg_steering_apply_ms']))
    if not mppi or not steer:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = [np.mean(mppi), np.mean(steer)]
    labels = ['MPPI', 'Steering (Ours)']
    colors = ['#5b7bb2', '#2e8b57']
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel('Mean apply latency (ms)')
    ax.set_title('Controller Step Latency (OOD zero-shot completed runs)')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f'{v:.2f} ms', ha='center', fontsize=9)
    speed = vals[0] / max(vals[1], 1e-9)
    ax.text(0.5, max(vals) * 0.82, f'Speedup = {speed:.1f}x', ha='center', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))
    ax.grid(axis='y', alpha=0.25)
    save(fig, '09_latency_speedup_ood.png')


def fig_latent_manifold_synth():
    # Synthetic placeholder because raw per-step latent trajectories are not yet exported.
    rng = np.random.default_rng(7)
    safe = np.vstack([
        rng.normal([-2.0, -0.5], [0.7, 0.5], (350, 2)),
        rng.normal([0.0, -1.2], [0.6, 0.4], (250, 2)),
    ])
    fail = np.vstack([
        rng.normal([1.6, 0.6], [0.45, 0.35], (220, 2)),
        rng.normal([2.5, -0.2], [0.4, 0.3], (120, 2)),
    ])
    traj = np.array([[-2.1, -1.0], [-1.4, -0.8], [-0.8, -0.4], [0.0, -0.1], [0.9, 0.3], [1.4, 0.6]])
    corr = np.array([[-0.7, -0.5], [-0.6, -0.4], [-0.6, -0.4], [-0.5, -0.4], [-0.5, -0.3], [-0.5, -0.3]])

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.scatter(safe[:, 0], safe[:, 1], s=8, alpha=0.35, label='Safe latent region', color='#3b6fb6')
    ax.scatter(fail[:, 0], fail[:, 1], s=10, alpha=0.45, label='Failure latent region', color='#d9534f')
    ax.plot(traj[:, 0], traj[:, 1], '-k', lw=1.8, label='Rollout trajectory (illustrative)')
    for p, c in zip(traj[2:], corr[2:]):
        ax.arrow(p[0], p[1], c[0], c[1], width=0.01, head_width=0.12, color='#2e8b57', alpha=0.9)
    ax.text(0.01, 0.98, 'Synthetic illustration (placeholder)\nReplace with true PCA/t-SNE after latent export',
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))
    ax.set_title('Latent Safety Manifold Visualization (Illustrative Placeholder)')
    ax.set_xlabel('Projection dim 1')
    ax.set_ylabel('Projection dim 2')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, '10_latent_manifold_placeholder.png')


def fig_fail_calibration_synth():
    rng = np.random.default_rng(12)
    t = np.linspace(0, 1, 240)
    p = 0.15 + 0.8 / (1 + np.exp(-14 * (t - 0.62))) + rng.normal(0, 0.03, len(t))
    p = np.clip(p, 0.01, 0.99)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(np.arange(len(t)), p, color='#2f4f7f', lw=2, label='Predicted fail probability')
    ax.axhline(0.85, color='#d9534f', ls='--', lw=1.6, label='Latent-stop threshold')
    idx = np.argmax(p >= 0.85)
    if p[idx] >= 0.85:
        ax.axvline(idx, color='#d9534f', ls=':', lw=1.6)
        ax.text(idx + 2, 0.08, f't={idx}', color='#a33', fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('$p_{fail}$')
    ax.set_title('Failure Probability Timeline and Deployment Threshold (Illustrative Placeholder)')
    ax.text(0.01, 0.95, 'Synthetic placeholder: replace with real risk timeline plots', transform=ax.transAxes,
            ha='left', va='top', fontsize=8, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')
    save(fig, '11_fail_probability_placeholder.png')


def main():
    fig_architecture()
    fig_zero_shot_seed_deltas()
    fig_task_mode_heatmap()
    fig_latency_speed()
    fig_latent_manifold_synth()
    fig_fail_calibration_synth()


if __name__ == '__main__':
    main()
