#!/usr/bin/env python3
"""Generate steering-focused visuals for paper integration."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    out = FIG / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


def fig_rollout_storyboard():
    pkl = ROOT / "research_data/rollouts/merged_all/success_rollouts.pkl"
    if not pkl.exists():
        return
    with pkl.open("rb") as f:
        data = pickle.load(f)
    if not data:
        return

    # choose a reasonably long successful rollout
    ep = None
    for r in data:
        if r.get("success", False) and len(r.get("actions", [])) >= 80:
            ep = r
            break
    if ep is None:
        ep = data[0]

    acts = np.asarray(ep.get("actions", []), dtype=np.float32)
    feats = np.asarray(ep.get("features", []), dtype=np.float32)
    rews = np.asarray(ep.get("rewards", []), dtype=np.float32)
    if acts.ndim != 2 or feats.ndim != 2 or acts.shape[0] == 0:
        return

    T = min(len(acts), len(feats), len(rews) if len(rews) else len(acts))
    acts = acts[:T]
    feats = feats[:T]
    if len(rews) >= T:
        rews = rews[:T]
    else:
        rews = np.zeros((T,), dtype=np.float32)

    step = np.arange(T)
    action_norm = np.linalg.norm(acts[:, :3], axis=1)
    feat_norm = np.linalg.norm(feats, axis=1)
    reward_cum = np.cumsum(rews)
    snap_idx = np.linspace(0, T - 1, num=4, dtype=int)

    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.3], hspace=0.35, wspace=0.25)

    # Top row: "frame cards" from timestep annotations (data-backed, image-free fallback)
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        t = int(snap_idx[i])
        ax.axis("off")
        txt = (
            f"Frame {i+1}\n"
            f"Step {t}\n"
            f"||a_xyz||={action_norm[t]:.3f}\n"
            f"||h||={feat_norm[t]:.1f}\n"
            f"cumR={reward_cum[t]:.2f}"
        )
        ax.text(
            0.5,
            0.5,
            txt,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(facecolor="#f6f7fb", edgecolor="#808080", boxstyle="round,pad=0.35"),
        )
        ax.set_title(f"Timestep snapshot {i+1}", fontsize=9)

    # Bottom row: metric trajectories
    axm = fig.add_subplot(gs[1, :])
    axm.plot(step, action_norm, label=r"$\|a_{xyz}\|$", color="#1f77b4", linewidth=1.8)
    axm.plot(step, feat_norm / max(feat_norm.mean(), 1e-6), label=r"$\|h\|$ (normalized)", color="#2ca02c", linewidth=1.6)
    axm.plot(step, reward_cum, label="Cumulative reward", color="#d62728", linewidth=1.6)
    for t in snap_idx:
        axm.axvline(int(t), color="gray", alpha=0.3, linestyle="--", linewidth=1.0)
    axm.set_xlabel("Step")
    axm.set_ylabel("Metric value")
    axm.set_title("Rollout metric timeline (synced merged_all rollout)")
    axm.legend(ncol=3, fontsize=8, loc="upper right")
    axm.grid(alpha=0.25)

    save(fig, "19_rollout_frame_metric_storyboard.png")


def fig_steering_effect_dashboard():
    p = ROOT / "hpc_mirror/results/closed_loop_alpha1.0/episode_details.json"
    if not p.exists():
        p = ROOT / "hpc_mirror/results/closed_loop/episode_details.json"
    if not p.exists():
        return

    d = json.loads(p.read_text())
    vanilla = d.get("vanilla", [])
    steering = d.get("steering", [])
    if not vanilla or not steering:
        return

    # aggregate by task
    task_ids = sorted({int(e["task_id"]) for e in vanilla if "task_id" in e})
    v_sr, s_sr, v_dev, s_dev, s_ir = [], [], [], [], []
    for tid in task_ids:
        vv = [e for e in vanilla if int(e.get("task_id", -1)) == tid]
        ss = [e for e in steering if int(e.get("task_id", -1)) == tid]
        if not vv or not ss:
            continue
        v_sr.append(100.0 * np.mean([1.0 if e.get("success") else 0.0 for e in vv]))
        s_sr.append(100.0 * np.mean([1.0 if e.get("success") else 0.0 for e in ss]))
        v_dev.append(float(np.mean([e.get("trajectory_deviation", np.nan) for e in vv])))
        s_dev.append(float(np.mean([e.get("trajectory_deviation", np.nan) for e in ss])))
        s_ir.append(float(np.mean([e.get("intervention_rate", 0.0) for e in ss])))

    n = len(v_sr)
    if n == 0:
        return
    x = np.arange(n)
    labels = [f"T{t}" for t in task_ids[:n]]

    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.9))

    w = 0.35
    axs[0].bar(x - w / 2, v_sr, width=w, label="Vanilla", color="#777777")
    axs[0].bar(x + w / 2, s_sr, width=w, label="Steering", color="#2e8b57")
    axs[0].set_title("Success rate by task")
    axs[0].set_ylabel("Success (%)")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    axs[0].grid(axis="y", alpha=0.25)
    axs[0].legend(fontsize=8)

    axs[1].plot(x, v_dev, marker="o", color="#777777", label="Vanilla")
    axs[1].plot(x, s_dev, marker="o", color="#2e8b57", label="Steering")
    axs[1].set_title("Trajectory deviation by task")
    axs[1].set_ylabel("Deviation")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].grid(alpha=0.25)

    axs[2].bar(x, np.asarray(s_ir) * 100.0, color="#1f77b4")
    axs[2].set_title("Steering intervention rate")
    axs[2].set_ylabel("IR (%)")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels)
    axs[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Steering behavior dashboard from closed-loop episode logs", fontsize=11, y=1.02)
    save(fig, "20_steering_task_dashboard.png")


def main():
    fig_rollout_storyboard()
    fig_steering_effect_dashboard()


if __name__ == "__main__":
    main()

