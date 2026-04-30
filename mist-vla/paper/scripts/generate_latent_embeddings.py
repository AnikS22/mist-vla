#!/usr/bin/env python3
"""Generate true latent-space embedding visuals from synced rollout features."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    out = FIG / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


def _load_rollouts(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _sample_latents(rollouts, label: int, max_points: int = 3000):
    xs = []
    ys = []
    # uniform-ish subsample across episodes
    per_ep = max(1, max_points // max(len(rollouts), 1))
    for ep in rollouts:
        feats = ep.get("features", [])
        if not feats:
            continue
        idx = np.linspace(0, len(feats) - 1, num=min(per_ep, len(feats)), dtype=int)
        for i in idx:
            v = np.asarray(feats[int(i)], dtype=np.float32)
            if v.ndim == 1 and v.size > 0:
                xs.append(v)
                ys.append(label)
    if not xs:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int32)
    return X, y


def _pca_2d(X: np.ndarray):
    # center then SVD (no sklearn dependency required)
    Xc = X - X.mean(axis=0, keepdims=True)
    # compute only top-2 right singular vectors
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    W = vt[:2].T
    Z = Xc @ W
    return Z


def fig_success_failure_pca():
    suc_path = ROOT / "research_data/rollouts/merged_all/success_rollouts.pkl"
    fail_path = ROOT / "research_data/rollouts/merged_all/failure_rollouts.pkl"
    if not (suc_path.exists() and fail_path.exists()):
        print("skip: missing merged_all rollout files")
        return

    succ = _load_rollouts(suc_path)
    fail = _load_rollouts(fail_path)
    Xs, ys = _sample_latents(succ, label=1, max_points=3000)
    Xf, yf = _sample_latents(fail, label=0, max_points=3000)
    X = np.concatenate([Xs, Xf], axis=0)
    y = np.concatenate([ys, yf], axis=0)
    if X.shape[0] < 10:
        print("skip: not enough latent points")
        return

    Z = _pca_2d(X)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    m_fail = y == 0
    m_succ = y == 1
    ax.scatter(Z[m_succ, 0], Z[m_succ, 1], s=8, alpha=0.35, label="Success rollouts", color="#2e8b57")
    ax.scatter(Z[m_fail, 0], Z[m_fail, 1], s=10, alpha=0.45, label="Failure rollouts", color="#d55e5e")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent Feature Space (PCA): Success vs Failure")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "17_latent_pca_success_failure.png")


def fig_task_colored_pca():
    suc_path = ROOT / "research_data/rollouts/merged_all/success_rollouts.pkl"
    if not suc_path.exists():
        return
    succ = _load_rollouts(suc_path)
    Xs, _ = _sample_latents(succ, label=1, max_points=4000)
    if Xs.shape[0] < 10:
        return

    # build a point-to-task map with same sampling rule
    task_ids = []
    per_ep = max(1, 4000 // max(len(succ), 1))
    for ep in succ:
        feats = ep.get("features", [])
        if not feats:
            continue
        tid = str(ep.get("task_id", "unknown"))
        idx = np.linspace(0, len(feats) - 1, num=min(per_ep, len(feats)), dtype=int)
        for _ in idx:
            task_ids.append(tid)
    task_ids = np.asarray(task_ids)

    Z = _pca_2d(Xs)
    uniq = sorted(list(set(task_ids.tolist())))
    # cap legend clutter by showing top 8 task ids by count
    counts = {u: int((task_ids == u).sum()) for u in uniq}
    top = sorted(uniq, key=lambda u: counts[u], reverse=True)[:8]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for t in top:
        m = task_ids == t
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.4, label=f"Task {t}")
    m_other = ~np.isin(task_ids, top)
    if np.any(m_other):
        ax.scatter(Z[m_other, 0], Z[m_other, 1], s=6, alpha=0.2, color="gray", label="Other tasks")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent Feature Space (PCA): Success Rollouts by Task")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    save(fig, "18_latent_pca_task_colored.png")


def main():
    fig_success_failure_pca()
    fig_task_colored_pca()


if __name__ == "__main__":
    main()

