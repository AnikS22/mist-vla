#!/usr/bin/env python3
"""Comprehensive data analysis for VLA failure prediction research.

Generates charts, metrics, and tables covering:
1. Dataset overview & per-task statistics
2. Hidden state analysis (PCA, t-SNE, separability)
3. Action distribution analysis (success vs failure)
4. Temporal dynamics (how hidden states evolve within episodes)
5. Cross-task generalization potential
6. Feature importance / discriminative dimensions
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path("analysis/comprehensive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]
TASK_NAMES = [f"Task {i}" for i in range(10)]

# ── Load Data ──────────────────────────────────────────────────────────────
print("Loading data...")
with open("data/combined/success_rollouts.pkl", "rb") as f:
    succ_rollouts = pickle.load(f)
with open("data/combined/failure_rollouts.pkl", "rb") as f:
    fail_rollouts = pickle.load(f)

all_rollouts = succ_rollouts + fail_rollouts
print(f"  {len(succ_rollouts)} success + {len(fail_rollouts)} failure = {len(all_rollouts)} rollouts")

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. DATASET OVERVIEW")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold")

# 1a. Per-task rollout counts
task_succ = Counter(r["task_id"] for r in succ_rollouts)
task_fail = Counter(r["task_id"] for r in fail_rollouts)
tasks = range(10)
x = np.arange(10)
width = 0.35

ax = axes[0, 0]
bars1 = ax.bar(x - width/2, [task_succ.get(t, 0) for t in tasks], width, label="Success", color="#2ecc71", alpha=0.85)
bars2 = ax.bar(x + width/2, [task_fail.get(t, 0) for t in tasks], width, label="Failure", color="#e74c3c", alpha=0.85)
ax.set_xlabel("Task ID")
ax.set_ylabel("Rollout Count")
ax.set_title("Rollouts per Task")
ax.set_xticks(x)
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

# 1b. Success rate per task
ax = axes[0, 1]
rates = []
for t in tasks:
    s = task_succ.get(t, 0)
    f = task_fail.get(t, 0)
    rates.append(s / (s + f) * 100 if (s + f) > 0 else 0)
colors = ["#2ecc71" if r >= 50 else "#e74c3c" for r in rates]
bars = ax.bar(x, rates, color=colors, alpha=0.85)
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Task ID")
ax.set_ylabel("Success Rate (%)")
ax.set_title("Success Rate per Task")
ax.set_xticks(x)
ax.set_ylim(0, 100)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)

# 1c. Episode length distribution
ax = axes[1, 0]
succ_lengths = [len(r["features"]) for r in succ_rollouts]
fail_lengths = [len(r["features"]) for r in fail_rollouts]
ax.hist(succ_lengths, bins=30, alpha=0.6, label=f"Success (μ={np.mean(succ_lengths):.0f})", color="#2ecc71", edgecolor="white")
ax.hist(fail_lengths, bins=30, alpha=0.6, label=f"Failure (μ={np.mean(fail_lengths):.0f})", color="#e74c3c", edgecolor="white")
ax.set_xlabel("Episode Length (steps)")
ax.set_ylabel("Count")
ax.set_title("Episode Length Distribution")
ax.legend()

# 1d. Steps per task
ax = axes[1, 1]
task_steps_s = defaultdict(int)
task_steps_f = defaultdict(int)
for r in succ_rollouts:
    task_steps_s[r["task_id"]] += len(r["features"])
for r in fail_rollouts:
    task_steps_f[r["task_id"]] += len(r["features"])
ax.bar(x - width/2, [task_steps_s.get(t, 0) for t in tasks], width, label="Success steps", color="#2ecc71", alpha=0.85)
ax.bar(x + width/2, [task_steps_f.get(t, 0) for t in tasks], width, label="Failure steps", color="#e74c3c", alpha=0.85)
ax.set_xlabel("Task ID")
ax.set_ylabel("Total Steps")
ax.set_title("Total Steps per Task")
ax.set_xticks(x)
ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "01_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 01_dataset_overview.png")

# Print summary table
print("\n  ┌─────────┬─────────┬─────────┬──────────┬────────────┬────────────┐")
print("  │  Task   │ Success │ Failure │ Rate (%) │ Succ Steps │ Fail Steps │")
print("  ├─────────┼─────────┼─────────┼──────────┼────────────┼────────────┤")
for t in tasks:
    s = task_succ.get(t, 0)
    f = task_fail.get(t, 0)
    rate = s / (s + f) * 100 if (s + f) > 0 else 0
    ss = task_steps_s.get(t, 0)
    fs = task_steps_f.get(t, 0)
    print(f"  │ Task {t:>2} │  {s:>5}  │  {f:>5}  │  {rate:>5.1f}   │   {ss:>7}  │   {fs:>7}  │")
print("  ├─────────┼─────────┼─────────┼──────────┼────────────┼────────────┤")
print(f"  │  TOTAL  │  {len(succ_rollouts):>5}  │  {len(fail_rollouts):>5}  │  {len(succ_rollouts)/(len(all_rollouts))*100:>5.1f}   │   {sum(task_steps_s.values()):>7}  │   {sum(task_steps_f.values()):>7}  │")
print("  └─────────┴─────────┴─────────┴──────────┴────────────┴────────────┘")

# ═══════════════════════════════════════════════════════════════════════════
# 2. HIDDEN STATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. HIDDEN STATE ANALYSIS")
print("="*70)

# Gather features: sample 1 random step per rollout for global analysis,
# plus first/last steps for temporal analysis
global_feats, global_labels, global_tasks = [], [], []
first_feats, first_labels = [], []
last_feats, last_labels = [], []

for r in all_rollouts:
    feats = np.array(r["features"])
    label = 0 if r["success"] else 1
    tid = r["task_id"]
    
    # Random step for global
    idx = np.random.randint(len(feats))
    global_feats.append(feats[idx])
    global_labels.append(label)
    global_tasks.append(tid)
    
    # First and last steps
    first_feats.append(feats[0])
    first_labels.append(label)
    last_feats.append(feats[-1])
    last_labels.append(label)

global_feats = np.array(global_feats)
global_labels = np.array(global_labels)
global_tasks = np.array(global_tasks)
first_feats = np.array(first_feats)
last_feats = np.array(last_feats)
first_labels = np.array(first_labels)
last_labels = np.array(last_labels)

# Also gather ALL steps for large-scale analysis (subsample for viz)
all_step_feats, all_step_labels, all_step_tasks, all_step_progress = [], [], [], []
for r in all_rollouts:
    feats = np.array(r["features"])
    label = 0 if r["success"] else 1
    tid = r["task_id"]
    T = len(feats)
    for t in range(T):
        all_step_feats.append(feats[t])
        all_step_labels.append(label)
        all_step_tasks.append(tid)
        all_step_progress.append(t / max(T - 1, 1))  # 0=start, 1=end

all_step_feats = np.array(all_step_feats)
all_step_labels = np.array(all_step_labels)
all_step_tasks = np.array(all_step_tasks)
all_step_progress = np.array(all_step_progress)
print(f"  Total step-level samples: {len(all_step_feats)}")

# 2a. PCA visualization (one point per rollout)
print("  Computing PCA...")
scaler = StandardScaler()
feats_scaled = scaler.fit_transform(global_feats)
pca = PCA(n_components=10)
feats_pca = pca.fit_transform(feats_scaled)

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("Hidden State PCA (1 sample per rollout)", fontsize=14, fontweight="bold")

# Color by success/failure
ax = axes[0]
succ_mask = global_labels == 0
fail_mask = global_labels == 1
ax.scatter(feats_pca[succ_mask, 0], feats_pca[succ_mask, 1], c="#2ecc71", alpha=0.7, s=50, label="Success", edgecolors="white", linewidth=0.5)
ax.scatter(feats_pca[fail_mask, 0], feats_pca[fail_mask, 1], c="#e74c3c", alpha=0.7, s=50, label="Failure", edgecolors="white", linewidth=0.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Colored by Outcome")
ax.legend()

# Color by task
ax = axes[1]
cmap = plt.cm.tab10
for t in range(10):
    mask = global_tasks == t
    if mask.any():
        ax.scatter(feats_pca[mask, 0], feats_pca[mask, 1], c=[cmap(t)], alpha=0.7, s=50, label=f"Task {t}", edgecolors="white", linewidth=0.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Colored by Task")
ax.legend(fontsize=7, ncol=2)

# Variance explained
ax = axes[2]
cumvar = np.cumsum(pca.explained_variance_ratio_)
ax.bar(range(1, 11), pca.explained_variance_ratio_ * 100, color="#3498db", alpha=0.7, label="Individual")
ax.plot(range(1, 11), cumvar * 100, "o-", color="#e74c3c", label="Cumulative")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained (%)")
ax.set_title("PCA Variance Explained")
ax.legend()
ax.set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig(OUT_DIR / "02_pca_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 02_pca_analysis.png")

# 2b. t-SNE visualization (subsample for speed)
print("  Computing t-SNE (may take a minute)...")
n_tsne = min(2000, len(all_step_feats))
tsne_idx = np.random.choice(len(all_step_feats), n_tsne, replace=False)
tsne_feats = StandardScaler().fit_transform(all_step_feats[tsne_idx])
tsne = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate="auto", init="pca")
tsne_emb = tsne.fit_transform(tsne_feats)

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle(f"t-SNE of Hidden States ({n_tsne} samples)", fontsize=14, fontweight="bold")

# By outcome
ax = axes[0]
s_mask = all_step_labels[tsne_idx] == 0
f_mask = all_step_labels[tsne_idx] == 1
ax.scatter(tsne_emb[s_mask, 0], tsne_emb[s_mask, 1], c="#2ecc71", alpha=0.4, s=10, label="Success")
ax.scatter(tsne_emb[f_mask, 0], tsne_emb[f_mask, 1], c="#e74c3c", alpha=0.4, s=10, label="Failure")
ax.set_title("Colored by Outcome")
ax.legend()

# By task
ax = axes[1]
for t in range(10):
    mask = all_step_tasks[tsne_idx] == t
    if mask.any():
        ax.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1], c=[cmap(t)], alpha=0.4, s=10, label=f"T{t}")
ax.set_title("Colored by Task")
ax.legend(fontsize=7, ncol=2)

# By episode progress
ax = axes[2]
sc = ax.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=all_step_progress[tsne_idx], cmap="viridis", alpha=0.4, s=10)
plt.colorbar(sc, ax=ax, label="Episode Progress (0=start, 1=end)")
ax.set_title("Colored by Episode Progress")

plt.tight_layout()
plt.savefig(OUT_DIR / "03_tsne_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 03_tsne_analysis.png")

# 2c. Feature statistics
print("\n  Hidden state statistics:")
succ_step_feats = all_step_feats[all_step_labels == 0]
fail_step_feats = all_step_feats[all_step_labels == 1]
print(f"    Success: mean={succ_step_feats.mean():.4f}, std={succ_step_feats.std():.4f}, shape={succ_step_feats.shape}")
print(f"    Failure: mean={fail_step_feats.mean():.4f}, std={fail_step_feats.std():.4f}, shape={fail_step_feats.shape}")

# Per-dimension mean difference (top discriminative hidden dims)
mean_diff = np.abs(succ_step_feats.mean(axis=0) - fail_step_feats.mean(axis=0))
top_dims = np.argsort(mean_diff)[-20:][::-1]
print(f"\n  Top 20 most discriminative hidden dimensions (by |mean_diff|):")
print(f"    Dims: {top_dims.tolist()}")
print(f"    Diffs: {mean_diff[top_dims].tolist()[:5]}...")

# ═══════════════════════════════════════════════════════════════════════════
# 3. SEPARABILITY ANALYSIS (Can we distinguish success from failure?)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. SEPARABILITY ANALYSIS")
print("="*70)

# 3a. Silhouette score
print("  Computing silhouette score...")
sil_idx = np.random.choice(len(all_step_feats), min(5000, len(all_step_feats)), replace=False)
sil_feats = StandardScaler().fit_transform(all_step_feats[sil_idx])
# Use PCA to reduce dimensionality for silhouette
sil_pca = PCA(n_components=50).fit_transform(sil_feats)
sil_score = silhouette_score(sil_pca, all_step_labels[sil_idx], sample_size=min(2000, len(sil_idx)))
print(f"    Silhouette score (success vs failure): {sil_score:.4f}")
print(f"    (0 = overlapping, 1 = perfectly separated)")

# 3b. Linear separability (logistic regression)
print("  Testing linear separability with logistic regression...")
# Use PCA features for speed
lr_feats = PCA(n_components=50).fit_transform(StandardScaler().fit_transform(all_step_feats))
lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
cv_scores = cross_val_score(lr, lr_feats, all_step_labels, cv=5, scoring="roc_auc")
print(f"    5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"    Per-fold: {[f'{s:.4f}' for s in cv_scores]}")

# 3c. Per-task separability
print("\n  Per-task linear separability (logistic regression AUC):")
print(f"    {'Task':>6} {'AUC':>8} {'N_succ':>8} {'N_fail':>8}")
print(f"    {'-'*36}")
task_aucs = {}
for t in range(10):
    t_mask = all_step_tasks == t
    t_feats = all_step_feats[t_mask]
    t_labels = all_step_labels[t_mask]
    n_s = (t_labels == 0).sum()
    n_f = (t_labels == 1).sum()
    if n_s < 10 or n_f < 10:
        print(f"    Task {t:>2}     N/A   {n_s:>6}   {n_f:>6}  (too few)")
        task_aucs[t] = None
        continue
    t_pca = PCA(n_components=min(50, t_feats.shape[0] // 3)).fit_transform(StandardScaler().fit_transform(t_feats))
    lr_t = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    try:
        scores = cross_val_score(lr_t, t_pca, t_labels, cv=min(5, min(n_s, n_f)), scoring="roc_auc")
        auc = scores.mean()
    except:
        auc = 0.5
    task_aucs[t] = auc
    print(f"    Task {t:>2}   {auc:.4f}   {n_s:>6}   {n_f:>6}")

# 3d. Cross-task generalization (leave-one-task-out)
print("\n  Cross-task generalization (leave-one-task-out):")
print(f"    {'Held-out':>10} {'AUC':>8} {'Train Tasks':>15}")
print(f"    {'-'*38}")
for held_out in range(10):
    train_mask = all_step_tasks != held_out
    test_mask = all_step_tasks == held_out
    
    n_test_s = ((all_step_labels[test_mask]) == 0).sum()
    n_test_f = ((all_step_labels[test_mask]) == 1).sum()
    
    if n_test_s < 5 or n_test_f < 5:
        print(f"    Task {held_out:>4}     N/A   (too few test samples)")
        continue
    
    train_feats = all_step_feats[train_mask]
    train_labels = all_step_labels[train_mask]
    test_feats = all_step_feats[test_mask]
    test_labels = all_step_labels[test_mask]
    
    sc = StandardScaler().fit(train_feats)
    pca_model = PCA(n_components=50).fit(sc.transform(train_feats))
    
    train_pca = pca_model.transform(sc.transform(train_feats))
    test_pca = pca_model.transform(sc.transform(test_feats))
    
    lr_lot = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    lr_lot.fit(train_pca, train_labels)
    probs = lr_lot.predict_proba(test_pca)[:, 1]
    try:
        auc = roc_auc_score(test_labels, probs)
    except:
        auc = 0.5
    print(f"    Task {held_out:>4}   {auc:.4f}   {','.join(str(i) for i in range(10) if i != held_out)}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. ACTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. ACTION ANALYSIS")
print("="*70)

succ_actions = np.concatenate([np.array(r["actions"]) for r in succ_rollouts], axis=0)
fail_actions = np.concatenate([np.array(r["actions"]) for r in fail_rollouts], axis=0)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Action Distributions: Success vs Failure", fontsize=14, fontweight="bold")

for i in range(7):
    ax = axes[i // 4, i % 4]
    ax.hist(succ_actions[:, i], bins=50, alpha=0.6, label="Success", color="#2ecc71", density=True, edgecolor="white")
    ax.hist(fail_actions[:, i], bins=50, alpha=0.6, label="Failure", color="#e74c3c", density=True, edgecolor="white")
    
    # Statistical test
    t_stat, p_val = stats.ttest_ind(succ_actions[:, i], fail_actions[:, i])
    ks_stat, ks_p = stats.ks_2samp(succ_actions[:, i], fail_actions[:, i])
    
    ax.set_title(f"{DIM_NAMES[i]}\nKS={ks_stat:.3f} (p={ks_p:.1e})")
    ax.set_xlabel("Action Value")
    ax.legend(fontsize=8)

# Summary stats in last subplot
ax = axes[1, 3]
ax.axis("off")
text = "Action Statistics\n" + "-"*40 + "\n"
text += f"{'Dim':>8} {'S_mean':>8} {'F_mean':>8} {'KS_stat':>8}\n"
for i in range(7):
    ks_stat, _ = stats.ks_2samp(succ_actions[:, i], fail_actions[:, i])
    text += f"{DIM_NAMES[i]:>8} {succ_actions[:, i].mean():>8.4f} {fail_actions[:, i].mean():>8.4f} {ks_stat:>8.3f}\n"
ax.text(0.1, 0.5, text, fontfamily="monospace", fontsize=10, va="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig(OUT_DIR / "04_action_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 04_action_distributions.png")

# Print action table
print("\n  Action dimension statistics (Success vs Failure):")
print(f"    {'Dim':>8} {'S_mean':>9} {'F_mean':>9} {'S_std':>8} {'F_std':>8} {'KS_stat':>8} {'p-value':>10} {'Signal':>8}")
print(f"    {'-'*72}")
for i in range(7):
    s_mean = succ_actions[:, i].mean()
    f_mean = fail_actions[:, i].mean()
    s_std = succ_actions[:, i].std()
    f_std = fail_actions[:, i].std()
    ks_stat, ks_p = stats.ks_2samp(succ_actions[:, i], fail_actions[:, i])
    signal = "STRONG" if ks_stat > 0.15 else ("WEAK" if ks_stat > 0.08 else "NONE")
    print(f"    {DIM_NAMES[i]:>8} {s_mean:>9.5f} {f_mean:>9.5f} {s_std:>8.5f} {f_std:>8.5f} {ks_stat:>8.3f} {ks_p:>10.2e} {signal:>8}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. TEMPORAL DYNAMICS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(21, 12))
fig.suptitle("Temporal Dynamics in Hidden States", fontsize=14, fontweight="bold")

# 5a. How hidden state norm evolves over episode
ax = axes[0, 0]
for label, rollouts, color, lbl in [(0, succ_rollouts, "#2ecc71", "Success"), (1, fail_rollouts, "#e74c3c", "Failure")]:
    norms_by_progress = defaultdict(list)
    for r in rollouts:
        feats = np.array(r["features"])
        T = len(feats)
        for t in range(T):
            prog = int(t / max(T-1, 1) * 20) / 20  # Bin to 5% increments
            norms_by_progress[prog].append(np.linalg.norm(feats[t]))
    
    progs = sorted(norms_by_progress.keys())
    means = [np.mean(norms_by_progress[p]) for p in progs]
    stds = [np.std(norms_by_progress[p]) for p in progs]
    ax.plot([p*100 for p in progs], means, color=color, label=lbl, linewidth=2)
    ax.fill_between([p*100 for p in progs], 
                     [m-s for m, s in zip(means, stds)],
                     [m+s for m, s in zip(means, stds)],
                     alpha=0.2, color=color)
ax.set_xlabel("Episode Progress (%)")
ax.set_ylabel("Hidden State L2 Norm")
ax.set_title("Hidden State Norm over Episode")
ax.legend()

# 5b. PCA trajectory (first 2 PCs over time, sample rollouts)
ax = axes[0, 1]
# Fit PCA on all data
all_feats_for_pca = StandardScaler().fit_transform(all_step_feats[:5000])
traj_pca = PCA(n_components=2).fit(all_feats_for_pca)

for r in succ_rollouts[:3]:
    feats = np.array(r["features"])
    projected = traj_pca.transform(StandardScaler().fit_transform(feats) if len(feats) > 1 else feats.reshape(1, -1))
    ax.plot(projected[:, 0], projected[:, 1], color="#2ecc71", alpha=0.5, linewidth=1)
    ax.scatter(projected[0, 0], projected[0, 1], color="#2ecc71", marker="o", s=30, zorder=5)
    ax.scatter(projected[-1, 0], projected[-1, 1], color="#2ecc71", marker="x", s=50, zorder=5)

for r in fail_rollouts[:3]:
    feats = np.array(r["features"])
    projected = traj_pca.transform(StandardScaler().fit_transform(feats) if len(feats) > 1 else feats.reshape(1, -1))
    ax.plot(projected[:, 0], projected[:, 1], color="#e74c3c", alpha=0.5, linewidth=1)
    ax.scatter(projected[0, 0], projected[0, 1], color="#e74c3c", marker="o", s=30, zorder=5)
    ax.scatter(projected[-1, 0], projected[-1, 1], color="#e74c3c", marker="x", s=50, zorder=5)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA Trajectories (o=start, x=end)")

# 5c. Action magnitude over time
ax = axes[0, 2]
for label, rollouts, color, lbl in [(0, succ_rollouts, "#2ecc71", "Success"), (1, fail_rollouts, "#e74c3c", "Failure")]:
    mag_by_progress = defaultdict(list)
    for r in rollouts:
        acts = np.array(r["actions"])
        T = len(acts)
        for t in range(T):
            prog = int(t / max(T-1, 1) * 20) / 20
            mag_by_progress[prog].append(np.linalg.norm(acts[t]))
    
    progs = sorted(mag_by_progress.keys())
    means = [np.mean(mag_by_progress[p]) for p in progs]
    ax.plot([p*100 for p in progs], means, color=color, label=lbl, linewidth=2)
ax.set_xlabel("Episode Progress (%)")
ax.set_ylabel("Action Magnitude (L2)")
ax.set_title("Action Magnitude over Episode")
ax.legend()

# 5d-f. Per-dimension action over time for key dims
for idx, (dim_i, dim_name) in enumerate([(5, "Yaw"), (6, "Gripper"), (4, "Pitch")]):
    ax = axes[1, idx]
    for label, rollouts, color, lbl in [(0, succ_rollouts, "#2ecc71", "Success"), (1, fail_rollouts, "#e74c3c", "Failure")]:
        vals_by_progress = defaultdict(list)
        for r in rollouts:
            acts = np.array(r["actions"])
            T = len(acts)
            for t in range(T):
                prog = int(t / max(T-1, 1) * 20) / 20
                vals_by_progress[prog].append(acts[t, dim_i])
        
        progs = sorted(vals_by_progress.keys())
        means = [np.mean(vals_by_progress[p]) for p in progs]
        stds = [np.std(vals_by_progress[p]) for p in progs]
        ax.plot([p*100 for p in progs], means, color=color, label=lbl, linewidth=2)
        ax.fill_between([p*100 for p in progs],
                         [m-s for m, s in zip(means, stds)],
                         [m+s for m, s in zip(means, stds)],
                         alpha=0.15, color=color)
    ax.set_xlabel("Episode Progress (%)")
    ax.set_ylabel(f"{dim_name} Action Value")
    ax.set_title(f"{dim_name} Action over Episode")
    ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "05_temporal_dynamics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 05_temporal_dynamics.png")

# ═══════════════════════════════════════════════════════════════════════════
# 6. FIRST vs LAST STEP ANALYSIS (early vs late detection)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("6. EARLY vs LATE FAILURE DETECTION")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Can We Detect Failure Early, Mid, or Late?", fontsize=14, fontweight="bold")

for idx, (frac_start, frac_end, title) in enumerate([
    (0.0, 0.2, "Early (0-20%)"),
    (0.3, 0.6, "Mid (30-60%)"),
    (0.8, 1.0, "Late (80-100%)")
]):
    feats_slice, labels_slice = [], []
    for r in all_rollouts:
        f = np.array(r["features"])
        T = len(f)
        label = 0 if r["success"] else 1
        t_start = int(frac_start * T)
        t_end = int(frac_end * T)
        if t_end <= t_start:
            t_end = t_start + 1
        for t in range(t_start, min(t_end, T)):
            feats_slice.append(f[t])
            labels_slice.append(label)
    
    feats_slice = np.array(feats_slice)
    labels_slice = np.array(labels_slice)
    
    pca_slice = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(feats_slice))
    
    ax = axes[idx]
    s_mask = labels_slice == 0
    f_mask = labels_slice == 1
    ax.scatter(pca_slice[s_mask, 0], pca_slice[s_mask, 1], c="#2ecc71", alpha=0.3, s=5, label="Success")
    ax.scatter(pca_slice[f_mask, 0], pca_slice[f_mask, 1], c="#e74c3c", alpha=0.3, s=5, label="Failure")
    
    # Compute AUC with logistic regression
    lr_slice = LogisticRegression(max_iter=300, random_state=42, class_weight="balanced")
    pca50 = PCA(n_components=min(50, feats_slice.shape[0]//3)).fit_transform(
        StandardScaler().fit_transform(feats_slice))
    try:
        cv_auc = cross_val_score(lr_slice, pca50, labels_slice, cv=3, scoring="roc_auc").mean()
    except:
        cv_auc = 0.5
    
    ax.set_title(f"{title}\nLogReg AUC = {cv_auc:.4f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "06_early_vs_late_detection.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 06_early_vs_late_detection.png")

# Print early/mid/late AUCs
print("\n  Detection AUC at different episode stages:")
for frac_start, frac_end, title in [
    (0.0, 0.1, "Very Early (0-10%)"),
    (0.0, 0.2, "Early (0-20%)"),
    (0.2, 0.5, "Mid (20-50%)"),
    (0.5, 0.8, "Late-Mid (50-80%)"),
    (0.8, 1.0, "Late (80-100%)")
]:
    feats_slice, labels_slice = [], []
    for r in all_rollouts:
        f = np.array(r["features"])
        T = len(f)
        label = 0 if r["success"] else 1
        t_start = int(frac_start * T)
        t_end = max(int(frac_end * T), t_start + 1)
        for t in range(t_start, min(t_end, T)):
            feats_slice.append(f[t])
            labels_slice.append(label)
    feats_slice = np.array(feats_slice)
    labels_slice = np.array(labels_slice)
    pca50 = PCA(n_components=min(50, feats_slice.shape[0]//3)).fit_transform(
        StandardScaler().fit_transform(feats_slice))
    lr_s = LogisticRegression(max_iter=300, random_state=42, class_weight="balanced")
    try:
        cv_auc = cross_val_score(lr_s, pca50, labels_slice, cv=3, scoring="roc_auc").mean()
    except:
        cv_auc = 0.5
    marker = "🟢" if cv_auc > 0.8 else ("🟡" if cv_auc > 0.65 else "🔴")
    print(f"    {marker} {title:>25}: AUC = {cv_auc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 7. HEATMAP: Cross-task transfer matrix
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("7. CROSS-TASK TRANSFER MATRIX")
print("="*70)

transfer_matrix = np.zeros((10, 10))
print("  Computing train-on-X / test-on-Y transfer matrix...")

for train_task in range(10):
    train_mask = all_step_tasks == train_task
    train_f = all_step_feats[train_mask]
    train_l = all_step_labels[train_mask]
    
    if len(np.unique(train_l)) < 2:
        transfer_matrix[train_task, :] = 0.5
        continue
    
    sc = StandardScaler().fit(train_f)
    pca_m = PCA(n_components=min(50, train_f.shape[0]//3)).fit(sc.transform(train_f))
    
    lr_tm = LogisticRegression(max_iter=300, random_state=42, class_weight="balanced")
    lr_tm.fit(pca_m.transform(sc.transform(train_f)), train_l)
    
    for test_task in range(10):
        test_mask = all_step_tasks == test_task
        test_f = all_step_feats[test_mask]
        test_l = all_step_labels[test_mask]
        
        if len(np.unique(test_l)) < 2:
            transfer_matrix[train_task, test_task] = 0.5
            continue
        
        probs = lr_tm.predict_proba(pca_m.transform(sc.transform(test_f)))[:, 1]
        try:
            transfer_matrix[train_task, test_task] = roc_auc_score(test_l, probs)
        except:
            transfer_matrix[train_task, test_task] = 0.5

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(transfer_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
ax.set_xlabel("Test Task")
ax.set_ylabel("Train Task")
ax.set_title("Cross-Task Transfer Matrix (AUC)\nTrain on row task → Test on column task", fontsize=13)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        color = "white" if transfer_matrix[i, j] < 0.6 else "black"
        ax.text(j, i, f"{transfer_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)
plt.colorbar(im, ax=ax, label="AUC")
plt.tight_layout()
plt.savefig(OUT_DIR / "07_transfer_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 07_transfer_matrix.png")

# Print summary
mean_diagonal = np.mean(np.diag(transfer_matrix))
mean_off_diag = np.mean(transfer_matrix[~np.eye(10, dtype=bool)])
print(f"\n  Transfer matrix summary:")
print(f"    Mean diagonal (within-task):     {mean_diagonal:.4f}")
print(f"    Mean off-diagonal (cross-task):  {mean_off_diag:.4f}")
print(f"    Gap:                             {mean_diagonal - mean_off_diag:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 8. COLLISION & ROBOT STATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("8. COLLISION & EPISODE METADATA")
print("="*70)

n_collisions = sum(1 for r in all_rollouts if r.get("collision_occurred"))
n_perturbed = sum(1 for r in all_rollouts if r.get("perturbed"))
print(f"  Rollouts with collisions: {n_collisions}/{len(all_rollouts)}")
print(f"  Rollouts with perturbations: {n_perturbed}/{len(all_rollouts)}")

if n_perturbed > 0:
    pert_succ = sum(1 for r in all_rollouts if r.get("perturbed") and r.get("success"))
    pert_fail = sum(1 for r in all_rollouts if r.get("perturbed") and not r.get("success"))
    no_pert_succ = sum(1 for r in all_rollouts if not r.get("perturbed") and r.get("success"))
    no_pert_fail = sum(1 for r in all_rollouts if not r.get("perturbed") and not r.get("success"))
    print(f"  Perturbed:     {pert_succ} success, {pert_fail} failure ({pert_fail/(pert_succ+pert_fail)*100:.0f}% fail)")
    print(f"  Not perturbed: {no_pert_succ} success, {no_pert_fail} failure ({no_pert_fail/(no_pert_succ+no_pert_fail)*100:.0f}% fail)")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
  Dataset:       {len(all_rollouts)} rollouts ({len(succ_rollouts)}S + {len(fail_rollouts)}F), {len(all_step_feats)} steps
  Tasks:         10 LIBERO-Spatial tasks
  Features:      4096-dim VLA last-layer hidden states (mean-pooled)
  
  KEY FINDINGS:
  1. Silhouette score:          {sil_score:.4f} (moderate cluster separation)
  2. Linear separability (AUC): {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (5-fold CV)
  3. Within-task AUC:           {mean_diagonal:.4f} (avg diagonal)
  4. Cross-task AUC:            {mean_off_diag:.4f} (avg off-diagonal)
  
  VERDICT:
  {"✅ STRONG" if cv_scores.mean() > 0.85 else "🟡 MODERATE" if cv_scores.mean() > 0.7 else "❌ WEAK"} failure signal in VLA hidden states
  {"✅ GENERALIZES" if mean_off_diag > 0.7 else "🟡 PARTIAL" if mean_off_diag > 0.6 else "❌ TASK-SPECIFIC"} across tasks
  
  All charts saved to: {OUT_DIR}/
""")

print("Charts generated:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  📊 {f}")
