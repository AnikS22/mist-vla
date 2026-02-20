#!/usr/bin/env python3
"""
CORE RESEARCH QUESTION: Can internal VLA hidden states predict WHERE the model
will fail — which specific action dimensions will deviate and cause failure?

This is NOT about binary success/failure classification.
This is about: given hidden_state[t], can we infer:
  1. Which dimension(s) will cause the failure?
  2. In which direction (positive/negative) will the deviation occur?
  3. How far in advance can we detect the directional failure?
  4. Does this directional prediction generalize across tasks?

Approach:
  - Define per-dimension "failure labels" by comparing failure trajectory actions
    to the success action distribution for that task at that episode stage
  - Train per-dimension probing classifiers: hidden_state → will_dim_i_deviate?
  - Measure per-dimension AUC, direction accuracy, and temporal resolution
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_predict
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path("analysis/directional_failure")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]

# ── Load Data ──────────────────────────────────────────────────────────────
print("Loading combined data...")
with open("data/combined/success_rollouts.pkl", "rb") as f:
    succ_rollouts = pickle.load(f)
with open("data/combined/failure_rollouts.pkl", "rb") as f:
    fail_rollouts = pickle.load(f)

all_rollouts = succ_rollouts + fail_rollouts
print(f"  {len(succ_rollouts)}S + {len(fail_rollouts)}F = {len(all_rollouts)} rollouts")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Build per-task, per-stage SUCCESS action baselines
# For each task and each episode progress bin (0-10%, 10-20%, etc.),
# compute the mean and std of each action dimension across SUCCESS rollouts.
# This tells us "what a good action looks like" at each stage.
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1: Building success action baselines per task × stage")
print("="*70)

N_BINS = 10  # 10% bins
baselines = {}  # baselines[(task_id, bin_idx)] = (mean_7d, std_7d)

for task_id in range(10):
    task_succ = [r for r in succ_rollouts if r["task_id"] == task_id]
    if not task_succ:
        continue
    for bin_idx in range(N_BINS):
        actions_in_bin = []
        for r in task_succ:
            acts = np.array(r["actions"])
            T = len(acts)
            t_start = int(bin_idx / N_BINS * T)
            t_end = int((bin_idx + 1) / N_BINS * T)
            if t_end > t_start:
                actions_in_bin.append(acts[t_start:t_end])
        if actions_in_bin:
            all_acts = np.concatenate(actions_in_bin, axis=0)
            baselines[(task_id, bin_idx)] = (all_acts.mean(axis=0), all_acts.std(axis=0) + 1e-6)

print(f"  Built baselines for {len(baselines)} (task, stage) pairs")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Label every step in failure rollouts with per-dimension deviation
# For each step in a failure rollout:
#   deviation_dim_i = (action_i - success_mean_i) / success_std_i
# If |deviation| > threshold → that dimension is "deviating" at this step
# Also track the SIGN of deviation (positive vs negative direction)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: Labeling per-dimension deviations in failure rollouts")
print("="*70)

DEVIATION_THRESHOLD = 1.5  # > 1.5 std from success mean = "deviating"

# Collect: (hidden_state, per_dim_is_deviating[7], per_dim_deviation_signed[7], 
#           per_dim_deviation_magnitude[7], task_id, episode_progress)
samples = []

for r in fail_rollouts:
    tid = r["task_id"]
    feats = np.array(r["features"])
    acts = np.array(r["actions"])
    T = min(len(feats), len(acts))
    
    for t in range(T):
        progress = t / max(T - 1, 1)
        bin_idx = min(int(progress * N_BINS), N_BINS - 1)
        
        key = (tid, bin_idx)
        if key not in baselines:
            continue
        
        mean, std = baselines[key]
        deviation = (acts[t] - mean) / std  # z-score per dimension
        is_deviating = (np.abs(deviation) > DEVIATION_THRESHOLD).astype(np.float32)
        
        samples.append({
            "hidden_state": feats[t],
            "deviation_zscore": deviation.astype(np.float32),
            "is_deviating": is_deviating,  # binary per-dim
            "deviation_sign": np.sign(deviation).astype(np.float32),  # direction
            "deviation_magnitude": np.abs(deviation).astype(np.float32),
            "task_id": tid,
            "progress": progress,
        })

# Also add success samples (should have low deviation by definition)
for r in succ_rollouts:
    tid = r["task_id"]
    feats = np.array(r["features"])
    acts = np.array(r["actions"])
    T = min(len(feats), len(acts))
    
    for t in range(T):
        progress = t / max(T - 1, 1)
        bin_idx = min(int(progress * N_BINS), N_BINS - 1)
        
        key = (tid, bin_idx)
        if key not in baselines:
            continue
        
        mean, std = baselines[key]
        deviation = (acts[t] - mean) / std
        is_deviating = (np.abs(deviation) > DEVIATION_THRESHOLD).astype(np.float32)
        
        samples.append({
            "hidden_state": feats[t],
            "deviation_zscore": deviation.astype(np.float32),
            "is_deviating": is_deviating,
            "deviation_sign": np.sign(deviation).astype(np.float32),
            "deviation_magnitude": np.abs(deviation).astype(np.float32),
            "task_id": tid,
            "progress": progress,
        })

print(f"  Total labeled samples: {len(samples)}")

# Analyze label distribution
all_is_deviating = np.array([s["is_deviating"] for s in samples])
print(f"\n  Per-dimension deviation rates (threshold={DEVIATION_THRESHOLD}σ):")
print(f"    {'Dim':>8} {'% Deviating':>12} {'Count':>8}")
print(f"    {'-'*32}")
for i in range(7):
    rate = all_is_deviating[:, i].mean() * 100
    count = int(all_is_deviating[:, i].sum())
    print(f"    {DIM_NAMES[i]:>8} {rate:>11.1f}% {count:>8}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Per-dimension probing — can hidden states predict which dim deviates?
# For each dimension, train a logistic regression probe:
#   hidden_state → will this dimension deviate?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: Per-dimension probing (hidden_state → deviation prediction)")
print("="*70)

hidden_states = np.array([s["hidden_state"] for s in samples])
task_ids = np.array([s["task_id"] for s in samples])
progress = np.array([s["progress"] for s in samples])
deviation_labels = np.array([s["is_deviating"] for s in samples])
deviation_zscores = np.array([s["deviation_zscore"] for s in samples])
deviation_signs = np.array([s["deviation_sign"] for s in samples])

# Standardize features
scaler = StandardScaler()
hidden_scaled = scaler.fit_transform(hidden_states)

# PCA for efficiency
pca = PCA(n_components=100)
hidden_pca = pca.fit_transform(hidden_scaled)
print(f"  PCA: 4096 → 100 dims, {pca.explained_variance_ratio_.sum()*100:.1f}% variance retained")

# 3a. Binary probing: can hidden states predict IF dimension i will deviate?
print("\n  === Per-Dimension Deviation Prediction (Binary) ===")
print(f"    {'Dim':>8} {'AUC':>8} {'Acc':>8} {'Pos%':>8} {'Result':>10}")
print(f"    {'-'*46}")

dim_aucs = {}
for i in range(7):
    y = deviation_labels[:, i]
    pos_rate = y.mean()
    
    if pos_rate < 0.01 or pos_rate > 0.99:
        print(f"    {DIM_NAMES[i]:>8} {'N/A':>8} {'N/A':>8} {pos_rate*100:>7.1f}% {'SKIP':>10}")
        dim_aucs[i] = None
        continue
    
    lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    # 5-fold cross-validated predictions
    probs = cross_val_predict(lr, hidden_pca, y, cv=5, method="predict_proba")[:, 1]
    preds = (probs > 0.5).astype(int)
    
    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, preds)
    dim_aucs[i] = auc
    
    verdict = "✅ STRONG" if auc > 0.75 else ("🟡 MODERATE" if auc > 0.65 else "❌ WEAK")
    print(f"    {DIM_NAMES[i]:>8} {auc:>8.4f} {acc:>8.4f} {pos_rate*100:>7.1f}% {verdict:>10}")

# 3b. Regression probing: can hidden states predict the MAGNITUDE of deviation?
print("\n  === Per-Dimension Deviation Magnitude Prediction (Regression) ===")
print(f"    {'Dim':>8} {'R²':>8} {'Corr':>8} {'Result':>10}")
print(f"    {'-'*38}")

for i in range(7):
    y = deviation_zscores[:, i]  # signed z-score
    
    ridge = Ridge(alpha=1.0)
    preds = cross_val_predict(ridge, hidden_pca, y, cv=5)
    
    r2 = r2_score(y, preds)
    corr = np.corrcoef(y, preds)[0, 1]
    
    verdict = "✅ STRONG" if abs(corr) > 0.4 else ("🟡 MOD" if abs(corr) > 0.25 else "❌ WEAK")
    print(f"    {DIM_NAMES[i]:>8} {r2:>8.4f} {corr:>8.4f} {verdict:>10}")

# 3c. Direction probing: can hidden states predict the SIGN of deviation?
print("\n  === Per-Dimension Direction Prediction (Positive vs Negative) ===")
print(f"    {'Dim':>8} {'AUC':>8} {'Acc':>8} {'Result':>10}")
print(f"    {'-'*38}")

# Only on samples where the dimension IS deviating
for i in range(7):
    mask = deviation_labels[:, i] > 0.5
    if mask.sum() < 50:
        print(f"    {DIM_NAMES[i]:>8} {'N/A':>8} {'N/A':>8} {'TOO FEW':>10}")
        continue
    
    y = (deviation_signs[mask, i] > 0).astype(int)  # 1=positive, 0=negative
    x = hidden_pca[mask]
    
    if len(np.unique(y)) < 2:
        print(f"    {DIM_NAMES[i]:>8} {'N/A':>8} {'N/A':>8} {'1 CLASS':>10}")
        continue
    
    lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    probs = cross_val_predict(lr, x, y, cv=min(5, min((y==0).sum(), (y==1).sum())), method="predict_proba")[:, 1]
    preds = (probs > 0.5).astype(int)
    
    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, preds)
    
    verdict = "✅" if auc > 0.7 else ("🟡" if auc > 0.6 else "❌")
    print(f"    {DIM_NAMES[i]:>8} {auc:>8.4f} {acc:>8.4f} {verdict:>10}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: TEMPORAL RESOLUTION — How early can we detect directional failure?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: Temporal Resolution — When does directional signal appear?")
print("="*70)

stages = [
    (0.0, 0.2, "Early (0-20%)"),
    (0.2, 0.4, "Mid-Early (20-40%)"),
    (0.4, 0.6, "Mid (40-60%)"),
    (0.6, 0.8, "Mid-Late (60-80%)"),
    (0.8, 1.0, "Late (80-100%)"),
]

temporal_results = {}

for stage_start, stage_end, stage_name in stages:
    mask = (progress >= stage_start) & (progress < stage_end)
    if mask.sum() < 100:
        continue
    
    x = hidden_pca[mask]
    stage_aucs = []
    
    for i in range(7):
        y = deviation_labels[mask, i]
        if y.mean() < 0.01 or y.mean() > 0.99 or mask.sum() < 50:
            stage_aucs.append(None)
            continue
        
        lr = LogisticRegression(max_iter=300, random_state=42, class_weight="balanced")
        try:
            probs = cross_val_predict(lr, x, y, cv=3, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, probs)
        except:
            auc = 0.5
        stage_aucs.append(auc)
    
    temporal_results[stage_name] = stage_aucs

print(f"\n  {'Stage':>20}", end="")
for d in DIM_NAMES:
    print(f" {d:>8}", end="")
print()
print(f"  {'-'*20}", end="")
for _ in range(7):
    print(f" {'-'*8}", end="")
print()

for stage_name, aucs in temporal_results.items():
    print(f"  {stage_name:>20}", end="")
    for auc in aucs:
        if auc is None:
            print(f" {'N/A':>8}", end="")
        else:
            marker = "★" if auc > 0.75 else ("·" if auc > 0.6 else " ")
            print(f" {auc:>7.3f}{marker}", end="")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: CROSS-TASK DIRECTIONAL GENERALIZATION
# Does the directional prediction transfer to unseen tasks?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 5: Cross-Task Directional Generalization (leave-one-task-out)")
print("="*70)

cross_task_dim_aucs = defaultdict(list)

print(f"\n  {'Held-out':>10}", end="")
for d in DIM_NAMES:
    print(f" {d:>8}", end="")
print(f" {'Mean':>8}")
print(f"  {'-'*10}", end="")
for _ in range(8):
    print(f" {'-'*8}", end="")
print()

for held_out in range(10):
    train_mask = task_ids != held_out
    test_mask = task_ids == held_out
    
    if test_mask.sum() < 30 or train_mask.sum() < 100:
        continue
    
    sc = StandardScaler().fit(hidden_states[train_mask])
    pca_m = PCA(n_components=100).fit(sc.transform(hidden_states[train_mask]))
    
    train_x = pca_m.transform(sc.transform(hidden_states[train_mask]))
    test_x = pca_m.transform(sc.transform(hidden_states[test_mask]))
    
    print(f"  Task {held_out:>4}  ", end="")
    task_dim_aucs = []
    
    for i in range(7):
        train_y = deviation_labels[train_mask, i]
        test_y = deviation_labels[test_mask, i]
        
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            print(f" {'N/A':>8}", end="")
            continue
        
        lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
        lr.fit(train_x, train_y)
        probs = lr.predict_proba(test_x)[:, 1]
        
        try:
            auc = roc_auc_score(test_y, probs)
        except:
            auc = 0.5
        
        task_dim_aucs.append(auc)
        cross_task_dim_aucs[i].append(auc)
        print(f" {auc:>8.3f}", end="")
    
    if task_dim_aucs:
        print(f" {np.mean(task_dim_aucs):>8.3f}")
    else:
        print(f" {'N/A':>8}")

print(f"\n  {'AVERAGE':>10}", end="")
for i in range(7):
    if cross_task_dim_aucs[i]:
        print(f" {np.mean(cross_task_dim_aucs[i]):>8.3f}", end="")
    else:
        print(f" {'N/A':>8}", end="")
print()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: FUTURE DEVIATION PREDICTION
# At timestep t, can we predict the deviation at t+k (lookahead)?
# This is the critical "early warning" capability.
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 6: Future Deviation Prediction (Lookahead)")
print("="*70)
print("  Can hidden_state[t] predict deviation at t+k steps ahead?")

lookahead_results = {}

for k in [1, 5, 10, 20, 30]:
    # Build (hidden_state[t], deviation_label[t+k]) pairs
    la_hidden = []
    la_labels = []
    
    for r in all_rollouts:
        feats = np.array(r["features"])
        acts = np.array(r["actions"])
        tid = r["task_id"]
        T = min(len(feats), len(acts))
        
        for t in range(T - k):
            t_future = t + k
            progress_future = t_future / max(T - 1, 1)
            bin_idx = min(int(progress_future * N_BINS), N_BINS - 1)
            
            key = (tid, bin_idx)
            if key not in baselines:
                continue
            
            mean, std = baselines[key]
            deviation = np.abs((acts[t_future] - mean) / std)
            is_dev = (deviation > DEVIATION_THRESHOLD).astype(np.float32)
            
            la_hidden.append(feats[t])
            la_labels.append(is_dev)
    
    la_hidden = np.array(la_hidden)
    la_labels = np.array(la_labels)
    
    if len(la_hidden) < 100:
        continue
    
    la_scaled = StandardScaler().fit_transform(la_hidden)
    la_pca = PCA(n_components=100).fit_transform(la_scaled)
    
    la_aucs = []
    for i in range(7):
        y = la_labels[:, i]
        if y.mean() < 0.01 or y.mean() > 0.99:
            la_aucs.append(None)
            continue
        lr = LogisticRegression(max_iter=300, random_state=42, class_weight="balanced")
        try:
            probs = cross_val_predict(lr, la_pca, y, cv=3, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, probs)
        except:
            auc = 0.5
        la_aucs.append(auc)
    
    lookahead_results[k] = la_aucs

print(f"\n  {'Lookahead':>12}", end="")
for d in DIM_NAMES:
    print(f" {d:>8}", end="")
print()
print(f"  {'-'*12}", end="")
for _ in range(7):
    print(f" {'-'*8}", end="")
print()

for k, aucs in sorted(lookahead_results.items()):
    print(f"  t+{k:>3} steps", end="")
    for auc in aucs:
        if auc is None:
            print(f" {'N/A':>8}", end="")
        else:
            marker = "★" if auc > 0.7 else ""
            print(f" {auc:>7.3f}{marker}", end="")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: HIDDEN STATE NEURON → DIMENSION MAPPING
# Which hidden dimensions are most predictive of each action dimension's failure?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 7: Hidden State Neuron → Action Dimension Mapping")
print("="*70)
print("  Which hidden dimensions encode each action dimension's failure?")

# Train per-dim logistic regression on full features and extract weights
neuron_importance = {}

for i in range(7):
    y = deviation_labels[:, i]
    if y.mean() < 0.01 or y.mean() > 0.99:
        continue
    
    lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced", C=0.1)
    lr.fit(hidden_scaled, y)
    
    weights = np.abs(lr.coef_[0])
    top_neurons = np.argsort(weights)[-10:][::-1]
    neuron_importance[DIM_NAMES[i]] = {
        "top_neurons": top_neurons.tolist(),
        "top_weights": weights[top_neurons].tolist(),
    }
    print(f"\n  {DIM_NAMES[i]:>8}: top neurons = {top_neurons[:5].tolist()}, weights = {[f'{w:.3f}' for w in weights[top_neurons[:5]]]}")

# Check neuron overlap between dimensions
print("\n  Neuron overlap between dimensions (shared top-50 neurons):")
for i, d1 in enumerate(DIM_NAMES):
    if d1 not in neuron_importance:
        continue
    for j, d2 in enumerate(DIM_NAMES):
        if j <= i or d2 not in neuron_importance:
            continue
        s1 = set(neuron_importance[d1]["top_neurons"])
        s2 = set(neuron_importance[d2]["top_neurons"])
        overlap = len(s1 & s2)
        print(f"    {d1:>8} ↔ {d2:<8}: {overlap}/10 shared")

# ═══════════════════════════════════════════════════════════════════════════
# GENERATE CHARTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("GENERATING CHARTS")
print("="*70)

# Chart 1: Per-dimension deviation prediction AUC (bar chart)
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("Can Internal Vectors Predict WHERE Failure Occurs?", fontsize=15, fontweight="bold")

ax = axes[0]
valid_dims = [(DIM_NAMES[i], dim_aucs[i]) for i in range(7) if dim_aucs[i] is not None]
if valid_dims:
    names, aucs = zip(*valid_dims)
    colors = ["#2ecc71" if a > 0.75 else ("#f39c12" if a > 0.65 else "#e74c3c") for a in aucs]
    bars = ax.bar(names, aucs, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.axhline(y=0.75, color="green", linestyle="--", alpha=0.3, label="Strong threshold")
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{auc:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Per-Dimension Deviation Prediction\n(hidden_state → will dim deviate?)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8)

# Chart 2: Temporal resolution heatmap
ax = axes[1]
stage_names = list(temporal_results.keys())
n_stages = len(stage_names)
heatmap_data = np.zeros((n_stages, 7))
for si, sn in enumerate(stage_names):
    for di in range(7):
        val = temporal_results[sn][di]
        heatmap_data[si, di] = val if val is not None else 0.5

im = ax.imshow(heatmap_data, cmap="RdYlGn", vmin=0.45, vmax=0.85, aspect="auto")
ax.set_xticks(range(7))
ax.set_xticklabels(DIM_NAMES, fontsize=9)
ax.set_yticks(range(n_stages))
ax.set_yticklabels([s.split("(")[1].rstrip(")") for s in stage_names], fontsize=9)
for si in range(n_stages):
    for di in range(7):
        val = heatmap_data[si, di]
        color = "white" if val < 0.6 else "black"
        ax.text(di, si, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
ax.set_title("Directional AUC by Episode Stage\n(rows=when, cols=which dim)")
plt.colorbar(im, ax=ax, label="AUC", shrink=0.8)

# Chart 3: Lookahead prediction
ax = axes[2]
if lookahead_results:
    ks = sorted(lookahead_results.keys())
    for i in range(7):
        vals = []
        valid_ks = []
        for k in ks:
            v = lookahead_results[k][i]
            if v is not None:
                vals.append(v)
                valid_ks.append(k)
        if vals:
            ax.plot(valid_ks, vals, "o-", label=DIM_NAMES[i], linewidth=2, markersize=6)
    
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lookahead (steps into future)")
    ax.set_ylabel("AUC")
    ax.set_title("Future Deviation Prediction\n(hidden_state[t] → deviation at t+k)")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 0.95)

plt.tight_layout()
plt.savefig(OUT_DIR / "01_directional_prediction.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 01_directional_prediction.png")

# Chart 2: Cross-task generalization heatmap
fig, ax = plt.subplots(figsize=(12, 8))
ct_matrix = np.full((10, 7), 0.5)
for held_out in range(10):
    train_mask = task_ids != held_out
    test_mask = task_ids == held_out
    
    if test_mask.sum() < 30:
        continue
    
    sc = StandardScaler().fit(hidden_states[train_mask])
    pca_m = PCA(n_components=100).fit(sc.transform(hidden_states[train_mask]))
    train_x = pca_m.transform(sc.transform(hidden_states[train_mask]))
    test_x = pca_m.transform(sc.transform(hidden_states[test_mask]))
    
    for i in range(7):
        train_y = deviation_labels[train_mask, i]
        test_y = deviation_labels[test_mask, i]
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            continue
        lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
        lr.fit(train_x, train_y)
        probs = lr.predict_proba(test_x)[:, 1]
        try:
            ct_matrix[held_out, i] = roc_auc_score(test_y, probs)
        except:
            pass

im = ax.imshow(ct_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
ax.set_xticks(range(7))
ax.set_xticklabels(DIM_NAMES, fontsize=11)
ax.set_yticks(range(10))
ax.set_yticklabels([f"Task {i}" for i in range(10)], fontsize=10)
for i in range(10):
    for j in range(7):
        color = "white" if ct_matrix[i, j] < 0.55 else "black"
        ax.text(j, i, f"{ct_matrix[i, j]:.2f}", ha="center", va="center", fontsize=10, color=color)
ax.set_title("Cross-Task Directional Generalization\n(Train on 9 tasks → Test on held-out task, per dimension)", fontsize=13)
ax.set_xlabel("Action Dimension")
ax.set_ylabel("Held-out Task")
plt.colorbar(im, ax=ax, label="AUC")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_cross_task_directional.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 02_cross_task_directional.png")

# Chart 3: Deviation z-score distributions for failure vs success
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Per-Dimension Deviation Z-Scores (Failure vs Success Rollouts)", fontsize=14, fontweight="bold")

fail_mask_samples = np.array([s["task_id"] in [r["task_id"] for r in fail_rollouts] for s in samples])
# Better: label by whether sample came from a failure rollout
sample_is_fail = []
idx = 0
for r in fail_rollouts:
    n = min(len(r["features"]), len(r["actions"]))
    sample_is_fail.extend([True] * n)
for r in succ_rollouts:
    n = min(len(r["features"]), len(r["actions"]))
    sample_is_fail.extend([False] * n)

# Actually rebuild properly
fail_zscores, succ_zscores = [], []
for r in fail_rollouts:
    tid = r["task_id"]
    acts = np.array(r["actions"])
    T = len(acts)
    for t in range(T):
        progress_t = t / max(T - 1, 1)
        bin_idx = min(int(progress_t * N_BINS), N_BINS - 1)
        key = (tid, bin_idx)
        if key not in baselines:
            continue
        mean, std = baselines[key]
        fail_zscores.append((acts[t] - mean) / std)

for r in succ_rollouts:
    tid = r["task_id"]
    acts = np.array(r["actions"])
    T = len(acts)
    for t in range(T):
        progress_t = t / max(T - 1, 1)
        bin_idx = min(int(progress_t * N_BINS), N_BINS - 1)
        key = (tid, bin_idx)
        if key not in baselines:
            continue
        mean, std = baselines[key]
        succ_zscores.append((acts[t] - mean) / std)

fail_zscores = np.array(fail_zscores)
succ_zscores = np.array(succ_zscores)

for i in range(7):
    ax = axes[i // 4, i % 4]
    ax.hist(succ_zscores[:, i], bins=80, alpha=0.6, density=True, color="#2ecc71", label="Success", range=(-5, 5))
    ax.hist(fail_zscores[:, i], bins=80, alpha=0.6, density=True, color="#e74c3c", label="Failure", range=(-5, 5))
    ax.axvline(x=-DEVIATION_THRESHOLD, color="orange", linestyle="--", alpha=0.7)
    ax.axvline(x=DEVIATION_THRESHOLD, color="orange", linestyle="--", alpha=0.7, label=f"±{DEVIATION_THRESHOLD}σ threshold")
    ax.set_title(f"{DIM_NAMES[i]}")
    ax.set_xlabel("Z-score (deviation from success baseline)")
    ax.legend(fontsize=7)

axes[1, 3].axis("off")
text = "Deviation Analysis\n" + "-"*35 + "\n"
text += f"Threshold: {DEVIATION_THRESHOLD}σ\n\n"
for i in range(7):
    f_dev = (np.abs(fail_zscores[:, i]) > DEVIATION_THRESHOLD).mean() * 100
    s_dev = (np.abs(succ_zscores[:, i]) > DEVIATION_THRESHOLD).mean() * 100
    text += f"{DIM_NAMES[i]:>7}: F={f_dev:.1f}% S={s_dev:.1f}%\n"
axes[1, 3].text(0.1, 0.5, text, fontfamily="monospace", fontsize=11, va="center",
                transform=axes[1, 3].transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig(OUT_DIR / "03_deviation_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 03_deviation_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VERDICT: Can Internal Vectors Predict WHERE Failure Occurs?")
print("="*70)

valid_aucs = [v for v in dim_aucs.values() if v is not None]
mean_dim_auc = np.mean(valid_aucs) if valid_aucs else 0.5

cross_task_valid = []
for i in range(7):
    if cross_task_dim_aucs[i]:
        cross_task_valid.extend(cross_task_dim_aucs[i])
mean_cross_task = np.mean(cross_task_valid) if cross_task_valid else 0.5

print(f"""
  Within-task directional AUC:  {mean_dim_auc:.4f} (avg across dims)
  Cross-task directional AUC:   {mean_cross_task:.4f} (leave-one-out avg)
  
  Per-dimension breakdown:""")
for i in range(7):
    within = dim_aucs.get(i)
    cross = np.mean(cross_task_dim_aucs[i]) if cross_task_dim_aucs[i] else None
    w_str = f"{within:.3f}" if within else "N/A"
    c_str = f"{cross:.3f}" if cross else "N/A"
    print(f"    {DIM_NAMES[i]:>8}: within={w_str}, cross-task={c_str}")

print(f"""
  CONCLUSION:
  {"✅ YES" if mean_dim_auc > 0.7 else "🟡 PARTIALLY" if mean_dim_auc > 0.6 else "❌ NO"} — internal vectors encode directional failure information
  {"✅ YES" if mean_cross_task > 0.65 else "🟡 PARTIALLY" if mean_cross_task > 0.55 else "❌ NO"} — directional prediction generalizes across tasks
  
  Charts saved to: {OUT_DIR}/
""")
