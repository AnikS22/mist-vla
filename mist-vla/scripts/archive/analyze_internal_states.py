"""
Analyze internal VLA states and correlate with movement patterns and failures.
"""
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_rollouts(data_dir, success_file="success_rollouts.pkl", failure_file="failure_rollouts.pkl"):
    """Load success and failure rollouts."""
    data_dir = Path(data_dir)
    
    success_path = data_dir / success_file
    failure_path = data_dir / failure_file
    
    success_rollouts = []
    failure_rollouts = []
    
    if success_path.exists():
        with open(success_path, 'rb') as f:
            success_rollouts = pickle.load(f)
        print(f"Loaded {len(success_rollouts)} success rollouts")
    
    if failure_path.exists():
        with open(failure_path, 'rb') as f:
            failure_rollouts = pickle.load(f)
        print(f"Loaded {len(failure_rollouts)} failure rollouts")
    
    return success_rollouts, failure_rollouts


def extract_features_and_actions(rollouts, label="success", max_rollouts=None):
    """Extract hidden states and actions from rollouts."""
    features_list = []
    actions_list = []
    labels_list = []
    rollout_ids = []
    step_indices = []
    
    n_rollouts = min(len(rollouts), max_rollouts) if max_rollouts else len(rollouts)
    
    for rollout_idx, rollout in enumerate(rollouts[:n_rollouts]):
        if not rollout.get('features') or not rollout.get('actions'):
            continue
        
        features = np.array(rollout['features'], dtype=np.float32)
        actions = np.array(rollout['actions'], dtype=np.float32)
        
        # Extract per-step data
        for step_idx in range(len(features)):
            features_list.append(features[step_idx])
            actions_list.append(actions[step_idx])
            labels_list.append(label)
            rollout_ids.append(rollout_idx)
            step_indices.append(step_idx)
    
    return {
        'features': np.array(features_list),
        'actions': np.array(actions_list),
        'labels': np.array(labels_list),
        'rollout_ids': np.array(rollout_ids),
        'step_indices': np.array(step_indices),
    }


def compute_action_magnitudes(actions):
    """Compute action magnitudes per dimension."""
    action_dims = {
        'x': actions[:, 0],
        'y': actions[:, 1],
        'z': actions[:, 2],
        'roll': actions[:, 3],
        'pitch': actions[:, 4],
        'yaw': actions[:, 5],
        'gripper': actions[:, 6],
    }
    
    magnitudes = {}
    for dim, values in action_dims.items():
        magnitudes[dim] = np.abs(values)
    
    return action_dims, magnitudes


def correlate_hidden_states_with_actions(features, actions, dim_names, output_dir):
    """Compute correlations between hidden state dimensions and action dimensions."""
    print("\n=== Computing Hidden State ↔ Action Correlations ===")
    
    n_features = features.shape[1]
    n_actions = actions.shape[1]
    
    # Sample features for correlation (use PCA to reduce dimensionality first)
    if n_features > 1000:
        print(f"  Reducing hidden state dims from {n_features} to 100 for correlation analysis...")
        pca = PCA(n_components=100)
        features_reduced = pca.fit_transform(features)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        features_reduced = features
    
    # Compute correlations
    correlations = np.zeros((features_reduced.shape[1], n_actions))
    p_values = np.zeros((features_reduced.shape[1], n_actions))
    
    for feat_idx in range(features_reduced.shape[1]):
        for action_idx in range(n_actions):
            corr, p_val = pearsonr(features_reduced[:, feat_idx], actions[:, action_idx])
            correlations[feat_idx, action_idx] = corr
            p_values[feat_idx, action_idx] = p_val
    
    # Visualize correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlations, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Action Dimension', fontsize=12)
    ax.set_ylabel('Hidden State Dimension (PCA-reduced)', fontsize=12)
    ax.set_title('Correlation: Hidden States ↔ Actions', fontsize=14)
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(dim_names)
    plt.colorbar(im, ax=ax, label='Pearson Correlation')
    plt.tight_layout()
    plt.savefig(output_dir / 'hidden_state_action_correlation.png', dpi=150)
    plt.close()
    
    # Find strongest correlations
    print("\n  Strongest correlations (top 10):")
    flat_corr = correlations.flatten()
    top_indices = np.argsort(np.abs(flat_corr))[-10:][::-1]
    for idx in top_indices:
        feat_idx = idx // n_actions
        action_idx = idx % n_actions
        corr_val = correlations[feat_idx, action_idx]
        p_val = p_values[feat_idx, action_idx]
        print(f"    Hidden[{feat_idx}] ↔ {dim_names[action_idx]}: {corr_val:.4f} (p={p_val:.4e})")
    
    return correlations, p_values


def analyze_failure_patterns(success_data, failure_data, dim_names, output_dir):
    """Analyze patterns that distinguish success from failure, per action dimension."""
    print("\n=== Analyzing Failure Patterns per Action Dimension ===")
    
    # Combine data
    all_features = np.vstack([success_data['features'], failure_data['features']])
    all_actions = np.vstack([success_data['actions'], failure_data['actions']])
    all_labels = np.hstack([
        np.ones(len(success_data['features'])),
        np.zeros(len(failure_data['features']))
    ])
    
    # Compute action magnitudes
    action_dims, magnitudes = compute_action_magnitudes(all_actions)
    
    # Analyze per dimension
    results = {}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for dim_idx, dim_name in enumerate(dim_names):
        ax = axes[dim_idx]
        
        # Get action values for this dimension
        action_values = action_dims[dim_name]
        action_mags = magnitudes[dim_name]
        
        # Split by success/failure
        success_actions = action_values[all_labels == 1]
        failure_actions = action_values[all_labels == 0]
        success_mags = action_mags[all_labels == 1]
        failure_mags = action_mags[all_labels == 0]
        
        # Statistics
        results[dim_name] = {
            'success_mean': np.mean(success_actions),
            'failure_mean': np.mean(failure_actions),
            'success_std': np.std(success_actions),
            'failure_std': np.std(failure_actions),
            'success_mag_mean': np.mean(success_mags),
            'failure_mag_mean': np.mean(failure_mags),
        }
        
        # Plot distribution
        ax.hist(success_actions, bins=50, alpha=0.5, label='Success', density=True, color='green')
        ax.hist(failure_actions, bins=50, alpha=0.5, label='Failure', density=True, color='red')
        ax.axvline(np.mean(success_actions), color='green', linestyle='--', linewidth=2, label='Success mean')
        ax.axvline(np.mean(failure_actions), color='red', linestyle='--', linewidth=2, label='Failure mean')
        ax.set_xlabel(f'{dim_name} Action Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{dim_name.upper()}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    axes[-1].axis('off')
    
    plt.suptitle('Action Distributions: Success vs. Failure', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'action_distributions_success_vs_failure.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n  Action statistics (Success vs. Failure):")
    for dim_name in dim_names:
        r = results[dim_name]
        print(f"    {dim_name.upper():8s}: Success μ={r['success_mean']:7.4f}±{r['success_std']:.4f}, "
              f"Failure μ={r['failure_mean']:7.4f}±{r['failure_std']:.4f}")
        print(f"             Magnitude: Success={r['success_mag_mean']:.4f}, Failure={r['failure_mag_mean']:.4f}")
    
    return results


def analyze_hidden_state_differences(success_data, failure_data, output_dir):
    """Analyze differences in hidden states between success and failure."""
    print("\n=== Analyzing Hidden State Differences ===")
    
    success_features = success_data['features']
    failure_features = failure_data['features']
    
    # Compute mean and std per dimension
    success_mean = np.mean(success_features, axis=0)
    failure_mean = np.mean(failure_features, axis=0)
    success_std = np.std(success_features, axis=0)
    failure_std = np.std(failure_features, axis=0)
    
    # Compute difference
    mean_diff = failure_mean - success_mean
    std_diff = failure_std - success_std
    
    # Find dimensions with largest differences
    top_diff_indices = np.argsort(np.abs(mean_diff))[-20:][::-1]
    
    print(f"  Hidden state dimensions with largest differences (top 20):")
    for idx in top_diff_indices[:10]:
        print(f"    Dim[{idx:4d}]: Success μ={success_mean[idx]:8.4f}±{success_std[idx]:.4f}, "
              f"Failure μ={failure_mean[idx]:8.4f}±{failure_std[idx]:.4f}, "
              f"Diff={mean_diff[idx]:8.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean difference
    axes[0, 0].plot(mean_diff, alpha=0.6)
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Hidden State Dimension', fontsize=12)
    axes[0, 0].set_ylabel('Mean Difference (Failure - Success)', fontsize=12)
    axes[0, 0].set_title('Mean Hidden State: Failure vs. Success', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of differences
    axes[0, 1].hist(mean_diff, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Mean Difference', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Distribution of Mean Differences', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Std difference
    axes[1, 0].plot(std_diff, alpha=0.6, color='orange')
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Hidden State Dimension', fontsize=12)
    axes[1, 0].set_ylabel('Std Difference (Failure - Success)', fontsize=12)
    axes[1, 0].set_title('Std Hidden State: Failure vs. Success', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Top differences
    top_20_indices = top_diff_indices[:20]
    top_20_diffs = mean_diff[top_20_indices]
    axes[1, 1].barh(range(len(top_20_indices)), top_20_diffs)
    axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_yticks(range(len(top_20_indices)))
    axes[1, 1].set_yticklabels([f'Dim[{idx}]' for idx in top_20_indices])
    axes[1, 1].set_xlabel('Mean Difference', fontsize=12)
    axes[1, 1].set_title('Top 20 Dimensions with Largest Differences', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hidden_state_differences.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return mean_diff, std_diff


def visualize_hidden_state_space(success_data, failure_data, output_dir, n_samples=5000):
    """Visualize hidden states in 2D using PCA and t-SNE."""
    print("\n=== Visualizing Hidden State Space ===")
    
    # Sample data
    success_features = success_data['features']
    failure_features = failure_data['features']
    
    n_success = min(len(success_features), n_samples // 2)
    n_failure = min(len(failure_features), n_samples // 2)
    
    success_sample = success_features[:n_success]
    failure_sample = failure_features[:n_failure]
    
    all_features = np.vstack([success_sample, failure_sample])
    labels = np.hstack([np.ones(n_success), np.zeros(n_failure)])
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # PCA
    print("  Computing PCA...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_scaled)
    print(f"  Explained variance (50 components): {pca.explained_variance_ratio_.sum():.3f}")
    
    # t-SNE on PCA-reduced features
    print("  Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA visualization
    axes[0].scatter(features_pca[labels == 1, 0], features_pca[labels == 1, 1], 
                    alpha=0.5, s=1, label='Success', color='green')
    axes[0].scatter(features_pca[labels == 0, 0], features_pca[labels == 0, 1], 
                    alpha=0.5, s=1, label='Failure', color='red')
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].set_title('PCA: Hidden States (Success vs. Failure)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE visualization
    axes[1].scatter(features_tsne[labels == 1, 0], features_tsne[labels == 1, 1], 
                   alpha=0.5, s=1, label='Success', color='green')
    axes[1].scatter(features_tsne[labels == 0, 0], features_tsne[labels == 0, 1], 
                   alpha=0.5, s=1, label='Failure', color='red')
    axes[1].set_xlabel('t-SNE 1', fontsize=12)
    axes[1].set_ylabel('t-SNE 2', fontsize=12)
    axes[1].set_title('t-SNE: Hidden States (Success vs. Failure)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hidden_state_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Visualization saved.")


def correlate_failure_with_action_magnitude(success_data, failure_data, dim_names, output_dir):
    """Correlate failure probability with action magnitude per dimension."""
    print("\n=== Correlating Failure with Action Magnitude ===")
    
    # Combine data
    all_actions = np.vstack([success_data['actions'], failure_data['actions']])
    all_labels = np.hstack([
        np.ones(len(success_data['actions'])),
        np.zeros(len(failure_data['actions']))
    ])
    
    action_dims, magnitudes = compute_action_magnitudes(all_actions)
    
    # Bin action magnitudes and compute failure rate
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    results = {}
    
    for dim_idx, dim_name in enumerate(dim_names):
        ax = axes[dim_idx]
        
        mags = magnitudes[dim_name]
        
        # Create bins
        n_bins = 20
        bins = np.linspace(0, np.percentile(mags, 99), n_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Compute failure rate per bin
        failure_rates = []
        bin_counts = []
        for i in range(len(bins) - 1):
            mask = (mags >= bins[i]) & (mags < bins[i+1])
            if np.sum(mask) > 10:  # Only include bins with enough samples
                failure_rate = 1 - np.mean(all_labels[mask])
                failure_rates.append(failure_rate)
                bin_counts.append(np.sum(mask))
            else:
                failure_rates.append(np.nan)
                bin_counts.append(0)
        
        failure_rates = np.array(failure_rates)
        bin_centers_valid = bin_centers[~np.isnan(failure_rates)]
        failure_rates_valid = failure_rates[~np.isnan(failure_rates)]
        
        # Plot
        ax.plot(bin_centers_valid, failure_rates_valid, 'o-', linewidth=2, markersize=4)
        ax.set_xlabel(f'{dim_name.upper()} Action Magnitude', fontsize=10)
        ax.set_ylabel('Failure Rate', fontsize=10)
        ax.set_title(f'{dim_name.upper()}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Compute correlation
        if len(failure_rates_valid) > 2:
            corr, p_val = pearsonr(bin_centers_valid, failure_rates_valid)
            results[dim_name] = {'correlation': corr, 'p_value': p_val}
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={p_val:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].axis('off')
    
    plt.suptitle('Failure Rate vs. Action Magnitude (per dimension)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'failure_rate_vs_action_magnitude.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n  Correlation between action magnitude and failure rate:")
    for dim_name in dim_names:
        if dim_name in results:
            r = results[dim_name]
            print(f"    {dim_name.upper():8s}: r={r['correlation']:7.4f}, p={r['p_value']:.4e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze internal VLA states and correlate with failures")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing rollout pickle files")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="Output directory for plots")
    parser.add_argument("--max-rollouts", type=int, default=None, help="Max rollouts to analyze (for speed)")
    parser.add_argument("--checkpoint", action="store_true", help="Use checkpoint files")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    success_file = "success_rollouts_partial.pkl" if args.checkpoint else "success_rollouts.pkl"
    failure_file = "failure_rollouts_partial.pkl" if args.checkpoint else "failure_rollouts.pkl"
    
    success_rollouts, failure_rollouts = load_rollouts(args.data_dir, success_file, failure_file)
    
    if len(success_rollouts) == 0 and len(failure_rollouts) == 0:
        print("Error: No rollouts found!")
        return
    
    # Extract features and actions
    print("\nExtracting features and actions...")
    success_data = extract_features_and_actions(success_rollouts, label="success", max_rollouts=args.max_rollouts)
    failure_data = extract_features_and_actions(failure_rollouts, label="failure", max_rollouts=args.max_rollouts)
    
    print(f"\nExtracted {len(success_data['features'])} success steps and {len(failure_data['features'])} failure steps")
    
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    
    # Run analyses
    correlate_hidden_states_with_actions(
        np.vstack([success_data['features'], failure_data['features']]),
        np.vstack([success_data['actions'], failure_data['actions']]),
        dim_names,
        output_dir
    )
    
    analyze_failure_patterns(success_data, failure_data, dim_names, output_dir)
    
    analyze_hidden_state_differences(success_data, failure_data, output_dir)
    
    correlate_failure_with_action_magnitude(success_data, failure_data, dim_names, output_dir)
    
    # Visualization (can be slow, so optional)
    try:
        visualize_hidden_state_space(success_data, failure_data, output_dir, n_samples=5000)
    except Exception as e:
        print(f"\nWarning: Visualization failed: {e}")
    
    print(f"\n=== Analysis complete! Results saved to {output_dir} ===")


if __name__ == "__main__":
    main()
