"""
Visualization utilities for MIST-VLA.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional
import torch


def plot_failure_scores(
    scores: List[float],
    thresholds: Optional[np.ndarray] = None,
    title: str = "Failure Scores Over Time"
):
    """Plot failure scores over episode timesteps."""
    plt.figure(figsize=(12, 6))

    timesteps = np.arange(len(scores))
    plt.plot(timesteps, scores, label='Failure Score', linewidth=2)

    if thresholds is not None:
        plt.plot(timesteps, thresholds[:len(scores)],
                label='Threshold', linestyle='--', linewidth=2)

    plt.xlabel('Timestep')
    plt.ylabel('Failure Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_attribution_heatmap(
    attributions: torch.Tensor,
    patch_size: int = 16,
    img_size: int = 224,
    title: str = "Visual Attribution Heatmap"
):
    """
    Plot attribution heatmap overlaid on image grid.

    Args:
        attributions: Tensor of shape [256] for image patches
        patch_size: Size of each patch
        img_size: Original image size
    """
    # Reshape to 2D grid
    # For 256 patches, we need 16x16 grid
    n_patches = attributions.numel()
    grid_size = int(np.sqrt(n_patches))  # 16 for 256 patches
    attr_grid = attributions.reshape(grid_size, grid_size).cpu().numpy()

    plt.figure(figsize=(8, 8))
    sns.heatmap(attr_grid, cmap='YlOrRd', cbar=True,
                square=True, annot=False)
    plt.title(title)
    plt.xlabel('Patch Column')
    plt.ylabel('Patch Row')
    plt.tight_layout()

    return plt.gcf()


def plot_recovery_analysis(
    results: Dict,
    save_path: Optional[str] = None
):
    """
    Plot comprehensive recovery analysis.

    Args:
        results: Dictionary with evaluation metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Success rate by task
    tasks = list(results['per_task_results'].keys())
    success_rates = [results['per_task_results'][t]['success_rate']
                    for t in tasks]

    axes[0, 0].bar(tasks, success_rates)
    axes[0, 0].set_xlabel('Task ID')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate by Task')
    axes[0, 0].set_ylim([0, 1])

    # 2. Failure detection frequency
    failures = [results['per_task_results'][t]['avg_failures_detected']
               for t in tasks]

    axes[0, 1].bar(tasks, failures, color='orange')
    axes[0, 1].set_xlabel('Task ID')
    axes[0, 1].set_ylabel('Avg Failures Detected')
    axes[0, 1].set_title('Failure Detection Frequency')

    # 3. Recovery metrics
    metrics = ['overall_success_rate', 'recovery_rate']
    values = [results['aggregate'][m] for m in metrics]

    axes[1, 0].bar(metrics, values, color=['green', 'blue'])
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].set_title('Aggregate Recovery Metrics')
    axes[1, 0].set_ylim([0, 1])

    # 4. Steps to completion
    steps = [results['per_task_results'][t]['avg_steps'] for t in tasks]

    axes[1, 1].plot(tasks, steps, marker='o', color='purple')
    axes[1, 1].set_xlabel('Task ID')
    axes[1, 1].set_ylabel('Avg Steps')
    axes[1, 1].set_title('Steps to Completion')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_steering_effects(
    original_actions: np.ndarray,
    corrected_actions: np.ndarray,
    action_labels: Optional[List[str]] = None
):
    """
    Visualize the effect of steering on actions.

    Args:
        original_actions: Original VLA actions [n_steps, action_dim]
        corrected_actions: Steered actions [n_steps, action_dim]
        action_labels: Labels for each action dimension
    """
    n_dims = original_actions.shape[1]

    if action_labels is None:
        action_labels = [f'Dim {i}' for i in range(n_dims)]

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3 * n_dims))

    if n_dims == 1:
        axes = [axes]

    timesteps = np.arange(len(original_actions))

    for i, (ax, label) in enumerate(zip(axes, action_labels)):
        ax.plot(timesteps, original_actions[:, i],
               label='Original', alpha=0.7, linewidth=2)
        ax.plot(timesteps, corrected_actions[:, i],
               label='Steered', alpha=0.7, linewidth=2)

        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title('Steering Effect on Actions')
        if i == n_dims - 1:
            ax.set_xlabel('Timestep')

    plt.tight_layout()
    return fig
