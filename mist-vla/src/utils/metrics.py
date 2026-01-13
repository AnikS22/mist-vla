"""
Evaluation metrics for MIST-VLA.
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def compute_detection_metrics(
    scores: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for failure detection.

    Args:
        scores: Predicted failure scores [N]
        labels: Ground truth labels [N] (0=success, 1=failure)

    Returns:
        Dictionary with ROC-AUC, PR-AUC, etc.
    """
    # ROC-AUC
    roc_auc = roc_auc_score(labels, scores)

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }


def compute_recovery_metrics(
    n_failures_detected: int,
    n_recoveries_successful: int,
    n_total_steps: int
) -> Dict[str, float]:
    """
    Compute recovery-specific metrics.

    Args:
        n_failures_detected: Number of detected failures
        n_recoveries_successful: Number of successful recoveries
        n_total_steps: Total steps in episode

    Returns:
        Dictionary with recovery metrics
    """
    if n_failures_detected == 0:
        recovery_rate = 1.0  # No failures = perfect
    else:
        recovery_rate = n_recoveries_successful / n_failures_detected

    failure_frequency = n_failures_detected / n_total_steps

    return {
        'recovery_rate': recovery_rate,
        'failure_frequency': failure_frequency,
        'n_failures': n_failures_detected,
        'n_recoveries': n_recoveries_successful
    }


def compute_task_metrics(
    episode_results: List[Dict]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple episodes.

    Args:
        episode_results: List of result dictionaries from episodes

    Returns:
        Aggregated metrics
    """
    success_rates = [r['success'] for r in episode_results]
    steps = [r['n_steps'] for r in episode_results]
    failures = [r['n_failures_detected'] for r in episode_results]
    recoveries = [r['n_recoveries_successful'] for r in episode_results]

    return {
        'success_rate': np.mean(success_rates),
        'success_rate_std': np.std(success_rates),
        'avg_steps': np.mean(steps),
        'avg_steps_std': np.std(steps),
        'avg_failures': np.mean(failures),
        'avg_recoveries': np.mean(recoveries),
        'overall_recovery_rate': sum(recoveries) / max(sum(failures), 1)
    }


def compute_latency_metrics(
    detection_timesteps: List[int],
    failure_occurred_timesteps: List[int]
) -> Dict[str, float]:
    """
    Compute latency of failure detection.

    Args:
        detection_timesteps: When failures were detected
        failure_occurred_timesteps: When failures actually occurred

    Returns:
        Detection latency metrics
    """
    latencies = np.array(detection_timesteps) - np.array(failure_occurred_timesteps)

    return {
        'mean_latency': latencies.mean(),
        'median_latency': np.median(latencies),
        'max_latency': latencies.max(),
        'min_latency': latencies.min()
    }


def print_evaluation_summary(results: Dict):
    """
    Pretty print evaluation results.

    Args:
        results: Results dictionary from evaluation
    """
    print("\n" + "="*60)
    print(f"MIST-VLA Evaluation Results: {results['task_suite']}")
    print("="*60)

    agg = results['aggregate']

    print(f"\nOverall Metrics:")
    print(f"  Success Rate:    {agg['overall_success_rate']:.2%}")
    print(f"  Recovery Rate:   {agg['recovery_rate']:.2%}")
    print(f"  Avg Failures:    {agg['avg_failures_per_episode']:.2f}")
    print(f"  Avg Recoveries:  {agg['avg_recoveries_per_episode']:.2f}")

    print(f"\nPer-Task Results:")
    for task_id, task_results in results['per_task_results'].items():
        print(f"\n  Task {task_id}: {task_results['instruction'][:50]}...")
        print(f"    Success Rate: {task_results['success_rate']:.2%}")
        print(f"    Avg Steps:    {task_results['avg_steps']:.1f}")
        print(f"    Avg Failures: {task_results['avg_failures_detected']:.2f}")

    print("\n" + "="*60)
