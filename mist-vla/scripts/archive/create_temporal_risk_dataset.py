#!/usr/bin/env python3
"""Create training dataset with temporal risk decay.

Key insight: Original labels are sparse because no collisions were detected.
Solution: Use temporal risk decay - assign risk based on distance to episode end.

For failure trajectories:
  - risk = exp(-step_distance / decay_constant) * action_attribution
  - Every step gets non-zero label, decaying further from failure

For success trajectories:
  - risk = 0 (these trajectories succeeded, no failure risk)
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


def compute_action_attribution(actions: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Attribute risk to action dimensions based on magnitude patterns.
    
    Uses action magnitude as proxy for which dimensions contributed to failure.
    Higher magnitude actions in the lead-up to failure are more likely culprits.
    """
    T, D = actions.shape
    attribution = np.zeros((T, D))
    
    for t in range(T):
        # Look at recent action window
        start = max(0, t - window)
        recent_actions = actions[start:t+1]
        
        # Compute per-dimension magnitude scores
        magnitudes = np.abs(recent_actions).mean(axis=0)
        
        # Normalize to get attribution weights (sum to 1)
        total = magnitudes.sum()
        if total > 0:
            attribution[t] = magnitudes / total
        else:
            attribution[t] = 1.0 / D  # Uniform if no actions
    
    return attribution


def compute_temporal_risk(episode_length: int, decay: float = 30.0) -> np.ndarray:
    """
    Compute temporal risk decay for a failure trajectory.
    
    risk(t) = exp(-(T - t) / decay)
    
    Steps near the end have risk ~1, steps at start have risk ~0.
    """
    risks = np.zeros(episode_length)
    for t in range(episode_length):
        distance_to_end = episode_length - 1 - t
        risks[t] = np.exp(-distance_to_end / decay)
    return risks


def process_rollout(rollout: dict, is_failure: bool, decay: float = 30.0) -> list:
    """Process a single rollout into training samples."""
    features = np.array(rollout['features'])
    actions = np.array(rollout['actions'])
    T = len(features)
    
    samples = []
    
    if is_failure:
        # Compute temporal risk decay
        temporal_risks = compute_temporal_risk(T, decay)
        
        # Compute per-dimension action attribution
        action_attribution = compute_action_attribution(actions)
        
        for t in range(T):
            # Per-dim risk = temporal_risk * action_attribution
            per_dim_risk = temporal_risks[t] * action_attribution[t]
            
            # Time to failure
            ttf = T - 1 - t  # Steps until end
            
            samples.append({
                'hidden_state': features[t].astype(np.float32),
                'risk_label': per_dim_risk.astype(np.float32),
                'action': actions[t].astype(np.float32),
                'time_to_failure': ttf,
                'is_failure': 1,
                'binary_risk': float(temporal_risks[t] > 0.1)  # Binary: will fail?
            })
    else:
        # Success trajectory - no risk
        for t in range(T):
            samples.append({
                'hidden_state': features[t].astype(np.float32),
                'risk_label': np.zeros(7, dtype=np.float32),
                'action': actions[t].astype(np.float32),
                'time_to_failure': -1,  # No failure
                'is_failure': 0,
                'binary_risk': 0.0
            })
    
    return samples


def create_balanced_dataset(fail_samples: list, succ_samples: list, 
                            fail_ratio: float = 0.5) -> list:
    """Create balanced dataset with specified failure ratio."""
    n_fail = len(fail_samples)
    n_succ = len(succ_samples)
    
    # Calculate how many samples we need
    if fail_ratio == 0.5:
        # Perfect balance
        n_each = min(n_fail, n_succ)
        fail_subset = np.random.choice(n_fail, n_each, replace=False)
        succ_subset = np.random.choice(n_succ, n_each, replace=False)
    else:
        # Use all failure, sample success to get ratio
        target_succ = int(n_fail * (1 - fail_ratio) / fail_ratio)
        fail_subset = np.arange(n_fail)
        succ_subset = np.random.choice(n_succ, min(target_succ, n_succ), 
                                       replace=target_succ > n_succ)
    
    dataset = []
    for i in fail_subset:
        dataset.append(fail_samples[i])
    for i in succ_subset:
        dataset.append(succ_samples[i])
    
    np.random.shuffle(dataset)
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Create temporal risk dataset')
    parser.add_argument('--failure', type=str, required=True,
                        help='Path to failure rollouts pickle')
    parser.add_argument('--success', type=str, required=True,
                        help='Path to success rollouts pickle')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for training dataset')
    parser.add_argument('--decay', type=float, default=30.0,
                        help='Decay constant for temporal risk (default: 30)')
    parser.add_argument('--fail-ratio', type=float, default=0.5,
                        help='Target ratio of failure samples (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print('Loading rollouts...')
    with open(args.failure, 'rb') as f:
        fail_rollouts = pickle.load(f)
    with open(args.success, 'rb') as f:
        succ_rollouts = pickle.load(f)
    
    print(f'Failure rollouts: {len(fail_rollouts)}')
    print(f'Success rollouts: {len(succ_rollouts)}')
    
    # Process rollouts
    print('\nProcessing failure rollouts...')
    fail_samples = []
    for rollout in fail_rollouts:
        fail_samples.extend(process_rollout(rollout, is_failure=True, decay=args.decay))
    print(f'  Total failure samples: {len(fail_samples)}')
    
    print('Processing success rollouts...')
    succ_samples = []
    for rollout in succ_rollouts:
        succ_samples.extend(process_rollout(rollout, is_failure=False, decay=args.decay))
    print(f'  Total success samples: {len(succ_samples)}')
    
    # Create balanced dataset
    print(f'\nCreating balanced dataset (fail_ratio={args.fail_ratio})...')
    dataset = create_balanced_dataset(fail_samples, succ_samples, args.fail_ratio)
    print(f'Final dataset size: {len(dataset)}')
    
    # Analyze label distribution
    risk_labels = np.array([s['risk_label'] for s in dataset])
    binary_risks = np.array([s['binary_risk'] for s in dataset])
    
    print('\n=== Label Distribution ===')
    print(f'Samples with binary_risk > 0: {(binary_risks > 0).sum()} ({100*(binary_risks > 0).mean():.1f}%)')
    print(f'Samples with any per_dim_risk > 0: {(risk_labels.any(axis=1)).sum()} ({100*(risk_labels.any(axis=1)).mean():.1f}%)')
    
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    print('\nPer-dimension positive rates:')
    for i, name in enumerate(dim_names):
        pos_rate = (risk_labels[:, i] > 0).mean() * 100
        mean_val = risk_labels[:, i].mean()
        print(f'  {name}: {pos_rate:.1f}% positive, mean={mean_val:.4f}')
    
    # Save dataset
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'dataset': dataset,
        'metadata': {
            'n_fail_rollouts': len(fail_rollouts),
            'n_succ_rollouts': len(succ_rollouts),
            'n_fail_samples': len(fail_samples),
            'n_succ_samples': len(succ_samples),
            'decay': args.decay,
            'fail_ratio': args.fail_ratio,
            'final_size': len(dataset)
        }
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)
    print(f'\nDataset saved to: {args.output}')


if __name__ == '__main__':
    main()
