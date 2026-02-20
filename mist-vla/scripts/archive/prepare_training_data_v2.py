"""
Improved training data preparation with better label balance.

Key improvements:
1. Binary classification: SUCCESS vs FAILURE trajectories (not per-step risk)
2. Trajectory-level features: aggregate hidden states over trajectory
3. Better balance: undersample successes or oversample failures
4. Auxiliary signals: use action patterns as input features, not labels
"""
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


def load_rollouts(path: Path) -> List[Dict]:
    """Load rollouts from pickle file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'trajectories' in data:
        return data['trajectories']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown data format in {path}")


def get_steps_from_rollout(rollout: Dict) -> List[Dict]:
    """Extract steps from rollout."""
    if 'steps' in rollout and rollout['steps']:
        return rollout['steps']
    
    # Fallback: construct from actions and features
    actions = rollout.get('actions', [])
    features = rollout.get('features', [])
    
    steps = []
    for i in range(min(len(actions), len(features))):
        steps.append({
            'action': np.asarray(actions[i], dtype=np.float32),
            'hidden_state': np.asarray(features[i], dtype=np.float32),
        })
    return steps


def create_binary_failure_dataset(
    success_rollouts: List[Dict],
    failure_rollouts: List[Dict],
    k_before_failure: int = 20,
    balance_ratio: float = 1.0,
) -> List[Dict]:
    """
    Create binary classification dataset: will this step lead to failure?
    
    Key insight: Use steps from BEFORE failure in failure rollouts as positive class,
    and all steps from success rollouts as negative class.
    
    Args:
        success_rollouts: List of successful rollouts
        failure_rollouts: List of failed rollouts
        k_before_failure: How many steps before failure to mark as "risky"
        balance_ratio: Ratio of negative to positive samples (1.0 = balanced)
    """
    positive_samples = []  # Steps that lead to failure
    negative_samples = []  # Steps from successful trajectories
    
    # Extract positive samples (steps before failure)
    for rollout in failure_rollouts:
        steps = get_steps_from_rollout(rollout)
        if not steps:
            continue
        
        n_steps = len(steps)
        
        # Mark last k steps as "risky" (will fail)
        for i in range(max(0, n_steps - k_before_failure), n_steps):
            step = steps[i]
            hidden_state = step.get('hidden_state', np.zeros(1))
            action = step.get('action', np.zeros(7))
            
            if len(hidden_state) < 10:  # Skip invalid hidden states
                continue
            
            # Steps from end
            steps_from_end = n_steps - i
            
            # Create per-dimension risk based on action magnitude
            # Higher action magnitude = higher risk in that dimension
            action_mag = np.abs(action)
            action_risk = action_mag / (action_mag.max() + 1e-8)
            
            # Weight by proximity to failure
            proximity = steps_from_end / k_before_failure
            per_dim_risk = action_risk * (1.0 - proximity * 0.5)  # Higher risk closer to failure
            
            positive_samples.append({
                'hidden_state': np.asarray(hidden_state, dtype=np.float32),
                'action': np.asarray(action, dtype=np.float32),
                'risk_label': per_dim_risk.astype(np.float32),
                'binary_label': 1,  # Will fail
                'time_to_failure': steps_from_end,
                'success': 0,
            })
    
    # Extract negative samples (steps from success)
    for rollout in success_rollouts:
        steps = get_steps_from_rollout(rollout)
        if not steps:
            continue
        
        for step in steps:
            hidden_state = step.get('hidden_state', np.zeros(1))
            action = step.get('action', np.zeros(7))
            
            if len(hidden_state) < 10:  # Skip invalid
                continue
            
            negative_samples.append({
                'hidden_state': np.asarray(hidden_state, dtype=np.float32),
                'action': np.asarray(action, dtype=np.float32),
                'risk_label': np.zeros(7, dtype=np.float32),  # Zero risk
                'binary_label': 0,  # Will succeed
                'time_to_failure': -1,
                'success': 1,
            })
    
    print(f"Positive samples (risky): {len(positive_samples)}")
    print(f"Negative samples (safe): {len(negative_samples)}")
    
    # Balance dataset
    n_positive = len(positive_samples)
    n_negative_target = int(n_positive * balance_ratio)
    
    if n_negative_target < len(negative_samples):
        # Undersample negatives
        indices = np.random.choice(len(negative_samples), n_negative_target, replace=False)
        negative_samples = [negative_samples[i] for i in indices]
        print(f"Undersampled negatives to: {len(negative_samples)}")
    
    # Combine
    dataset = positive_samples + negative_samples
    np.random.shuffle(dataset)
    
    return dataset


def create_temporal_dataset(
    success_rollouts: List[Dict],
    failure_rollouts: List[Dict],
    window_size: int = 10,
) -> List[Dict]:
    """
    Create dataset with temporal context - aggregate features over window.
    
    Instead of predicting from a single hidden state, use a window of states.
    """
    dataset = []
    
    for rollout in failure_rollouts + success_rollouts:
        steps = get_steps_from_rollout(rollout)
        if len(steps) < window_size:
            continue
        
        success = rollout.get('success', False)
        n_steps = len(steps)
        
        # Extract hidden states
        hidden_states = np.array([
            step.get('hidden_state', np.zeros(4096)) for step in steps
        ], dtype=np.float32)
        
        actions = np.array([
            step.get('action', np.zeros(7)) for step in steps
        ], dtype=np.float32)
        
        # Create samples with temporal context
        for i in range(window_size, n_steps):
            # Window of hidden states
            window_hidden = hidden_states[i-window_size:i]
            window_actions = actions[i-window_size:i]
            
            # Aggregate features
            features = np.concatenate([
                window_hidden.mean(axis=0),  # Mean hidden state
                window_hidden.std(axis=0),   # Variance
                window_hidden[-1] - window_hidden[0],  # Trend
                window_actions.mean(axis=0),  # Mean action
                window_actions.std(axis=0),   # Action variance
            ])
            
            # Label
            if not success:
                steps_from_end = n_steps - i
                if steps_from_end <= 20:
                    binary_label = 1
                    risk = 1.0 - (steps_from_end / 20.0)
                else:
                    binary_label = 0
                    risk = 0.0
            else:
                binary_label = 0
                risk = 0.0
            
            dataset.append({
                'features': features,
                'hidden_state': hidden_states[i],  # Current hidden state
                'action': actions[i],
                'risk_label': np.full(7, risk, dtype=np.float32),
                'binary_label': binary_label,
                'success': int(success),
            })
    
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", type=str, required=True)
    parser.add_argument("--failure", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mode", type=str, default="binary", 
                       choices=["binary", "temporal"])
    parser.add_argument("--k-before-failure", type=int, default=20)
    parser.add_argument("--balance-ratio", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=10)
    args = parser.parse_args()
    
    print(f"Loading success rollouts from {args.success}...")
    success_rollouts = load_rollouts(Path(args.success))
    print(f"Loaded {len(success_rollouts)} success rollouts")
    
    print(f"Loading failure rollouts from {args.failure}...")
    failure_rollouts = load_rollouts(Path(args.failure))
    print(f"Loaded {len(failure_rollouts)} failure rollouts")
    
    if args.mode == "binary":
        dataset = create_binary_failure_dataset(
            success_rollouts,
            failure_rollouts,
            k_before_failure=args.k_before_failure,
            balance_ratio=args.balance_ratio,
        )
    else:
        dataset = create_temporal_dataset(
            success_rollouts,
            failure_rollouts,
            window_size=args.window_size,
        )
    
    # Statistics
    risk_labels = np.array([d['risk_label'] for d in dataset])
    binary_labels = np.array([d['binary_label'] for d in dataset])
    
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Positive (risky): {binary_labels.sum()} ({100*binary_labels.mean():.1f}%)")
    print(f"  Negative (safe): {len(binary_labels) - binary_labels.sum()}")
    print(f"  Risk mean per dim: {risk_labels.mean(axis=0)}")
    print(f"  Risk std per dim: {risk_labels.std(axis=0)}")
    print(f"  Non-zero risk samples: {(risk_labels.sum(axis=1) > 0).sum()}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'dataset': dataset,
        'metadata': {
            'mode': args.mode,
            'num_samples': len(dataset),
            'num_positive': int(binary_labels.sum()),
            'num_negative': int(len(binary_labels) - binary_labels.sum()),
            'k_before_failure': args.k_before_failure,
            'balance_ratio': args.balance_ratio,
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
