"""
Prepare comprehensive training dataset using research-backed techniques.

Creates samples with:
1. Binary labels (success vs failure trajectory)
2. Time-to-failure (continuous)
3. Per-dimension risk (based on action patterns + trajectory outcome)
4. Auxiliary features (action statistics, trajectory features)
"""
import argparse
import pickle
from pathlib import Path
from typing import List, Dict
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
        raise ValueError(f"Unknown data format")


def get_steps(rollout: Dict) -> List[Dict]:
    """Extract steps from rollout."""
    if 'steps' in rollout and rollout['steps']:
        return rollout['steps']
    
    actions = rollout.get('actions', [])
    features = rollout.get('features', [])
    
    steps = []
    for i in range(min(len(actions), len(features))):
        steps.append({
            'action': np.asarray(actions[i], dtype=np.float32),
            'hidden_state': np.asarray(features[i], dtype=np.float32),
        })
    return steps


def compute_action_features(actions: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute action-based features for risk estimation."""
    if len(actions) < 2:
        return {
            'velocity': np.zeros_like(actions[0]),
            'acceleration': np.zeros_like(actions[0]),
            'cumulative': np.zeros_like(actions[0]),
        }
    
    # Velocity and acceleration
    velocity = np.diff(actions, axis=0, prepend=actions[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    
    # Cumulative displacement
    cumulative = np.cumsum(actions, axis=0)
    
    return {
        'velocity': velocity,
        'acceleration': acceleration,
        'cumulative': cumulative,
    }


def create_comprehensive_dataset(
    success_rollouts: List[Dict],
    failure_rollouts: List[Dict],
    k_before_failure: int = 30,
    include_all_failure_steps: bool = True,
    include_success_negative: bool = True,
    risk_decay: float = 0.9,
) -> List[Dict]:
    """
    Create comprehensive dataset with rich labels.
    
    Key innovations:
    1. Use ALL failure steps (not just last k) with decaying risk
    2. Per-dimension risk based on action contribution to failure
    3. Include auxiliary features (velocity, acceleration)
    """
    dataset = []
    
    print("Processing failure rollouts...")
    for rollout in tqdm(failure_rollouts):
        steps = get_steps(rollout)
        if not steps:
            continue
        
        n_steps = len(steps)
        
        # Extract actions
        actions = np.array([s.get('action', np.zeros(7)) for s in steps], dtype=np.float32)
        action_features = compute_action_features(actions)
        
        # Compute trajectory-level action statistics
        action_mean = np.abs(actions).mean(axis=0)
        action_max = np.abs(actions).max(axis=0)
        action_std = actions.std(axis=0)
        
        # Infer failure direction from final actions
        lookback = min(10, n_steps)
        final_actions = actions[-lookback:] if lookback > 0 else actions
        failure_direction = np.mean(final_actions, axis=0)
        failure_direction = failure_direction / (np.linalg.norm(failure_direction) + 1e-8)
        
        for i, step in enumerate(steps):
            hidden = step.get('hidden_state', np.zeros(1))
            action = actions[i]
            
            if len(hidden) < 10:
                continue
            
            # Time to failure
            steps_from_end = n_steps - i
            
            # Compute per-dimension risk
            # Risk is higher for dimensions that:
            # 1. Have large action magnitude
            # 2. Align with failure direction
            # 3. Are close to failure
            
            if steps_from_end <= k_before_failure:
                # Within window: high risk with decay
                proximity = steps_from_end / k_before_failure
                base_risk = 1.0 - (proximity ** 0.5)  # Non-linear decay
                
                # Action magnitude contribution
                action_mag = np.abs(action)
                action_contrib = action_mag / (action_max + 1e-8)
                
                # Alignment with failure direction
                alignment = np.abs(action * failure_direction)
                
                # Velocity contribution (sudden changes = risk)
                vel_mag = np.abs(action_features['velocity'][i])
                vel_contrib = vel_mag / (np.abs(action_features['velocity']).max(axis=0) + 1e-8)
                
                # Combine signals
                per_dim_risk = (
                    0.4 * base_risk * action_contrib +
                    0.3 * base_risk * alignment +
                    0.2 * base_risk * vel_contrib +
                    0.1 * base_risk
                )
                per_dim_risk = np.clip(per_dim_risk, 0, 1)
                
                binary_label = 1
            else:
                # Earlier in trajectory: lower risk with exponential decay
                if include_all_failure_steps:
                    decay_steps = n_steps - k_before_failure - i
                    decay = risk_decay ** decay_steps
                    
                    action_mag = np.abs(action)
                    action_contrib = action_mag / (action_max + 1e-8)
                    
                    per_dim_risk = 0.3 * decay * action_contrib
                    binary_label = 1 if decay > 0.1 else 0
                else:
                    per_dim_risk = np.zeros(7, dtype=np.float32)
                    binary_label = 0
            
            dataset.append({
                'hidden_state': np.asarray(hidden, dtype=np.float32),
                'action': np.asarray(action, dtype=np.float32),
                'risk_label': np.asarray(per_dim_risk, dtype=np.float32),
                'binary_label': binary_label,
                'time_to_failure': steps_from_end,
                'success': 0,
            })
    
    print("Processing success rollouts...")
    for rollout in tqdm(success_rollouts):
        if not include_success_negative:
            continue
            
        steps = get_steps(rollout)
        if not steps:
            continue
        
        for step in steps:
            hidden = step.get('hidden_state', np.zeros(1))
            action = step.get('action', np.zeros(7))
            
            if len(hidden) < 10:
                continue
            
            dataset.append({
                'hidden_state': np.asarray(hidden, dtype=np.float32),
                'action': np.asarray(action, dtype=np.float32),
                'risk_label': np.zeros(7, dtype=np.float32),
                'binary_label': 0,
                'time_to_failure': -1,
                'success': 1,
            })
    
    return dataset


def balance_dataset(
    dataset: List[Dict],
    target_ratio: float = 1.0,
) -> List[Dict]:
    """Balance dataset by undersampling majority class."""
    positives = [d for d in dataset if d['binary_label'] > 0]
    negatives = [d for d in dataset if d['binary_label'] == 0]
    
    n_pos = len(positives)
    n_neg_target = int(n_pos * target_ratio)
    
    if n_neg_target < len(negatives):
        indices = np.random.choice(len(negatives), n_neg_target, replace=False)
        negatives = [negatives[i] for i in indices]
    
    balanced = positives + negatives
    np.random.shuffle(balanced)
    
    return balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", type=str, required=True)
    parser.add_argument("--failure", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--k-before-failure", type=int, default=30)
    parser.add_argument("--risk-decay", type=float, default=0.9)
    parser.add_argument("--balance-ratio", type=float, default=1.0)
    parser.add_argument("--include-all-failure-steps", action="store_true")
    args = parser.parse_args()
    
    print(f"Loading success rollouts from {args.success}...")
    success_rollouts = load_rollouts(Path(args.success))
    print(f"Loaded {len(success_rollouts)} success rollouts")
    
    print(f"Loading failure rollouts from {args.failure}...")
    failure_rollouts = load_rollouts(Path(args.failure))
    print(f"Loaded {len(failure_rollouts)} failure rollouts")
    
    print("\nCreating comprehensive dataset...")
    dataset = create_comprehensive_dataset(
        success_rollouts,
        failure_rollouts,
        k_before_failure=args.k_before_failure,
        include_all_failure_steps=args.include_all_failure_steps,
        risk_decay=args.risk_decay,
    )
    
    print("\nBalancing dataset...")
    dataset = balance_dataset(dataset, target_ratio=args.balance_ratio)
    
    # Statistics
    risk_labels = np.array([d['risk_label'] for d in dataset])
    binary_labels = np.array([d['binary_label'] for d in dataset])
    
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    print(f"Positive (risky): {binary_labels.sum()} ({100*binary_labels.mean():.1f}%)")
    print(f"Negative (safe): {len(binary_labels) - binary_labels.sum()}")
    print(f"\nPer-dimension risk:")
    dims = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    for i, dim in enumerate(dims):
        mean = risk_labels[:, i].mean()
        std = risk_labels[:, i].std()
        nonzero = (risk_labels[:, i] > 0).sum()
        print(f"  {dim:8s}: mean={mean:.4f}, std={std:.4f}, non-zero={nonzero}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'dataset': dataset,
        'metadata': {
            'num_samples': len(dataset),
            'num_positive': int(binary_labels.sum()),
            'num_negative': int(len(binary_labels) - binary_labels.sum()),
            'k_before_failure': args.k_before_failure,
            'risk_decay': args.risk_decay,
            'balance_ratio': args.balance_ratio,
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
