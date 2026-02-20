"""
Prepare training data from collected rollouts.

This script:
1. Loads success and failure rollouts
2. Applies labels (if not already labeled)
3. Builds per-step dataset for training
4. Supports multiple model sources
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
            'collision': False,
        })
    
    return steps


def prepare_dataset(
    success_path: Path,
    failure_path: Path,
    output_path: Path,
    require_labels: bool = False
) -> None:
    """
    Prepare training dataset from rollouts.
    
    Args:
        success_path: Path to success rollouts
        failure_path: Path to failure rollouts
        output_path: Output dataset path
        require_labels: Whether to require per_dim_risk labels
    """
    print(f"Loading success rollouts from {success_path}...")
    success_rollouts = load_rollouts(success_path)
    print(f"Loaded {len(success_rollouts)} success rollouts")
    
    print(f"Loading failure rollouts from {failure_path}...")
    failure_rollouts = load_rollouts(failure_path)
    print(f"Loaded {len(failure_rollouts)} failure rollouts")
    
    all_rollouts = success_rollouts + failure_rollouts
    print(f"Total rollouts: {len(all_rollouts)}")
    
    # Extract per-step data
    dataset = []
    n_with_labels = 0
    n_without_labels = 0
    
    for rollout in tqdm(all_rollouts, desc="Processing rollouts"):
        steps = get_steps_from_rollout(rollout)
        if not steps:
            continue
        
        success = rollout.get('success', False)
        collision_occurred = rollout.get('collision_occurred', False)
        collision_step = rollout.get('collision_step')
        
        for i, step in enumerate(steps):
            hidden_state = np.asarray(step.get('hidden_state', np.zeros(1)), dtype=np.float32)
            action = np.asarray(step.get('action', np.zeros(7)), dtype=np.float32)
            
            # Check if labels exist (can be in step directly or in step['labels'])
            labels = step.get('labels', {})
            if 'per_dim_risk' in step:
                risk_label = np.asarray(step['per_dim_risk'], dtype=np.float32)
                time_to_failure = step.get('time_to_failure', -1)
                fail_within_k = step.get('fail_within_k', False)
                n_with_labels += 1
            elif 'per_dim_risk' in labels:
                risk_label = np.asarray(labels['per_dim_risk'], dtype=np.float32)
                time_to_failure = labels.get('time_to_failure', -1)
                fail_within_k = labels.get('fail_within_k', False)
                n_with_labels += 1
            elif require_labels:
                # Skip if labels required but not present
                continue
            else:
                # Generate heuristic labels
                # For failures: high risk near the end, based on action magnitude
                # For successes: low risk throughout
                if not success:
                    # Failure rollouts: risk increases as we approach the end
                    total_steps = len(steps)
                    steps_from_end = total_steps - i
                    
                    # Risk is higher in last 20% of failed episodes
                    if steps_from_end <= max(10, total_steps * 0.2):
                        # Risk proportional to action magnitude and proximity to failure
                        action_mag = np.abs(action)
                        proximity_weight = 1.0 - (steps_from_end / max(10, total_steps * 0.2))
                        risk_label = (action_mag * proximity_weight).astype(np.float32)
                        # Normalize to [0, 1]
                        risk_label = np.clip(risk_label / (risk_label.max() + 1e-8), 0, 1)
                        time_to_failure = steps_from_end
                        fail_within_k = steps_from_end <= 10
                    else:
                        # Lower risk earlier in episode
                        risk_label = (np.abs(action) * 0.1).astype(np.float32)
                        risk_label = np.clip(risk_label, 0, 1)
                        time_to_failure = steps_from_end
                        fail_within_k = False
                else:
                    # Success rollouts: very low risk
                    risk_label = np.zeros(7, dtype=np.float32)
                    time_to_failure = -1
                    fail_within_k = False
                n_without_labels += 1
            
            dataset.append({
                'hidden_state': hidden_state,
                'risk_label': risk_label,
                'action': action,
                'time_to_failure': int(time_to_failure),
                'fail_within_k': int(fail_within_k),
                'success': int(success),
                'collision': int(step.get('collision', False)),
            })
    
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  With labels: {n_with_labels}")
    print(f"  Without labels (heuristic): {n_without_labels}")
    
    # Compute statistics
    risk_labels = np.array([d['risk_label'] for d in dataset])
    print(f"\nRisk label statistics:")
    print(f"  Mean per dimension: {risk_labels.mean(axis=0)}")
    print(f"  Std per dimension: {risk_labels.std(axis=0)}")
    print(f"  Non-zero samples: {(risk_labels > 0).any(axis=1).sum()} / {len(dataset)}")
    
    # Save dataset
    output_data = {
        'dataset': dataset,
        'metadata': {
            'num_samples': len(dataset),
            'num_success_rollouts': len(success_rollouts),
            'num_failure_rollouts': len(failure_rollouts),
            'num_with_labels': n_with_labels,
            'num_without_labels': n_without_labels,
            'hidden_dim': dataset[0]['hidden_state'].shape[0] if dataset else 0,
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n✓ Saved dataset to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from rollouts")
    parser.add_argument("--success", type=str, required=True, help="Path to success rollouts")
    parser.add_argument("--failure", type=str, required=True, help="Path to failure rollouts")
    parser.add_argument("--output", type=str, required=True, help="Output dataset path")
    parser.add_argument("--require-labels", action="store_true", help="Require per_dim_risk labels")
    args = parser.parse_args()
    
    prepare_dataset(
        Path(args.success),
        Path(args.failure),
        Path(args.output),
        require_labels=args.require_labels
    )


if __name__ == "__main__":
    main()
