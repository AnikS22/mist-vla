"""
Improved labeling using multiple signals:
1. Action magnitude patterns
2. Failure proximity
3. Action direction vs. failure direction
4. Velocity/acceleration patterns
5. Collision geometry (when available)
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List
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


def compute_action_velocity(actions: np.ndarray) -> np.ndarray:
    """Compute action velocity (change between steps)."""
    if len(actions) < 2:
        return np.zeros_like(actions)
    
    velocity = np.diff(actions, axis=0, prepend=actions[0:1])
    return velocity


def compute_action_acceleration(actions: np.ndarray) -> np.ndarray:
    """Compute action acceleration (change in velocity)."""
    velocity = compute_action_velocity(actions)
    if len(velocity) < 2:
        return np.zeros_like(velocity)
    
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    return acceleration


def label_with_multi_signal(
    rollout: Dict,
    k_fail: int = 10,
    k_collision: int = 10,
    use_action_patterns: bool = True,
    use_velocity: bool = True,
    use_failure_direction: bool = True,
) -> None:
    """
    Label rollout using multiple signals.
    
    Signals used:
    1. Failure proximity (time-to-failure)
    2. Action magnitude patterns
    3. Action velocity/acceleration
    4. Failure direction (if we can infer it)
    5. Collision geometry (if available)
    """
    steps = get_steps_from_rollout(rollout)
    if not steps:
        return
    
    success = rollout.get('success', False)
    failure_step = None if success else len(steps) - 1
    collision_step = rollout.get('collision_step')
    
    # Extract all actions
    actions = np.array([step.get('action', np.zeros(7)) for step in steps], dtype=np.float32)
    
    # Compute velocity and acceleration
    if use_velocity and len(actions) > 1:
        velocities = compute_action_velocity(actions)
        accelerations = compute_action_acceleration(actions)
    else:
        velocities = np.zeros_like(actions)
        accelerations = np.zeros_like(actions)
    
    # Compute action statistics over trajectory
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0)
    action_max = np.abs(actions).max(axis=0)
    
    # Infer failure direction from last N steps before failure
    failure_direction = None
    if failure_step is not None and use_failure_direction:
        lookback = min(10, failure_step)
        if lookback > 0:
            # Average action direction in steps leading to failure
            failure_actions = actions[max(0, failure_step - lookback):failure_step]
            failure_direction = np.mean(failure_actions, axis=0)
            failure_direction = failure_direction / (np.linalg.norm(failure_direction) + 1e-8)
    
    for i, step in enumerate(steps):
        action = actions[i]
        action_mag = np.abs(action)
        
        # Time-to-failure signal
        time_to_failure = -1 if failure_step is None else max(failure_step - i, 0)
        time_to_collision = -1 if collision_step is None else max(collision_step - i, 0)
        
        fail_within_k = 1 if (failure_step is not None and time_to_failure <= k_fail) else 0
        collision_within_k = 1 if (collision_step is not None and time_to_collision <= k_collision) else 0
        
        # Initialize per-dimension risk
        per_dim_risk = np.zeros(7, dtype=np.float32)
        
        # Signal 1: Failure proximity (stronger near failure)
        if fail_within_k:
            proximity_weight = 1.0 - (time_to_failure / k_fail) if time_to_failure > 0 else 1.0
            
            # Signal 2: Action magnitude (larger actions = higher risk when near failure)
            action_mag_normalized = action_mag / (action_max + 1e-8)
            per_dim_risk += proximity_weight * action_mag_normalized * 0.5
            
            # Signal 3: Action direction alignment with failure direction
            if failure_direction is not None:
                # Risk is higher if current action aligns with failure direction
                alignment = np.abs(action * failure_direction)
                alignment_normalized = alignment / (np.abs(action).max() + 1e-8)
                per_dim_risk += proximity_weight * alignment_normalized * 0.3
            
            # Signal 4: Velocity patterns (sudden changes = risk)
            if use_velocity and i > 0:
                vel_mag = np.abs(velocities[i])
                vel_normalized = vel_mag / (np.abs(velocities).max(axis=0) + 1e-8)
                per_dim_risk += proximity_weight * vel_normalized * 0.2
            
            # Signal 5: Acceleration (rapid changes = risk)
            if use_velocity and i > 1:
                accel_mag = np.abs(accelerations[i])
                accel_normalized = accel_mag / (np.abs(accelerations).max(axis=0) + 1e-8)
                per_dim_risk += proximity_weight * accel_normalized * 0.1
        
        # Signal 6: Collision geometry (if available)
        collision_normal = step.get('collision_normal')
        robot_state = step.get('robot_state', {})
        if collision_normal is not None and robot_state and 'eef_quat' in robot_state:
            # Transform collision normal to EEF frame
            from scripts.label_failure_data import _rotate_world_to_eef
            world_n = np.array(collision_normal, dtype=np.float32)
            eef_quat = np.array(robot_state['eef_quat'], dtype=np.float32)
            eef_n = _rotate_world_to_eef(world_n, eef_quat)
            
            # High risk for dimensions aligned with collision normal
            for axis in range(3):
                if abs(eef_n[axis]) >= 0.3:
                    per_dim_risk[axis] = max(per_dim_risk[axis], abs(eef_n[axis]))
        
        # Signal 7: Action patterns (deviations from normal)
        if use_action_patterns:
            # Risk if action is unusually large compared to trajectory mean
            deviation = (action_mag - action_mean) / (action_std + 1e-8)
            deviation_risk = np.clip(deviation / 2.0, 0, 1)  # Normalize to [0, 1]
            per_dim_risk += deviation_risk * 0.2 if fail_within_k else deviation_risk * 0.05
        
        # Normalize to [0, 1]
        per_dim_risk = np.clip(per_dim_risk, 0, 1)
        
        # If no failure signal, set to zero
        if not fail_within_k and not collision_within_k:
            per_dim_risk = np.zeros(7, dtype=np.float32)
        
        # Add labels to step
        if 'labels' not in step:
            step['labels'] = {}
        
        step['labels'].update({
            'time_to_failure': int(time_to_failure),
            'time_to_collision': int(time_to_collision),
            'fail_within_k': int(fail_within_k),
            'collision_within_k': int(collision_within_k),
            'per_dim_risk': per_dim_risk.tolist(),
        })


def main():
    parser = argparse.ArgumentParser(description="Improve labels using multiple signals")
    parser.add_argument("--input", type=str, required=True, help="Input rollouts pickle")
    parser.add_argument("--output", type=str, required=True, help="Output labeled pickle")
    parser.add_argument("--k-fail", type=int, default=10, help="Look-ahead window for failure")
    parser.add_argument("--k-collision", type=int, default=10, help="Look-ahead window for collision")
    parser.add_argument("--use-action-patterns", action="store_true", help="Use action pattern analysis")
    parser.add_argument("--use-velocity", action="store_true", help="Use velocity/acceleration signals")
    parser.add_argument("--use-failure-direction", action="store_true", help="Use failure direction inference")
    args = parser.parse_args()
    
    print(f"Loading rollouts from {args.input}...")
    rollouts = load_rollouts(Path(args.input))
    print(f"Loaded {len(rollouts)} rollouts")
    
    print("\nLabeling rollouts with multi-signal approach...")
    for rollout in tqdm(rollouts, desc="Labeling"):
        label_with_multi_signal(
            rollout,
            k_fail=args.k_fail,
            k_collision=args.k_collision,
            use_action_patterns=args.use_action_patterns,
            use_velocity=args.use_velocity,
            use_failure_direction=args.use_failure_direction,
        )
    
    # Save labeled rollouts
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(rollouts, f)
    
    # Statistics
    total_steps = 0
    steps_with_risk = 0
    risk_by_dim = np.zeros(7)
    
    for rollout in rollouts:
        steps = get_steps_from_rollout(rollout)
        for step in steps:
            if 'labels' in step and 'per_dim_risk' in step['labels']:
                total_steps += 1
                risk = np.array(step['labels']['per_dim_risk'])
                if risk.sum() > 0:
                    steps_with_risk += 1
                risk_by_dim += risk
    
    print(f"\nLabeling statistics:")
    print(f"  Total steps: {total_steps}")
    print(f"  Steps with risk: {steps_with_risk} ({100*steps_with_risk/total_steps:.1f}%)")
    print(f"  Mean risk per dimension: {risk_by_dim / total_steps}")
    print(f"  Non-zero risk samples: {steps_with_risk} / {total_steps}")
    
    print(f"\n✓ Saved labeled rollouts to {args.output}")


if __name__ == "__main__":
    main()
