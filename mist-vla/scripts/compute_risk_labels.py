#!/usr/bin/env python3
"""
Phase 1.4: Compute per-dimension risk labels.

This script processes the collected trajectories and computes per-dimension
risk labels based on collision geometry:

For each timestep within K steps of a collision:
    risk_i = max(0, action_i * v_i)

where v is the unit vector from end-effector to collision point.

This creates directional risk labels:
- If moving right (action[0] > 0) toward collision (v[0] > 0) → high risk
- If moving left (action[0] < 0) away from collision → zero risk
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_risk_labels(trajectory, K=10, alpha=1.0):
    """
    Compute per-dimension risk labels for a trajectory.

    Args:
        trajectory: Trajectory dictionary with 'steps' list
        K: Number of steps before collision to label as risky
        alpha: Scaling factor for risk labels

    Returns:
        List of risk vectors (one per step)
    """
    steps = trajectory['steps']
    risk_labels = []

    # Find collision timestep and position
    collision_step = trajectory.get('collision_step', None)
    collision_pos = None

    if collision_step is not None and collision_step < len(steps):
        collision_pos = steps[collision_step]['collision_pos']

    # If no collision, all risk labels are zero
    if collision_pos is None:
        return [np.zeros(7) for _ in steps]

    # Compute risk for each step
    for t, step in enumerate(steps):
        # Default: no risk
        risk = np.zeros(7)

        # Only compute risk within K steps before collision
        if collision_step is not None and t <= collision_step <= t + K:
            # Get end-effector position at this step
            ee_pos = step['ee_pos']

            # Direction from EE to collision point
            direction = collision_pos - ee_pos
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 1e-6:  # Avoid division by zero
                direction_unit = direction / direction_norm

                # Get action at this step
                action = step['action']

                # Compute per-dimension risk
                # risk_i = max(0, action_i * direction_i) * alpha
                # This means:
                # - If moving toward collision → positive risk
                # - If moving away from collision → zero risk
                for i in range(min(3, len(action))):  # Only x, y, z dimensions
                    risk[i] = max(0, action[i] * direction_unit[i]) * alpha

                # For rotation dimensions (roll, pitch, yaw), we don't have
                # geometric direction, so set to zero for now
                # Could be extended with rotation collision analysis

        risk_labels.append(risk)

    return risk_labels


def process_trajectories(input_path, output_path, K=10, alpha=1.0):
    """
    Process all trajectories and compute risk labels.

    Args:
        input_path: Path to collected rollouts (from Phase 1.3)
        output_path: Path to save labeled data
        K: Number of steps before collision to label
        alpha: Scaling factor for risk labels
    """
    print("=" * 60)
    print("Phase 1.4: Compute Per-Dimension Risk Labels")
    print("=" * 60)

    # Load data
    print(f"\n[1/4] Loading data from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    trajectories = data['trajectories']
    metadata = data['metadata']
    print(f"  ✓ Loaded {len(trajectories)} trajectories")
    print(f"  ✓ Collisions: {metadata['collision_count']}")

    # Compute risk labels
    print(f"\n[2/4] Computing risk labels (K={K}, alpha={alpha})...")
    print("  Formula: risk_i = max(0, action_i * direction_i) * alpha")
    print("  where direction = (collision_pos - ee_pos) / ||...||")

    labeled_trajectories = []
    total_risky_steps = 0
    risk_stats = {
        'x': [], 'y': [], 'z': [],
        'roll': [], 'pitch': [], 'yaw': [], 'gripper': []
    }

    for traj in tqdm(trajectories, desc="Computing labels"):
        risk_labels = compute_risk_labels(traj, K=K, alpha=alpha)

        # Add risk labels to trajectory
        traj_with_labels = traj.copy()
        for step, risk in zip(traj_with_labels['steps'], risk_labels):
            step['risk_label'] = risk

            # Track statistics
            if risk.sum() > 0:
                total_risky_steps += 1
                risk_stats['x'].append(risk[0])
                risk_stats['y'].append(risk[1])
                risk_stats['z'].append(risk[2])
                risk_stats['roll'].append(risk[3])
                risk_stats['pitch'].append(risk[4])
                risk_stats['yaw'].append(risk[5])
                risk_stats['gripper'].append(risk[6])

        labeled_trajectories.append(traj_with_labels)

    # Print statistics
    print("\n[3/4] Risk Label Statistics:")
    total_steps = sum(len(t['steps']) for t in labeled_trajectories)
    print(f"  Total steps: {total_steps}")
    print(f"  Risky steps: {total_risky_steps} ({total_risky_steps/total_steps:.2%})")

    print("\n  Per-dimension risk (mean of non-zero values):")
    for dim_name, risks in risk_stats.items():
        if len(risks) > 0:
            mean_risk = np.mean(risks)
            max_risk = np.max(risks)
            print(f"    {dim_name:8s}: mean={mean_risk:.4f}, max={max_risk:.4f}, count={len(risks)}")
        else:
            print(f"    {dim_name:8s}: (no risky steps)")

    # Create dataset format for training
    print("\n[4/4] Creating training dataset...")
    dataset = []

    for traj in labeled_trajectories:
        for step in traj['steps']:
            sample = {
                'hidden_state': step['hidden_state'],
                'action': step['action'],
                'risk_label': step['risk_label'],
                'collision': step['collision'],
                'instruction': traj['instruction'],
            }
            dataset.append(sample)

    print(f"  ✓ Created dataset with {len(dataset)} samples")

    # Save labeled data
    print(f"\n[5/5] Saving labeled data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        'trajectories': labeled_trajectories,
        'dataset': dataset,
        'metadata': {
            **metadata,
            'K': K,
            'alpha': alpha,
            'total_steps': total_steps,
            'risky_steps': total_risky_steps,
            'risk_rate': total_risky_steps / total_steps,
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"  ✓ Data saved to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("✅ Phase 1.4 Complete - Phase 1 Done!")
    print("=" * 60)
    print("\nData collection complete! Next steps:")
    print("  1. Create dataset for training (Phase 2.1)")
    print("  2. Train per-dimension risk predictor (Phase 2.2)")
    print("\nNext command:")
    print("  python scripts/create_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="Compute per-dimension risk labels")
    parser.add_argument('--input', type=str, default='data/phase1/rollouts.pkl',
                        help='Input path for collected rollouts')
    parser.add_argument('--output', type=str, default='data/phase1/labeled_data.pkl',
                        help='Output path for labeled data')
    parser.add_argument('--K', type=int, default=10,
                        help='Number of steps before collision to label as risky')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Scaling factor for risk labels')

    args = parser.parse_args()

    process_trajectories(
        input_path=args.input,
        output_path=args.output,
        K=args.K,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
