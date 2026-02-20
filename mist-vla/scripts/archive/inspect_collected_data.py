"""
Inspect collected rollout data to see what internal states are being captured.
"""
import argparse
import pickle
import numpy as np
from pathlib import Path


def inspect_rollout(rollout, rollout_idx=0):
    """Print detailed information about a single rollout."""
    print(f"\n{'='*80}")
    print(f"ROLLOUT {rollout_idx}")
    print(f"{'='*80}")
    
    print(f"\nMetadata:")
    print(f"  Success: {rollout.get('success', 'N/A')}")
    print(f"  Collision occurred: {rollout.get('collision_occurred', 'N/A')}")
    print(f"  Collision steps: {rollout.get('collision_steps', 'N/A')}")
    print(f"  Collision step: {rollout.get('collision_step', 'N/A')}")
    print(f"  Instruction: {rollout.get('instruction', 'N/A')}")
    print(f"  Task ID: {rollout.get('task_id', 'N/A')}")
    
    # Check what data is available
    print(f"\nData Arrays:")
    print(f"  Actions: {len(rollout.get('actions', []))} steps")
    print(f"  Features (hidden states): {len(rollout.get('features', []))} steps")
    print(f"  Robot states: {len(rollout.get('robot_states', []))} steps")
    print(f"  Rewards: {len(rollout.get('rewards', []))} steps")
    print(f"  Steps (detailed): {len(rollout.get('steps', []))} steps")
    
    # Show sample data from first step
    if rollout.get('steps') and len(rollout['steps']) > 0:
        first_step = rollout['steps'][0]
        print(f"\nFirst Step Details:")
        print(f"  Action shape: {np.array(first_step.get('action', [])).shape}")
        print(f"  Action values: {np.array(first_step.get('action', []))}")
        print(f"  Hidden state shape: {np.array(first_step.get('hidden_state', [])).shape}")
        print(f"  Hidden state dtype: {np.array(first_step.get('hidden_state', [])).dtype}")
        print(f"  Hidden state sample (first 10 dims): {np.array(first_step.get('hidden_state', []))[:10]}")
        print(f"  Collision: {first_step.get('collision', 'N/A')}")
        
        robot_state = first_step.get('robot_state', {})
        if robot_state:
            print(f"  Robot state keys: {list(robot_state.keys())}")
            for key, val in robot_state.items():
                if isinstance(val, np.ndarray):
                    print(f"    {key}: shape={val.shape}, dtype={val.dtype}, sample={val.flat[:5] if val.size > 0 else 'empty'}")
                else:
                    print(f"    {key}: {val}")
    
    # Show sample from middle step
    if rollout.get('steps') and len(rollout['steps']) > 5:
        mid_idx = len(rollout['steps']) // 2
        mid_step = rollout['steps'][mid_idx]
        print(f"\nMiddle Step ({mid_idx}) Details:")
        print(f"  Action: {np.array(mid_step.get('action', []))}")
        print(f"  Hidden state shape: {np.array(mid_step.get('hidden_state', [])).shape}")
        print(f"  Collision: {mid_step.get('collision', 'N/A')}")
        if mid_step.get('collision'):
            print(f"  Collision pos: {mid_step.get('collision_pos', 'N/A')}")
            print(f"  Collision normal: {mid_step.get('collision_normal', 'N/A')}")
            print(f"  Collision geoms: {mid_step.get('collision_geoms', 'N/A')}")
    
    # Statistics
    if rollout.get('features'):
        features_array = np.array(rollout['features'])
        print(f"\nHidden States Statistics:")
        print(f"  Shape: {features_array.shape}")
        print(f"  Dtype: {features_array.dtype}")
        print(f"  Mean: {features_array.mean():.6f}")
        print(f"  Std: {features_array.std():.6f}")
        print(f"  Min: {features_array.min():.6f}")
        print(f"  Max: {features_array.max():.6f}")
        print(f"  Non-zero elements: {np.count_nonzero(features_array)} / {features_array.size}")
    
    if rollout.get('actions'):
        actions_array = np.array(rollout['actions'])
        print(f"\nActions Statistics:")
        print(f"  Shape: {actions_array.shape}")
        print(f"  Dtype: {actions_array.dtype}")
        print(f"  Mean per dimension: {actions_array.mean(axis=0)}")
        print(f"  Std per dimension: {actions_array.std(axis=0)}")
        print(f"  Min per dimension: {actions_array.min(axis=0)}")
        print(f"  Max per dimension: {actions_array.max(axis=0)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect collected rollout data")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing rollout pickle files")
    parser.add_argument("--file", type=str, default="success_rollouts.pkl", help="Pickle file to inspect (success_rollouts.pkl or failure_rollouts.pkl)")
    parser.add_argument("--n-rollouts", type=int, default=3, help="Number of rollouts to inspect")
    parser.add_argument("--checkpoint", action="store_true", help="Check checkpoint files instead")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.checkpoint:
        file_path = data_dir / f"{args.file.replace('.pkl', '_partial.pkl')}"
    else:
        file_path = data_dir / args.file
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Loading data from: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        rollouts = data
    elif isinstance(data, dict) and 'trajectories' in data:
        rollouts = data['trajectories']
    else:
        print(f"Unknown data format. Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        return
    
    print(f"\nLoaded {len(rollouts)} rollouts")
    
    # Inspect first N rollouts
    n_inspect = min(args.n_rollouts, len(rollouts))
    for i in range(n_inspect):
        inspect_rollout(rollouts[i], rollout_idx=i)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    successes = sum(1 for r in rollouts if r.get('success', False))
    failures = len(rollouts) - successes
    collisions = sum(1 for r in rollouts if r.get('collision_occurred', False))
    
    print(f"\nTotal rollouts: {len(rollouts)}")
    print(f"  Successes: {successes} ({successes/len(rollouts)*100:.1f}%)")
    print(f"  Failures: {failures} ({failures/len(rollouts)*100:.1f}%)")
    print(f"  With collisions: {collisions} ({collisions/len(rollouts)*100:.1f}%)")
    
    # Hidden state dimensions
    if rollouts and rollouts[0].get('features'):
        hidden_dims = [len(r['features'][0]) if r.get('features') and len(r['features']) > 0 else 0 for r in rollouts]
        if hidden_dims and hidden_dims[0] > 0:
            print(f"\nHidden state dimensions: {hidden_dims[0]} (consistent: {all(d == hidden_dims[0] for d in hidden_dims)})")
    
    # Action dimensions
    if rollouts and rollouts[0].get('actions'):
        action_dims = [len(r['actions'][0]) if r.get('actions') and len(r['actions']) > 0 else 0 for r in rollouts]
        if action_dims and action_dims[0] > 0:
            print(f"Action dimensions: {action_dims[0]} (consistent: {all(d == action_dims[0] for d in action_dims)})")


if __name__ == "__main__":
    main()
