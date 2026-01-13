"""
Collect successful and failed rollouts for training the failure detector.
Based on FailSafe methodology for generating diverse failure cases.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Import LIBERO
import libero.libero.envs
from libero.libero import benchmark

# Import VLA
from src.models.vla_wrapper import create_vla_wrapper


def collect_rollouts(
    env_name: str,
    policy,
    n_success: int = 100,
    n_failure: int = 100,
    max_steps: int = 200,
    save_dir: str = "data/rollouts"
):
    """
    Collect both successful and failed rollouts.

    Failure generation strategies:
    1. Natural failures from policy
    2. Injected perturbations (translation, rotation, no-op)
    3. OOD scenarios (new objects, positions)
    """

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[env_name]()

    success_rollouts = []
    failure_rollouts = []

    for task_id in range(len(task_suite.tasks)):
        env = task_suite.make_env(task_id)
        instruction = task_suite.get_task_instruction(task_id)

        # Collect successful rollouts
        while len(success_rollouts) < n_success:
            rollout = run_episode(env, policy, instruction, max_steps)

            if rollout['success']:
                success_rollouts.append(rollout)
                print(f"Success: {len(success_rollouts)}/{n_success}")

        # Collect natural failures
        while len(failure_rollouts) < n_failure // 2:
            rollout = run_episode(env, policy, instruction, max_steps)

            if not rollout['success']:
                failure_rollouts.append(rollout)
                print(f"Natural failure: {len(failure_rollouts)}/{n_failure}")

        # Collect injected failures
        while len(failure_rollouts) < n_failure:
            rollout = run_episode_with_perturbation(
                env, policy, instruction, max_steps
            )

            if not rollout['success']:
                failure_rollouts.append(rollout)
                print(f"Injected failure: {len(failure_rollouts)}/{n_failure}")

    # Save rollouts
    with open(save_path / "success_rollouts.pkl", "wb") as f:
        pickle.dump(success_rollouts, f)

    with open(save_path / "failure_rollouts.pkl", "wb") as f:
        pickle.dump(failure_rollouts, f)

    print(f"Saved {len(success_rollouts)} success and {len(failure_rollouts)} failure rollouts")

    return success_rollouts, failure_rollouts


def run_episode(env, policy, instruction, max_steps):
    """Run a single episode and collect trajectory data."""

    obs = env.reset()
    trajectory = {
        'observations': [],
        'actions': [],
        'features': [],  # VLA latent features
        'rewards': [],
        'success': False,
        'instruction': instruction
    }

    for step in range(max_steps):
        image = obs['agentview_image']

        # Get action and features from policy
        action, features = policy.get_action_with_features(image, instruction)

        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['features'].append(features.cpu().numpy())

        obs, reward, done, info = env.step(action)
        trajectory['rewards'].append(reward)

        if done:
            trajectory['success'] = info.get('success', False)
            break

    return trajectory


def run_episode_with_perturbation(env, policy, instruction, max_steps):
    """Run episode with injected perturbations to create failures."""

    obs = env.reset()
    trajectory = {
        'observations': [],
        'actions': [],
        'features': [],
        'rewards': [],
        'success': False,
        'instruction': instruction,
        'perturbation_type': None,
        'perturbation_step': None
    }

    # Choose perturbation type and timing
    perturbation_type = np.random.choice(['translation', 'rotation', 'noop'])
    perturbation_step = np.random.randint(10, max_steps // 2)

    trajectory['perturbation_type'] = perturbation_type
    trajectory['perturbation_step'] = perturbation_step

    for step in range(max_steps):
        image = obs['agentview_image']
        action, features = policy.get_action_with_features(image, instruction)

        # Apply perturbation
        if step >= perturbation_step and step < perturbation_step + 10:
            action = apply_perturbation(action, perturbation_type)

        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['features'].append(features.cpu().numpy())

        obs, reward, done, info = env.step(action)
        trajectory['rewards'].append(reward)

        if done:
            trajectory['success'] = info.get('success', False)
            break

    return trajectory


def apply_perturbation(action, perturbation_type):
    """Apply perturbation to action."""

    if perturbation_type == 'translation':
        # Add random translation offset
        offset = np.random.uniform(-0.1, 0.1, size=3)
        action[:3] += offset

    elif perturbation_type == 'rotation':
        # Add random rotation offset
        offset = np.random.uniform(-0.2, 0.2, size=3)
        action[3:6] += offset

    elif perturbation_type == 'noop':
        # Zero out action
        action = np.zeros_like(action)

    return np.clip(action, -1, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--n_success", type=int, default=100)
    parser.add_argument("--n_failure", type=int, default=100)
    parser.add_argument("--save_dir", default="data/rollouts")
    args = parser.parse_args()

    # Load policy
    policy = create_vla_wrapper("openvla", "openvla/openvla-7b")

    collect_rollouts(
        args.env,
        policy,
        args.n_success,
        args.n_failure,
        save_dir=args.save_dir
    )
