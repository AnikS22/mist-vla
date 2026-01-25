"""
Collect successful and failed rollouts for training the failure detector.
Uses current LIBERO API (OffScreenRenderEnv + task.language).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Some diffusers versions expect torch.xpu to exist (even if unused).
if not hasattr(torch, "xpu"):
    class _XPU:
        def empty_cache(self):
            return None

        def device_count(self):
            return 0

        def is_available(self):
            return False

        def manual_seed(self, *args, **kwargs):
            return None

        def manual_seed_all(self, *args, **kwargs):
            return None

        def device(self, *args, **kwargs):
            return None

        def current_device(self):
            return 0

    torch.xpu = _XPU()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.models.vla_wrapper import create_vla_wrapper
from src.data_collection.collision_detection import CollisionDetector


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            return Image.fromarray(image)
        return Image.fromarray((image * 255).astype(np.uint8))
    if torch.is_tensor(image):
        return Image.fromarray(image.detach().cpu().numpy())
    return image


def _check_success(env, info):
    if hasattr(env, "check_success"):
        try:
            return bool(env.check_success())
        except Exception:
            pass
    if isinstance(info, dict):
        if info.get("success") is True:
            return True
        if info.get("is_success") is True:
            return True
    return False


def _get_robot_state(env):
    sim = None
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        sim = env.env.sim
    elif hasattr(env, "sim"):
        sim = env.sim
    if sim is None:
        return {}

    state = {
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
    }

    model = sim.model
    site_id = None
    for name in ("gripper0_grip_site", "robot0_eef", "eef", "ee_site", "right_gripper"):
        try:
            site_id = model.site_name2id(name)
            break
        except Exception:
            continue

    if site_id is not None:
        state["eef_pos"] = sim.data.site_xpos[site_id].copy()
        if hasattr(sim.data, "site_xquat"):
            state["eef_quat"] = sim.data.site_xquat[site_id].copy()
        if hasattr(sim.data, "site_xvelp"):
            state["eef_vel"] = sim.data.site_xvelp[site_id].copy()
        if hasattr(sim.data, "site_xvelr"):
            state["eef_ang_vel"] = sim.data.site_xvelr[site_id].copy()

    return state


def collect_rollouts(
    env_name: str,
    policy,
    n_success: int = 100,
    n_failure: int = 100,
    max_steps: int = 200,
    save_dir: str = "data/rollouts",
    camera_height: int = 128,
    camera_width: int = 128,
    max_attempts_per_task: int = 200,
    seed: int = 0,
    shuffle_tasks: bool = True,
    checkpoint_every: int = 25,
    disable_perturbations: bool = False,
):
    """
    Collect both successful and failed rollouts.

    Failure generation strategies:
    1. Natural failures from policy
    2. Injected perturbations (translation, rotation, no-op)
    """

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[env_name]()
    num_tasks = task_suite.get_num_tasks()
    task_ids = list(range(num_tasks))
    if shuffle_tasks:
        rng.shuffle(task_ids)

    success_rollouts = []
    failure_rollouts = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        instruction = task.language
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=camera_height,
            camera_widths=camera_width,
        )
        init_states = task_suite.get_task_init_states(task_id)
        env.reset()
        env.set_init_state(init_states[0])

        print(f"\nTask {task_id}: {instruction}", flush=True)

        # Collect successful rollouts (bounded per task to avoid getting stuck)
        attempts = 0
        while len(success_rollouts) < n_success and attempts < max_attempts_per_task:
            attempts += 1
            print(
                f"[progress] task={task_id} success_attempt={attempts} "
                f"successes={len(success_rollouts)}/{n_success} "
                f"failures={len(failure_rollouts)}/{n_failure}",
                flush=True,
            )
            rollout = run_episode(env, policy, instruction, max_steps, init_states, rng)

            if rollout["success"]:
                success_rollouts.append(rollout)
                print(f"Success: {len(success_rollouts)}/{n_success}", flush=True)
                if len(success_rollouts) % checkpoint_every == 0:
                    _write_checkpoint(save_path, success_rollouts, failure_rollouts)

        # Collect natural failures (bounded per task)
        attempts = 0
        while len(failure_rollouts) < n_failure // 2 and attempts < max_attempts_per_task:
            attempts += 1
            print(
                f"[progress] task={task_id} collecting natural failures "
                f"{len(failure_rollouts)}/{n_failure}",
                flush=True,
            )
            rollout = run_episode(env, policy, instruction, max_steps, init_states, rng)
            if not rollout["success"]:
                failure_rollouts.append(rollout)
                print(f"Natural failure: {len(failure_rollouts)}/{n_failure}", flush=True)
                if len(failure_rollouts) % checkpoint_every == 0:
                    _write_checkpoint(save_path, success_rollouts, failure_rollouts)

        # Collect injected failures (bounded per task) unless disabled
        if not disable_perturbations:
            attempts = 0
            while len(failure_rollouts) < n_failure and attempts < max_attempts_per_task:
                attempts += 1
                print(
                    f"[progress] task={task_id} collecting injected failures "
                    f"{len(failure_rollouts)}/{n_failure}",
                    flush=True,
                )
                rollout = run_episode_with_perturbation(
                    env, policy, instruction, max_steps, init_states, rng
                )
                if not rollout["success"]:
                    failure_rollouts.append(rollout)
                    print(f"Injected failure: {len(failure_rollouts)}/{n_failure}", flush=True)
                    if len(failure_rollouts) % checkpoint_every == 0:
                        _write_checkpoint(save_path, success_rollouts, failure_rollouts)

        env.close()

        if len(success_rollouts) >= n_success and len(failure_rollouts) >= n_failure:
            break

    with open(save_path / "success_rollouts.pkl", "wb") as f:
        pickle.dump(success_rollouts, f)

    with open(save_path / "failure_rollouts.pkl", "wb") as f:
        pickle.dump(failure_rollouts, f)

    print(
        f"\nSaved {len(success_rollouts)} success and {len(failure_rollouts)} "
        f"failure rollouts to {save_path}",
        flush=True,
    )

    return success_rollouts, failure_rollouts


def run_episode(env, policy, instruction, max_steps, init_states, rng):
    obs = env.reset()
    if init_states is not None and len(init_states) > 0:
        env.set_init_state(init_states[rng.integers(0, len(init_states))])
    detector = CollisionDetector(env)
    trajectory = {
        "observations": [],
        "actions": [],
        "features": [],
        "rewards": [],
        "robot_states": [],
        "success": False,
        "collision_occurred": False,
        "collision_steps": 0,
        "instruction": instruction,
    }

    for step in range(max_steps):
        image = obs.get("agentview_image")
        if image is None:
            image = obs.get("image")
        image = _to_pil(image)

        action, features = policy.get_action_with_features(image, instruction, obs=obs)

        if torch.is_tensor(action):
            action = action.cpu().numpy()
        if torch.is_tensor(features):
            features = features.float().cpu().numpy()

        trajectory["observations"].append(obs)
        trajectory["actions"].append(action)
        trajectory["features"].append(features)
        trajectory["robot_states"].append(_get_robot_state(env))

        has_collision, _ = detector.check_collision()
        if has_collision:
            trajectory["collision_occurred"] = True
            trajectory["collision_steps"] += 1

        obs, reward, done, info = env.step(action)
        trajectory["rewards"].append(reward)

        if _check_success(env, info):
            trajectory["success"] = True
            break
        if done:
            break

    return trajectory


def run_episode_with_perturbation(env, policy, instruction, max_steps, init_states, rng):
    obs = env.reset()
    if init_states is not None and len(init_states) > 0:
        env.set_init_state(init_states[rng.integers(0, len(init_states))])
    detector = CollisionDetector(env)
    trajectory = {
        "observations": [],
        "actions": [],
        "features": [],
        "rewards": [],
        "robot_states": [],
        "success": False,
        "collision_occurred": False,
        "collision_steps": 0,
        "instruction": instruction,
        "perturbation_type": None,
        "perturbation_step": None,
    }

    perturbation_type = rng.choice(["translation", "rotation", "noop"])
    upper = max(1, max_steps // 2)
    lower = 0 if upper <= 10 else 10
    perturbation_step = rng.integers(lower, upper)

    trajectory["perturbation_type"] = perturbation_type
    trajectory["perturbation_step"] = perturbation_step

    for step in range(max_steps):
        image = obs.get("agentview_image")
        if image is None:
            image = obs.get("image")
        image = _to_pil(image)

        action, features = policy.get_action_with_features(image, instruction, obs=obs)

        if torch.is_tensor(action):
            action = action.cpu().numpy()
        if torch.is_tensor(features):
            features = features.float().cpu().numpy()

        if step >= perturbation_step and step < perturbation_step + 10:
            action = apply_perturbation(action, perturbation_type)

        trajectory["observations"].append(obs)
        trajectory["actions"].append(action)
        trajectory["features"].append(features)
        trajectory["robot_states"].append(_get_robot_state(env))

        has_collision, _ = detector.check_collision()
        if has_collision:
            trajectory["collision_occurred"] = True
            trajectory["collision_steps"] += 1

        obs, reward, done, info = env.step(action)
        trajectory["rewards"].append(reward)

        if _check_success(env, info):
            trajectory["success"] = True
            break
        if done:
            break

    return trajectory


def apply_perturbation(action, perturbation_type):
    if perturbation_type == "translation":
        offset = np.random.uniform(-0.1, 0.1, size=3)
        action[:3] += offset
    elif perturbation_type == "rotation":
        offset = np.random.uniform(-0.2, 0.2, size=3)
        action[3:6] += offset
    elif perturbation_type == "noop":
        action = np.zeros_like(action)

    return np.clip(action, -1, 1)


def _write_checkpoint(save_path: Path, success_rollouts, failure_rollouts) -> None:
    ckpt_path = save_path / "rollouts_checkpoint.pkl"
    payload = {
        "success_rollouts": success_rollouts,
        "failure_rollouts": failure_rollouts,
        "success_count": len(success_rollouts),
        "failure_count": len(failure_rollouts),
    }
    with open(ckpt_path, "wb") as f:
        pickle.dump(payload, f)
    print(
        f"[checkpoint] saved {len(success_rollouts)} success / "
        f"{len(failure_rollouts)} failure rollouts",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--n_success", type=int, default=100)
    parser.add_argument("--n_failure", type=int, default=100)
    parser.add_argument("--save_dir", default="data/rollouts")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--max-attempts-per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle-tasks", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--model-type", default="openvla", choices=["openvla", "openvla_oft"])
    parser.add_argument("--model-name", default="openvla/openvla-7b")
    parser.add_argument("--no-perturbations", action="store_true")
    args = parser.parse_args()

    policy = create_vla_wrapper(args.model_type, args.model_name)

    collect_rollouts(
        args.env,
        policy,
        args.n_success,
        args.n_failure,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        max_attempts_per_task=args.max_attempts_per_task,
        seed=args.seed,
        shuffle_tasks=not args.no_shuffle_tasks,
        checkpoint_every=args.checkpoint_every,
        disable_perturbations=args.no_perturbations,
    )
