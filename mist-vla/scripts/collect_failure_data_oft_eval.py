"""
Collect rollouts using the official OpenVLA-OFT LIBERO evaluation pipeline,
while logging MIST-VLA signals (actions, hidden states, collisions, robot state).
"""
import argparse
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

# Ensure torch.load maps to CPU when CUDA is unavailable.
if not torch.cuda.is_available():
    _orig_torch_load = torch.load

    def _cpu_torch_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = torch.device("cpu")
        return _orig_torch_load(*args, **kwargs)

    torch.load = _cpu_torch_load

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    normalize_proprio,
    prepare_images_for_vla,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    TASK_MAX_STEPS,
    prepare_observation,
    process_action,
)

from src.data_collection.collision_detection import CollisionDetector
from src.data_collection.hooks import HiddenStateCollector


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
    elif hasattr(env, "env") and hasattr(env.env, "env") and hasattr(env.env.env, "sim"):
        sim = env.env.env.sim
    elif hasattr(env, "_env") and hasattr(env._env, "sim"):
        sim = env._env.sim
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


# Use official prepare_observation and process_action from run_libero_eval.py


def _resolve_unnorm_key(cfg, vla):
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key in vla.norm_stats:
        cfg.unnorm_key = unnorm_key


def _write_checkpoint(save_dir, success_rollouts, failure_rollouts):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "success_rollouts_partial.pkl", "wb") as f:
        pickle.dump(success_rollouts, f)
    with open(save_path / "failure_rollouts_partial.pkl", "wb") as f:
        pickle.dump(failure_rollouts, f)
    print(
        f"[checkpoint] success={len(success_rollouts)} failure={len(failure_rollouts)}",
        flush=True,
    )


def _write_status(save_dir, message):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "status.txt", "a") as f:
        f.write(message + "\n")


def collect_rollouts(
    cfg,
    save_dir,
    n_success,
    n_failure,
    max_attempts_per_task,
    seed,
    checkpoint_every=25,
    store_observations=False,
    save_video=False,
    video_dir=None,
    video_fps=30,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    print("[status] loading VLA...", flush=True)
    _write_status(save_dir, "[status] loading VLA...")
    vla = get_vla(cfg)
    print("[status] VLA loaded", flush=True)
    _write_status(save_dir, "[status] VLA loaded")
    _resolve_unnorm_key(cfg, vla)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    collector = HiddenStateCollector(vla)
    collector.register_hooks()

    resize_size = get_image_resize_size(cfg)
    success_rollouts, failure_rollouts = [], []

    total_rollouts = 0
    for task_id in range(num_taskxplain s):
        task = task_suite.get_task(task_id)
        print(f"[status] init env task={task_id}", flush=True)
        _write_status(save_dir, f"[status] init env task={task_id}")
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        print(f"[status] env ready task={task_id}", flush=True)
        _write_status(save_dir, f"[status] env ready task={task_id}")
        init_states = task_suite.get_task_init_states(task_id)

        # Count per-task successes and failures
        task_successes = 0
        task_failures = 0
        attempts = 0
        episode_idx = 0

        # Collect n_success successes AND n_failure failures for THIS task
        while (task_successes < n_success or task_failures < n_failure) and attempts < max_attempts_per_task:
            attempts += 1
            initial_state = init_states[episode_idx % len(init_states)] if init_states is not None else None
            episode_idx += 1
            rollout = _run_episode(
                cfg,
                env,
                task_description,
                resize_size,
                vla,
                processor,
                action_head,
                proprio_projector,
                collector,
                initial_state=initial_state,
                store_observations=store_observations,
                save_video=save_video,
                video_dir=video_dir,
                video_fps=video_fps,
            )
            rollout["task_id"] = task_id

            if rollout["success"]:
                if task_successes < n_success:
                    success_rollouts.append(rollout)
                    task_successes += 1
            else:
                if task_failures < n_failure:
                    failure_rollouts.append(rollout)
                    task_failures += 1

            total_rollouts += 1
            print(
                f"[progress] task={task_id} total={total_rollouts} "
                f"task_succ={task_successes}/{n_success} task_fail={task_failures}/{n_failure} "
                f"global_succ={len(success_rollouts)} global_fail={len(failure_rollouts)}",
                flush=True,
            )
            if checkpoint_every and total_rollouts % checkpoint_every == 0:
                _write_checkpoint(save_dir, success_rollouts, failure_rollouts)

        env.close()

        # Free GPU/CPU cache between tasks to prevent OOM
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[status] task={task_id} done: {task_successes} success, {task_failures} failure "
            f"in {attempts} attempts",
            flush=True,
        )
        _write_status(save_dir, f"[status] task={task_id} done: {task_successes}S/{task_failures}F")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "success_rollouts.pkl", "wb") as f:
        pickle.dump(success_rollouts, f)
    with open(save_path / "failure_rollouts.pkl", "wb") as f:
        pickle.dump(failure_rollouts, f)

    return success_rollouts, failure_rollouts


def _run_episode(
    cfg,
    env,
    task_description,
    resize_size,
    vla,
    processor,
    action_head,
    proprio_projector,
    collector,
    initial_state=None,
    store_observations=False,
    save_video=False,
    video_dir=None,
    video_fps=30,
):
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    detector = CollisionDetector(env)

    trajectory = {
        "actions": [],
        "features": [],
        "rewards": [],
        "robot_states": [],
        "steps": [],
        "success": False,
        "collision_occurred": False,
        "collision_steps": 0,
        "collision_step": None,
        "instruction": task_description,
    }

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    t = 0
    last_features = None

    video_writer = None
    video_path = None
    if save_video and video_dir:
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        video_path = video_dir / f"episode_{stamp}.mp4"
        video_writer = imageio.get_writer(str(video_path), fps=video_fps)

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, raw_img = prepare_observation(obs, resize_size)

            if len(action_queue) == 0:
                collector.clear()
                with collector:
                    actions = get_action(
                        cfg,
                        vla,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=None,
                        use_film=cfg.use_film,
                    )
                action_queue.extend(actions)
                feats = collector.get_last_layer(pool="mean")
                last_features = None if feats is None else feats.detach().cpu().float().numpy()[0]
                features = last_features
            else:
                features = last_features

            action = np.asarray(action_queue.popleft(), dtype=np.float32)
            action = process_action(action, cfg.model_family)

            if features is None:
                features = np.zeros((1,), dtype=np.float32)

            if video_writer is not None and raw_img is not None:
                video_writer.append_data(raw_img)

            # Execute action in environment (matching official OFT eval order)
            obs, reward, done, info = env.step(action.tolist())
            
            # Check collision AFTER step (on new state)
            has_collision, pos, normal, geom1, geom2 = detector.check_collision_details()
            if has_collision:
                trajectory["collision_occurred"] = True
                trajectory["collision_steps"] += 1
                if trajectory["collision_step"] is None:
                    trajectory["collision_step"] = len(trajectory["actions"])

            # Get robot state AFTER step
            robot_state = _get_robot_state(env)
            
            # Log action and features (from BEFORE step, as they were used to generate the action)
            if store_observations:
                trajectory.setdefault("observations", []).append(obs)
            trajectory["actions"].append(action)
            trajectory["features"].append(features)
            trajectory["robot_states"].append(robot_state)
            trajectory["rewards"].append(reward)
            
            # Log step data
            trajectory["steps"].append(
                {
                    "action": np.array(action, dtype=np.float32),
                    "hidden_state": np.array(features, dtype=np.float32),
                    "collision": bool(has_collision),
                    "collision_pos": None if pos is None else pos.tolist(),
                    "collision_normal": None if normal is None else normal.tolist(),
                    "collision_geoms": [geom1, geom2],
                    "robot_state": robot_state,
                    "done": bool(done),
                }
            )
            
            # Match official OFT eval: if done, treat as success and break immediately
            if done:
                trajectory["success"] = True
                break
            t += 1
    finally:
        if video_writer is not None:
            video_writer.close()
        if video_path is not None:
            trajectory["video_path"] = str(video_path)

    return trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--n_success", type=int, default=20)
    parser.add_argument("--n_failure", type=int, default=40)
    parser.add_argument("--max-attempts-per-task", type=int, default=20)
    parser.add_argument("--camera-res", type=int, default=256)
    parser.add_argument("--save_dir", default="data/rollouts")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--store-observations", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", default="data/rollouts_videos")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = GenerateConfig(
        pretrained_checkpoint=args.model_name,
        task_suite_name=args.env,
        env_img_res=args.camera_res,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        num_trials_per_task=args.max_attempts_per_task,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        center_crop=True,
        num_images_in_input=args.num_images,
    )

    success_rollouts, failure_rollouts = collect_rollouts(
        cfg,
        args.save_dir,
        args.n_success,
        args.n_failure,
        args.max_attempts_per_task,
        args.seed,
        args.checkpoint_every,
        args.store_observations,
        args.save_video,
        args.video_dir,
        args.video_fps,
    )

    print(
        f"Saved {len(success_rollouts)} success and {len(failure_rollouts)} failure rollouts to {args.save_dir}",
        flush=True,
    )
