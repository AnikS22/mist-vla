"""
Collect rollouts using the official OpenVLA-OFT LIBERO evaluation pipeline,
while logging MIST-VLA signals (actions, hidden states, collisions, robot state).
"""
import argparse
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch

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
from experiments.robot.robot_utils import get_image_resize_size
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from experiments.robot.libero.run_libero_eval import GenerateConfig, TASK_MAX_STEPS

from src.data_collection.collision_detection import CollisionDetector


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


def _prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def _get_vla_action_with_features(cfg, vla, processor, obs, task_label, action_head, proprio_projector):
    with torch.inference_mode():
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        device = next(vla.parameters()).device
        inputs = processor(prompt, primary_image).to(device, dtype=torch.bfloat16)
        if all_images:
            all_wrist_inputs = [processor(prompt, image_wrist).to(device, dtype=torch.bfloat16) for image_wrist in all_images]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(proprio, proprio_norm_stats)

        actions, actions_hidden_states = vla.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=cfg.use_film,
        )

    actions = [actions[i] for i in range(len(actions))]
    if actions_hidden_states is not None and actions_hidden_states.ndim == 3:
        features = actions_hidden_states.mean(dim=1).detach().cpu().float().numpy()[0]
    elif actions_hidden_states is not None:
        features = actions_hidden_states.detach().cpu().float().numpy()
    else:
        features = None
    return actions, features


def _process_action(action, model_family):
    from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action

    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def _resolve_unnorm_key(cfg, vla):
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key in vla.norm_stats:
        cfg.unnorm_key = unnorm_key


def collect_rollouts(cfg, save_dir, n_success, n_failure, max_attempts_per_task, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    vla = get_vla(cfg)
    _resolve_unnorm_key(cfg, vla)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)

    resize_size = get_image_resize_size(cfg)
    success_rollouts, failure_rollouts = [], []

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        init_states = task_suite.get_task_init_states(task_id)

        attempts = 0
        while len(success_rollouts) < n_success and attempts < max_attempts_per_task:
            attempts += 1
            rollout = _run_episode(
                cfg,
                env,
                task_description,
                init_states,
                resize_size,
                vla,
                processor,
                action_head,
                proprio_projector,
            )
            rollout["task_id"] = task_id
            if rollout["success"]:
                success_rollouts.append(rollout)
            else:
                failure_rollouts.append(rollout)

        attempts = 0
        while len(failure_rollouts) < n_failure and attempts < max_attempts_per_task:
            attempts += 1
            rollout = _run_episode(
                cfg,
                env,
                task_description,
                init_states,
                resize_size,
                vla,
                processor,
                action_head,
                proprio_projector,
            )
            rollout["task_id"] = task_id
            if not rollout["success"]:
                failure_rollouts.append(rollout)

        env.close()
        if len(success_rollouts) >= n_success and len(failure_rollouts) >= n_failure:
            break

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "success_rollouts.pkl", "wb") as f:
        pickle.dump(success_rollouts, f)
    with open(save_path / "failure_rollouts.pkl", "wb") as f:
        pickle.dump(failure_rollouts, f)

    return success_rollouts, failure_rollouts


def _run_episode(cfg, env, task_description, init_states, resize_size, vla, processor, action_head, proprio_projector):
    env.reset()
    obs = env.set_init_state(init_states[np.random.randint(0, len(init_states))])
    detector = CollisionDetector(env)

    trajectory = {
        "observations": [],
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

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation, _ = _prepare_observation(obs, resize_size)

        if len(action_queue) == 0:
            actions, features = _get_vla_action_with_features(
                cfg, vla, processor, observation, task_description, action_head, proprio_projector
            )
            action_queue.extend(actions)
            last_features = features
        else:
            features = last_features

        action = action_queue.popleft()
        action = _process_action(action, cfg.model_family)

        if features is None:
            features = np.zeros((1,), dtype=np.float32)

        robot_state = _get_robot_state(env)
        trajectory["observations"].append(obs)
        trajectory["actions"].append(action)
        trajectory["features"].append(features)
        trajectory["robot_states"].append(robot_state)

        has_collision, pos, normal, geom1, geom2 = detector.check_collision_details()
        if has_collision:
            trajectory["collision_occurred"] = True
            trajectory["collision_steps"] += 1
            if trajectory["collision_step"] is None:
                trajectory["collision_step"] = len(trajectory["actions"])

        obs, reward, done, info = env.step(action.tolist())
        trajectory["rewards"].append(reward)
        trajectory["steps"].append(
            {
                "action": np.array(action, dtype=np.float32),
                "hidden_state": np.array(features, dtype=np.float32),
                "collision": bool(has_collision),
                "collision_pos": None if pos is None else pos.tolist(),
                "collision_normal": None if normal is None else normal.tolist(),
                "collision_geoms": [geom1, geom2],
                "robot_state": robot_state,
            }
        )
        if _check_success(env, info):
            trajectory["success"] = True
            break
        if done:
            break
        t += 1

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
        num_images_in_input=2,
    )

    success_rollouts, failure_rollouts = collect_rollouts(
        cfg,
        args.save_dir,
        args.n_success,
        args.n_failure,
        args.max_attempts_per_task,
        args.seed,
    )

    print(
        f"Saved {len(success_rollouts)} success and {len(failure_rollouts)} failure rollouts to {args.save_dir}",
        flush=True,
    )
