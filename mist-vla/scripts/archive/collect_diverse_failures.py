"""
Collect diverse failure data by applying controlled perturbations to induce failures.

This script collects rollouts with:
1. Action noise/perturbations to induce failures
2. Better collision detection
3. More diverse failure modes
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import deque
import time

import torch

# Ensure torch.load maps to CPU when CUDA is unavailable
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
from experiments.robot.libero.run_libero_eval import GenerateConfig, TASK_MAX_STEPS, prepare_observation, process_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from src.data_collection.collision_detection import CollisionDetector
from src.data_collection.hooks import HiddenStateCollector


def apply_perturbation(action: np.ndarray, perturbation_type: str, strength: float) -> np.ndarray:
    """
    Apply perturbation to action to induce failures.
    
    Perturbation types:
    - 'noise': Add Gaussian noise
    - 'scale': Scale action magnitude
    - 'bias': Add constant bias to specific dimensions
    - 'override': Override specific dimensions with large values
    """
    perturbed = action.copy()
    
    if perturbation_type == 'noise':
        noise = np.random.normal(0, strength, size=action.shape)
        perturbed = perturbed + noise
        # Clip to valid range
        perturbed = np.clip(perturbed, -1, 1)
    
    elif perturbation_type == 'scale':
        perturbed = perturbed * (1.0 + strength * np.random.uniform(-1, 1))
        perturbed = np.clip(perturbed, -1, 1)
    
    elif perturbation_type == 'bias':
        # Add bias to random dimension
        dim = np.random.randint(0, len(action))
        perturbed[dim] += strength * np.random.uniform(-1, 1)
        perturbed = np.clip(perturbed, -1, 1)
    
    elif perturbation_type == 'override':
        # Override random dimension with large value
        dim = np.random.randint(0, len(action))
        perturbed[dim] = strength * np.random.choice([-1, 1])
    
    return perturbed


def collect_diverse_rollouts(
    cfg,
    save_dir,
    n_rollouts,
    perturbation_prob: float = 0.3,
    perturbation_strength: float = 0.5,
    seed: int = 0,
    checkpoint_every: int = 25,
):
    """Collect rollouts with perturbations to induce diverse failures."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    print("[status] loading VLA...", flush=True)
    vla = get_vla(cfg)
    print("[status] VLA loaded", flush=True)

    # Resolve unnorm_key (required for finetuned models)
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key in vla.norm_stats:
        cfg.unnorm_key = unnorm_key
    print(f"[status] unnorm_key={cfg.unnorm_key}", flush=True)

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    collector = HiddenStateCollector(vla)
    collector.register_hooks()
    
    resize_size = get_image_resize_size(cfg)
    rollouts = []
    
    # Distribute rollouts evenly across tasks
    rollouts_per_task = max(1, n_rollouts // num_tasks)
    total_rollouts = 0
    
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        print(f"[status] init env task={task_id} (target={rollouts_per_task} rollouts)", flush=True)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        init_states = task_suite.get_task_init_states(task_id)
        
        task_count = 0
        episode_idx = 0
        while task_count < rollouts_per_task:
            initial_state = init_states[episode_idx % len(init_states)] if init_states is not None and len(init_states) > 0 else None
            episode_idx += 1
            
            # Decide whether to apply perturbation this episode
            apply_pert = np.random.random() < perturbation_prob
            
            rollout = _run_episode_with_perturbation(
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
                should_perturb=apply_pert,
                perturbation_strength=perturbation_strength,
            )
            
            rollout["task_id"] = task_id
            rollouts.append(rollout)
            task_count += 1
            total_rollouts += 1
            
            n_succ = sum(1 for r in rollouts if r.get("success"))
            n_fail = sum(1 for r in rollouts if not r.get("success"))
            n_coll = sum(1 for r in rollouts if r.get("collision_occurred"))
            print(
                f"[progress] task={task_id} task_count={task_count}/{rollouts_per_task} "
                f"total={total_rollouts} succ={n_succ} fail={n_fail} coll={n_coll}",
                flush=True,
            )
            
            if checkpoint_every and total_rollouts % checkpoint_every == 0:
                _write_checkpoint(save_dir, rollouts)
        
        env.close()
        print(f"[status] task={task_id} done: {task_count} rollouts collected", flush=True)
    
    # Save final rollouts
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "diverse_failure_rollouts.pkl", "wb") as f:
        pickle.dump(rollouts, f)
    
    return rollouts


def _run_episode_with_perturbation(
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
    should_perturb=False,
    perturbation_strength=0.5,
):
    """Run episode with optional perturbations."""
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
        "perturbed": should_perturb,
    }
    
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    t = 0
    last_features = None
    
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
            
            # Apply perturbation if enabled
            if should_perturb:
                perturbation_type = np.random.choice(['noise', 'bias', 'override'])
                action = apply_perturbation(action, perturbation_type, perturbation_strength)
            
            action = process_action(action, cfg.model_family)
            
            if features is None:
                features = np.zeros((1,), dtype=np.float32)
            
            # Execute action
            obs, reward, done, info = env.step(action.tolist())
            
            # Check collision
            has_collision, pos, normal, geom1, geom2 = detector.check_collision_details()
            if has_collision:
                trajectory["collision_occurred"] = True
                trajectory["collision_steps"] += 1
                if trajectory["collision_step"] is None:
                    trajectory["collision_step"] = len(trajectory["actions"])
            
            # Get robot state
            robot_state = _get_robot_state(env)
            
            # Log data
            trajectory["actions"].append(action)
            trajectory["features"].append(features)
            trajectory["robot_states"].append(robot_state)
            trajectory["rewards"].append(reward)
            
            trajectory["steps"].append({
                "action": np.array(action, dtype=np.float32),
                "hidden_state": np.array(features, dtype=np.float32),
                "collision": bool(has_collision),
                "collision_pos": None if pos is None else pos.tolist(),
                "collision_normal": None if normal is None else normal.tolist(),
                "collision_geoms": [geom1, geom2],
                "robot_state": robot_state,
                "done": bool(done),
            })
            
            if done:
                trajectory["success"] = True
                break
            t += 1
    finally:
        pass
    
    return trajectory


def _get_robot_state(env):
    """Get robot state from environment."""
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


def _write_checkpoint(save_dir, rollouts):
    """Write checkpoint of rollouts."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "diverse_failure_rollouts_partial.pkl", "wb") as f:
        pickle.dump(rollouts, f)
    print(f"[checkpoint] saved {len(rollouts)} rollouts", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--perturbation-prob", type=float, default=0.3, help="Probability of applying perturbation")
    parser.add_argument("--perturbation-strength", type=float, default=0.5, help="Strength of perturbation")
    parser.add_argument("--camera-res", type=int, default=256)
    parser.add_argument("--save-dir", default="data/diverse_failures")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    cfg = GenerateConfig(
        pretrained_checkpoint=args.model_name,
        task_suite_name=args.env,
        env_img_res=args.camera_res,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        center_crop=True,
        num_images_in_input=2,
    )
    
    rollouts = collect_diverse_rollouts(
        cfg,
        args.save_dir,
        args.n_rollouts,
        perturbation_prob=args.perturbation_prob,
        perturbation_strength=args.perturbation_strength,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )
    
    print(f"\n✓ Collected {len(rollouts)} rollouts")
    print(f"  Successes: {sum(1 for r in rollouts if r.get('success'))}")
    print(f"  Failures: {sum(1 for r in rollouts if not r.get('success'))}")
    print(f"  With collisions: {sum(1 for r in rollouts if r.get('collision_occurred'))}")
