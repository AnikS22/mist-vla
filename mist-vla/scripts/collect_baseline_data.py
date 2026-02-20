#!/usr/bin/env python3
"""
Collect rollout data from baseline models (Diffusion Policy, ACT) on LIBERO,
in the SAME format as the OpenVLA SafeVLA data collector.

For each rollout we save:
  - features:      list of model internal embeddings per timestep
  - robot_states:  list of {eef_pos, qpos, qvel} per timestep
  - actions:       list of 7-D action vectors
  - rewards:       list of floats
  - success:       bool
  - task_id:       int
  - instruction:   str
  - collision_occurred / collision_step / collision_steps
  - steps:         list of per-step detail dicts

This produces the EXACT same pickle format as collect_failure_data_oft_eval.py
so the MLP training script can consume data from any model.

Usage:
  python scripts/collect_baseline_data.py \
      --model-type diffusion_policy \
      --checkpoint checkpoints/diffusion_policy/best_model.pt \
      --env libero_spatial \
      --n-success 50 --n-failure 50 \
      --save-dir data/multi_model/dp_spatial
"""

import argparse
import gc
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from libero.libero import benchmark as libero_benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ══════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

IMG_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING + FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

class DPWrapper:
    """Wraps Diffusion Policy model for inference + feature extraction."""

    def __init__(self, checkpoint_path, device):
        from train_diffusion_policy_libero import SimpleDiffusionPolicy
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        self.model = SimpleDiffusionPolicy(
            obs_dim=ckpt['proprio_dim'],
            action_dim=ckpt['action_dim'],
            action_horizon=ckpt['action_horizon'],
        ).to(device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.action_queue = []
        self.last_features = None
        n = sum(p.numel() for p in self.model.parameters())
        print(f"  DP loaded: {n:,} params", flush=True)

    def reset(self):
        self.action_queue = []
        self.last_features = None

    def predict_and_extract(self, obs):
        """Get action + internal features from observation."""
        img_tensor, proprio_tensor = self._obs_to_input(obs)

        if len(self.action_queue) == 0:
            with torch.no_grad():
                # Extract features from the observation encoder
                imgs = img_tensor.unsqueeze(1).expand(-1, 2, -1, -1, -1)
                proprios = proprio_tensor.unsqueeze(1).expand(-1, 2, -1)
                features = self.model.encode_obs(
                    imgs[:, -1], proprios[:, -1])  # (1, 256)
                self.last_features = features[0].cpu().numpy()

                # Get action chunk
                actions = self.model.predict(imgs, proprios)
                self.action_queue = list(actions[0].cpu().numpy())

        action = self.action_queue.pop(0)
        return np.clip(action, -1.0, 1.0), self.last_features.copy()

    def _obs_to_input(self, obs):
        img = obs.get('agentview_image', obs.get('agentview_rgb'))
        if img is None:
            for k in obs:
                if 'image' in k.lower() or 'rgb' in k.lower():
                    img = obs[k]
                    break
        img_tensor = IMG_TRANSFORM(img).unsqueeze(0).to(self.device)
        proprio = obs.get('robot0_eef_pos', obs.get('ee_pos'))
        if proprio is None:
            proprio = obs.get('robot0_joint_pos', np.zeros(8))
        proprio_tensor = torch.FloatTensor(proprio).unsqueeze(0).to(
            self.device)
        return img_tensor, proprio_tensor


class ACTWrapper:
    """Wraps ACT model for inference + feature extraction."""

    def __init__(self, checkpoint_path, device):
        from train_act_libero import ACTPolicy
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        self.model = ACTPolicy(
            obs_dim=ckpt['proprio_dim'],
            action_dim=ckpt['action_dim'],
            action_horizon=ckpt['action_horizon'],
        ).to(device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.action_queue = []
        self.last_features = None
        n = sum(p.numel() for p in self.model.parameters())
        print(f"  ACT loaded: {n:,} params", flush=True)

    def reset(self):
        self.action_queue = []
        self.last_features = None

    def predict_and_extract(self, obs):
        """Get action + internal features from observation."""
        img_tensor, proprio_tensor = self._obs_to_input(obs)

        if len(self.action_queue) == 0:
            with torch.no_grad():
                # Extract features from the observation encoder
                features = self.model.encode_obs(
                    img_tensor, proprio_tensor)  # (1, 256)
                self.last_features = features[0].cpu().numpy()

                # Get action chunk
                actions = self.model.predict(img_tensor, proprio_tensor)
                self.action_queue = list(actions[0].cpu().numpy())

        action = self.action_queue.pop(0)
        return np.clip(action, -1.0, 1.0), self.last_features.copy()

    def _obs_to_input(self, obs):
        img = obs.get('agentview_image', obs.get('agentview_rgb'))
        if img is None:
            for k in obs:
                if 'image' in k.lower() or 'rgb' in k.lower():
                    img = obs[k]
                    break
        img_tensor = IMG_TRANSFORM(img).unsqueeze(0).to(self.device)
        proprio = obs.get('robot0_eef_pos', obs.get('ee_pos'))
        if proprio is None:
            proprio = obs.get('robot0_joint_pos', np.zeros(8))
        proprio_tensor = torch.FloatTensor(proprio).unsqueeze(0).to(
            self.device)
        return img_tensor, proprio_tensor


# ══════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_eef_pos(env):
    """Extract end-effector position from LIBERO environment."""
    try:
        sim = getattr(env, 'sim', None)
        if sim is None:
            inner = getattr(env, 'env', None)
            if inner is not None:
                sim = getattr(inner, 'sim', None)
        if sim is not None:
            site_id = sim.model.site_name2id("gripper0_grip_site")
            return sim.data.site_xpos[site_id].copy()
    except Exception:
        pass
    return np.zeros(3)


def get_robot_state(env):
    """Extract full robot state from LIBERO environment."""
    state = {"eef_pos": get_eef_pos(env)}
    try:
        sim = getattr(env, 'sim', None)
        if sim is None:
            inner = getattr(env, 'env', None)
            if inner is not None:
                sim = getattr(inner, 'sim', None)
        if sim is not None:
            state["qpos"] = sim.data.qpos.copy()
            state["qvel"] = sim.data.qvel.copy()
    except Exception:
        state["qpos"] = np.zeros(48)
        state["qvel"] = np.zeros(43)
    return state


def check_collision(env):
    """Check if a collision occurred in the environment."""
    try:
        sim = getattr(env, 'sim', None)
        if sim is None:
            inner = getattr(env, 'env', None)
            if inner is not None:
                sim = getattr(inner, 'sim', None)
        if sim is not None and sim.data.ncon > 0:
            for i in range(sim.data.ncon):
                contact = sim.data.contact[i]
                geom1 = sim.model.geom_id2name(contact.geom1)
                geom2 = sim.model.geom_id2name(contact.geom2)
                if geom1 and geom2:
                    # Check for gripper-table or gripper-obstacle contacts
                    gripper_geoms = ['gripper0', 'finger']
                    table_geoms = ['table', 'floor']
                    g1_grip = any(g in str(geom1) for g in gripper_geoms)
                    g2_grip = any(g in str(geom2) for g in gripper_geoms)
                    g1_table = any(g in str(geom1) for g in table_geoms)
                    g2_table = any(g in str(geom2) for g in table_geoms)
                    if (g1_grip and g2_table) or (g2_grip and g1_table):
                        pos = contact.pos.copy()
                        normal = contact.frame[:3].copy()
                        return True, [geom1, geom2], pos, normal
    except Exception:
        pass
    return False, [None, None], None, None


# ══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_episode(env, model_wrapper, max_steps=300):
    """Run one episode, collecting full rollout data."""
    obs = env.reset()
    model_wrapper.reset()

    trajectory = {
        "features": [],
        "robot_states": [],
        "actions": [],
        "rewards": [],
        "steps": [],
        "success": False,
        "collision_occurred": False,
        "collision_step": None,
        "collision_steps": 0,
    }

    for step in range(max_steps):
        # Get action + internal features
        action, features = model_wrapper.predict_and_extract(obs)

        # Get robot state before stepping
        robot_state = get_robot_state(env)

        # Step environment
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            print(f"    ⚠ env.step failed at step {step}: {e}", flush=True)
            break

        # Check collision
        collision, c_geoms, c_pos, c_normal = check_collision(env)
        if collision and not trajectory["collision_occurred"]:
            trajectory["collision_occurred"] = True
            trajectory["collision_step"] = step
        if collision:
            trajectory["collision_steps"] += 1

        # Record per-step data (same format as OpenVLA collector)
        step_data = {
            "action": action.astype(np.float32),
            "hidden_state": features.astype(np.float32),
            "robot_state": {
                "qpos": robot_state.get("qpos", np.zeros(48)),
                "qvel": robot_state.get("qvel", np.zeros(43)),
                "eef_pos": robot_state["eef_pos"],
            },
            "collision": collision,
            "collision_geoms": c_geoms,
            "collision_pos": c_pos if c_pos is not None else None,
            "collision_normal": c_normal if c_normal is not None else None,
            "done": bool(done),
        }

        trajectory["features"].append(features.astype(np.float32))
        trajectory["robot_states"].append(robot_state)
        trajectory["actions"].append(action.astype(np.float32))
        trajectory["rewards"].append(float(reward))
        trajectory["steps"].append(step_data)

        # Check success
        success = False
        if done:
            success = True
        elif isinstance(info, dict):
            success = info.get("success", False) or \
                info.get("is_success", False)

        if success:
            trajectory["success"] = True
            break

    return trajectory


# ══════════════════════════════════════════════════════════════════════════
#  MAIN COLLECTION LOOP
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Collect rollout data from baseline models")
    parser.add_argument("--model-type", required=True,
                        choices=["diffusion_policy", "act"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--n-success", type=int, default=50,
                        help="Target successful rollouts per task")
    parser.add_argument("--n-failure", type=int, default=50,
                        help="Target failed rollouts per task")
    parser.add_argument("--max-attempts-per-task", type=int, default=200,
                        help="Max episodes to try per task")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"  ROLLOUT DATA COLLECTION — {args.model_type.upper()}", flush=True)
    print("=" * 70, flush=True)
    print(f"  Checkpoint: {args.checkpoint}", flush=True)
    print(f"  Suite:      {args.env}", flush=True)
    print(f"  Target:     {args.n_success}S + {args.n_failure}F per task",
          flush=True)
    print(f"  Device:     {device}", flush=True)
    print(f"  Save:       {args.save_dir}", flush=True)
    print(flush=True)

    # Load model
    print("[1/2] Loading model...", flush=True)
    if args.model_type == "diffusion_policy":
        wrapper = DPWrapper(args.checkpoint, device)
    elif args.model_type == "act":
        wrapper = ACTWrapper(args.checkpoint, device)

    # Setup LIBERO
    print("[2/2] Setting up LIBERO...", flush=True)
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.env]()
    num_tasks = task_suite.n_tasks
    print(f"  {args.env}: {num_tasks} tasks", flush=True)
    print(flush=True)

    all_successes = []
    all_failures = []
    t0 = time.time()

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_name = task.language
        init_states = task_suite.get_task_init_states(task_id)

        task_bddl_file = str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env = OffScreenRenderEnv(**env_args)

        print(f"━━━ Task {task_id}: {task_name[:60]}... ━━━", flush=True)

        successes = []
        failures = []
        attempt = 0

        while (len(successes) < args.n_success or
               len(failures) < args.n_failure) and \
                attempt < args.max_attempts_per_task:

            if init_states is not None and len(init_states) > 0:
                init_state = init_states[attempt % len(init_states)]
                env.reset()
                try:
                    env.set_init_state(init_state)
                except Exception:
                    pass

            traj = run_episode(env, wrapper)
            traj["task_id"] = task_id
            traj["instruction"] = task_name

            if traj["success"] and len(successes) < args.n_success:
                successes.append(traj)
                tag = "✓"
            elif not traj["success"] and len(failures) < args.n_failure:
                failures.append(traj)
                tag = "✗"
            else:
                tag = "⊘"

            n_steps = len(traj["actions"])
            attempt += 1
            print(f"    [{tag}] attempt {attempt}: steps={n_steps}  "
                  f"S={len(successes)}/{args.n_success}  "
                  f"F={len(failures)}/{args.n_failure}", flush=True)

        all_successes.extend(successes)
        all_failures.extend(failures)

        env.close()
        gc.collect()

        # Checkpoint after each task
        with open(save_dir / "success_rollouts_partial.pkl", "wb") as f:
            pickle.dump(all_successes, f)
        with open(save_dir / "failure_rollouts_partial.pkl", "wb") as f:
            pickle.dump(all_failures, f)
        print(f"    [saved] {len(all_successes)}S + {len(all_failures)}F",
              flush=True)
        print(flush=True)

    # Save final
    with open(save_dir / "success_rollouts.pkl", "wb") as f:
        pickle.dump(all_successes, f)
    with open(save_dir / "failure_rollouts.pkl", "wb") as f:
        pickle.dump(all_failures, f)

    elapsed = time.time() - t0

    # Compute baseline success rates
    print("=" * 70, flush=True)
    print(f"  {args.model_type.upper()} COLLECTION COMPLETE", flush=True)
    print("=" * 70, flush=True)

    success_rates = {}
    tc_s = Counter(r["task_id"] for r in all_successes)
    tc_f = Counter(r["task_id"] for r in all_failures)

    for tid in range(num_tasks):
        s = tc_s.get(tid, 0)
        f = tc_f.get(tid, 0)
        total = s + f
        rate = (s / total * 100) if total > 0 else 0
        success_rates[tid] = round(rate, 1)
        task = task_suite.get_task(tid)
        print(f"  Task {tid}: {s}S + {f}F = {rate:.1f}%  "
              f"{task.language[:50]}", flush=True)

    avg_rate = np.mean(list(success_rates.values()))
    print(f"\n  AVERAGE SUCCESS RATE: {avg_rate:.1f}%", flush=True)
    print(f"  Total: {len(all_successes)}S + {len(all_failures)}F = "
          f"{len(all_successes) + len(all_failures)} rollouts", flush=True)
    print(f"  Time:  {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    feat_dim = len(all_successes[0]["features"][0]) if all_successes and \
        all_successes[0]["features"] else "?"
    print(f"  Feature dim: {feat_dim}", flush=True)

    # Save metadata
    meta = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "suite": args.env,
        "embedding_dim": int(feat_dim) if isinstance(feat_dim, (int, np.integer)) else 0,
        "n_success": len(all_successes),
        "n_failure": len(all_failures),
        "num_tasks": num_tasks,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "per_task_success_rate": success_rates,
        "avg_success_rate": round(avg_rate, 1),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved to: {save_dir}/", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
