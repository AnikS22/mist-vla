#!/usr/bin/env python3
"""
Evaluate baseline models (Diffusion Policy, ACT) on LIBERO tasks.

Just runs closed-loop episodes and reports success rate per task.
These models don't need the Safety MLP — they're comparison baselines.

Usage:
  python scripts/eval_baseline_model.py \
      --model-type diffusion_policy \
      --checkpoint checkpoints/diffusion_policy/best_model.pt \
      --env libero_spatial \
      --episodes-per-task 20 \
      --save-dir results/paper_table/diffusion_policy
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from libero.libero import benchmark as libero_benchmark
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T


# ══════════════════════════════════════════════════════════════════════════
#  IMPORT MODEL CLASSES
# ══════════════════════════════════════════════════════════════════════════

def load_model(model_type, checkpoint_path, device):
    """Load a trained baseline model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == "diffusion_policy":
        from train_diffusion_policy_libero import SimpleDiffusionPolicy
        model = SimpleDiffusionPolicy(
            obs_dim=ckpt['proprio_dim'],
            action_dim=ckpt['action_dim'],
            action_horizon=ckpt['action_horizon'],
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

    elif model_type == "act":
        from train_act_libero import ACTPolicy
        model = ACTPolicy(
            obs_dim=ckpt['proprio_dim'],
            action_dim=ckpt['action_dim'],
            action_horizon=ckpt['action_horizon'],
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model_type}: {n_params:,} params, "
          f"epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

from torchvision import transforms

IMG_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def obs_to_input(obs, device):
    """Convert LIBERO observation to model input tensors."""
    # Image
    img = obs.get('agentview_image', obs.get('agentview_rgb'))
    if img is None:
        for k in obs:
            if 'image' in k.lower() or 'rgb' in k.lower():
                img = obs[k]
                break
    if img is None:
        raise ValueError(f"No image found in obs keys: {list(obs.keys())}")

    img_tensor = IMG_TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, 128, 128)

    # Proprio
    proprio = obs.get('robot0_eef_pos', obs.get('ee_pos'))
    if proprio is None:
        proprio = obs.get('robot0_joint_pos', np.zeros(8))
    proprio_tensor = torch.FloatTensor(proprio).unsqueeze(0).to(device)

    return img_tensor, proprio_tensor


# ══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_episode(env, model, model_type, device, max_steps=300,
                action_horizon=8):
    """Run one closed-loop episode."""
    obs = env.reset()
    action_queue = []
    success = False

    for step in range(max_steps):
        if len(action_queue) == 0:
            # Get new action chunk
            img_tensor, proprio_tensor = obs_to_input(obs, device)

            with torch.no_grad():
                if model_type == "diffusion_policy":
                    # DP needs obs_horizon images; use current frame repeated
                    imgs = img_tensor.unsqueeze(1).expand(
                        -1, 2, -1, -1, -1)  # (1, 2, 3, 128, 128)
                    proprios = proprio_tensor.unsqueeze(1).expand(
                        -1, 2, -1)  # (1, 2, proprio_dim)
                    actions = model.predict(imgs, proprios)
                elif model_type == "act":
                    actions = model.predict(img_tensor, proprio_tensor)
                else:
                    raise ValueError(f"Unknown: {model_type}")

            # actions shape: (1, action_horizon, action_dim)
            action_chunk = actions[0].cpu().numpy()
            action_queue = list(action_chunk)

        action = action_queue.pop(0)
        action = np.clip(action, -1.0, 1.0)
        obs, reward, done, info = env.step(action)

        if done:
            success = True
            break

    return success


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True,
                        choices=["diffusion_policy", "act"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--tasks", type=int, nargs="+",
                        default=list(range(10)))
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="results/baselines")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"  BASELINE EVALUATION — {args.model_type}", flush=True)
    print(f"  Tasks: {args.tasks}  Episodes: {args.episodes_per_task}",
          flush=True)
    print("=" * 60, flush=True)

    # Load model
    model = load_model(args.model_type, args.checkpoint, device)

    # Setup LIBERO
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.env]()

    results = {}
    t0 = time.time()

    for task_id in args.tasks:
        task = task_suite.get_task(task_id)
        task_name = task.language
        init_states = task_suite.get_task_init_states(task_id)

        # Create environment
        env_args = {
            "bddl_file_name": task.problem_folder,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env = OffScreenRenderEnv(**env_args)

        print(f"\n━━━ Task {task_id}: {task_name[:50]}... ━━━", flush=True)
        successes = 0
        print(f"  ", end="", flush=True)

        for ep in range(args.episodes_per_task):
            if init_states is not None:
                init_state = init_states[ep % len(init_states)]
                env.reset()
                env.set_init_state(init_state)

            ok = run_episode(env, model, args.model_type, device)
            print("✓" if ok else "✗", end="", flush=True)
            if ok:
                successes += 1

        rate = successes / args.episodes_per_task * 100
        print(f"  {successes}/{args.episodes_per_task} ({rate:.0f}%)",
              flush=True)

        results[task_id] = {
            "task_name": task_name,
            "success_rate_pct": round(rate, 1),
            "n_successes": successes,
            "n_episodes": args.episodes_per_task,
        }
        env.close()
        gc.collect()

    total_time = time.time() - t0

    # Summary
    rates = [r["success_rate_pct"] for r in results.values()]
    avg = np.mean(rates) if rates else 0

    print(f"\n{'=' * 60}", flush=True)
    print(f"  {args.model_type.upper()} RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    for tid in args.tasks:
        if tid in results:
            r = results[tid]
            print(f"  Task {tid}: {r['success_rate_pct']:>5.1f}%  "
                  f"{r['task_name'][:40]}", flush=True)
    print(f"  {'─' * 50}", flush=True)
    print(f"  AVERAGE: {avg:.1f}%", flush=True)
    print(f"  Time: {total_time:.0f}s", flush=True)

    # Save
    report = {
        "model_type": args.model_type,
        "env": args.env,
        "episodes_per_task": args.episodes_per_task,
        "avg_success_pct": round(avg, 1),
        "per_task": {str(k): v for k, v in results.items()},
    }
    out_path = save_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
