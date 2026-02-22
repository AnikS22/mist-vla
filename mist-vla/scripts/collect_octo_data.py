"""
Collect rollouts using Octo model on LIBERO,
while logging hidden states (bottleneck embeddings), EEF positions, and outcomes.

Octo is a JAX-based diffusion transformer VLA. We extract the transformer
bottleneck embeddings as the "hidden state" for our Safety MLP.

Usage:
    python scripts/collect_octo_data.py \
        --model-name hf://rail-berkeley/octo-base-1.5 \
        --env libero_spatial \
        --n_success 50 --n_failure 50 \
        --save_dir data/multi_model/octo_spatial
"""

import argparse
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

# JAX / Octo imports
import jax
import jax.numpy as jnp

# ── Compatibility shim: Octo references jax.random.KeyArray which was
#    removed in JAX 0.4.25+. Patch it before importing Octo. ──
if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jax.Array

from octo.model.octo_model import OctoModel

# LIBERO imports
from pathlib import Path as _Path
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from PIL import Image


def get_libero_env(task, resolution=256):
    """Create a LIBERO environment for a given task."""
    task_description = task.language
    # Full BDDL path — just task.bddl_file alone causes "does not exist" errors
    task_bddl_file = str(
        _Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": ["agentview"],
        "render_gpu_device_id": 0,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_eef_pos(env):
    """Extract end-effector position from LIBERO environment."""
    try:
        sim = getattr(env, 'sim', None)
        if sim is None:
            sim = getattr(env, 'env', None)
            if sim is not None:
                sim = getattr(sim, 'sim', None)
        if sim is not None:
            eef_pos = sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")].copy()
            return eef_pos
    except Exception:
        pass
    return np.zeros(3)


# ── Cache for JIT-compiled embedding function ──
_embed_fn = None


def extract_octo_embeddings(model, task_input, observation):
    """
    Run ONLY Octo's transformer (not the diffusion action head) to extract
    readout token embeddings.

    The diffusion head requires (time, noisy_actions) which we don't have,
    so we call model.module.octo_transformer directly via `method=`.
    """
    global _embed_fn

    try:
        pad_mask = observation["timestep_pad_mask"]

        if _embed_fn is None:
            @jax.jit
            def _forward(params, obs, task_in, mask):
                return model.module.apply(
                    {"params": params}, obs, task_in, mask, train=False,
                    method=model.module.octo_transformer,
                )
            _embed_fn = _forward

        transformer_out = _embed_fn(
            model.params, observation, task_input, pad_mask
        )

        # transformer_out is a TokenGroup with .tokens: (batch, n_tokens, embed_dim)
        if hasattr(transformer_out, "tokens"):
            embed = transformer_out.tokens
        elif isinstance(transformer_out, dict):
            embed = list(transformer_out.values())[0]
        else:
            embed = transformer_out

        # Mean pool across token dim → (batch, embed_dim)
        pooled = np.array(jnp.mean(embed, axis=1))
        return pooled[0]  # (embed_dim,)

    except Exception as e:
        print(f"  ⚠ Embedding extraction failed: {e}", flush=True)
        return np.zeros(512, dtype=np.float32)


def run_octo_episode(model, env, task_description, max_steps=250, resolution=256):
    """Run one episode with Octo, collecting features and robot states."""
    obs = env.reset()

    trajectory = {
        "features": [],
        "robot_states": [],
        "actions": [],
        "rewards": [],
        "steps": 0,
        "success": False,
        "collision_occurred": False,
        "collision_step": -1,
        "collision_steps": [],
    }

    # Create task encoding once
    task_input = model.create_tasks(texts=[task_description])

    # Window of recent images for Octo (it uses history)
    image_history = []
    WINDOW_SIZE = 2  # Octo typically uses 2-frame history

    for step in range(max_steps):
        # Get observation image
        img = obs.get("agentview_image", obs.get("image", None))
        if img is None:
            for k, v in obs.items():
                if "image" in k.lower() or "rgb" in k.lower():
                    img = v
                    break
        if img is None:
            img = np.zeros((resolution, resolution, 3), dtype=np.uint8)

        # Ensure correct format
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        image_history.append(img)
        if len(image_history) > WINDOW_SIZE:
            image_history = image_history[-WINDOW_SIZE:]

        # Pad if we don't have enough history yet
        while len(image_history) < WINDOW_SIZE:
            image_history.insert(0, image_history[0])

        images = np.stack(image_history, axis=0)  # (T, H, W, 3)

        # Build Octo observation dict (note: "timestep_pad_mask", NOT "pad_mask")
        observation = {
            "image_primary": images[np.newaxis],        # (1, T, H, W, 3)
            "timestep_pad_mask": np.ones((1, WINDOW_SIZE), dtype=bool),
        }

        # Get action
        try:
            rng = jax.random.PRNGKey(step)
            actions = model.sample_actions(observation, task_input, rng=rng)
            action = np.array(actions[0, 0])  # First batch, first timestep
        except Exception as e:
            if step == 0:
                print(f"  ⚠ Action failed at step {step}: {e}", flush=True)
            action = np.zeros(7, dtype=np.float32)

        # Extract embeddings every step
        try:
            features = extract_octo_embeddings(model, task_input, observation)
        except Exception:
            features = np.zeros(512, dtype=np.float32)

        # Get robot state
        eef_pos = get_eef_pos(env)
        robot_state = {"eef_pos": eef_pos.copy()}

        # Step environment
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            print(f"  ⚠ Env step failed: {e}", flush=True)
            break

        # Record
        trajectory["features"].append(features.astype(np.float32))
        trajectory["robot_states"].append(robot_state)
        trajectory["actions"].append(action.astype(np.float32))
        trajectory["rewards"].append(float(reward))
        trajectory["steps"] = step + 1

        # Check success
        success = False
        if done:
            success = True
        elif isinstance(info, dict):
            success = info.get("success", False) or info.get("is_success", False)

        if success:
            trajectory["success"] = True
            break

    return trajectory


def collect_rollouts(model, env, task_description, task_id, n_success, n_failure,
                     max_attempts, init_states=None, resolution=256):
    """Collect success and failure rollouts for one task."""
    successes = []
    failures = []
    attempt = 0

    while (len(successes) < n_success or len(failures) < n_failure) and attempt < max_attempts:
        # Set initial state if available
        if init_states is not None and len(init_states) > 0:
            init_state = init_states[attempt % len(init_states)]
            env.set_init_state(init_state)

        traj = run_octo_episode(model, env, task_description, resolution=resolution)
        traj["task_id"] = task_id
        traj["instruction"] = task_description

        if traj["success"] and len(successes) < n_success:
            successes.append(traj)
            tag = "✓"
        elif not traj["success"] and len(failures) < n_failure:
            failures.append(traj)
            tag = "✗"
        else:
            tag = "⊘"  # Discarded (quota met)

        attempt += 1
        print(f"    [{tag}] Task {task_id} attempt {attempt}: "
              f"steps={traj['steps']}  "
              f"S={len(successes)}/{n_success}  F={len(failures)}/{n_failure}",
              flush=True)

    return successes, failures


def main():
    parser = argparse.ArgumentParser(description="Octo data collection for LIBERO")
    parser.add_argument("--model-name", default="hf://rail-berkeley/octo-base-1.5",
                        help="Octo model path (HuggingFace or local)")
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--n_success", type=int, default=50)
    parser.add_argument("--n_failure", type=int, default=50)
    parser.add_argument("--max-attempts-per-task", type=int, default=30)
    parser.add_argument("--save_dir", default="data/multi_model/octo_spatial")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    np.random.seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("OCTO DATA COLLECTION FOR SAFETY MLP", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: {args.model_name}", flush=True)
    print(f"  Suite: {args.env}", flush=True)
    print(f"  Target: {args.n_success}S + {args.n_failure}F per task", flush=True)
    print(f"  Save:  {args.save_dir}", flush=True)
    print(flush=True)

    # Load Octo model
    print("[1/2] Loading Octo model...", flush=True)
    model = OctoModel.load_pretrained(args.model_name)
    print(f"  ✓ Octo loaded", flush=True)

    # Check embedding dimension
    print("  Probing embedding dimension...", flush=True)
    dummy_img = np.zeros((2, args.resolution, args.resolution, 3), dtype=np.uint8)
    dummy_obs = {
        "image_primary": dummy_img[np.newaxis],
        "timestep_pad_mask": np.ones((1, 2), dtype=bool),
    }
    dummy_task = model.create_tasks(texts=["test"])
    try:
        embed = extract_octo_embeddings(model, dummy_task, dummy_obs)
        embed_dim = embed.shape[-1]
        print(f"  ✓ Embedding dimension: {embed_dim}", flush=True)
    except Exception as e:
        embed_dim = 512
        print(f"  ⚠ Could not probe embedding dim ({e}), assuming {embed_dim}", flush=True)

    # Setup LIBERO
    print("[2/2] Setting up LIBERO...", flush=True)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.env]()
    num_tasks = task_suite.n_tasks
    print(f"  ✓ {args.env}: {num_tasks} tasks", flush=True)
    print(flush=True)

    all_successes = []
    all_failures = []
    t0 = time.time()

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        env, task_desc = get_libero_env(task, resolution=args.resolution)
        init_states = task_suite.get_task_init_states(task_id)

        print(f"━━━ Task {task_id}: {task_desc[:60]}... ━━━", flush=True)

        s, f = collect_rollouts(
            model, env, task_desc, task_id,
            args.n_success, args.n_failure,
            args.max_attempts_per_task * (args.n_success + args.n_failure),
            init_states=init_states,
            resolution=args.resolution,
        )
        all_successes.extend(s)
        all_failures.extend(f)

        env.close()

        # Checkpoint
        total = len(all_successes) + len(all_failures)
        if total > 0 and total % args.checkpoint_every == 0:
            with open(save_dir / "success_rollouts_partial.pkl", "wb") as fp:
                pickle.dump(all_successes, fp)
            with open(save_dir / "failure_rollouts_partial.pkl", "wb") as fp:
                pickle.dump(all_failures, fp)
            print(f"  [checkpoint] {len(all_successes)}S + {len(all_failures)}F saved",
                  flush=True)

        print(flush=True)

    # Save final
    with open(save_dir / "success_rollouts.pkl", "wb") as fp:
        pickle.dump(all_successes, fp)
    with open(save_dir / "failure_rollouts.pkl", "wb") as fp:
        pickle.dump(all_failures, fp)

    elapsed = time.time() - t0

    # Summary
    print("=" * 70, flush=True)
    print("COLLECTION COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"  Total: {len(all_successes)}S + {len(all_failures)}F = "
          f"{len(all_successes) + len(all_failures)} rollouts", flush=True)
    print(f"  Time:  {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"  Embed: dim={embed_dim}", flush=True)

    # Per-task summary
    tc_s = Counter(r["task_id"] for r in all_successes)
    tc_f = Counter(r["task_id"] for r in all_failures)
    print(flush=True)
    for tid in sorted(set(list(tc_s.keys()) + list(tc_f.keys()))):
        print(f"    Task {tid}: {tc_s.get(tid,0)}S + {tc_f.get(tid,0)}F", flush=True)

    print(f"\n  Saved to: {save_dir}/", flush=True)
    print("=" * 70, flush=True)

    # Save metadata
    meta = {
        "model": args.model_name,
        "model_type": "octo",
        "suite": args.env,
        "embedding_dim": int(embed_dim),
        "n_success": len(all_successes),
        "n_failure": len(all_failures),
        "num_tasks": num_tasks,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
    }
    import json
    with open(save_dir / "metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2)


if __name__ == "__main__":
    main()
