#!/usr/bin/env python3
"""
Phase 1.3: Collect rollouts with collision labels.

This script:
1. Loads OpenVLA model
2. Creates LIBERO environment
3. Collects rollout trajectories with:
   - Hidden states (for risk prediction)
   - Actions taken
   - Observations
   - Collision detection
   - End-effector positions
4. Saves data for Phase 1.4 (computing risk labels)

Target: Collect ~2000 trajectories (mix of collision and non-collision)
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.hooks import HiddenStateCollector
from src.data_collection.collision_detection import CollisionDetector


def collect_rollout(
    env,
    model,
    processor,
    collector,
    detector,
    instruction,
    max_steps=200,
    device='cuda',
    debug_collision=False,
    debug_actions=False,
    debug_interval=10,
    debug_info=False,
    camera_height=128,
    camera_width=128,
):
    """
    Collect a single rollout trajectory.

    Args:
        env: LIBERO environment
        model: OpenVLA model
        processor: OpenVLA processor
        collector: HiddenStateCollector instance
        detector: CollisionDetector instance
        instruction: Task instruction text
        max_steps: Maximum steps per rollout
        device: Device for inference

    Returns:
        Dictionary containing trajectory data
    """
    trajectory = {
        'instruction': instruction,
        'steps': [],
        'success': False,
        'failure': False,
        'collision_occurred': False,
        'collision_step': None,
    }

    obs = env.reset()
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        # Get observation image
        if 'agentview_image' in obs:
            image = obs['agentview_image']
        elif 'image' in obs:
            image = obs['image']
        else:
            # Use first available image key
            image_keys = [k for k in obs.keys() if 'image' in k.lower()]
            if image_keys:
                image = obs[image_keys[0]]
            else:
                raise ValueError(f"No image found in observation keys: {obs.keys()}")

        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))

        # Get end-effector position before action
        ee_pos = detector.get_end_effector_position()

        # Check for collision
        has_collision, collision_pos = detector.check_collision()

        # Predict action with hidden state collection
        collector.clear()
        with collector:
            # Process inputs
            inputs = processor(images=image, text=instruction, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Convert pixel_values to match model dtype (bfloat16 on GPU)
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model.dtype)

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Extract action
            # OpenVLA outputs action directly or via action_head
            if hasattr(outputs, 'action'):
                action_tensor = outputs.action
            elif hasattr(outputs, 'logits'):
                # May need to extract action from logits
                action_tensor = outputs.logits[:, -1, :7]  # Last token, 7 dims
            else:
                raise ValueError("Cannot extract action from model outputs")

            action = action_tensor.cpu().float().numpy()[0]

            # Get hidden state (convert bfloat16 to float32 for numpy compatibility)
            hidden_state = collector.get_last_layer().cpu().float().numpy()[0]

        # Optional debug: print action magnitude
        if debug_actions and step_count % debug_interval == 0:
            action_norm = float(np.linalg.norm(action))
            print(f"  [debug] step={step_count} action_norm={action_norm:.4f} "
                  f"min={action.min():.4f} max={action.max():.4f}")

        # Store step data
        step_data = {
            'observation': {k: v for k, v in obs.items()},
            'action': action.copy(),
            'hidden_state': hidden_state.copy(),
            'ee_pos': ee_pos.copy(),
            'collision': has_collision,
            'collision_pos': collision_pos.copy() if collision_pos is not None else None,
        }

        # Execute action
        obs, reward, done, info = env.step(action)

        # Update step data with results
        step_data['reward'] = reward
        step_data['done'] = done
        step_data['info'] = info
        # Optional debug: print contact geoms
        if debug_collision and step_count % debug_interval == 0:
            details = detector.get_collision_details()
            if details['contacts']:
                print(f"  [debug] step={step_count} contacts={details['num_contacts']} "
                      f"first_contact={details['contacts'][0]}")
            else:
                print(f"  [debug] step={step_count} contacts=0")

        # Optional debug: print info summary (requires info from env.step)
        if debug_info and step_count % debug_interval == 0:
            info_keys = list(info.keys()) if isinstance(info, dict) else []
            success_val = info.get("success") if isinstance(info, dict) else None
            is_success_val = info.get("is_success") if isinstance(info, dict) else None
            env_success = None
            if hasattr(env, "check_success"):
                try:
                    env_success = bool(env.check_success())
                except Exception:
                    env_success = None
            print(
                f"  [debug] step={step_count} reward={reward:.4f} done={done} "
                f"success={success_val} is_success={is_success_val} "
                f"env_success={env_success} info_keys={info_keys}"
            )

        # Benchmark-defined success metric (prefer env.check_success when available)
        success_flag = False
        if hasattr(env, "check_success"):
            try:
                success_flag = bool(env.check_success())
            except Exception:
                success_flag = False
        if not success_flag and isinstance(info, dict):
            if info.get('success') is True:
                success_flag = True
            elif info.get('is_success') is True:
                success_flag = True
        if success_flag:
            trajectory['success'] = True
            done = True

        trajectory['steps'].append(step_data)

        # Track collision
        if has_collision and not trajectory['collision_occurred']:
            trajectory['collision_occurred'] = True
            trajectory['collision_step'] = step_count

        step_count += 1

    # Check success one more time in case info only appears at the end
    if isinstance(info, dict):
    if 'success' in info:
        trajectory['success'] = info['success']
    elif 'is_success' in info:
        trajectory['success'] = info['is_success']

    trajectory['failure'] = not trajectory['success']
    return trajectory


def collect_data(
    output_path,
    num_rollouts=2000,
    benchmark_name='libero_spatial',
    max_steps=200,
    device='cuda',
    debug_collision=False,
    debug_actions=False,
    debug_interval=10,
    debug_info=False,
):
    """
    Collect rollout data.

    Args:
        output_path: Path to save data
        num_rollouts: Number of rollouts to collect
        benchmark_name: LIBERO benchmark name
        max_steps: Max steps per rollout
        device: Device for inference
    """
    print("=" * 60)
    print("Phase 1.3: Collect Rollouts with Collision Labels")
    print("=" * 60)

    # Load OpenVLA
    print("\n[1/6] Loading OpenVLA model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    print("  ✓ Processor loaded")

    # Prefer FlashAttention2 when available; otherwise force eager to avoid SDPA
    # attribute errors in some OpenVLA builds.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "eager"

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        attn_implementation=attn_impl,
    )
    if device == "cpu":
        model = model.to(device=device, dtype=torch.float32)
    else:
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model loaded on {device}")

    # Create hidden state collector
    collector = HiddenStateCollector(model)
    print("  ✓ Hidden state collector created")

    # Load LIBERO
    print("\n[2/6] Loading LIBERO environment...")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    # PyTorch 2.6+ defaults to weights_only=True for torch.load, which breaks
    # LIBERO's init state loading. Allowlist numpy reconstruct to proceed.
    import numpy as np
    try:
        torch.serialization.add_safe_globals(
            [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
        )
    except Exception:
        pass
    # Force weights_only=False for LIBERO init state torch.load calls.
    _orig_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[benchmark_name]()
    num_tasks = task_suite.get_num_tasks()
    print(f"  ✓ Loaded {benchmark_name} with {num_tasks} tasks")

    # Sanity check: Verify action extraction works
    print("\n[3/6] Sanity check: Verifying action extraction...")
    try:
        # Create test environment
        task = task_suite.get_task(0)
        instruction = task.language
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file
        )
        test_env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=camera_height,
            camera_widths=camera_width,
        )
        test_env.reset()
        init_states = task_suite.get_task_init_states(0)
        test_env.set_init_state(init_states[0])

        # Get test observation
        obs = test_env.reset()
        image = obs['agentview_image']
        image_pil = Image.fromarray(image)

        # Test model inference
        inputs = processor(images=image_pil, text=instruction, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model.dtype)

        with torch.no_grad():
            outputs = model(**inputs)

        # Verify action extraction
        action_extracted = False
        action_shape = None
        extraction_method = None

        if hasattr(outputs, 'action'):
            action_tensor = outputs.action
            action_extracted = True
            extraction_method = "outputs.action"
            action_shape = tuple(action_tensor.shape)
        elif hasattr(outputs, 'logits'):
            # Try to extract from logits
            action_tensor = outputs.logits[:, -1, :7]
            action_extracted = True
            extraction_method = "outputs.logits[:, -1, :7]"
            action_shape = tuple(action_tensor.shape)

        if not action_extracted:
            print(f"  ✗ FATAL: Cannot extract action from model outputs!")
            print(f"    Available attributes: {[a for a in dir(outputs) if not a.startswith('_')]}")
            print(f"  ")
            print(f"  This means data collection will fail.")
            print(f"  Please verify OpenVLA model outputs on this HPC environment.")
            test_env.close()
            return

        # Verify action is valid
        action = action_tensor.cpu().float().numpy()[0]
        if len(action) != 7:
            print(f"  ✗ WARNING: Action has {len(action)} dimensions, expected 7")
            print(f"    This may indicate incorrect action extraction")

        print(f"  ✓ Action extraction verified!")
        print(f"    Method: {extraction_method}")
        print(f"    Action shape: {action_shape}")
        print(f"    Action dims: {len(action)} (expected: 7)")
        print(f"    Action range: [{action.min():.3f}, {action.max():.3f}]")

        test_env.close()

    except Exception as e:
        print(f"  ✗ FATAL: Sanity check failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"  ")
        print(f"  Cannot proceed with data collection.")
        return

    # Collect rollouts
    print(f"\n[4/6] Collecting {num_rollouts} rollouts...")
    print(f"  Target: ~{num_rollouts} trajectories")
    print(f"  Max steps per rollout: {max_steps}")

    trajectories = []
    collision_count = 0
    success_count = 0
    failure_count = 0

    pbar = tqdm(total=num_rollouts, desc="Collecting rollouts")

    for rollout_idx in range(num_rollouts):
        # Round-robin through tasks
        task_id = rollout_idx % num_tasks

        # Get task from task suite
        task = task_suite.get_task(task_id)
        instruction = task.language

        # Build BDDL file path
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file
        )

        # Create environment
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=camera_height,
            camera_widths=camera_width,
        )
        env.seed(task_id)

        # Initialize environment with task init state
        init_states = task_suite.get_task_init_states(task_id)
        env.reset()
        env.set_init_state(init_states[0])

        # Create collision detector
        detector = CollisionDetector(env)

        try:
            # Collect rollout
            trajectory = collect_rollout(
                env=env,
                model=model,
                processor=processor,
                collector=collector,
                detector=detector,
                instruction=instruction,
                max_steps=max_steps,
                device=device,
                debug_collision=debug_collision,
                debug_actions=debug_actions,
                debug_interval=debug_interval,
                debug_info=debug_info,
            )

            trajectories.append(trajectory)

            # Update stats
            if trajectory['collision_occurred']:
                collision_count += 1
            if trajectory['success']:
                success_count += 1
            if trajectory.get('failure'):
                failure_count += 1

            # Update progress bar
            pbar.set_postfix({
                'collisions': collision_count,
                'successes': success_count,
                'failures': failure_count,
                'collision_rate': f"{collision_count/(rollout_idx+1):.2%}",
            })
            pbar.update(1)

            # Checkpoint every 50 rollouts
            if (rollout_idx + 1) % 50 == 0:
                checkpoint_path = output_path.replace('.pkl', f'_checkpoint_{rollout_idx+1}.pkl')
                checkpoint_data = {
                    'trajectories': trajectories,
                    'metadata': {
                        'num_rollouts': rollout_idx + 1,
                        'benchmark': benchmark_name,
                        'max_steps': max_steps,
                        'checkpoint': True,
                    }
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"\n  ✓ Checkpoint saved: {rollout_idx + 1} rollouts")

        except Exception as e:
            print(f"\n  ! Error in rollout {rollout_idx}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up environment
            env.close()
            del env
            del detector

            # Clear CUDA cache every 10 rollouts to prevent memory fragmentation
            if device == 'cuda' and (rollout_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    pbar.close()

    # Print statistics
    print("\n[5/6] Collection Statistics:")
    print(f"  Total rollouts: {len(trajectories)}")
    total_steps = 0
    if len(trajectories) > 0:
        print(f"  Collisions: {collision_count} ({collision_count/len(trajectories):.2%})")
        print(f"  Successes: {success_count} ({success_count/len(trajectories):.2%})")
        print(f"  Failures: {failure_count} ({failure_count/len(trajectories):.2%})")

        total_steps = sum(len(t['steps']) for t in trajectories)
        print(f"  Total steps: {total_steps}")
        print(f"  Avg steps/rollout: {total_steps/len(trajectories):.1f}")
    else:
        print(f"  WARNING: No trajectories collected!")

    # Save data
    print(f"\n[6/6] Saving data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        'trajectories': trajectories,
        'metadata': {
            'num_rollouts': len(trajectories),
            'benchmark_name': benchmark_name,
            'max_steps': max_steps,
            'collision_count': collision_count,
            'success_count': success_count,
            'failure_count': failure_count,
            'total_steps': total_steps,
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  ✓ Data saved to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("✅ Phase 1.3 Complete!")
    print("=" * 60)
    print("\nNext step: Compute per-dimension risk labels")
    print("  python scripts/compute_risk_labels.py")


def main():
    parser = argparse.ArgumentParser(description="Collect rollout data with collision labels")
    parser.add_argument('--output', type=str, default='data/phase1/rollouts.pkl',
                        help='Output path for collected data')
    parser.add_argument('--num-rollouts', type=int, default=2000,
                        help='Number of rollouts to collect')
    parser.add_argument('--benchmark', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal',
                                 'libero_10', 'libero_90'],
                        help='LIBERO benchmark to use')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Maximum steps per rollout')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda or cpu)')
    parser.add_argument('--debug-collision', action='store_true',
                        help='Print contact geom names periodically')
    parser.add_argument('--debug-actions', action='store_true',
                        help='Print action norms periodically')
    parser.add_argument('--debug-info', action='store_true',
                        help='Print reward/done/success/info keys periodically')
    parser.add_argument('--debug-interval', type=int, default=10,
                        help='Steps between debug prints')
    parser.add_argument('--camera-height', type=int, default=128,
                        help='Camera height for LIBERO rendering')
    parser.add_argument('--camera-width', type=int, default=128,
                        help='Camera width for LIBERO rendering')

    args = parser.parse_args()

    collect_data(
        output_path=args.output,
        num_rollouts=args.num_rollouts,
        benchmark_name=args.benchmark,
        max_steps=args.max_steps,
        device=args.device,
        debug_collision=args.debug_collision,
        debug_actions=args.debug_actions,
        debug_interval=args.debug_interval,
        debug_info=args.debug_info,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )


if __name__ == "__main__":
    main()
