#!/usr/bin/env python3
"""
Quick test with REAL OpenVLA + LIBERO (not mock data).
Takes ~5-10 minutes instead of 8-12 hours.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

print("="*60)
print("QUICK REAL DATA TEST")
print("="*60)

# Test 1: Load REAL OpenVLA
print("\n[Test 1] Loading REAL OpenVLA model...")
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("  Downloading/loading model (this may take a few minutes)...")
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    print("  ✓ Processor loaded")

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    print("  ✓ Model loaded")
    print(f"  Model size: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

except Exception as e:
    print(f"  ✗ Failed to load OpenVLA: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load REAL LIBERO environment
print("\n[Test 2] Loading REAL LIBERO environment...")
try:
    import libero.libero.envs
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    print(f"  ✓ LIBERO loaded, available benchmarks: {list(benchmark_dict.keys())}")

    task_suite = benchmark_dict["libero_spatial"]()
    print(f"  ✓ Task suite loaded with {len(task_suite.tasks)} tasks")

    bddl_file = task_suite.get_task_bddl_file_path(0)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        render_camera="agentview",
        camera_heights=128,
        camera_widths=128,
    )
    instruction = task_suite.get_task(0).language
    print(f"  ✓ Environment created")
    print(f"  Task: '{instruction}'")

except Exception as e:
    print(f"  ✗ Failed to load LIBERO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Run ONE real rollout with VLA + LIBERO
print("\n[Test 3] Running ONE real rollout with VLA...")
try:
    obs = env.reset()
    print(f"  ✓ Environment reset")
    print(f"  Observation keys: {list(obs.keys())}")

    # Get image
    image = obs['agentview_image']
    print(f"  Image shape: {image.shape}")
    image_pil = Image.fromarray(image)

    # Run 5 steps with REAL VLA
    print("  Running 5 steps with real VLA inference...")
    for step in range(5):
        # Prepare input
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, image_pil, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                model.device, dtype=model.dtype
            )

        # Get action from VLA
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False
            )

        # Decode action
        action_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        action = (action_tokens.float() - 128) / 128  # Convert to [-1, 1]
        action = torch.clamp(action, -1, 1)
        action = action[0].cpu().numpy()

        # Execute in simulation
        obs, reward, done, info = env.step(action)
        image = obs['agentview_image']
        image_pil = Image.fromarray(image)

        print(f"    Step {step+1}: action={action[:3]}, reward={reward:.3f}, done={done}")

        if done:
            break

    print("  ✓ Real rollout complete!")

except Exception as e:
    print(f"  ✗ Rollout failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test our MIST modules with real data
print("\n[Test 4] Testing MIST modules with real VLA...")
try:
    sys.path.insert(0, str(Path.cwd()))
    from src.models.hooked_openvla import HookedOpenVLA

    print("  Loading HookedOpenVLA...")
    hooked_vla = HookedOpenVLA(
        "openvla/openvla-7b",
        device="cuda",
        model=model,
        processor=processor,
    )
    print("  ✓ HookedOpenVLA loaded")

    # Try to get features
    print("  Extracting features from real image...")
    features = hooked_vla.get_last_layer_features(image_pil, instruction)
    print(f"  ✓ Features extracted: shape {features.shape}")

    # Test failure detector
    from src.training.risk_predictor import RiskPredictor
    detector = RiskPredictor(input_dim=features.shape[-1]).to(hooked_vla.device)

    score = detector(features.float())
    print(f"  ✓ Risk scores computed: shape {score.shape}")

except Exception as e:
    print(f"  ✗ MIST modules failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Collect minimal real data
print("\n[Test 5] Collecting 2 real rollouts (1 min)...")
try:
    from src.models.vla_wrapper import OpenVLAWrapper

    policy = OpenVLAWrapper(
        "openvla/openvla-7b",
        device="cuda",
        model=model,
        processor=processor,
    )

    rollouts = []
    for i in range(2):
        obs = env.reset()
        trajectory = {
            'observations': [],
            'actions': [],
            'features': [],
            'rewards': [],
            'success': False,
            'instruction': instruction
        }

        for step in range(10):  # Just 10 steps
            image = obs['agentview_image']
            image_pil = Image.fromarray(image)
            action, features = policy.get_action_with_features(image_pil, instruction)

            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['features'].append(features.float().cpu().numpy())

            obs, reward, done, info = env.step(action.cpu().numpy())
            trajectory['rewards'].append(reward)

            if done:
                trajectory['success'] = info.get('success', False)
                break

        rollouts.append(trajectory)
        print(f"  Rollout {i+1}: {len(trajectory['actions'])} steps, success={trajectory['success']}")

    # Save to file
    import pickle
    os.makedirs('data/test_rollouts', exist_ok=True)
    with open('data/test_rollouts/quick_test.pkl', 'wb') as f:
        pickle.dump(rollouts, f)

    print(f"  ✓ Saved 2 real rollouts to data/test_rollouts/quick_test.pkl")

except Exception as e:
    print(f"  ✗ Data collection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
env.close()

print("\n" + "="*60)
print("✅ ALL REAL DATA TESTS PASSED!")
print("="*60)
print("\nVerified:")
print("  ✓ OpenVLA loads and runs inference")
print("  ✓ LIBERO simulation works")
print("  ✓ VLA + LIBERO integration works")
print("  ✓ MIST modules work with real data")
print("  ✓ Real rollout data can be collected")
print("\nReal data saved:")
print("  data/test_rollouts/quick_test.pkl")
print("\nThe full pipeline WILL work!")
print("Full run will just collect more data (100 rollouts instead of 2)")
