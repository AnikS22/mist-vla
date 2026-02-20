#!/usr/bin/env python3
"""
Test LIBERO simulation environment (no VLA needed).
"""

import numpy as np

print("="*60)
print("LIBERO Simulation Test")
print("="*60)

# Test 1: Import LIBERO
print("\n[Test 1] Importing LIBERO...")
try:
    import libero.libero.envs
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    print("✓ LIBERO imports successful")
except Exception as e:
    print(f"✗ LIBERO import failed: {e}")
    exit(1)

# Test 2: Get benchmarks
print("\n[Test 2] Getting available benchmarks...")
try:
    benchmark_dict = benchmark.get_benchmark_dict()
    print(f"✓ Available benchmarks: {list(benchmark_dict.keys())}")
except Exception as e:
    print(f"✗ Failed to get benchmarks: {e}")
    exit(1)

# Test 3: Load a task suite
print("\n[Test 3] Loading LIBERO-Spatial task suite...")
try:
    task_suite = benchmark_dict["libero_spatial"]()
    n_tasks = len(task_suite.tasks)
    print(f"✓ Task suite loaded with {n_tasks} tasks")
except Exception as e:
    print(f"✗ Failed to load task suite: {e}")
    exit(1)

# Test 4: Create environment
print("\n[Test 4] Creating environment for first task...")
try:
    bddl_file = task_suite.get_task_bddl_file_path(0)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        render_camera="agentview",
        camera_heights=128,
        camera_widths=128,
    )
    instruction = task_suite.get_task(0).language
    print(f"✓ Environment created")
    print(f"  Task instruction: '{instruction}'")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Reset environment
print("\n[Test 5] Resetting environment...")
try:
    obs = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation keys: {list(obs.keys())}")
    if 'agentview_image' in obs:
        img_shape = obs['agentview_image'].shape
        print(f"  Image shape: {img_shape}")
except Exception as e:
    print(f"✗ Failed to reset environment: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Take random actions
print("\n[Test 6] Taking random actions in simulation...")
try:
    low, high = env.env.action_spec
    action_dim = low.shape[0]
    print(f"  Action dimension: {action_dim}")

    for step in range(5):
        # Random action within env bounds
        action = np.random.uniform(low, high)
        obs, reward, done, info = env.step(action)
        print(f"  Step {step+1}: reward={reward:.3f}, done={done}")

        if done:
            break

    print("✓ Simulation steps successful")
except Exception as e:
    print(f"✗ Failed to step environment: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Close environment
print("\n[Test 7] Closing environment...")
try:
    env.close()
    print("✓ Environment closed successfully")
except Exception as e:
    print(f"✗ Failed to close environment: {e}")

print("\n" + "="*60)
print("✅ LIBERO SIMULATION TEST PASSED")
print("="*60)
print("\nLibero is working! You can run simulations.")
print("Next: Test with actual VLA on HPC")
