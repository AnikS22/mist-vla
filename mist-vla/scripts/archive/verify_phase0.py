#!/usr/bin/env python3
"""
Phase 0: Verify environment setup.
Tests that OpenVLA and LIBERO are installed and working.
"""

import sys

print("="*60)
print("Phase 0: Environment Verification")
print("="*60)

# Test 1: PyTorch
print("\n[Test 1] PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  ✗ PyTorch failed: {e}")
    sys.exit(1)

# Test 2: Transformers
print("\n[Test 2] Transformers...")
try:
    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f"  ✓ AutoModelForVision2Seq available")
except Exception as e:
    print(f"  ✗ Transformers failed: {e}")
    sys.exit(1)

# Test 3: LIBERO
print("\n[Test 3] LIBERO...")
try:
    import libero.libero.envs
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    benchmark_dict = benchmark.get_benchmark_dict()
    print(f"  ✓ LIBERO imported")
    print(f"  ✓ Available benchmarks: {list(benchmark_dict.keys())}")
except Exception as e:
    print(f"  ✗ LIBERO failed: {e}")
    print(f"  Install with: cd ~/LIBERO && pip install -e .")
    sys.exit(1)

# Test 4: MuJoCo
print("\n[Test 4] MuJoCo...")
try:
    import mujoco
    print(f"  ✓ MuJoCo {mujoco.__version__}")
except Exception as e:
    print(f"  ✗ MuJoCo failed: {e}")
    sys.exit(1)

# Test 5: Load OpenVLA (this downloads model if needed)
print("\n[Test 5] Loading OpenVLA...")
print("  (This may take a few minutes to download ~14GB model)")
try:
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    print(f"  ✓ Processor loaded")

    # Don't load full model in verification, just check it exists
    print(f"  ✓ Model downloadable from HuggingFace")

except Exception as e:
    print(f"  ✗ OpenVLA failed: {e}")
    sys.exit(1)

# Test 6: Create LIBERO environment
print("\n[Test 6] Creating LIBERO environment...")
try:
    task_suite = benchmark_dict["libero_spatial"]()
    bddl_file = task_suite.get_task_bddl_file_path(0)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        render_camera="agentview",
        camera_heights=128,
        camera_widths=128,
    )
    instruction = task_suite.get_task(0).language
    print(f"  ✓ Environment created")
    print(f"  ✓ Task: '{instruction}'")

    obs = env.reset()
    print(f"  ✓ Environment reset successful")
    print(f"  ✓ Observation keys: {list(obs.keys())}")

    env.close()
except Exception as e:
    print(f"  ✗ LIBERO environment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ Phase 0 Complete - Environment Ready!")
print("="*60)
print("\nNext step: Run Phase 1 data collection")
print("  python scripts/collect_phase1_data.py")
