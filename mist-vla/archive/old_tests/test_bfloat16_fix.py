#!/usr/bin/env python3
"""Test BFloat16 dtype fix for OpenVLA"""

import sys
import torch
from PIL import Image
import numpy as np

print("="*60)
print("Testing BFloat16 Dtype Fix")
print("="*60)

# Test 1: Load OpenVLA in bfloat16
print("\n[1/3] Loading OpenVLA in bfloat16...")
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    print("  ✓ Processor loaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model loaded on {device} with dtype {model.dtype}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load LIBERO and get a real image
print("\n[2/3] Loading LIBERO environment...")
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    import os

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_spatial']()
    task = task_suite.get_task(0)
    instruction = task.language

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=128,
        camera_widths=128,
    )

    obs = env.reset()
    image = obs['agentview_image']
    image_pil = Image.fromarray(image)
    print(f"  ✓ Environment created, got image: {image.shape}")
    print(f"  Task: {instruction}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Process image WITH dtype conversion (the fix!)
print("\n[3/3] Testing image processing with dtype conversion...")
try:
    # Process inputs
    inputs = processor(images=image_pil, text=instruction, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"  Before dtype conversion:")
    print(f"    pixel_values dtype: {inputs['pixel_values'].dtype}")
    print(f"    model dtype: {model.dtype}")

    # THE FIX: Convert pixel_values to match model dtype
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model.dtype)

    print(f"  After dtype conversion:")
    print(f"    pixel_values dtype: {inputs['pixel_values'].dtype}")

    # Run forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"  ✓ Forward pass successful!")
    print(f"  Output type: {type(outputs)}")
    if hasattr(outputs, 'logits'):
        print(f"  Output shape: {outputs.logits.shape}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

env.close()

print("\n" + "="*60)
print("✅ BFloat16 DTYPE FIX VERIFIED!")
print("="*60)
print("\nThe fix works correctly:")
print("  ✓ Model loads in bfloat16")
print("  ✓ pixel_values converted to bfloat16")
print("  ✓ Forward pass succeeds without dtype errors")
print("\nThe same fix is now in scripts/collect_phase1_data.py")
