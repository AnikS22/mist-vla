#!/usr/bin/env python3
"""
Visualize OpenVLA controlling a LIBERO task using offscreen frames.
Displays frames with matplotlib (or saves frames if no display).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure repo root is on sys.path for src imports when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.models.vla_wrapper import OpenVLAWrapper


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            return Image.fromarray(image)
        return Image.fromarray((image * 255).astype(np.uint8))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize OpenVLA on LIBERO")
    parser.add_argument("--benchmark", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--save-frames", type=str, default="")
    args = parser.parse_args()

    # PyTorch 2.6+ weights_only default breaks LIBERO init states
    try:
        torch.serialization.add_safe_globals(
            [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
        )
    except Exception:
        pass
    _orig_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _torch_load_compat

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.benchmark]()
    task = task_suite.get_task(args.task_id)
    instruction = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        render_gpu_device_id=args.gpu_id,
    )
    init_states = task_suite.get_task_init_states(args.task_id)
    env.reset()
    env.set_init_state(init_states[0])

    if args.device == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
    policy = OpenVLAWrapper("openvla/openvla-7b", device=args.device)

    save_dir = Path(args.save_frames) if args.save_frames else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    show = False
    try:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots()
        show = True
    except Exception:
        show = False

    obs = env.reset()
    print(f"Instruction: {instruction}")

    for step in range(args.max_steps):
        image = obs.get("agentview_image")
        if image is None:
            image = obs.get("image")
        image = _to_pil(image)
        action, _ = policy.get_action_with_features(image, instruction)
        action_np = action.detach().cpu().numpy()

        if show:
            ax.clear()
            ax.imshow(image)
            ax.set_title(f"Step {step}")
            ax.axis("off")
            plt.pause(0.001)

        if save_dir:
            image.save(save_dir / f"frame_{step:04d}.png")

        obs, reward, done, info = env.step(action_np)
        if done:
            print(f"Done at step {step}, info={info}")
            break

    if show:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()

    env.close()
    policy.close()


if __name__ == "__main__":
    main()
