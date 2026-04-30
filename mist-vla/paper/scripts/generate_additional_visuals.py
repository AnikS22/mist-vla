#!/usr/bin/env python3
"""Generate additional paper visuals:
1) Annotated LIBERO benchmark scene panels
2) Task-level safety scatter from frozen eval JSONs
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data"
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    out = FIG / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


def _collect_task_points(eval_file: Path):
    d = json.loads(eval_file.read_text())
    per = d.get("per_task", {})
    xs, ys, labels = [], [], []
    if not isinstance(per, dict):
        return xs, ys, labels
    for task_id, row in sorted(per.items(), key=lambda kv: int(kv[0])):
        if not isinstance(row, dict):
            continue
        steering = row.get("steering", {})
        if not isinstance(steering, dict):
            continue
        ir = steering.get("mean_ir")
        corr = steering.get("mean_corr_mag_m")
        if isinstance(ir, (int, float)) and isinstance(corr, (int, float)):
            xs.append(float(ir))
            ys.append(float(corr) * 1000.0)  # m -> mm
            labels.append(str(task_id))
    return xs, ys, labels


def fig_safety_phase_space() -> None:
    # Use one frozen representative run with per-task steering statistics.
    cand = DATA / "category1_eval_results.json"
    if not cand.exists():
        return
    xs, ys, labels = _collect_task_points(cand)
    if not xs:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(xs, ys, color="#2e8b57", s=52, alpha=0.9)
    for x, y, t in zip(xs, ys, labels):
        ax.text(x + 0.005, y + 0.02, f"T{t}", fontsize=8)
    ax.set_xlabel("Intervention rate (mean_ir)")
    ax.set_ylabel("Mean correction magnitude (mm)")
    ax.set_title("Task-Level Safety Controller Operating Regime")
    ax.grid(alpha=0.25)
    save(fig, "15_task_safety_phase_space.png")


def fig_libero_annotated_panels() -> None:
    try:
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as e:
        print("skip LIBERO panel generation:", e)
        return

    # PyTorch 2.6+ compatibility for LIBERO init-state loading.
    try:
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype])
    except Exception:
        pass
    _orig_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _torch_load_compat

    bench = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_ids = [0, 1, 2, 3]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    axes = axes.flatten()

    for ax, task_id in zip(axes, task_ids):
        task = bench.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=0,
        )
        try:
            obs = env.reset()
            init_states = bench.get_task_init_states(task_id)
            obs = env.set_init_state(init_states[0])
            img = obs.get("agentview_image")
            if img is None:
                img = obs.get("image")
            if img is None:
                ax.text(0.5, 0.5, "No frame", ha="center", va="center")
                ax.axis("off")
                continue

            # Handle upside-down convention used in some LIBERO obs streams.
            frame = np.asarray(img)
            if frame.ndim == 3:
                frame = frame[::-1]

            ax.imshow(frame)
            wrapped = "\n".join(textwrap.wrap(task.language, width=34))
            ax.set_title(f"Task {task_id}: {wrapped}", fontsize=9)
            ax.text(
                0.01,
                0.02,
                "LIBERO-Spatial",
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, pad=2),
            )
            ax.axis("off")
        finally:
            env.close()

    plt.suptitle("Annotated LIBERO Benchmark Scenes (Initial States)", fontsize=12, y=0.99)
    save(fig, "16_libero_annotated_panels.png")


def main() -> None:
    fig_safety_phase_space()
    fig_libero_annotated_panels()


if __name__ == "__main__":
    main()

