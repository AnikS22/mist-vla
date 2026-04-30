#!/usr/bin/env bash
# Local smoke runs mirroring mist-vla/scripts/hpc/*.slurm (no Slurm).
#
# The HPC jobs mostly call:
#   scripts/eval_tuning.py          (OpenVLA-OFT + LIBERO Franka, EGL)
#   scripts/eval_act_steering.py    (ACT + LIBERO)
#   scripts/collect_baseline_data.py
#
# **GUI vs headless**
#   - Commands ending in *-gui* open a native viewer (needs $DISPLAY, MUJOCO_GL=glfw for LIBERO).
#   - HPC-style eval uses EGL/offscreen (set in eval-tuning-one-episode).
#
# Usage:
#   ./scripts/local_run_from_hpc.sh libero-gui              # LIBERO Franka — interactive MuJoCo window
#   ./scripts/local_run_from_hpc.sh libero-smoke            # headless LIBERO check (fast)
#   ./scripts/local_run_from_hpc.sh libero-frame             # save one agentview PNG (+ xdg-open)
#   ./scripts/local_run_from_hpc.sh maniskill-gui           # xArm ManiSkill + SAPIEN viewer
#   ./scripts/local_run_from_hpc.sh libero-record            # headless MP4 (LIBERO Franka, random policy)
#   ./scripts/local_run_from_hpc.sh eval-tuning-one-episode  # full VLA eval (GPU + HF); headless
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export PYTHONPATH="${REPO}:${REPO}/../openvla-oft:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODE="${1:-help}"

# Default headless MuJoCo (matches Slurm EGL-style local runs)
export_headless_mujoco() {
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  if [[ -z "${PYOPENGL_PLATFORM:-}" ]] && [[ "${MUJOCO_GL}" == "egl" ]]; then
    export PYOPENGL_PLATFORM=egl
  fi
}

# Interactive desktop window (robosuite / MuJoCo glfw)
export_gui_mujoco() {
  if [[ -z "${DISPLAY:-}" ]] && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "No DISPLAY or WAYLAND_DISPLAY — cannot open a GUI. Use a desktop session or SSH -X." >&2
    exit 1
  fi
  export MUJOCO_GL=glfw
  unset PYOPENGL_PLATFORM || true
}

libero_smoke() {
  export_headless_mujoco
  python3 archive/old_tests/test_libero_sim.py
}

libero_gui() {
  export_gui_mujoco
  python3 scripts/libero_gui_smoke.py "${@:2}"
}

libero_frame() {
  export_headless_mujoco
  python3 << 'PY'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(r"${REPO}")
sys.path.insert(0, str(REPO))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
bddl_file = task_suite.get_task_bddl_file_path(0)
env = OffScreenRenderEnv(
    bddl_file_name=bddl_file,
    render_camera="agentview",
    camera_heights=256,
    camera_widths=256,
)
obs = env.reset()
img = obs["agentview_image"]
if img.dtype != np.uint8:
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
out = REPO / "figures" / "local_libero_agentview.png"
out.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(img).save(out)
env.close()
print(f"Saved: {out}")
PY
  if [[ -n "${DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${REPO}/figures/local_libero_agentview.png" 2>/dev/null || true
  fi
}

eval_tuning_one_episode() {
  export_headless_mujoco
  CKPT="${MLP_CKPT:-${REPO}/hpc_mirror/checkpoints/eef_correction_mlp/best_model.pt}"
  MODEL="${OPENVLA_MODEL:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
  if [[ ! -f "$CKPT" ]]; then
    echo "Missing probe checkpoint: $CKPT" >&2
    exit 1
  fi
  echo "Using model=$MODEL  mlp=$CKPT"
  echo "(First run may download HF weights — same stack as scripts/hpc/eval_tuning.slurm)"
  python3 -u scripts/eval_tuning.py \
    --model-name "$MODEL" \
    --mlp-checkpoint "$CKPT" \
    --env libero_spatial \
    --tasks 0 \
    --episodes-per-task 1 \
    --modes steering \
    --alpha 0.1 \
    --ema-beta 0.7 \
    --action-scale 0.05 \
    --correction-threshold 0.005 \
    --max-correction 0.02 \
    --use-fail-gate \
    --fail-threshold 0.5 \
    --seed 42 \
    --save-dir "${REPO}/results/local_smoke_eval_tuning"
}

libero_record() {
  export_headless_mujoco
  python3 scripts/libero_record_mp4.py -o "${REPO}/figures/local_libero_recorded.mp4" --frames 120 --fps 12
}

maniskill_demo() {
  local gui_flag=()
  if [[ "${MODE}" == "maniskill-gui" ]]; then
    if [[ -z "${DISPLAY:-}" ]] && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
      echo "No DISPLAY — use maniskill-demo (headless) or connect a desktop session." >&2
      exit 1
    fi
    gui_flag=(--gui)
  fi
  # Avoid interactive asset download prompts in non-interactive shells.
  export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
  python3 scripts/demo_sim_vs_realworld.py \
    --output figures/local_maniskill_demo.mp4 \
    --steps 80 \
    --fps 10 \
    --mode steering \
    "${gui_flag[@]}" \
    "${@:2}"
}

case "$MODE" in
  libero-smoke) libero_smoke ;;
  libero-gui) libero_gui "$@" ;;
  libero-frame) libero_frame ;;
  libero-record) libero_record ;;
  eval-tuning-one-episode) eval_tuning_one_episode ;;
  maniskill-demo) maniskill_demo "$@" ;;
  maniskill-gui) maniskill_demo "$@" ;;
  help|*)
    sed -n '1,28p' "$0" | tail -n +2
    echo "Commands: libero-gui | libero-smoke | libero-frame | libero-record | eval-tuning-one-episode | maniskill-demo | maniskill-gui"
    ;;
esac
