#!/bin/bash
#SBATCH --job-name=octo_col
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/octo_collect_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/octo_collect_%j.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7

###############################################################################
#  OCTO DATA COLLECTION — JAX-based VLA
#
#  Phase 1: Build octo-env from scratch (if missing or broken)
#  Phase 2: Collect rollouts on libero_spatial
###############################################################################

set -eo pipefail

echo "================================================================"
echo "  OCTO DATA COLLECTION"
echo "================================================================"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST:-unknown}"
echo "Date:    $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================================"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"

cd /mnt/onefs/home/asahai2024/mist-vla

# ─── Phase 1: Ensure Octo env works ───
ENV_NAME="octo-env"

# Quick check: can we import the core deps?
echo ""
echo "Testing existing Octo environment..."
ENV_OK=false
if conda env list | grep -q "${ENV_NAME}"; then
    conda activate "${ENV_NAME}" 2>/dev/null || true
    if python3 -c "
import jax
if not hasattr(jax.random, 'KeyArray'):
    jax.random.KeyArray = jax.Array
import jaxlib; assert '0.4.28' in jaxlib.__version__
import tensorflow
import dlimp
from octo.model.octo_model import OctoModel
print('ENV OK')
" 2>/dev/null | grep -q "ENV OK"; then
        ENV_OK=true
        echo "  Existing env is healthy, skipping rebuild."
    else
        echo "  Existing env is broken, rebuilding..."
        conda deactivate 2>/dev/null || true
    fi
fi

if [ "${ENV_OK}" = false ]; then
    echo "Building Octo environment from scratch..."
    bash scripts/hpc/setup_octo_env.sh
    conda activate "${ENV_NAME}"
fi

# ─── GPU / Rendering setup ───
# With --gres=gpu:1, SLURM cgroup isolates exactly 1 GPU as device 0.
# Do NOT use SLURM_JOB_GPUS (physical ID) — it doesn't match the cgroup view.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# JAX GPU detection — point to CUDA toolkit and nvidia-cudnn-cu12 pip package
CUDA_DIR=$(dirname $(dirname $(which nvcc)))
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_DIR}"
# Include nvidia pip packages in library path (cuDNN .so lives here)
NVIDIA_PKGS="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
CUDNN_LIB=""
if [ -d "${NVIDIA_PKGS}/cudnn/lib" ]; then
    CUDNN_LIB="${NVIDIA_PKGS}/cudnn/lib"
fi
export LD_LIBRARY_PATH="${CUDNN_LIB}:${CONDA_PREFIX}/lib:${CUDA_DIR}/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# HuggingFace cache (shared across jobs)
CACHE_ROOT="/mnt/onefs/home/asahai2024/hf_cache"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:${PYTHONPATH:-}"

mkdir -p logs data/multi_model/octo_spatial

echo ""
echo "=== JAX/GPU Verification (on compute node) ==="
python3 << 'VERIFY'
import jax
# Compat shim: Octo uses jax.random.KeyArray removed in JAX 0.4.25+
if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jax.Array

import jaxlib
print(f"  jaxlib: {jaxlib.__version__}")
devs = jax.devices()
print(f"  JAX devices: {devs}")
gpu_devs = [d for d in devs if d.platform == "gpu"]
if gpu_devs:
    print(f"  GPU devices: {gpu_devs}")
else:
    print("  ERROR: No GPU detected!")
    import sys; sys.exit(1)

import tensorflow as tf
print(f"  TF: {tf.__version__}")

from octo.model.octo_model import OctoModel
print(f"  OctoModel: OK")
print("  All checks passed.")
VERIFY
echo ""

# ─── Phase 2: Collect Data ───
echo "Starting Octo data collection on libero_spatial..."
echo ""

python -u scripts/collect_octo_data.py \
    --model-name "hf://rail-berkeley/octo-base-1.5" \
    --env libero_spatial \
    --n_success 50 \
    --n_failure 50 \
    --max-attempts-per-task 500 \
    --save_dir data/multi_model/octo_spatial \
    --seed 42

echo ""
echo "================================================================"
echo "  OCTO COLLECTION COMPLETE"
echo "================================================================"
echo "  Data: data/multi_model/octo_spatial/"
echo "  Date: $(date)"
echo "================================================================"
