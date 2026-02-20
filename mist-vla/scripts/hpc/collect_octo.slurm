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
#  OCTO DATA COLLECTION — Separate JAX-based VLA
#
#  Step 1: Setup octo conda env (rebuild to fix deps)
#  Step 2: Collect rollouts on libero_spatial using Octo-Base
###############################################################################

set -eo pipefail

echo "================================================================"
echo "  OCTO DATA COLLECTION"
echo "================================================================"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST:-unknown}"
echo "Date:    $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================================"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"

cd /mnt/onefs/home/asahai2024/mist-vla

# ─── Step 1: Rebuild Octo env ───
ENV_NAME="octo-env"

echo "Rebuilding Octo environment to fix JAX/CUDA + flax deps..."
conda remove -n "${ENV_NAME}" --all -y 2>/dev/null || true
bash scripts/hpc/setup_octo_env.sh

conda activate "${ENV_NAME}"

# GPU / rendering
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | head -n1)"
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# JAX GPU detection
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$(dirname $(dirname $(which nvcc)))"
export LD_LIBRARY_PATH="/mnt/beegfs/home/asahai2024/.conda/envs/octo-env/lib:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# HuggingFace cache
CACHE_ROOT="/mnt/onefs/home/asahai2024/hf_cache"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:${PYTHONPATH:-}"

mkdir -p logs data/multi_model/octo_spatial

echo ""
echo "JAX/GPU verification:"
python3 -c "
import jax
print(f'  JAX version: {jax.__version__}')
print(f'  JAX devices: {jax.devices()}')
try:
    gpu_devs = jax.devices('gpu')
    print(f'  GPU devices: {gpu_devs}')
except:
    print('  WARNING: No GPU found - will try anyway')
import flax
print(f'  Flax: {flax.__version__}')
from octo.model.octo_model import OctoModel
print(f'  OctoModel: importable')
"
echo ""

# ─── Step 2: Collect Data ───
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
