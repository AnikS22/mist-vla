#!/bin/bash
#SBATCH --job-name=train_act
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/train_act_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/train_act_%j.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7

###############################################################################
#  ACT — Train on LIBERO demos → Collect full rollout data
#  (SafeVLA format: features + EEF + actions + success/failure)
#
#  Row 6 of the paper table
###############################################################################

set -eo pipefail
export PYTHONUNBUFFERED=1

echo "================================================================"
echo "  ACT — Train + Collect Data"
echo "================================================================"
echo "  Job:  ${SLURM_JOB_ID}"
echo "  Node: ${SLURM_NODELIST:-unknown}"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "================================================================"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2
eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla

# GPU / rendering
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | head -n1)"
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
fi
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d"

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla/scripts:${PYTHONPATH:-}"

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda.device_count()} GPUs')"

DATASET_DIR=$(python3 -c "from libero.libero import get_libero_path; print(get_libero_path('datasets'))")
echo "  Dataset dir: ${DATASET_DIR}"

N_HDF5=$(find "${DATASET_DIR}" -name "*.hdf5" 2>/dev/null | wc -l)
if [ "${N_HDF5}" -eq 0 ]; then
    echo "ERROR: No LIBERO demo HDF5 files found!"
    exit 1
fi
echo "  Found ${N_HDF5} HDF5 demo files"

mkdir -p checkpoints/act data/multi_model/act_spatial

# ─── Phase 1: Train ───
echo ""
echo "================================================================"
echo "  Phase 1: Training ACT on LIBERO spatial"
echo "================================================================"

python -u scripts/train_act_libero.py \
    --dataset-dir "${DATASET_DIR}" \
    --benchmark LIBERO_SPATIAL \
    --output-dir checkpoints/act \
    --epochs 300 \
    --batch-size 128 \
    --seed 42

# ─── Phase 2: Collect rollout data (SafeVLA format) ───
echo ""
echo "================================================================"
echo "  Phase 2: Collecting rollout data (SafeVLA format)"
echo "================================================================"

python -u scripts/collect_baseline_data.py \
    --model-type act \
    --checkpoint checkpoints/act/best_model.pt \
    --env libero_spatial \
    --n-success 50 --n-failure 50 \
    --max-attempts-per-task 500 \
    --save-dir data/multi_model/act_spatial

echo ""
echo "================================================================"
echo "  ACT — COMPLETE"
echo "  Data:    data/multi_model/act_spatial/"
echo "  Model:   checkpoints/act/best_model.pt"
echo "================================================================"
