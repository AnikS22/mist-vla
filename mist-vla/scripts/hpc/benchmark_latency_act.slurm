#!/bin/bash
#SBATCH --job-name=act_latency
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/act_latency_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/act_latency_%j.err
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7

set -eo pipefail

echo "================================================================"
echo "  ACT LATENCY BENCHMARK (MPPI vs Steering)"
echo "================================================================"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Date:   $(date)"
echo "================================================================"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2
eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d"
export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python3 - <<'PY'
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f"CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
sys.exit(0 if ok else 2)
PY

mkdir -p logs results/eval_act_latency

ACT_CKPT="checkpoints/act/best_model.pt"
MLP_CKPT="checkpoints/eef_correction_mlp_act_honest/best_model.pt"

python -u scripts/eval_act_steering.py \
    --act-checkpoint "${ACT_CKPT}" \
    --mlp-checkpoint "${MLP_CKPT}" \
    --env libero_spatial \
    --tasks 0 1 \
    --modes mppi steering \
    --episodes-per-task 4 \
    --alpha 0.12 \
    --ema-beta 0.9 \
    --max-correction 0.003 \
    --correction-threshold 0.002 \
    --use-fail-gate \
    --fail-threshold 0.65 \
    --mppi-samples 16 \
    --mppi-temperature 5.0 \
    --save-dir results/eval_act_latency/job_${SLURM_JOB_ID}

echo "Saved: results/eval_act_latency/job_${SLURM_JOB_ID}/eval_results.json"
echo "Done: $(date)"
