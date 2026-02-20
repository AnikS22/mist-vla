#!/bin/bash
#SBATCH --job-name=collect_all
#SBATCH --output=logs/collect_all_%j.out
#SBATCH --error=logs/collect_all_%j.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu
#SBATCH --constraint=a100

set -eo pipefail

echo "=== Collect All Tasks Data ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Start: $(date)"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
HF_CACHE_DIR="${SLURM_TMPDIR:-/mnt/onefs/home/asahai2024}/.cache/huggingface"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"

cd /mnt/onefs/home/asahai2024/mist-vla
export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:$PYTHONPATH"
export MUJOCO_GL=egl

echo "Collecting 5 success + 5 failure PER TASK across all 10 LIBERO-Spatial tasks..."
python scripts/collect_failure_data_oft_eval.py \
    --env libero_spatial \
    --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
    --n_success 5 \
    --n_failure 5 \
    --max-attempts-per-task 50 \
    --camera-res 256 \
    --save_dir data/rollouts_all_tasks \
    --seed 42 \
    --num-images 2

echo ""
echo "=== Collection Complete ==="
echo "End: $(date)"
