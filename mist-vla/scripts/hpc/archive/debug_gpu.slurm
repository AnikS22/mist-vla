#!/bin/bash
#SBATCH --job-name=mistvla_gpu_debug
#SBATCH --output=logs/mistvla_gpu_debug_%j.out
#SBATCH --error=logs/mistvla_gpu_debug_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu
#SBATCH --exclude=nodegpu041

set -euo pipefail

echo "=== GPU Debug ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "=== Devices ==="
ls -l /dev/nvidia* || true
ls -l /dev/dri || true

echo "=== nvidia-smi ==="
nvidia-smi -L || true
nvidia-smi || true

echo "=== lspci ==="
lspci | grep -i nvidia || true

echo "Done: $(date)"
