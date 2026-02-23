#!/bin/bash
#SBATCH --job-name=act_mlp
#SBATCH --partition=shortq7
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/act_mlp_%j.out
#SBATCH --error=logs/act_mlp_%j.err

set -euo pipefail
cd /mnt/onefs/home/asahai2024/mist-vla
mkdir -p logs

# ── Environment ──
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
eval "$(conda shell.bash hook)"
conda activate mist-vla

export CUDA_VISIBLE_DEVICES=0

echo "================================================================"
echo "  TRAIN EEF CORRECTION MLP ON ACT DATA (HONEST — CHUNK SUBSAMPLED)"
echo "  Input: 256-dim ACT encoder hidden states"
echo "  Data:  499S + 500F (libero_spatial, 10 tasks)"
echo "  Mode:  --subsample-chunks (only unique feature vectors)"
echo "         ACT uses 8-step action chunks → ~38 unique features/episode"
echo "         This prevents inflated metrics from duplicated features"
echo "================================================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Date: $(date)"
echo ""

# ── Train (honest — chunk subsampled) ──
python scripts/train_eef_correction_mlp.py \
    --success-data data/multi_model/act_spatial/success_rollouts.pkl \
    --failure-data data/multi_model/act_spatial/failure_rollouts.pkl \
    --save-dir checkpoints/eef_correction_mlp_act_honest \
    --epochs 80 \
    --lr 5e-4 \
    --batch-size 256 \
    --input-noise 0.01 \
    --corr-mag-penalty 0.1 \
    --subsample-chunks \
    --seed 42

echo ""
echo "================================================================"
echo "  ACT MLP TRAINING COMPLETE (HONEST)"
echo "  Checkpoint: checkpoints/eef_correction_mlp_act_honest/"
echo "  Date: $(date)"
echo "================================================================"
