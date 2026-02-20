#!/bin/bash
#SBATCH --job-name=steer_sim
#SBATCH --output=logs/steering_sim_%j.out
#SBATCH --error=logs/steering_sim_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu

set -eo pipefail

echo "===== OFFLINE STEERING SIMULATOR ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "======================================"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:$PYTHONPATH"
cd /mnt/onefs/home/asahai2024/mist-vla

mkdir -p logs results/steering_sim

# ── Step 1: Merge multi-suite data if needed ──
echo ""
echo "Step 1: Merging multi-suite data..."
python -c "
import pickle, glob, os
from pathlib import Path

out_dir = Path('data/combined')
out_dir.mkdir(parents=True, exist_ok=True)

succ_out = out_dir / 'success_rollouts.pkl'
fail_out = out_dir / 'failure_rollouts.pkl'

all_succ, all_fail = [], []
for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
    d = Path(f'data/multi_suite/{suite}')
    for pkl in d.glob('*.pkl'):
        if '_partial' in pkl.name:
            continue
        with open(pkl, 'rb') as f:
            rols = pickle.load(f)
        s = [r for r in rols if r.get('success', False)]
        f_ = [r for r in rols if not r.get('success', False)]
        print(f'  {pkl}: {len(s)}S + {len(f_)}F')
        all_succ.extend(s)
        all_fail.extend(f_)

print(f'  TOTAL: {len(all_succ)}S + {len(all_fail)}F = {len(all_succ)+len(all_fail)}')

with open(succ_out, 'wb') as f:
    pickle.dump(all_succ, f)
with open(fail_out, 'wb') as f:
    pickle.dump(all_fail, f)
print(f'  Saved to {out_dir}/')
"

# ── Step 2: Retrain EEF Correction MLP on all 4 suites ──
echo ""
echo "Step 2: Training EEF Correction MLP on merged multi-suite data..."
python scripts/train_eef_correction_mlp.py \
    --success-data data/combined/success_rollouts.pkl \
    --failure-data data/combined/failure_rollouts.pkl \
    --epochs 80 \
    --lr 5e-4 \
    --batch-size 256 \
    --hidden-dim 512 \
    --save-dir checkpoints/eef_correction_mlp \
    --input-noise 0.01 \
    --corr-mag-penalty 0.1 \
    --run-loo \
    --seed 42

# ── Step 3: Run Offline Steering Simulator ──
echo ""
echo "Step 3: Running offline steering simulation..."
python scripts/sim_steering_math.py \
    --checkpoint checkpoints/eef_correction_mlp/best_model.pt \
    --data-dirs data/multi_suite \
    --alpha 1.0 \
    --sweep-alpha \
    --run-loo \
    --save-dir results/steering_sim

echo ""
echo "===== COMPLETE ====="
echo "Results: results/steering_sim/steering_report.json"
echo "Model:   checkpoints/eef_correction_mlp/best_model.pt"
date
