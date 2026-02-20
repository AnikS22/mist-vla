#!/bin/bash
# Test the improved training pipeline locally

set -e

cd /home/mpcr/Desktop/SalusV5/mist-vla

# Activate environment
source /home/mpcr/miniconda/etc/profile.d/conda.sh
conda activate mist-vla

# Prepare data with balanced labels
echo "=== Preparing balanced dataset ==="
python scripts/prepare_training_data_v2.py \
    --success data/rollouts_oft_eval_test/success_rollouts.pkl \
    --failure data/rollouts_oft_eval_test/failure_rollouts.pkl \
    --output data/training_datasets/local_test_v2.pkl \
    --mode binary \
    --k-before-failure 20 \
    --balance-ratio 1.0

# Train improved model
echo ""
echo "=== Training improved risk predictor ==="
python scripts/train_risk_predictor_v2.py \
    --data data/training_datasets/local_test_v2.pkl \
    --output-dir checkpoints/risk_predictor_local_test \
    --epochs 20 \
    --batch-size 64 \
    --lr 5e-4 \
    --device cpu

echo ""
echo "=== Done! ==="
