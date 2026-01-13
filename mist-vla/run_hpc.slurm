#!/bin/bash
#SBATCH --job-name=mist-vla
#SBATCH --output=logs/mist_vla_%j.out
#SBATCH --error=logs/mist_vla_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Adjust the above based on your HPC's configuration
# Common partitions: gpu, gpu-long, a100, v100, etc.

echo "========================================="
echo "MIST-VLA Experiment on HPC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Load modules (adjust for your HPC)
# module load cuda/12.1
# module load gcc/11.2.0
# module load miniconda3

# Activate conda environment
source activate mist-vla

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p data/rollouts
mkdir -p data/steering_vectors
mkdir -p results

# Print GPU info
nvidia-smi

echo "========================================="
echo "Step 1: Collecting Failure Data"
echo "========================================="
python scripts/collect_failure_data.py \
    --env libero_spatial \
    --n_success 100 \
    --n_failure 100 \
    --save_dir data/rollouts

echo "========================================="
echo "Step 2: Training Failure Detector"
echo "========================================="
python scripts/train_failure_detector.py \
    --data_dir data/rollouts \
    --detector_type mlp \
    --epochs 50

echo "========================================="
echo "Step 3: Extracting Steering Vectors"
echo "========================================="
python scripts/extract_steering_vectors.py \
    --model openvla/openvla-7b \
    --save_path data/steering_vectors

echo "========================================="
echo "Step 4: Running Evaluation"
echo "========================================="
python scripts/run_libero_eval.py \
    --task_suite libero_spatial \
    --model_path openvla/openvla-7b \
    --detector_path checkpoints/best_detector.pt \
    --steering_path data/steering_vectors/all_vectors.pt \
    --n_episodes 50

echo "========================================="
echo "Experiment Complete!"
echo "========================================="

# Copy results to a timestamped directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p results_archive/${TIMESTAMP}
cp -r results/* results_archive/${TIMESTAMP}/
cp -r checkpoints results_archive/${TIMESTAMP}/

echo "Results saved to: results_archive/${TIMESTAMP}/"
