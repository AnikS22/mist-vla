# HPC Transfer and Execution Guide

## Quick Summary
✅ Local: Use for code development and editing
🚀 HPC: Use for running the full pipeline (OpenVLA + LIBERO)

## Step 1: Transfer Code to HPC

```bash
# From your local machine
cd /home/mpcr/Desktop/SalusV5

# Transfer to HPC (adjust username and path)
rsync -avz --progress mist-vla/ username@hpc-address:/home/username/mist-vla/

# OR use scp
scp -r mist-vla/ username@hpc-address:/home/username/
```

## Step 2: SSH into HPC

```bash
ssh username@hpc-address
cd ~/mist-vla
```

## Step 3: Run Setup Script

```bash
# Make it executable
chmod +x HPC_SETUP.sh

# Run setup
bash HPC_SETUP.sh
```

This will:
- Create conda environment
- Install all dependencies
- Clone LIBERO and OpenVLA
- Download model weights
- Verify everything works

## Step 4: Submit Job

### Option A: Interactive Session (for testing)
```bash
# Request interactive GPU node
srun --gres=gpu:1 --mem=64G --time=2:00:00 --pty bash

# Activate environment
conda activate mist-vla

# Run a quick test
python scripts/verify_setup.py

# Try one component
python scripts/extract_steering_vectors.py
```

### Option B: Batch Job (for full pipeline)
```bash
# Create logs directory
mkdir -p logs

# Submit the job
sbatch run_hpc.slurm

# Check job status
squeue -u $USER

# Monitor output
tail -f logs/mist_vla_*.out
```

## Step 5: Monitor Progress

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/mist_vla_JOBID.out

# Check GPU usage
ssh compute-node  # Get node from squeue
nvidia-smi
```

## Expected Timeline

| Step | Time | Output |
|------|------|--------|
| Data Collection | 2-4 hours | `data/rollouts/*.pkl` |
| Train Detector | 1-2 hours | `checkpoints/best_detector.pt` |
| Extract Steering | 0.5-1 hour | `data/steering_vectors/all_vectors.pt` |
| Evaluation | 2-4 hours | `results/libero_spatial_results.json` |
| **TOTAL** | **~8-12 hours** | Complete results |

## HPC-Specific Adjustments

### 1. Adjust SLURM Parameters

Edit `run_hpc.slurm`:

```bash
#SBATCH --partition=gpu        # Change to your HPC partition
#SBATCH --gres=gpu:1          # Or gpu:a100:1, gpu:v100:1, etc.
#SBATCH --time=12:00:00       # Increase if needed
```

### 2. Load Required Modules

Some HPCs require module loads. Add to `run_hpc.slurm`:

```bash
module load cuda/12.1
module load gcc/11.2.0
module load miniconda3
```

Check your HPC documentation for correct module names.

### 3. Adjust CUDA Version

If your HPC has different CUDA version:

```bash
# In HPC_SETUP.sh, change:
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# Or
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
```

## Troubleshooting

### Job Fails Immediately
```bash
# Check error log
cat logs/mist_vla_JOBID.err

# Common issues:
# 1. Wrong partition name
# 2. Insufficient memory
# 3. CUDA version mismatch
```

### Out of Memory
```bash
# In run_hpc.slurm, increase:
#SBATCH --mem=128G  # Instead of 64G

# Or request larger GPU:
#SBATCH --gres=gpu:a100:1  # A100 has more memory
```

### Module Not Found
```bash
# Check available modules
module avail

# Load correct modules before running
module load cuda
module load gcc
```

## Retrieve Results

After job completes:

```bash
# From HPC, check results
cd ~/mist-vla
ls -lh results/

# Transfer results back to local
# On local machine:
rsync -avz username@hpc:/home/username/mist-vla/results/ ./results/
rsync -avz username@hpc:/home/username/mist-vla/checkpoints/ ./checkpoints/
```

## Quick Commands Cheatsheet

```bash
# Submit job
sbatch run_hpc.slurm

# Check status
squeue -u $USER

# Cancel job
scancel JOBID

# View output
tail -f logs/mist_vla_*.out

# Check remaining time
squeue -j JOBID --format="%.18i %.9P %.8j %.8u %.2t %.10M %.10L %.6D %R"

# Request interactive session
srun --gres=gpu:1 --mem=64G --time=2:00:00 --pty bash
```

## What to Expect

### Successful Run Output:
```
Step 1: Collecting Failure Data
Success: 100/100
Natural failure: 50/100
Injected failure: 100/100

Step 2: Training Failure Detector
Epoch 49: Train Loss = 0.234, Val Loss = 0.267
Best model saved!

Step 3: Extracting Steering Vectors
Computing steering vectors...
Saved to data/steering_vectors/all_vectors.pt

Step 4: Running Evaluation
=== Results for libero_spatial ===
Overall Success Rate: 78.5%
Avg Failures Detected: 2.3
Recovery Rate: 67.8%

Experiment Complete!
```

## Next Steps After HPC Run

1. **Download results**: `rsync` results back to local
2. **Analyze**: Use Jupyter notebooks to visualize
3. **Iterate**: Modify code locally, re-run on HPC
4. **Write paper**: Use results for publication

Good luck! 🚀
