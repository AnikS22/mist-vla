# HPC Quick Start Guide

## Step 1: Connect to FAU VPN (If Required)

If you're off-campus, connect to FAU VPN first.

## Step 2: Transfer Code to HPC

### Option A: Using the transfer script
```bash
cd /home/mpcr/Desktop/SalusV5/mist-vla
./TRANSFER_TO_HPC.sh
```

### Option B: Manual rsync
```bash
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    /home/mpcr/Desktop/SalusV5/mist-vla/ \
    asahai2024@athene-login.fau.edu:~/mist-vla/
```

### Option C: Using scp
```bash
scp -r /home/mpcr/Desktop/SalusV5/mist-vla \
    asahai2024@athene-login.fau.edu:~/
```

## Step 3: SSH into HPC

```bash
ssh asahai2024@athene-login.fau.edu
```

## Step 4: Setup Environment on HPC

```bash
cd ~/mist-vla

# Make setup script executable
chmod +x HPC_SETUP.sh

# Run setup (takes ~30-60 minutes)
bash HPC_SETUP.sh
```

**What this does:**
- Creates conda environment `mist-vla`
- Installs all dependencies
- Clones LIBERO and OpenVLA
- Downloads model weights
- Verifies everything works

## Step 5: Check FAU HPC Specifics

Before submitting job, check your HPC's specifics:

```bash
# Check available partitions
sinfo

# Check GPU types
sinfo -o "%20P %10G %10c %10m %25f %15N"

# Check your account/allocation
sacctmgr show associations user=$USER
```

## Step 6: Modify SLURM Script (If Needed)

Edit `run_hpc.slurm` to match your HPC:

```bash
nano run_hpc.slurm
```

Common changes needed:
```bash
#SBATCH --partition=gpu          # Change to your partition name
#SBATCH --gres=gpu:1            # Or gpu:a100:1, gpu:v100:1, etc.
#SBATCH --account=YOUR_ACCOUNT  # If required by your HPC
```

## Step 7: Submit Job

```bash
# Create log directory
mkdir -p logs

# Submit the job
sbatch run_hpc.slurm

# Note the job ID
```

## Step 8: Monitor Job

```bash
# Check job status
squeue -u $USER

# Watch live output
tail -f logs/mist_vla_JOBID.out

# Check error log if something fails
tail -f logs/mist_vla_JOBID.err

# Cancel job if needed
scancel JOBID
```

## Step 9: Wait for Results (~8-12 hours)

The pipeline will:
1. Collect failure data (2-4 hours)
2. Train detector (1-2 hours)
3. Extract steering vectors (0.5-1 hour)
4. Run evaluation (2-4 hours)

## Step 10: Download Results

From your LOCAL machine:

```bash
# Download results
rsync -avz asahai2024@athene-login.fau.edu:~/mist-vla/results/ \
    /home/mpcr/Desktop/SalusV5/mist-vla/results/

# Download checkpoints
rsync -avz asahai2024@athene-login.fau.edu:~/mist-vla/checkpoints/ \
    /home/mpcr/Desktop/SalusV5/mist-vla/checkpoints/

# Download logs
rsync -avz asahai2024@athene-login.fau.edu:~/mist-vla/logs/ \
    /home/mpcr/Desktop/SalusV5/mist-vla/logs/
```

## Troubleshooting

### Connection Issues

```bash
# Test SSH connection
ssh asahai2024@athene-login.fau.edu

# If fails, check:
# 1. Are you on VPN?
# 2. Is the hostname correct?
# 3. Try pinging: ping athene-login.fau.edu
```

### Job Not Starting

```bash
# Check job queue position
squeue -u $USER --start

# Check job details
scontrol show job JOBID

# Common issues:
# - No resources available (wait)
# - Wrong partition name (check with sinfo)
# - Exceeded allocation (check with sacctmgr)
```

### Out of Memory

If job crashes with OOM:

```bash
# Edit run_hpc.slurm and increase:
#SBATCH --mem=128G  # Instead of 64G

# Or request bigger GPU:
#SBATCH --gres=gpu:a100:1
```

### Setup Fails

```bash
# Check which step failed
less logs/setup_output.log

# Common fixes:
# - CUDA version mismatch: adjust in HPC_SETUP.sh
# - Module not found: check 'module avail'
# - Permission denied: check disk quota with 'quota'
```

## Quick Commands Reference

```bash
# Submit job
sbatch run_hpc.slurm

# Check status
squeue -u $USER

# Cancel job
scancel JOBID

# Interactive session (for testing)
srun --gres=gpu:1 --mem=64G --time=2:00:00 --pty bash

# Check GPU
nvidia-smi

# Monitor GPU usage live
watch -n 1 nvidia-smi

# Check disk usage
du -sh ~/mist-vla
quota
```

## Expected Output

When successful, you should see:

```
=== Results for libero_spatial ===
Overall Success Rate: 75-85%
Avg Failures Detected: 2-4
Recovery Rate: 60-80%

Experiment Complete!
Results saved to: results_archive/YYYYMMDD_HHMMSS/
```

## Next Steps After Completion

1. Download all results to local machine
2. Analyze with Jupyter notebooks
3. Generate figures for paper
4. Run ablation studies if needed

---

**Need Help?**

Check these files:
- `TEST_SUMMARY.md` - What was tested locally
- `HPC_TRANSFER_GUIDE.md` - Detailed transfer guide
- `GETTING_STARTED.md` - Full setup instructions

Good luck! 🚀
