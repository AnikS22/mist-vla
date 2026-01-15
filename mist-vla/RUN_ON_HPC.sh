#!/bin/bash
# Quick start script for HPC
# Copy-paste these commands on HPC

echo "========================================="
echo "MIST-VLA Quick Start on HPC"
echo "========================================="
echo ""

echo "Step 1: Remove old broken environment"
echo "---------------------------------------"
echo "Run: conda env remove -n mist-vla"
echo ""
read -p "Press Enter after you've done this..."

echo ""
echo "Step 2: Run new setup (takes ~30 mins)"
echo "---------------------------------------"
echo "Run: cd ~/mist-vla && bash HPC_SETUP.sh"
echo ""
echo "Watch for:"
echo "  ✓ Environment activated: mist-vla"
echo "  ✓ PyTorch installed"
echo "  ✓ Transformers installed"
echo "  ..."
echo "  ✅ Setup Complete!"
echo ""
read -p "Press Enter after setup completes successfully..."

echo ""
echo "Step 3: Submit job"
echo "---------------------------------------"
echo "Run: sbatch run_hpc.slurm"
echo ""
echo "Note the job ID, then monitor with:"
echo "  tail -f logs/mist_vla_JOBID.out"
echo ""

echo "========================================="
echo "✅ Ready to Go!"
echo "========================================="
