#!/bin/bash
###############################################################################
#  Setup Diffusion Policy + ACT baselines for LIBERO evaluation
#
#  These are Category 2 models (SOTA Imitation Learning baselines) needed
#  for the paper results table.
#
#  1. Diffusion Policy  — github.com/real-stanford/diffusion_policy
#  2. ACT               — github.com/tonyzhaozh/act
#
#  Both will be trained on LIBERO demonstration HDF5 data.
###############################################################################

set -eo pipefail

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2
eval "$(conda shell.bash hook)"

cd /mnt/onefs/home/asahai2024

# ═══════════════════════════════════════════════════════════════════
#  1. Diffusion Policy
# ═══════════════════════════════════════════════════════════════════
echo "================================================================"
echo "  Setting up Diffusion Policy"
echo "================================================================"

if [ ! -d "diffusion_policy" ]; then
    git clone https://github.com/real-stanford/diffusion_policy.git
    echo "  ✓ Cloned diffusion_policy"
else
    echo "  ✓ diffusion_policy already exists"
fi

# Create conda env
ENV_DP="dp-env"
if ! conda env list | grep -q "${ENV_DP}"; then
    echo "Creating ${ENV_DP} environment..."
    conda create -n "${ENV_DP}" python=3.9 -y
    conda activate "${ENV_DP}"
    cd diffusion_policy
    pip install -e .
    pip install robomimic==0.2.0
    pip install mujoco
    pip install libero
    cd ..
else
    echo "  ✓ ${ENV_DP} already exists"
fi

# ═══════════════════════════════════════════════════════════════════
#  2. ACT (Action Chunking with Transformers)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  Setting up ACT"
echo "================================================================"

if [ ! -d "act" ]; then
    git clone https://github.com/tonyzhaozh/act.git
    echo "  ✓ Cloned act"
else
    echo "  ✓ act already exists"
fi

# ACT shares the same env as DP for simplicity
# (both need PyTorch + robosuite/robomimic)

# ═══════════════════════════════════════════════════════════════════
#  3. Download LIBERO demonstration datasets
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  Downloading LIBERO demonstration datasets"
echo "================================================================"

conda activate mist-vla
cd /mnt/onefs/home/asahai2024/LIBERO

# LIBERO provides official demo datasets for training baselines
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_spatial \
    --download_dir /mnt/onefs/home/asahai2024/mist-vla/data/libero_demos 2>/dev/null \
    || echo "  ⚠ Demo download failed — may need manual download"

echo ""
echo "================================================================"
echo "  Setup complete"
echo "================================================================"
echo "  Diffusion Policy: /mnt/onefs/home/asahai2024/diffusion_policy"
echo "  ACT:              /mnt/onefs/home/asahai2024/act"
echo "  LIBERO demos:     /mnt/onefs/home/asahai2024/mist-vla/data/libero_demos"
echo "================================================================"
