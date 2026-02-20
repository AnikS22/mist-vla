#!/bin/bash
###############################################################################
#  Setup Octo environment on HPC (JAX + Octo model)
#  Run this interactively on a GPU node, or as a short SLURM job.
#
#  Octo is JAX-based (not PyTorch), so we need a separate conda env.
###############################################################################

set -eo pipefail

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"

ENV_NAME="octo-env"

echo "================================================================"
echo "  Setting up Octo environment: ${ENV_NAME}"
echo "================================================================"

# Create new env if it doesn't exist
if ! conda env list | grep -q "${ENV_NAME}"; then
    echo "Creating conda environment: ${ENV_NAME}..."
    conda create -n "${ENV_NAME}" python=3.10 -y
else
    echo "Environment ${ENV_NAME} already exists, updating..."
fi

conda activate "${ENV_NAME}"

# Install JAX with CUDA support (use cuda12_pip to bundle CUDA libs)
echo "Installing JAX with bundled CUDA 12 support..."
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Also install nvidia-cusparse-cu12 explicitly in case it's missing
pip install nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 2>/dev/null || true

# Install Octo from GitHub (not on PyPI)
echo "Installing Octo from source..."
pip install "git+https://github.com/octo-models/octo.git"

# Install LIBERO dependencies (needed for env)
echo "Installing LIBERO + MuJoCo dependencies..."
pip install mujoco
pip install robosuite
pip install libero
pip install imageio[ffmpeg]
pip install torch torchvision  # For data processing utilities
pip install scikit-learn
pip install numpy scipy

# Verify installation
echo ""
echo "=== Verification ==="
python3 -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
import octo
print(f'Octo: installed')
from octo.model.octo_model import OctoModel
print(f'OctoModel: importable')
print('✓ All good!')
"

echo ""
echo "================================================================"
echo "  Octo environment ready: ${ENV_NAME}"
echo "================================================================"
