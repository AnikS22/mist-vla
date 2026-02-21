#!/bin/bash
###############################################################################
#  Setup Octo environment on HPC (JAX + Octo model)
#  Run this interactively on a GPU node, or as a short SLURM job.
#
#  Octo is JAX-based (not PyTorch), so we need a separate conda env.
#  Pinned to JAX 0.4.35 for Octo compatibility + CUDA 12 support.
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

# ─── Install JAX 0.4.x with CUDA 12 support ───
# Octo requires JAX 0.4.x; 0.5+ breaks API compatibility
# Use the jax_cuda_releases index for proper CUDA-bundled jaxlib
# Available CUDA builds: 0.4.28+cuda12.cudnn89 is the last stable one
echo "Installing JAX 0.4.28 with CUDA 12 support..."
pip install "jax==0.4.28" "jaxlib==0.4.28+cuda12.cudnn89" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ─── Install Octo core deps that the pip install misses ───
echo "Installing flax and other Octo dependencies..."
pip install "flax>=0.7.0,<0.9.0" "orbax-checkpoint>=0.5.0,<0.7.0" \
    "distrax>=0.1.3" "chex>=0.1.80" "optax>=0.1.7" \
    "tensorflow_probability>=0.22.0,<0.25.0"

# ─── Install Octo from GitHub ───
echo "Installing Octo from source..."
pip install "git+https://github.com/octo-models/octo.git" --no-deps
# Now install remaining deps that Octo needs but we haven't covered
pip install dlimp absl-py ml-collections einops 2>/dev/null || true

# ─── Install LIBERO dependencies (needed for env) ───
echo "Installing LIBERO + MuJoCo dependencies..."
pip install mujoco
pip install "robosuite==1.4.0"
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
try:
    gpu_devs = jax.devices('gpu')
    print(f'GPU devices: {gpu_devs}')
except:
    print('WARNING: No GPU devices found')
import flax
print(f'Flax version: {flax.__version__}')
from octo.model.octo_model import OctoModel
print(f'OctoModel: importable')
print('✓ All good!')
"

echo ""
echo "================================================================"
echo "  Octo environment ready: ${ENV_NAME}"
echo "================================================================"
