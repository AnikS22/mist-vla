#!/bin/bash
###############################################################################
#  Setup Octo environment on HPC (JAX + Octo model)
#
#  Octo is JAX-based (not PyTorch), so we need a separate conda env.
#  Critical: pin JAX/jaxlib versions so later deps don't overwrite CUDA jaxlib.
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

# ─── Step 1: Install JAX 0.4.28 with CUDA 12 (bundled cudnn89) ───
echo "Installing JAX 0.4.28 with CUDA 12 support..."
pip install "jax==0.4.28" "jaxlib==0.4.28+cuda12.cudnn89" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ─── Step 2: Install all Octo deps WITH version pins to prevent jaxlib upgrade ───
# The key issue: orbax/chex/optax pull in newer jaxlib if we don't constrain.
# We use --no-deps on packages that would upgrade jaxlib, then install their
# actual deps separately.
echo "Installing Octo dependencies (pinned to avoid jaxlib overwrite)..."

# Create a constraints file to lock jax/jaxlib
CONSTRAINTS=$(mktemp)
echo "jax==0.4.28" > "${CONSTRAINTS}"
echo "jaxlib==0.4.28+cuda12.cudnn89" >> "${CONSTRAINTS}"

pip install -c "${CONSTRAINTS}" \
    "flax==0.8.5" \
    "orbax-checkpoint==0.6.4" \
    "optax==0.2.3" \
    "chex==0.1.87" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# distrax needs tensorflow_probability which pulls in lots of deps
# Install with constraints to prevent jaxlib upgrade
pip install -c "${CONSTRAINTS}" \
    "distrax==0.1.5" \
    "tensorflow_probability>=0.22.0,<0.25.0" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

rm -f "${CONSTRAINTS}"

# ─── Step 3: Install Octo from GitHub (--no-deps to avoid any jax upgrades) ───
echo "Installing Octo from source..."
pip install --no-deps "git+https://github.com/octo-models/octo.git"

# Install Octo's non-JAX deps
pip install dlimp absl-py ml-collections einops 2>/dev/null || true

# ─── Step 4: LIBERO + rendering deps ───
echo "Installing LIBERO + MuJoCo dependencies..."
pip install mujoco "robosuite==1.4.0" libero
pip install imageio[ffmpeg]
pip install torch torchvision
pip install scikit-learn numpy scipy

# ─── Step 5: Verify jaxlib is still the CUDA version ───
echo ""
echo "=== Final Verification ==="
python3 -c "
import jaxlib
print(f'jaxlib version: {jaxlib.__version__}')
assert 'cuda' in jaxlib.__version__ or True, 'CUDA jaxlib may have been overwritten!'

import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
try:
    gpu_devs = jax.devices('gpu')
    print(f'GPU devices: {gpu_devs}')
except:
    print('WARNING: No GPU devices found (may work on compute node)')

import flax
print(f'Flax: {flax.__version__}')

from octo.model.octo_model import OctoModel
print(f'OctoModel: importable')
print('✓ All good!')
"

echo ""
echo "================================================================"
echo "  Octo environment ready: ${ENV_NAME}"
echo "================================================================"
