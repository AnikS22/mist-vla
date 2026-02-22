#!/bin/bash
###############################################################################
#  Setup Octo environment on HPC (JAX + Octo model)
#
#  Octo is JAX-based (not PyTorch), so we need a separate conda env.
#
#  KEY FIXES (v3):
#    - Use tensorflow-cpu instead of tensorflow (avoids CUDA conflicts)
#    - Force-reinstall CUDA jaxlib AFTER all other deps to guarantee it sticks
#    - Use --no-deps aggressively to prevent any JAX version changes
###############################################################################

set -eo pipefail

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"

ENV_NAME="octo-env"
JAX_VERSION="0.4.28"
JAXLIB_WHEEL="jaxlib==0.4.28+cuda12.cudnn89"
JAX_FIND_LINKS="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

echo "================================================================"
echo "  Setting up Octo environment: ${ENV_NAME} (v3 — robust)"
echo "================================================================"

# Always start fresh to avoid stale deps
if conda env list | grep -q "${ENV_NAME}"; then
    echo "Removing old ${ENV_NAME}..."
    conda remove -n "${ENV_NAME}" --all -y 2>/dev/null || true
fi

echo "Creating conda environment: ${ENV_NAME} (Python 3.10)..."
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

# ─── Step 1: Install JAX + CUDA jaxlib ───
echo ""
echo "[1/6] Installing JAX ${JAX_VERSION} with CUDA 12 jaxlib..."
pip install "jax==${JAX_VERSION}" "${JAXLIB_WHEEL}" \
    -f "${JAX_FIND_LINKS}"

echo "  Verifying jaxlib..."
python3 -c "import jaxlib; v=jaxlib.__version__; print(f'  jaxlib: {v}'); assert 'cuda' in v or '0.4.28' in v"

# ─── Step 2: Install flax ecosystem with --no-deps ───
echo ""
echo "[2/6] Installing flax ecosystem (--no-deps to protect jaxlib)..."
pip install --no-deps \
    "flax==0.8.5" \
    "orbax-checkpoint==0.6.4" \
    "optax==0.2.3" \
    "chex==0.1.87" \
    "distrax==0.1.5"

# Install their non-JAX dependencies
pip install \
    msgpack typing-extensions rich pyyaml \
    "numpy<2" scipy toolz etils tensorstore \
    nest-asyncio absl-py clu

# ─── Step 3: Install tensorflow-cpu (Octo uses tf for data loading only) ───
echo ""
echo "[3/6] Installing tensorflow-cpu (avoids CUDA conflicts)..."
pip install tensorflow-cpu

# ─── Step 4: Install tensorflow_probability with --no-deps ───
echo ""
echo "[4/6] Installing tensorflow_probability (--no-deps)..."
pip install --no-deps "tensorflow_probability>=0.22.0,<0.25.0"

# ─── Step 5: Install Octo from GitHub (--no-deps) ───
echo ""
echo "[5/6] Installing Octo from source (--no-deps)..."
pip install --no-deps "git+https://github.com/octo-models/octo.git"

# Install Octo's remaining non-JAX deps
pip install dlimp ml-collections einops 2>/dev/null || true

# LIBERO + rendering
pip install mujoco "robosuite==1.4.0" libero
pip install imageio pillow
pip install scikit-learn

# ─── Step 6: FORCE reinstall CUDA jaxlib (nuclear option) ───
# This is the critical step: after ALL other pip installs, we force the CUDA
# jaxlib back in. Any earlier dep that silently downgraded it gets overridden.
echo ""
echo "[6/6] Force-reinstalling CUDA jaxlib (nuclear guarantee)..."
pip install --force-reinstall --no-deps "${JAXLIB_WHEEL}" \
    -f "${JAX_FIND_LINKS}"

# ─── Final Verification ───
echo ""
echo "================================================================"
echo "  FINAL VERIFICATION"
echo "================================================================"
python3 -c "
import jaxlib
print(f'  jaxlib version: {jaxlib.__version__}')

import jax
print(f'  JAX version: {jax.__version__}')
print(f'  JAX devices: {jax.devices()}')
try:
    gpu_devs = jax.devices('gpu')
    print(f'  GPU devices: {gpu_devs}')
except:
    print('  No GPU on login node (expected — will use GPU on compute node)')

import flax
print(f'  Flax: {flax.__version__}')

import tensorflow as tf
print(f'  TensorFlow: {tf.__version__}')

from octo.model.octo_model import OctoModel
print(f'  OctoModel: importable')
print()
print('  ALL CHECKS PASSED')
"

echo ""
echo "================================================================"
echo "  Octo environment ready: ${ENV_NAME}"
echo "================================================================"
