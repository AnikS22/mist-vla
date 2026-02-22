#!/bin/bash
###############################################################################
#  Setup Octo environment on HPC (JAX + Octo model)
#
#  Octo is JAX-based (not PyTorch), so we need a separate conda env.
#
#  KEY FIXES (v4):
#    - torch CPU-ONLY to avoid nvidia-cudnn-cu12 v9 conflicting with
#      jaxlib's bundled cuDNN 8.9
#    - tensorflow-cpu instead of tensorflow (avoids CUDA conflicts)
#    - Force-reinstall CUDA jaxlib AFTER all other deps
#    - No silent || true on critical deps
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
# CPU-only PyTorch so nvidia-cudnn-cu12 v9 doesn't overwrite jaxlib's cuDNN 8.9
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

echo "================================================================"
echo "  Setting up Octo environment: ${ENV_NAME} (v4 — torch-cpu)"
echo "================================================================"

# Always start fresh
if conda env list | grep -q "${ENV_NAME}"; then
    echo "Removing old ${ENV_NAME}..."
    conda remove -n "${ENV_NAME}" --all -y 2>/dev/null || true
fi

echo "Creating conda environment: ${ENV_NAME} (Python 3.10)..."
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

# ─── Step 1: Install JAX + CUDA jaxlib ───
echo ""
echo "[1/7] Installing JAX ${JAX_VERSION} with CUDA 12 jaxlib..."
pip install "jax==${JAX_VERSION}" "${JAXLIB_WHEEL}" \
    -f "${JAX_FIND_LINKS}"

echo "  Verifying jaxlib is CUDA..."
python3 -c "
import jaxlib
print(f'  jaxlib: {jaxlib.__version__}')
# The CUDA wheel puts .so files in jaxlib/cuda/
import os, jaxlib as jl
jl_dir = os.path.dirname(jl.__file__)
cuda_dir = os.path.join(jl_dir, 'cuda')
if os.path.isdir(cuda_dir):
    print(f'  jaxlib/cuda/ exists: {os.listdir(cuda_dir)[:5]}...')
else:
    print(f'  WARNING: jaxlib/cuda/ not found at {cuda_dir}')
"

# ─── Step 2: Install flax ecosystem with --no-deps ───
echo ""
echo "[2/7] Installing flax ecosystem (--no-deps to protect jaxlib)..."
pip install --no-deps \
    "flax==0.8.5" \
    "orbax-checkpoint==0.6.4" \
    "optax==0.2.3" \
    "chex==0.1.87" \
    "distrax==0.1.5"

# Their non-JAX dependencies
pip install \
    msgpack typing-extensions rich pyyaml \
    "numpy<2" scipy toolz etils tensorstore \
    nest-asyncio absl-py clu

# ─── Step 3: tensorflow-cpu (Octo uses tf for data loading only) ───
echo ""
echo "[3/7] Installing tensorflow-cpu..."
pip install --no-deps tensorflow-cpu
# TF's non-GPU deps
pip install flatbuffers gast google-pasta h5py keras libclang \
    ml-dtypes opt-einsum protobuf termcolor wrapt grpcio \
    tensorboard markdown werkzeug 2>/dev/null || true

# ─── Step 4: tensorflow_probability (--no-deps) ───
echo ""
echo "[4/7] Installing tensorflow_probability (--no-deps)..."
pip install --no-deps "tensorflow_probability>=0.22.0,<0.25.0"
pip install decorator dm-tree  # its actual deps

# ─── Step 5: Install Octo from GitHub (--no-deps) ───
echo ""
echo "[5/7] Installing Octo from source (--no-deps)..."
pip install --no-deps "git+https://github.com/octo-models/octo.git"

# Octo's remaining deps — these MUST succeed
echo "  Installing Octo's required deps (dlimp, ml-collections, einops)..."
pip install dlimp
pip install ml-collections einops

# ─── Step 6: LIBERO + rendering + torch CPU ───
echo ""
echo "[6/7] Installing LIBERO + MuJoCo + torch (CPU-only)..."
pip install mujoco "robosuite==1.4.0" libero
pip install imageio pillow scikit-learn

# CPU-only torch — does NOT install nvidia-cudnn-cu12 v9
pip install torch torchvision --index-url "${TORCH_CPU_INDEX}"

# ─── Step 7: FORCE reinstall CUDA jaxlib (nuclear guarantee) ───
echo ""
echo "[7/7] Force-reinstalling CUDA jaxlib..."
pip install --force-reinstall --no-deps "${JAXLIB_WHEEL}" \
    -f "${JAX_FIND_LINKS}"

# ─── Final Verification ───
echo ""
echo "================================================================"
echo "  FINAL VERIFICATION"
echo "================================================================"
python3 << 'PYEOF'
import jaxlib
print(f"  jaxlib version: {jaxlib.__version__}")

import jax
print(f"  JAX version: {jax.__version__}")
devs = jax.devices()
print(f"  JAX devices: {devs}")
gpu_devs = [d for d in devs if d.platform == "gpu"]
if gpu_devs:
    print(f"  GPU devices: {gpu_devs}")
else:
    print("  No GPU on this node (expected on login, will work on compute)")

import flax
print(f"  Flax: {flax.__version__}")

import tensorflow as tf
print(f"  TensorFlow: {tf.__version__}")

import dlimp
print(f"  dlimp: OK")

from octo.model.octo_model import OctoModel
print(f"  OctoModel: importable")

import torch
print(f"  PyTorch: {torch.__version__} (CPU-only)")

print()
print("  ALL CHECKS PASSED")
PYEOF

echo ""
echo "================================================================"
echo "  Octo environment ready: ${ENV_NAME}"
echo "================================================================"
