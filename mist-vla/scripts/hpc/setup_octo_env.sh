#!/bin/bash
###############################################################################
#  Setup Octo environment on HPC  (v6 — global constraints file)
#
#  The dependency hell:
#    - jaxlib 0.4.28 bundles cuDNN 8.9
#    - torch (from pip) brings nvidia-cudnn-cu12 v9 → breaks JAX
#    - tensorflow-cpu 2.15 needs numpy<2 and ml-dtypes 0.3.x
#    - tensorstore 0.1.78 needs numpy>=2 and ml-dtypes>=0.5
#
#  Solution: CPU-only torch, tf-cpu 2.15, pin EVERYTHING via constraints.
###############################################################################

set -eo pipefail

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"

ENV_NAME="octo-env"
JAXLIB_WHEEL="jaxlib==0.4.28+cuda12.cudnn89"
JAX_FIND_LINKS="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

echo "================================================================"
echo "  Setting up Octo environment: ${ENV_NAME}  (v6)"
echo "================================================================"

# Always start fresh
if conda env list | grep -q "${ENV_NAME}"; then
    echo "Removing old ${ENV_NAME}..."
    conda remove -n "${ENV_NAME}" --all -y 2>/dev/null || true
fi

conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

# ─── Global constraints file ───────────────────────────────────────────
# This prevents ANY pip install from upgrading/downgrading these packages.
CONSTRAINTS="${CONDA_PREFIX}/pip-constraints.txt"
cat > "${CONSTRAINTS}" << 'EOF'
jax==0.4.28
numpy>=1.24,<2.0
ml-dtypes>=0.3.0,<0.4.0
EOF

echo "Using constraints file: ${CONSTRAINTS}"
cat "${CONSTRAINTS}"
echo ""

# ─── 1. Core numeric stack (numpy < 2 first!) ─────────────────────────
echo "[1/7] Installing core numeric stack..."
pip install -c "${CONSTRAINTS}" "numpy>=1.24,<2" scipy

# ─── 2. JAX + CUDA jaxlib + cuDNN 8.9 ─────────────────────────────────
# jaxlib 0.4.28+cuda12.cudnn89 does NOT bundle cuDNN; it expects
# nvidia-cudnn-cu12 to provide the .so files. We pin v8.9 specifically
# (GPU torch would install v9 which is incompatible).
echo ""
echo "[2/7] Installing JAX + CUDA jaxlib + cuDNN 8.9..."
pip install -c "${CONSTRAINTS}" "jax==0.4.28" "${JAXLIB_WHEEL}" \
    "nvidia-cudnn-cu12==8.9.7.29" \
    -f "${JAX_FIND_LINKS}"

# ─── 3. CPU-only PyTorch (BEFORE libero so libero doesn't pull GPU torch) ──
echo ""
echo "[3/7] Installing torch CPU-only..."
pip install -c "${CONSTRAINTS}" torch torchvision \
    --index-url "${TORCH_CPU_INDEX}"

# ─── 4. tensorflow-cpu 2.15 + dlimp ───────────────────────────────────
echo ""
echo "[4/7] Installing tensorflow-cpu 2.15 + dlimp..."
pip install -c "${CONSTRAINTS}" "tensorflow-cpu>=2.15,<2.16"

# dlimp from GitHub (--no-deps so it doesn't pull tensorflow GPU 2.15)
pip install --no-deps "git+https://github.com/kvablack/dlimp.git"
# dlimp needs tensorflow_datasets at import time
pip install -c "${CONSTRAINTS}" tensorflow_datasets simple_parsing immutabledict

# tensorflow_probability (--no-deps to avoid pulling incompatible jax)
pip install --no-deps "tensorflow_probability>=0.23,<0.24"
pip install -c "${CONSTRAINTS}" decorator dm-tree cloudpickle

# ─── 5. Flax ecosystem (--no-deps to protect jax/jaxlib) ──────────────
echo ""
echo "[5/7] Installing flax ecosystem..."
pip install --no-deps \
    "flax==0.8.5" \
    "orbax-checkpoint==0.5.23" \
    "optax==0.2.3" \
    "chex==0.1.87" \
    "distrax==0.1.5"

# Their non-JAX deps (with constraints)
pip install -c "${CONSTRAINTS}" \
    msgpack typing-extensions rich pyyaml \
    toolz etils nest-asyncio absl-py clu \
    "tensorstore>=0.1.45,<0.1.65"

# ─── 6. Octo (--no-deps) ──────────────────────────────────────────────
echo ""
echo "[6/7] Installing Octo + LIBERO..."
pip install --no-deps "git+https://github.com/octo-models/octo.git"
pip install -c "${CONSTRAINTS}" ml-collections einops

# LIBERO + rendering (torch already satisfied from step 3)
pip install -c "${CONSTRAINTS}" mujoco "robosuite==1.4.0" libero
pip install -c "${CONSTRAINTS}" imageio pillow scikit-learn

# Pin transformers to 4.x (Octo needs FlaxAutoModel, removed in transformers 5.x)
pip install -c "${CONSTRAINTS}" "transformers>=4.36,<5.0"

# ─── 7. FORCE reinstall CUDA jaxlib + cuDNN 8.9 (nuclear guarantee) ───
echo ""
echo "[7/7] Force-reinstalling CUDA jaxlib + cuDNN 8.9..."
pip install --force-reinstall --no-deps "${JAXLIB_WHEEL}" \
    -f "${JAX_FIND_LINKS}"
pip install --force-reinstall "nvidia-cudnn-cu12==8.9.7.29"

# ─── Final Verification ───────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  FINAL VERIFICATION"
echo "================================================================"
python3 << 'PYEOF'
import sys

import numpy as np
print(f"  numpy: {np.__version__}")
assert np.__version__.startswith("1."), f"Expected numpy 1.x, got {np.__version__}"

import jaxlib
print(f"  jaxlib: {jaxlib.__version__}")

import jax
print(f"  JAX: {jax.__version__}")
devs = jax.devices()
print(f"  JAX devices: {devs}")
gpu_devs = [d for d in devs if d.platform == "gpu"]
if gpu_devs:
    print(f"  GPU: {gpu_devs}")
else:
    print("  No GPU on this node (expected on login node)")

import flax
print(f"  Flax: {flax.__version__}")

import tensorflow as tf
print(f"  TensorFlow: {tf.__version__}")

import dlimp
print(f"  dlimp: OK")

import torch
print(f"  PyTorch: {torch.__version__} (CUDA={torch.cuda.is_available()})")

from octo.model.octo_model import OctoModel
print(f"  OctoModel: importable")

print()
print("  ALL CHECKS PASSED")
PYEOF

echo ""
echo "================================================================"
echo "  Octo environment ready: ${ENV_NAME}"
echo "================================================================"
