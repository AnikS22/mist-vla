#!/bin/bash
# HPC Setup Script for MIST-VLA

set -e  # Exit on any error

echo "========================================="
echo "MIST-VLA HPC Setup"
echo "========================================="

# Get conda base path
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 1. Create conda environment
echo "Creating conda environment..."
conda create -n mist-vla python=3.10 -y
conda activate mist-vla

# Verify environment is active
if [[ "$CONDA_DEFAULT_ENV" != "mist-vla" ]]; then
    echo "ERROR: Failed to activate mist-vla environment"
    exit 1
fi
echo "✓ Environment activated: $CONDA_DEFAULT_ENV"

# 2. Install PyTorch
echo ""
echo "Installing PyTorch..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch installed"

# 3. Install transformers (CRITICAL - must be this version)
echo ""
echo "Installing transformers..."
pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 accelerate
echo "✓ Transformers installed"

# 4. Test transformers import
echo ""
echo "Testing transformers import..."
python -c "from transformers import AutoProcessor, AutoModel; print('✓ Transformers working')" || {
    echo "ERROR: Transformers import failed"
    exit 1
}

# 5. Install interpretability tools
echo ""
echo "Installing interpretability tools..."
pip install captum scikit-learn
echo "✓ Interpretability tools installed"

# 6. Install utilities
echo ""
echo "Installing utilities..."
pip install matplotlib seaborn tqdm einops pyyaml
echo "✓ Utilities installed"

# 7. Install simulation libraries
echo ""
echo "Installing simulation libraries..."
pip install gymnasium
pip install mujoco
echo "✓ Simulation base installed"

# 8. Install robosuite
echo ""
echo "Installing robosuite..."
pip install robosuite
echo "✓ Robosuite installed"

# 9. Clone and install LIBERO
echo ""
echo "Installing LIBERO..."
if [ ! -d "$HOME/LIBERO" ]; then
    cd $HOME
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO
    if [ ! -f "libero/__init__.py" ]; then
        echo "# LIBERO package marker" > libero/__init__.py
    fi
    pip install -e .
else
    echo "LIBERO already exists, skipping clone"
    cd $HOME/LIBERO
    if [ ! -f "libero/__init__.py" ]; then
        echo "# LIBERO package marker" > libero/__init__.py
    fi
    pip install -e .
fi
echo "✓ LIBERO installed"

# 10. Clone and install OpenVLA
echo ""
echo "Installing OpenVLA..."
if [ ! -d "$HOME/openvla" ]; then
    cd $HOME
    git clone https://github.com/openvla/openvla.git
    cd openvla
    pip install -e .
else
    echo "OpenVLA already exists, skipping clone"
    cd $HOME/openvla
    pip install -e .
fi
echo "✓ OpenVLA installed"

# 11. Install MIST-VLA
echo ""
echo "Installing MIST-VLA..."
cd $HOME/mist-vla
pip install -e .
echo "✓ MIST-VLA installed"

# 12. Download OpenVLA model
echo ""
echo "Downloading OpenVLA model (this may take a while)..."
python -c "
from transformers import AutoProcessor
try:
    processor = AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
    print('✓ Model downloaded successfully')
except Exception as e:
    print(f'WARNING: Model download failed: {e}')
    print('It will be downloaded when first needed')
"

# 13. Final verification
echo ""
echo "========================================="
echo "Final Verification"
echo "========================================="

python -c "
import sys
print('Testing all imports...')

try:
    import torch
    print('✓ PyTorch:', torch.__version__)

    import transformers
    print('✓ Transformers:', transformers.__version__)

    from transformers import AutoProcessor
    print('✓ AutoProcessor available')

    import captum
    print('✓ Captum available')

    import libero
    print('✓ LIBERO available')

    sys.path.insert(0, '$HOME/mist-vla')
    from src.models.hooked_openvla import HookedOpenVLA
    print('✓ MIST-VLA imports work')

    print('')
    print('✅ ALL IMPORTS SUCCESSFUL!')

except Exception as e:
    print(f'✗ Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Setup Complete!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "1. Submit job: sbatch run_hpc.slurm"
    echo "2. Monitor: squeue -u \$USER"
    echo "3. Watch logs: tail -f logs/mist_vla_*.out"
else
    echo ""
    echo "❌ Setup failed - check errors above"
    exit 1
fi
