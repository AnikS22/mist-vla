#!/bin/bash
# HPC Setup Script for MIST-VLA

echo "========================================="
echo "MIST-VLA HPC Setup"
echo "========================================="

# 1. Create conda environment
echo "Creating conda environment..."
conda create -n mist-vla python=3.10 -y
conda activate mist-vla

# 2. Install PyTorch (check HPC CUDA version first!)
echo "Installing PyTorch..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install transformers and vision libs
echo "Installing transformers..."
pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10

# 4. Install flash attention (if available on HPC)
echo "Installing flash attention..."
pip install flash-attn==2.5.5 --no-build-isolation || echo "Flash attention install failed, continuing..."

# 5. Install interpretability tools
echo "Installing interpretability tools..."
pip install transformer-lens captum

# 6. Install simulation
echo "Installing simulation libraries..."
pip install robosuite mujoco gymnasium

# 7. Install utilities
echo "Installing utilities..."
pip install wandb matplotlib seaborn scikit-learn tqdm einops pyyaml

# 8. Clone and install LIBERO
echo "Cloning LIBERO..."
cd ~/
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .

# 9. Clone and install OpenVLA
echo "Cloning OpenVLA..."
cd ~/
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# 10. Install MIST-VLA
echo "Installing MIST-VLA..."
cd ~/mist-vla
pip install -e .

# 11. Download OpenVLA model (will be cached)
echo "Downloading OpenVLA model..."
python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('openvla/openvla-7b', trust_remote_code=True)"

# 12. Verify setup
echo "Verifying setup..."
python scripts/verify_setup.py

echo "========================================="
echo "Setup complete!"
echo "========================================="
