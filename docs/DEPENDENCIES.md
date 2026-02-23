# External Dependencies

This document lists the external repositories and dependencies required for MIST-VLA.

## Required Repositories

### LIBERO
- **Purpose**: Benchmark environment for robotic manipulation tasks
- **Repository**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **Setup**: Clone and follow LIBERO installation instructions
- **Configuration**: Requires `~/.libero/config.yaml` with valid paths

### OpenVLA-OFT
- **Purpose**: OpenVLA model with OFT (Orthogonal Fine-Tuning) evaluation pipeline
- **Repository**: https://github.com/openvla/openvla-oft
- **Usage**: Add to `PYTHONPATH` for model evaluation
- **Note**: Required for `collect_failure_data_oft_eval.py` script

## Optional Dependencies

### FailSafe
- **Purpose**: Baseline safety methods for comparison
- **Repository**: [Add FailSafe repo URL]
- **Usage**: Used for comparative evaluation

### SimplerEnv
- **Purpose**: Simplified simulation environments
- **Repository**: https://github.com/simpler-env/SimplerEnv
- **Usage**: Alternative environment for testing

### TransformerLens
- **Purpose**: Mechanistic interpretability tools for transformers
- **Repository**: https://github.com/neelnanda-io/TransformerLens
- **Usage**: Internal analysis and activation steering

### SAELens
- **Purpose**: Sparse Autoencoder tools for interpretability
- **Repository**: https://github.com/jbloomAus/SAELens
- **Usage**: Advanced interpretability analysis

### Open-PI-Zero
- **Purpose**: Physical intelligence models and benchmarks
- **Repository**: https://github.com/Physical-Intelligence/open-pi-zero
- **Usage**: Additional model baselines

## Installation

To set up all dependencies:

```bash
# Create a workspace directory
mkdir -p ~/mist-vla-workspace
cd ~/mist-vla-workspace

# Clone main repo
git clone https://github.com/yourusername/MIST-VLA.git
cd MIST-VLA

# Clone required dependencies
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git clone https://github.com/openvla/openvla-oft.git

# Clone optional dependencies (as needed)
git clone https://github.com/simpler-env/SimplerEnv.git
git clone https://github.com/neelnanda-io/TransformerLens.git
git clone https://github.com/jbloomAus/SAELens.git

# Set up PYTHONPATH
export PYTHONPATH=$PWD/mist-vla:$PWD/openvla-oft:$PYTHONPATH
```

## Version Requirements

- Python: 3.8+
- PyTorch: 2.0+
- CUDA: 11.8+ (for GPU support)

See `mist-vla/requirements.txt` for detailed Python package dependencies.
