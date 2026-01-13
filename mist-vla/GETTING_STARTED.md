# Getting Started with MIST-VLA

This guide will walk you through the complete setup and execution of MIST-VLA from scratch.

## Prerequisites

- NVIDIA GPU with ≥24GB VRAM (recommended: A100, A6000, or RTX 4090)
- CUDA 12.1+
- Python 3.10+
- 64GB+ RAM
- 500GB+ disk space

## Step-by-Step Setup

### 1. Clone Required Repositories

```bash
# Create project directory
mkdir -p ~/vla-research && cd ~/vla-research

# Clone MIST-VLA (this repository)
git clone <mist-vla-repo> mist-vla
cd mist-vla

# Clone dependencies
cd ..

# OpenVLA - Primary VLA model
git clone https://github.com/openvla/openvla.git

# LIBERO - Primary benchmark
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# TransformerLens - Interpretability library
git clone https://github.com/TransformerLensOrg/TransformerLens.git
```

### 2. Environment Setup

```bash
# Create conda environment
conda create -n mist-vla python=3.10 -y
conda activate mist-vla

# Core PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Transformers and vision
pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10

# Flash Attention (optional but recommended)
pip install flash-attn==2.5.5 --no-build-isolation

# Interpretability tools
pip install transformer-lens captum

# Simulation
pip install robosuite mujoco gymnasium

# Utilities
pip install wandb matplotlib seaborn scikit-learn tqdm einops pyyaml

# Install LIBERO
cd LIBERO
pip install -r requirements.txt
pip install -e .
cd ..

# Install OpenVLA
cd openvla
pip install -e .
cd ..

# Install MIST-VLA
cd mist-vla
pip install -e .
```

### 3. Download Models and Data

```bash
# Download OpenVLA model (will cache to ~/.cache/huggingface)
python -c "from transformers import AutoModelForVision2Seq; \
    AutoModelForVision2Seq.from_pretrained('openvla/openvla-7b', trust_remote_code=True)"

# Download LIBERO datasets
cd ../LIBERO
python libero/scripts/download_datasets.py
cd ../mist-vla
```

### 4. Quick Test

Test that everything is installed correctly:

```bash
python -c "
import torch
from src.models.hooked_openvla import HookedOpenVLA
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('All imports successful!')
"
```

## Running the Full Pipeline

### Phase 1: Data Collection

Collect successful and failed rollouts for training the failure detector:

```bash
# Collect 100 success and 100 failure trajectories
python scripts/collect_failure_data.py \
    --env libero_spatial \
    --n_success 100 \
    --n_failure 100 \
    --save_dir data/rollouts

# This will take ~2-4 hours depending on your hardware
```

### Phase 2: Train Failure Detector

Train the SAFE-style failure detection model:

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Train MLP detector
python scripts/train_failure_detector.py \
    --data_dir data/rollouts \
    --detector_type mlp \
    --epochs 50

# Outputs: checkpoints/best_detector.pt
#          checkpoints/conformal_calibration.pt
```

### Phase 3: Extract Steering Vectors

Pre-compute steering vectors for all semantic directions:

```bash
python scripts/extract_steering_vectors.py \
    --model openvla/openvla-7b \
    --save_path data/steering_vectors

# Outputs: data/steering_vectors/all_vectors.pt
# This takes ~30-60 minutes
```

### Phase 4: Evaluation

Evaluate MIST-VLA on LIBERO benchmark:

```bash
# Create results directory
mkdir -p results

# Run evaluation on LIBERO-Spatial suite
python scripts/run_libero_eval.py \
    --task_suite libero_spatial \
    --model_path openvla/openvla-7b \
    --detector_path checkpoints/best_detector.pt \
    --steering_path data/steering_vectors/all_vectors.pt \
    --n_episodes 50

# Outputs: results/libero_spatial_results.json
```

## Expected Results

After running the full pipeline, you should see:

```
=== Results for libero_spatial ===
Overall Success Rate: ~75-85%
Avg Failures Detected: ~2-4 per episode
Recovery Rate: ~60-80%
```

These are typical results. Your actual numbers may vary based on:
- Random seed
- Hardware (GPU precision)
- LIBERO environment version
- Number of training epochs

## Troubleshooting

### Out of Memory

If you run out of GPU memory:

1. Reduce batch size in training:
   ```bash
   python scripts/train_failure_detector.py --batch_size 16
   ```

2. Use gradient checkpointing (modify hooked_openvla.py)

3. Use smaller model variant (if available)

### LIBERO Environment Issues

If LIBERO environments fail:

1. Check mujoco installation:
   ```bash
   python -c "import mujoco; print(mujoco.__version__)"
   ```

2. Verify datasets are downloaded:
   ```bash
   ls ~/LIBERO/datasets/
   ```

### Slow Inference

If inference is too slow:

1. Ensure flash-attention is installed
2. Use bfloat16 precision (default)
3. Reduce sequence length if applicable

## Next Steps

### Run Ablation Studies

Test different components:

```bash
# Detection only (no steering)
python scripts/run_libero_eval.py --no_steering

# Different steering coefficients
python scripts/run_libero_eval.py --steering_coef 0.5
python scripts/run_libero_eval.py --steering_coef 2.0
```

### Try Different Tasks

Evaluate on other LIBERO suites:

```bash
# LIBERO-Object
python scripts/run_libero_eval.py --task_suite libero_object

# LIBERO-Goal
python scripts/run_libero_eval.py --task_suite libero_goal
```

### Analyze Failure Cases

Use notebooks to analyze specific failures:

```bash
jupyter notebook notebooks/04_intervention_experiments.ipynb
```

## Additional Resources

- [Full Blueprint](claude.md) - Complete implementation details
- [README](README.md) - Project overview
- [OpenVLA Docs](https://github.com/openvla/openvla)
- [LIBERO Docs](https://github.com/Lifelong-Robot-Learning/LIBERO)

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review the full blueprint in `claude.md`
3. Ensure all dependencies are correctly installed
4. Try with a minimal example first

## Estimated Timeline

- Setup: 1-2 hours
- Data collection: 2-4 hours
- Training detector: 1-2 hours
- Extract steering vectors: 0.5-1 hour
- Evaluation: 2-4 hours

**Total: ~8-14 hours for complete pipeline**

Good luck with your research!
