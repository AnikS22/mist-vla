# Getting Started with MIST-VLA

This guide will help you get up and running with MIST-VLA.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8 or higher** installed
- **CUDA-capable GPU** (recommended, especially for training)
- **Git** for version control
- At least **20GB of free disk space** for models and data

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MIST-VLA.git
cd MIST-VLA
```

### 2. Set Up Python Environment

We recommend using a virtual environment:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n mist-vla python=3.10
conda activate mist-vla
```

### 3. Install Dependencies

```bash
cd mist-vla
pip install -r requirements.txt
pip install -e .
```

### 4. Set Up LIBERO

LIBERO requires a configuration file:

```bash
mkdir -p ~/.libero
```

Create `~/.libero/config.yaml`:

```yaml
# Example LIBERO configuration
libero_path: /path/to/LIBERO
dataset_path: /path/to/datasets
asset_path: /path/to/assets
```

### 5. Configure Python Path

Add the necessary directories to your Python path:

```bash
export PYTHONPATH=$PWD:../openvla-oft:$PYTHONPATH
```

To make this permanent, add it to your `~/.bashrc` or `~/.zshrc`.

## Verify Installation

Test your installation:

```bash
python -c "import mist_vla; print('MIST-VLA installed successfully!')"
```

## Next Steps

1. **Explore Examples**: Check out the example scripts in `mist-vla/scripts/`
2. **Read Documentation**: Browse the docs in `docs/`
3. **Run Your First Experiment**: Follow the [Quick Start](../README.md#-quick-start) guide

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'mist_vla'`

**Solution**: Make sure you've installed the package and set PYTHONPATH correctly.

```bash
cd mist-vla
pip install -e .
```

**Issue**: LIBERO configuration not found

**Solution**: Ensure `~/.libero/config.yaml` exists with valid paths.

**Issue**: CUDA out of memory

**Solution**: Reduce batch size or use a smaller model variant.

## Getting Help

- Check the [FAQ](FAQ.md)
- Open an [issue](https://github.com/yourusername/MIST-VLA/issues)
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup

## What's Next?

- [Data Collection Guide](DATA_COLLECTION.md)
- [Training Guide](TRAINING.md)
- [Evaluation Guide](EVALUATION.md)
