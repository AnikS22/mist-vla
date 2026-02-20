<div align="center">

# 🛡️ MIST-VLA

### **M**echanistic **I**nterpretability for **S**afer **T**argeted Steering in **V**ision-**L**anguage-**A**ction Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Per-dimension collision forecasting and targeted activation steering for safer Vision-Language-Action models on LIBERO*

[Key Features](#-key-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Documentation](#-project-structure) •
[Citation](#-citation)

</div>

---

## 📋 Overview

MIST-VLA is a research framework for improving the safety and reliability of Vision-Language-Action (VLA) models through mechanistic interpretability and targeted activation steering. This project focuses on:

- **Collision Forecasting**: Per-dimension risk prediction for robotic manipulation tasks
- **Activation Steering**: Targeted interventions to mitigate unsafe behaviors
- **Failure Analysis**: SAFE-style labeling and comprehensive failure characterization
- **Empirical Evaluation**: Rigorous testing on LIBERO benchmark tasks

## ✨ Key Features

- 🎯 **Per-Dimension Risk Prediction**: Train predictors for fine-grained collision forecasting
- 🔄 **Data Collection Pipeline**: Automated rollout collection with internal signal logging
- 🧠 **Activation Steering**: Extract and apply steering vectors for targeted safety interventions
- 📊 **Comprehensive Evaluation**: Success rate, collision metrics, and recovery rate analysis
- 🔧 **Modular Design**: Clean interfaces for model wrappers, data collection, and evaluation
- 🚀 **Production Ready**: Supports both local development and HPC cluster deployments

## 🗂️ Project Structure

```
MIST-VLA/
├── mist-vla/              # Main package
│   ├── scripts/           # Runnable entry points
│   ├── src/               # Core implementation
│   │   ├── data/         # Data collection and processing
│   │   ├── models/       # Model wrappers and interfaces
│   │   ├── steering/     # Activation steering implementation
│   │   └── evaluation/   # Evaluation metrics and utilities
│   ├── configs/          # Configuration files
│   ├── requirements.txt  # Python dependencies
│   └── setup.py         # Package installation
├── FailSafe_code/        # FailSafe baseline implementation
├── LIBERO/              # LIBERO benchmark environment
├── openvla-oft/         # OpenVLA-OFT evaluation pipeline
└── README.md            # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training and large-scale rollouts)
- LIBERO environment properly configured

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MIST-VLA.git
cd MIST-VLA
```

2. **Install dependencies**
```bash
cd mist-vla
pip install -r requirements.txt
pip install -e .
```

3. **Configure LIBERO**
Ensure LIBERO config exists at `~/.libero/config.yaml` with valid paths.

4. **Set up Python path**
```bash
export PYTHONPATH=$PWD:../openvla-oft:$PYTHONPATH
```

## 🎬 Quick Start

### Collect Failure Data

```bash
cd mist-vla
python scripts/collect_failure_data_oft_eval.py \
  --env libero_spatial \
  --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
  --n_success 10 \
  --n_failure 10 \
  --max-attempts-per-task 5 \
  --camera-res 256 \
  --save_dir data/rollouts_oft_eval \
  --seed 0
```

### Train Risk Predictor

```bash
python scripts/train_risk_predictor.py \
  --data_dir data/rollouts_oft_eval \
  --output_dir experiments/risk_predictor \
  --epochs 50
```

### Extract Steering Vectors

```bash
python scripts/extract_steering_vectors.py \
  --data_dir data/rollouts_oft_eval \
  --model_name moojink/openvla-7b-oft-finetuned-libero-spatial \
  --output_dir data/steering_vectors
```

### Run Evaluation

```bash
python scripts/run_evaluation.py \
  --env libero_spatial \
  --model_name moojink/openvla-7b-oft-finetuned-libero-spatial \
  --steering_vectors data/steering_vectors \
  --n_episodes 100
```

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- [Getting Started Guide](docs/GETTING_STARTED.md) - Detailed installation and setup
- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [API Reference](docs/API.md) - Complete API documentation
- [FAQ](docs/FAQ.md) - Frequently asked questions
- [Dependencies](DEPENDENCIES.md) - External dependencies guide
- [Changelog](CHANGELOG.md) - Version history

## 📚 Key Scripts

| Script | Description |
|--------|-------------|
| `collect_failure_data_oft_eval.py` | Uses OpenVLA-OFT eval pipeline and logs MIST-VLA signals (actions, hidden states, collisions, robot states) |
| `collect_failure_data.py` | Custom data collector with optional perturbation support |
| `collect_phase1_data.py` | Phase 1 data collection with collision labels |
| `train_risk_predictor.py` | Train per-dimension failure predictor |
| `extract_steering_vectors.py` | Build steering vectors for targeted mitigation |
| `run_evaluation.py` | Evaluate success rate, collisions, and recovery metrics |

## ⚙️ Configuration

Key configuration options can be found in `mist-vla/configs/`. Customize:
- Model parameters
- Data collection settings
- Training hyperparameters
- Evaluation metrics

## 🔬 Research

This project explores mechanistic interpretability techniques applied to VLA models to improve safety in robotic manipulation. Key research directions include:

- Understanding failure modes in VLA models
- Developing targeted interventions without full model retraining
- Scaling interpretability techniques to large vision-language-action models

## 📝 Notes

- LIBERO requires a config file at `~/.libero/config.yaml` with valid asset and dataset paths
- For OpenVLA-OFT integration, ensure `openvla-oft` is on your `PYTHONPATH`
- GPU acceleration is strongly recommended for large-scale rollouts and training
- All experiments are reproducible with fixed random seeds

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{mist-vla2025,
  title={MIST-VLA: Mechanistic Interpretability for Safer Targeted Steering in Vision-Language-Action Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/MIST-VLA}
}
```

## 🙏 Acknowledgments

- [LIBERO](https://libero-project.github.io/) - Benchmark environment
- [OpenVLA](https://openvla.github.io/) - Vision-Language-Action model
- FailSafe - Baseline safety methods

---

<div align="center">
Made with ❤️ for safer robotics
</div>
