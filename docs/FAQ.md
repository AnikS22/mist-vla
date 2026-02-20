# Frequently Asked Questions

## General Questions

### What is MIST-VLA?

MIST-VLA is a research framework for improving the safety and reliability of Vision-Language-Action (VLA) models through mechanistic interpretability and targeted activation steering.

### What are the main use cases?

- Per-dimension collision forecasting in robotic manipulation
- Targeted activation steering for safety interventions
- Failure analysis and characterization of VLA models
- Research in mechanistic interpretability for robotics

### Do I need a GPU?

While not strictly required, a CUDA-capable GPU is highly recommended for:
- Training risk predictors
- Large-scale data collection
- Model evaluation

CPU-only usage is possible but will be significantly slower.

## Installation & Setup

### Which Python version should I use?

Python 3.8 or higher is required. We recommend Python 3.10 for the best compatibility.

### How do I set up LIBERO?

1. Clone the LIBERO repository
2. Follow LIBERO's installation instructions
3. Create `~/.libero/config.yaml` with valid paths
4. Verify installation by running LIBERO examples

### What if dependencies conflict?

If you encounter dependency conflicts:
1. Use a fresh virtual environment
2. Install dependencies in the order specified in `requirements.txt`
3. Check the [DEPENDENCIES.md](../DEPENDENCIES.md) for version requirements

## Data Collection

### How much data do I need?

For initial experiments:
- 50-100 rollouts per task is a good starting point
- Aim for a mix of successful and failed rollouts (e.g., 50/50 split)

For robust training:
- 500+ rollouts per task recommended
- More diverse failure modes improve predictor quality

### How long does data collection take?

Typical collection rates:
- ~1-2 minutes per rollout (depends on task complexity)
- 100 rollouts: ~2-3 hours
- Use parallel collection on HPC for large-scale data

### Can I use my own robot environment?

Yes! MIST-VLA is designed to be extensible. You'll need to:
1. Implement a model wrapper in `src/models/`
2. Create a data collector in `src/data/`
3. Define collision detection for your environment

## Training

### How long does training take?

Training times vary based on:
- Dataset size: 100-1000 rollouts
- Model complexity: 10-50 epochs
- Hardware: 1-4 hours on a modern GPU

### What if my model isn't improving?

Common fixes:
1. Check data quality and balance
2. Adjust learning rate
3. Increase training data
4. Verify collision labels are correct

### Can I use custom architectures?

Yes! Implement your architecture in `src/models/` and follow the existing model interface.

## Evaluation

### What metrics should I track?

Key metrics:
- **Success rate**: Overall task completion
- **Collision rate**: Frequency of collisions
- **Recovery rate**: Ability to recover from near-collisions
- **Precision/Recall**: For risk prediction

### How do I compare with baselines?

See the evaluation scripts:
- Compare against vanilla VLA (no steering)
- Compare against FailSafe baseline
- Run statistical significance tests

## Deployment

### Can I deploy this on a real robot?

MIST-VLA is primarily designed for simulation. Real robot deployment requires:
- Real-time inference optimization
- Safety verification
- Extensive real-world testing

### How do I scale to multiple GPUs?

Multi-GPU support can be added:
- Use PyTorch DistributedDataParallel for training
- Parallelize data collection across nodes
- See HPC setup documentation

## Research & Development

### Can I extend this for my research?

Absolutely! MIST-VLA is open source. Common extensions:
- New environments beyond LIBERO
- Different VLA model architectures
- Alternative steering methods
- Novel interpretability techniques

### How do I cite this work?

See the [Citation](../README.md#-citation) section in the README.

### Where can I find the paper?

[Add paper link when available]

## Troubleshooting

### My installation fails

1. Ensure Python version is 3.8+
2. Update pip: `pip install --upgrade pip`
3. Install one dependency at a time to identify conflicts
4. Check [GitHub Issues](https://github.com/yourusername/MIST-VLA/issues)

### Data collection crashes

Common causes:
- LIBERO configuration issues
- Out of memory (reduce batch size)
- Invalid model checkpoints
- Missing environment assets

### Results don't match the paper

Potential reasons:
- Different random seeds
- Different dataset split
- Hyperparameter differences
- Check experiment configurations

## Still Have Questions?

- Open an [issue](https://github.com/yourusername/MIST-VLA/issues)
- Check existing [discussions](https://github.com/yourusername/MIST-VLA/discussions)
- Contact the maintainers
