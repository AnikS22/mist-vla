# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Real-time inference optimization
- Multi-GPU training support
- Additional steering methods
- Interactive visualization dashboard
- Real robot deployment tools

## [0.1.0] - 2025-01-25

### Added
- Initial release of MIST-VLA framework
- Data collection pipeline with OpenVLA-OFT integration
- Per-dimension risk prediction model
- Activation steering implementation
- Comprehensive evaluation metrics
- LIBERO environment support
- Documentation and examples
- Unit tests for core components
- Configuration management system
- HPC deployment scripts

### Features
- **Data Collection**:
  - Automated rollout collection
  - Internal signal logging (actions, activations, collisions)
  - Support for success/failure data collection
  - Configurable camera resolution and seeds

- **Risk Prediction**:
  - Per-dimension collision forecasting
  - Configurable network architecture
  - Training with validation monitoring
  - Model checkpointing

- **Activation Steering**:
  - Multiple extraction methods (mean_diff, CAA, PCA)
  - Targeted layer selection
  - Coefficient sweeping for evaluation

- **Evaluation**:
  - Success rate tracking
  - Collision detection and logging
  - Recovery rate computation
  - Statistical analysis tools

### Documentation
- Comprehensive README with quick start guide
- API reference documentation
- Architecture overview
- Getting started guide
- FAQ section
- Contribution guidelines
- Issue and PR templates

### Infrastructure
- MIT License
- GitHub Actions workflows (planned)
- Docker support (planned)
- Pre-commit hooks configuration (planned)

## [0.0.1] - 2025-01-13

### Added
- Project initialization
- Basic directory structure
- Core dependencies setup
- Initial LIBERO integration
- Proof-of-concept scripts

---

## Release Notes

### Version 0.1.0 Highlights

This is the first official release of MIST-VLA, providing a complete framework for:

1. **Safety-focused VLA research**: Tools for understanding and improving Vision-Language-Action model safety
2. **Mechanistic interpretability**: Extract and analyze internal representations
3. **Targeted interventions**: Apply precise activation steering for collision avoidance
4. **Rigorous evaluation**: Comprehensive metrics for safety and performance

### Known Issues

- GPU memory consumption can be high for large-scale data collection
- LIBERO configuration requires manual setup
- Limited to simulation environments (real robot support planned)
- Some edge cases in collision detection may need refinement

### Migration Guide

N/A - First release

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## Questions or Feedback?

Open an issue on [GitHub](https://github.com/yourusername/MIST-VLA/issues) or start a discussion.
