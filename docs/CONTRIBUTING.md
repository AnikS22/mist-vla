# Contributing to MIST-VLA

Thank you for your interest in contributing to MIST-VLA! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/MIST-VLA.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit with clear, descriptive messages
7. Push to your fork and submit a pull request

## Development Setup

```bash
cd mist-vla
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install black flake8 pytest mypy
```

## Code Style

- Follow PEP 8 guidelines
- Use `black` for code formatting: `black .`
- Run `flake8` for linting
- Add type hints where appropriate
- Write clear, descriptive variable and function names

## Testing

- Add tests for new features in the `tests/` directory
- Ensure all tests pass before submitting a PR
- Run tests with: `pytest tests/`

## Pull Request Guidelines

1. **Clear Description**: Explain what your PR does and why
2. **Small, Focused Changes**: Keep PRs focused on a single feature or fix
3. **Tests**: Include tests for new functionality
4. **Documentation**: Update relevant documentation
5. **No Breaking Changes**: Avoid breaking existing functionality without discussion

## Commit Messages

Use clear, descriptive commit messages:
- Start with a verb in present tense (Add, Fix, Update, Remove)
- Keep the first line under 72 characters
- Add details in the body if needed

Examples:
```
Add per-dimension risk prediction module

Fix collision detection in libero_spatial environment

Update README with installation instructions
```

## Reporting Issues

When reporting issues, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, GPU)
- Error messages or logs

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them get started
- Focus on the technical merits of contributions
- Assume good intentions

## Questions?

Feel free to open an issue for questions or join discussions in existing issues.

Thank you for contributing!
