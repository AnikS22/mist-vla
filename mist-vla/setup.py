from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mist-vla",
    version="0.1.0",
    description="Mechanistic Interpretability for Safer Targeted Steering in Vision-Language-Action Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MIST-VLA Contributors",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/MIST-VLA",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/MIST-VLA/issues",
        "Documentation": "https://github.com/yourusername/MIST-VLA/tree/master/docs",
        "Source Code": "https://github.com/yourusername/MIST-VLA",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "captum>=0.7.0",
        "transformer-lens>=1.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "wandb>=0.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    python_requires=">=3.8",
    keywords="robotics vla interpretability mechanistic-interpretability activation-steering safety libero",
    license="MIT",
)
