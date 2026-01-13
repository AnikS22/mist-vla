from setuptools import setup, find_packages

setup(
    name="mist-vla",
    version="0.1.0",
    description="Mechanistic Interpretability for Steering and Transparent VLA Failure Recovery",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "captum",
        "transformer-lens",
        "scikit-learn",
        "numpy",
        "tqdm",
        "pyyaml",
    ],
    python_requires=">=3.10",
)
