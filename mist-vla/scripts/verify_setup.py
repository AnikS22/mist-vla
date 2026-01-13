#!/usr/bin/env python3
"""
Verify that MIST-VLA setup is complete and functional.
"""

import sys
from pathlib import Path


def check_imports():
    """Check that all required packages are installed."""
    print("Checking imports...")

    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('captum', 'Captum'),
        ('sklearn', 'scikit-learn'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]

    missing = []

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False

    print("✓ All required packages installed\n")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("Checking CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ Device: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️  CUDA not available - will use CPU (very slow)")

    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

    print()
    return True


def check_directories():
    """Check that required directories exist."""
    print("Checking directory structure...")

    required_dirs = [
        'src/models',
        'src/failure_detection',
        'src/attribution',
        'src/steering',
        'src/recovery',
        'scripts',
        'configs',
        'data',
    ]

    missing = []

    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            missing.append(dir_path)

    if missing:
        print(f"\n⚠️  Missing directories: {', '.join(missing)}")
        print("   Create with: mkdir -p " + " ".join(missing))
        return False

    print("✓ All required directories exist\n")
    return True


def check_mist_imports():
    """Check that MIST-VLA modules can be imported."""
    print("Checking MIST-VLA imports...")

    mist_modules = [
        ('src.models.hooked_openvla', 'HookedOpenVLA'),
        ('src.failure_detection.safe_detector', 'SAFEDetector'),
        ('src.attribution.failure_localizer', 'FailureLocalizer'),
        ('src.steering.activation_steerer', 'ActivationSteerer'),
        ('src.recovery.recovery_orchestrator', 'MISTRecoveryOrchestrator'),
    ]

    missing = []

    # Add current directory to path
    sys.path.insert(0, str(Path.cwd()))

    for module, component in mist_modules:
        try:
            __import__(module)
            print(f"  ✓ {component}")
        except ImportError as e:
            print(f"  ✗ {component} - {e}")
            missing.append(component)

    if missing:
        print(f"\n⚠️  Cannot import: {', '.join(missing)}")
        print("   Make sure you're in the mist-vla directory")
        print("   Run: pip install -e .")
        return False

    print("✓ All MIST-VLA modules can be imported\n")
    return True


def check_model_access():
    """Check if we can access the model."""
    print("Checking model access...")

    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b",
            trust_remote_code=True
        )

        print("  ✓ Can access openvla/openvla-7b")
        print("  ✓ Model is downloaded/cached")

    except Exception as e:
        print(f"  ⚠️  Cannot access model: {e}")
        print("     Model will be downloaded on first use")

    print()
    return True


def main():
    """Run all verification checks."""
    print("="*60)
    print("MIST-VLA Setup Verification")
    print("="*60)
    print()

    checks = [
        check_imports(),
        check_cuda(),
        check_directories(),
        check_mist_imports(),
        check_model_access(),
    ]

    print("="*60)

    if all(checks):
        print("✅ Setup verification PASSED")
        print("\nYou're ready to start using MIST-VLA!")
        print("Next steps:")
        print("  1. python scripts/collect_failure_data.py")
        print("  2. python scripts/train_failure_detector.py")
        print("  3. python scripts/extract_steering_vectors.py")
        print("  4. python scripts/run_libero_eval.py")
    else:
        print("⚠️  Setup verification FAILED")
        print("\nPlease fix the issues above and run again.")
        print("See GETTING_STARTED.md for detailed setup instructions.")
        sys.exit(1)

    print("="*60)


if __name__ == "__main__":
    main()
