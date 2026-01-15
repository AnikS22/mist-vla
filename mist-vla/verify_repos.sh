#!/bin/bash
# Verify which GitHub repos are being used

echo "========================================="
echo "Verifying GitHub Repositories Usage"
echo "========================================="
echo ""

echo "1. OpenVLA Repository"
echo "---------------------"
if [ -d "$HOME/openvla" ]; then
    echo "✓ OpenVLA installed at: $HOME/openvla"
    cd $HOME/openvla
    echo "  Git remote:"
    git remote -v | head -2
    echo "  Latest commit:"
    git log -1 --oneline
    echo ""
else
    echo "✗ OpenVLA not found"
fi

echo "2. LIBERO Repository"
echo "--------------------"
if [ -d "$HOME/LIBERO" ]; then
    echo "✓ LIBERO installed at: $HOME/LIBERO"
    cd $HOME/LIBERO
    echo "  Git remote:"
    git remote -v | head -2
    echo "  Latest commit:"
    git log -1 --oneline
    echo ""
else
    echo "✗ LIBERO not found"
fi

echo "3. Python Package Locations"
echo "---------------------------"
python -c "
import sys

# Check OpenVLA
try:
    import openvla
    print(f'✓ openvla imported from: {openvla.__file__}')
except ImportError:
    print('✗ openvla not importable')

# Check LIBERO
try:
    import libero
    print(f'✓ libero imported from: {libero.__file__}')
except ImportError:
    print('✗ libero not importable')

# Check transformers (for OpenVLA model)
try:
    import transformers
    print(f'✓ transformers: {transformers.__version__}')
except ImportError:
    print('✗ transformers not found')

print('')
"

echo "4. Model Cache Location"
echo "-----------------------"
echo "OpenVLA models are cached in:"
if [ -d "$HOME/.cache/huggingface" ]; then
    echo "  $HOME/.cache/huggingface"
    du -sh $HOME/.cache/huggingface/hub/models--openvla* 2>/dev/null | head -3 || echo "  (Model not downloaded yet)"
else
    echo "  (Cache directory doesn't exist yet)"
fi
echo ""

echo "5. MIST-VLA Code"
echo "----------------"
if [ -d "$HOME/mist-vla" ]; then
    echo "✓ MIST-VLA at: $HOME/mist-vla"
    echo "  Files:"
    ls -1 $HOME/mist-vla/src/*/*.py | wc -l
    echo "  Python modules found"
else
    echo "✗ MIST-VLA not found"
fi
echo ""

echo "========================================="
echo "Usage in Code"
echo "========================================="
echo ""
echo "When you run scripts, they use:"
echo ""
echo "1. OpenVLA (from ~/openvla or HuggingFace):"
echo "   - Model: openvla/openvla-7b"
echo "   - Used in: src/models/hooked_openvla.py"
echo "   - Function: VLA inference for robot actions"
echo ""
echo "2. LIBERO (from ~/LIBERO):"
echo "   - Tasks: libero_spatial, libero_object, etc."
echo "   - Used in: scripts/collect_failure_data.py"
echo "   - Function: Robot simulation environment"
echo ""
echo "3. MIST-VLA (from ~/mist-vla):"
echo "   - Our code: failure detection, steering, etc."
echo "   - Used in: all scripts"
echo "   - Function: Main contribution of this project"
echo ""

echo "To verify repos are actually being used:"
echo "  1. Run: python test_real_quick.py"
echo "  2. Watch for 'Loading REAL OpenVLA' and 'Loading REAL LIBERO'"
echo "  3. Check data/test_rollouts/ for actual output"
