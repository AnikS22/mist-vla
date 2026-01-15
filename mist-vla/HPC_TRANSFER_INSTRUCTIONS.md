# HPC Transfer Instructions

## Issue: Cannot Connect to HPC

The automatic transfer failed because `athene-login.fau.edu` is not reachable from this machine.

**Possible reasons:**
- Not connected to FAU network
- VPN not running
- Hostname is incorrect

---

## Option 1: Run Transfer Script (Recommended)

When connected to FAU network or VPN:

```bash
cd /home/mpcr/Desktop/SalusV5/mist-vla
./transfer_to_hpc.sh
```

This script will:
1. Clean cache files
2. Test HPC connection
3. Transfer all code files
4. Show next steps

---

## Option 2: Manual Transfer

### Step 1: Connect to VPN (if off-campus)

If you're off-campus, connect to FAU VPN first.

### Step 2: Transfer Files

```bash
cd /home/mpcr/Desktop/SalusV5/mist-vla

rsync -avz --progress \
    --exclude='data/' \
    --exclude='models/' \
    --exclude='results/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    . asahai2024@athene-login.fau.edu:~/mist-vla/
```

**Alternative using scp:**

```bash
cd /home/mpcr/Desktop/SalusV5/mist-vla
tar czf mist-vla.tar.gz \
    --exclude='data' \
    --exclude='models' \
    --exclude='results' \
    --exclude='__pycache__' \
    .

scp mist-vla.tar.gz asahai2024@athene-login.fau.edu:~/

# Then on HPC:
# ssh asahai2024@athene-login.fau.edu
# mkdir -p mist-vla
# tar xzf mist-vla.tar.gz -C mist-vla/
```

---

## Option 3: Use Different Transfer Method

### Via Git (if you have a repo)

```bash
# On local machine
cd /home/mpcr/Desktop/SalusV5/mist-vla
git init
git add .
git commit -m "Initial MIST-VLA implementation"
git push origin main

# On HPC
git clone <your-repo-url> ~/mist-vla
```

### Via USB/External Drive

1. Copy project to USB drive
2. Transfer USB to machine with HPC access
3. Upload from there

---

## What Gets Transferred

**Included:**
- ✅ All Python source code (`src/`, `scripts/`)
- ✅ Documentation (`.md` files)
- ✅ Configuration files
- ✅ SLURM scripts
- ✅ Test scripts

**Excluded (will be generated on HPC):**
- ❌ `data/` directory (empty or cached data)
- ❌ `models/` directory (will be created during training)
- ❌ `results/` directory (will be created during evaluation)
- ❌ Python cache files (`__pycache__`, `*.pyc`)

**Total size:** ~50-100 KB (just code)

---

## After Transfer: Next Steps on HPC

### 1. SSH to HPC

```bash
ssh asahai2024@athene-login.fau.edu
```

### 2. Navigate to Project

```bash
cd ~/mist-vla
```

### 3. Check Files Transferred

```bash
ls -lh
ls -R src/
ls scripts/
```

Expected output:
```
✓ src/data_collection/
✓ src/training/
✓ src/steering/
✓ src/evaluation/
✓ scripts/collect_phase1_data.py
✓ scripts/train_risk_predictor.py
✓ scripts/extract_steering_vectors.py
✓ scripts/run_evaluation.py
✓ claude.md
✓ README.md
```

### 4. Verify Environment

```bash
conda activate mist-vla
python scripts/verify_phase0.py
```

### 5. Quick Test (Recommended First)

```bash
# Test with just 5 rollouts (~1 hour)
python scripts/collect_phase1_data.py \
    --num-rollouts 5 \
    --max-steps 50 \
    --output data/phase1/test_rollouts.pkl
```

If this works, you're ready for full pipeline!

### 6. Full Pipeline

```bash
# Option A: Run phases individually
python scripts/collect_phase1_data.py --num-rollouts 2000
python scripts/compute_risk_labels.py
python scripts/train_risk_predictor.py --epochs 50
python scripts/extract_steering_vectors.py
python scripts/run_evaluation.py

# Option B: Submit SLURM job
sbatch run_hpc.slurm
```

---

## Troubleshooting Transfer

### Connection Timeout

**Error:** `ssh: connect to host athene-login.fau.edu port 22: Connection timed out`

**Solutions:**
1. Check VPN connection
2. Verify hostname: `ping athene-login.fau.edu`
3. Try alternative hostname (if HPC has multiple login nodes)

### Permission Denied

**Error:** `Permission denied (publickey,password)`

**Solutions:**
1. Check username: `asahai2024`
2. Ensure SSH key is set up or enter password when prompted
3. Check SSH config: `~/.ssh/config`

### Hostname Unknown

**Error:** `ssh: Could not resolve hostname`

**Solutions:**
1. Connect to FAU network/VPN
2. Check if hostname is correct
3. Ask IT for correct HPC login node address

---

## Verify Transfer Success

On HPC, run:

```bash
cd ~/mist-vla

# Check Python files
find src/ -name "*.py" | wc -l
# Expected: ~15-20 files

# Check scripts
ls scripts/*.py | wc -l
# Expected: ~6 files

# Test imports
python -c "from src.training.risk_predictor import RiskPredictor; print('✓ Imports work')"
# Expected: ✓ Imports work

# Run test suite
python test_implementation.py
# Expected: ✅ All tests passed!
```

---

## Quick Status Check

After transfer, you can quickly check status:

```bash
cd ~/mist-vla
cat QUICK_STATUS.txt
```

This shows:
- What's been tested
- What's ready to run
- Expected timeline
- Common issues and fixes

---

## Need Help?

1. **Check documentation:**
   - `README.md` - Quick start
   - `claude.md` - Complete specification
   - `NEXT_STEPS.md` - Detailed HPC guide
   - `TEST_REPORT.md` - What works and what doesn't

2. **Test locally first:**
   ```bash
   python test_implementation.py
   ```

3. **Check HPC status:**
   ```bash
   squeue -u $USER  # Check running jobs
   tail -f slurm-*.out  # Monitor job output
   ```

---

## Summary

**Current Status:** ✅ Code ready, ❌ Transfer pending

**Required:** Connection to FAU network/VPN

**Next Action:** Run `./transfer_to_hpc.sh` when connected

**Time:** ~30 seconds to transfer (~50 KB)

**After Transfer:** Run quick test (5 rollouts, 1 hour)
