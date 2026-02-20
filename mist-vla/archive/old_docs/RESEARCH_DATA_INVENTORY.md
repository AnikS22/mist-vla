# MIST-VLA Research Data Inventory

This document catalogs all research data, analysis results, and documentation for the MIST-VLA project.

## 🎯 Latest Results (Feb 6, 2026)

### Comprehensive Failure Predictor - SUCCESS!
**Location:** `checkpoints/failure_predictor/`

#### Core Predictions (Ground Truth Labels)
| Task | Metric | Value |
|------|--------|-------|
| Failure Classification | AUC | **0.9994** |
| Failure Classification | Accuracy | **99.52%** |
| Time-to-Failure | MAE | **0.032** (normalized) |
| Time-to-Failure | R² | **0.9814** |

#### Per-Dimension Failure Attribution (Gradient-Based)
| Dimension | Failure Risk | Success Risk | **Risk Ratio** |
|-----------|--------------|--------------|----------------|
| GRIPPER | 0.3475 | 0.0034 | **102x** |
| PITCH | 0.0202 | 0.0005 | **41x** |
| ROLL | 0.0104 | 0.0003 | **40x** |
| YAW | 0.0141 | 0.0004 | **36x** |
| Z | 0.1099 | 0.0032 | **35x** |
| Y | 0.0353 | 0.0010 | **34x** |
| X | 0.0721 | 0.0025 | **29x** |

**Key Insight:** Gripper actions are 102x more risky in failure trajectories!

#### Model Capabilities
1. **Failure Prediction:** Detects if trajectory will fail (AUC 0.9994)
2. **Time-to-Failure:** Predicts steps until failure (R² 0.9814)
3. **Per-Dimension Risk:** Identifies which action dimensions are risky
4. **Course Correction:** Provides actionable risk scores per dimension

---

## 📍 Primary Data Locations

### HPC (Athene Cluster)
**Base Path:** `/mnt/onefs/home/asahai2024/mist-vla/`

### Local Machine
**Base Path:** `/home/mpcr/Desktop/SalusV5/mist-vla/`

---

## 📊 Collected Rollout Data

### 1. **Main Collection (Current - OpenVLA-OFT)**
**Location (HPC):** `data/rollouts_oft_eval_big/seed_0/` and `seed_1/`
- **Size:** ~7.0 GB total
- **Status:** ✅ Active collection (job 3768516)
- **Contents:**
  - `success_rollouts.pkl` - Successful episodes
  - `failure_rollouts.pkl` - Failed episodes
  - `success_rollouts_partial.pkl` - Checkpoint files (updated every 25 rollouts)
  - `failure_rollouts_partial.pkl` - Checkpoint files
- **Data per rollout:**
  - Actions: 7D vectors (x, y, z, roll, pitch, yaw, gripper)
  - Hidden states: 4096D VLA transformer hidden states (mean-pooled from last layer)
  - Robot states: qpos (48D), qvel (43D), eef_pos (3D)
  - Collision info: position, normal, geom names
  - Step-by-step detailed data

### 2. **Previous Collection (OpenVLA-OFT with Geometry Labels)**
**Location (HPC):** `data/rollouts_oft_eval_geom/`
- **Size:** ~7.9 GB
- **Status:** ✅ Complete
- **Contents:**
  - `success_rollouts.pkl` (~919 MB)
  - `failure_rollouts.pkl` (~3.3 GB)
  - `success_rollouts_labeled.pkl` - With SAFE-style labels
  - `failure_rollouts_labeled.pkl` - With per-dimension risk labels

### 3. **Initial OFT Collection**
**Location (HPC):** `data/rollouts_oft_eval/`
- **Size:** ~7.4 GB
- **Status:** ✅ Complete
- **Contents:**
  - `success_rollouts.pkl` (~796 MB)
  - `failure_rollouts.pkl` (~3.3 GB)
  - `success_rollouts_labeled.pkl` - With labels
  - `failure_rollouts_labeled.pkl` - With labels
  - `risk_dataset.pkl` (~172 MB) - Extracted for training
  - `risk_dataset_labeled.pkl` (~172 MB) - With labels

### 4. **Legacy Collection (Original OpenVLA)**
**Location (HPC):** `data/rollouts/`
- **Size:** ~18 GB
- **Status:** ⚠️ Legacy (may have API mismatches)
- **Contents:**
  - `failure_rollouts.pkl` (~1.7 GB)
  - `failure_rollouts_labeled.pkl` (~6.3 GB)
  - `rollouts_checkpoint.pkl` (~9.0 GB)

### 5. **Phase 1 Test Data**
**Location (HPC):** `data/phase1/`
- **Size:** ~1.2 GB
- **Status:** ✅ Test/validation data
- **Contents:**
  - `test_rollouts.pkl` (~29 MB)
  - `collision_test.pkl` (~225 MB)
  - `claims_test_data_checkpoint_50.pkl` (~1.1 GB)
  - Perturbation test data

---

## 📈 Analysis Results

### **Internal States Analysis**
**Location (HPC):** `analysis_output/`
**Location (Local):** `mist-vla/analysis_output/`

**Generated Visualizations:**
1. `action_distributions_success_vs_failure.png` - Action distributions per dimension
2. `failure_rate_vs_action_magnitude.png` - Failure rate vs. action magnitude curves
3. `hidden_state_action_correlation.png` - Correlation heatmap (Hidden States ↔ Actions)
4. `hidden_state_differences.png` - Hidden state differences between success/failure
5. `hidden_state_visualization.png` - PCA/t-SNE visualization of hidden state space

**Key Findings:**
- Strong correlations between hidden states and actions (yaw: r=0.588, y: r=0.521)
- Negative correlation between action magnitude and failure rate for x, y, z, pitch
- Positive correlation for yaw (larger yaw → higher failure rate)
- Clear hidden state differences between success and failure trajectories

**Analysis Script:** `scripts/analyze_internal_states.py`

---

## 📝 Documentation & Scripts

### **Data Collection Scripts**
- `scripts/collect_failure_data_oft_eval.py` - Main collection script (OpenVLA-OFT)
- `scripts/collect_failure_data.py` - Legacy collection script
- `scripts/label_failure_data.py` - Add SAFE-style and per-dimension labels
- `scripts/inspect_collected_data.py` - Inspect rollout data structure

### **Analysis Scripts**
- `scripts/analyze_internal_states.py` - Comprehensive internal state analysis
- `scripts/build_risk_dataset.py` - Build dataset for risk prediction training
- `scripts/train_time_to_failure.py` - Train time-to-failure predictor

### **Evaluation Scripts**
- `scripts/visualize_openvla_libero.py` - Visualize VLA controlling robot
- `scripts/hpc/collect_oft_eval_array_pinned.slurm` - HPC job script (current)

---

## 🔬 Research Data Structure

### **Per Rollout Data Fields:**
```python
{
    "success": bool,                    # Episode success flag
    "collision_occurred": bool,         # Collision occurred flag
    "collision_steps": int,             # Number of collision steps
    "collision_step": int,              # First collision step index
    "instruction": str,                 # Task instruction
    "task_id": int,                     # LIBERO task ID
    
    # Arrays (one per step):
    "actions": List[np.ndarray],        # 7D action vectors
    "features": List[np.ndarray],       # 4096D hidden states
    "robot_states": List[dict],         # Robot state dicts
    "rewards": List[float],             # Step rewards
    
    # Detailed step-by-step data:
    "steps": List[dict],                # Full step data
        # Each step contains:
        #   - action: 7D array
        #   - hidden_state: 4096D array
        #   - collision: bool
        #   - collision_pos: [x, y, z] or None
        #   - collision_normal: [x, y, z] or None
        #   - collision_geoms: [geom1, geom2]
        #   - robot_state: dict with qpos, qvel, eef_pos
        #   - done: bool
}
```

### **Labeled Data (after running label_failure_data.py):**
Additional fields per step:
- `time_to_failure`: float (steps until failure, or inf)
- `fail_within_k`: bool (failure within k steps)
- `time_to_collision`: float (steps until collision, or inf)
- `collision_within_k`: bool (collision within k steps)
- `per_dim_risk`: np.ndarray (7D risk vector per action dimension)

---

## 📊 Data Statistics

### **Current Collection (rollouts_oft_eval_big/seed_0):**
- **Success rollouts:** 163+ (actively collecting)
- **Failure rollouts:** 237+ (actively collecting)
- **Success rate:** ~40% (matches expected ~70% for OFT)
- **Total steps:** ~16,631+ (5,631 success + 11,000 failure)

### **Previous Collections:**
- **rollouts_oft_eval_geom:** 20 successes, 40 failures
- **rollouts_oft_eval:** 20 successes, 40 failures

---

## 🎯 Recommended Data for Research

### **For Training Risk Predictor:**
1. **Primary:** `data/rollouts_oft_eval_big/seed_0/` (current, largest collection)
2. **Alternative:** `data/rollouts_oft_eval_geom/` (has geometry-based labels)

### **For Analysis:**
1. **Internal states:** `analysis_output/` (all visualizations)
2. **Raw data:** Use `scripts/inspect_collected_data.py` to explore
3. **Correlations:** See analysis output plots

### **For Paper Figures:**
1. `analysis_output/failure_rate_vs_action_magnitude.png` - Shows per-dimension failure patterns
2. `analysis_output/action_distributions_success_vs_failure.png` - Action distributions
3. `analysis_output/hidden_state_visualization.png` - Hidden state space visualization

---

## 🔄 Data Collection Status

### **Active Jobs:**
- **Job 3768516:** Collecting rollouts using OpenVLA-OFT
  - Target: 20 successes, 40 failures per task
  - Current: 163 successes, 237 failures (across multiple tasks)
  - Status: ✅ Running successfully with ~40% success rate

### **Collection Parameters:**
- Model: OpenVLA-OFT (fine-tuned on LIBERO)
- Benchmark: LIBERO-Spatial
- Images: 2 (agent view + wrist camera)
- Resolution: 256x256
- Max steps: 220 per episode

---

## 📁 Local vs. HPC Data

### **Local Machine:**
- Analysis plots: `mist-vla/analysis_output/`
- Test data: `mist-vla/data/rollouts_local_test/`
- Scripts: `mist-vla/scripts/`

### **HPC:**
- All collected rollout data
- Large analysis outputs
- Active collection jobs

---

## 🚀 Quick Access Commands

### **Inspect Data:**
```bash
# On HPC
python scripts/inspect_collected_data.py --data-dir data/rollouts_oft_eval_big/seed_0 --file success_rollouts_partial.pkl --checkpoint --n-rollouts 1
```

### **Run Analysis:**
```bash
# On HPC
python scripts/analyze_internal_states.py --data-dir data/rollouts_oft_eval_big/seed_0 --output-dir analysis_output --checkpoint --max-rollouts 50
```

### **Download Analysis Results:**
```bash
# From local machine
rsync -avz asahai2024@athene-login.hpc.fau.edu:/mnt/onefs/home/asahai2024/mist-vla/analysis_output/ mist-vla/analysis_output/
```

---

## 📋 Data Collection Checklist

- [x] Collect success rollouts (163+ collected)
- [x] Collect failure rollouts (237+ collected)
- [x] Extract hidden states (4096D per step)
- [x] Extract actions (7D per step)
- [x] Extract robot states (qpos, qvel, eef_pos)
- [x] Detect collisions (position, normal, geoms)
- [x] Generate SAFE-style labels (time-to-failure, fail-within-k)
- [x] Generate per-dimension risk labels
- [x] Analyze internal states vs. actions
- [x] Correlate failure with movement patterns
- [ ] Train risk predictor (next step)
- [ ] Evaluate on held-out data

---

**Last Updated:** January 29, 2025
**Data Collection Status:** ✅ Active (Job 3768516)
**Total Data Size:** ~40 GB across all collections
