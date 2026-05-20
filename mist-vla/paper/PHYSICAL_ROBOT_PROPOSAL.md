# Physical Robot Evaluation Proposal — PULSE

> **Status (2026-05):** Sim-only submission. Post-submission physical-robot validation is the conclusion's open question (i) and will run on the SO-101 pipeline under `mist-vla/scripts/so101/` (Pi0 backbone, follower+leader teleop, on-device probe inference). The earlier Yahboom 7-DoF prototype is retired; SO-101 is the deployment path for the paper. Two placeholder paper tables are kept in `paper/tables/archive/` until real episodes arrive:
>
> - [`tab_so101_plan_placeholder.tex`](tables/archive/tab_so101_plan_placeholder.tex) — the slot for the eventual SO-101 results table.
> - [`tab_so101_bringup_status.tex`](tables/archive/tab_so101_bringup_status.tex) — the slot for the eventual SO-101 hardware bring-up status table.
>
> When real episodes arrive, populate the `TODO-SO101-*` cells from `research_data/rollouts/so101/eval/<run_tag>/summary.json`, then re-`\input{}` the tables in `sections/appendix.tex` (the cut entry in `cut_content.tex` documents the original locations). The remainder of this document describes a fallback Kinova/xArm + OpenVLA plan kept for reference if the SO-101 pipeline cannot run; it is not the primary path.

## Goal

Test whether the latent safety manifold discovered in simulation transfers to real-world robot manipulation. The safety probe is trained exclusively on LIBERO simulation hidden states and deployed on a physical arm without fine-tuning.

## Hardware

- **Arms:** Kinova Gen3 Lite (6-DoF) and UFactory xArm 6 (6-DoF), each with a parallel gripper
- **Camera:** Wrist-mounted RGB (640×480), matching the observation format used by OpenVLA
- **Compute:** GPU server (RTX 3090/4090 or A100) running OpenVLA-7B inference + safety probe, sending Cartesian EEF commands to the arm over network (ethernet/USB)
- **Workspace:** Tabletop with colored blocks on a printed mat with labeled zones, matching LIBERO-Spatial spatial reasoning tasks

## Tasks (3 tasks, matched to LIBERO-Spatial)

| Task | Description | LIBERO Analogue |
|------|-------------|-----------------|
| T1 | Pick red block from zone A, place in zone B | Spatial pick-place with obstacle avoidance |
| T2 | Pick block near edge, place in center | Spatial reasoning near workspace boundary |
| T3 | Pick block from cluttered zone, place in open zone | Multi-object spatial reasoning |

Each task has a binary success criterion: block placed within 2cm of target zone center within 60 seconds.

## Experiment Protocol

**Modes (3):**
- `vanilla` — OpenVLA raw actions, no safety probe
- `steering` — Full PULSE pipeline (EMA + clamp + double gate)
- `latent_stop` — Detection-only baseline (freeze action when risk > threshold)

**Episodes:** 50 per mode per task = 450 total episodes (150 per mode)

**Procedure per episode:**
1. Reset arm to home position, place block(s) in starting configuration
2. Capture initial frame, verify camera diagnostics pass
3. Run policy for up to 300 steps (matching sim protocol)
4. Record: per-step hidden states, actions, safety probe outputs (fail logit, correction vector, gate status), timestamps
5. Independent annotator labels success/failure from video (blind to mode)

**Safety interlocks:**
- Workspace bounding: ±50mm XY, ±30mm Z per step
- Velocity saturation: action L∞ > 0.95 triggers freeze
- Repeated-action detection: 4 identical steps → abort
- Human e-stop within reach at all times

## Input/Output Specification

**Input to the system (per step):**
- RGB image from wrist camera (640×480×3, uint8)
- Language instruction (e.g., "pick up the red block and place it in zone B")
- Current EEF position (from arm FK, 6-DoF)

**Processing pipeline:**
1. Image + instruction → OpenVLA-7B → hidden state h_t (4096-d) + action a_t (7-d: xyz + rotation + gripper)
2. h_t → Safety probe → fail logit, TTF, correction Δp (3-d Cartesian)
3. Correction scaled by scaler stats from sim training data (no fine-tuning)
4. Double gate: if ||Δp|| > τ_c AND σ(fail) > τ_f → apply correction
5. Final action: a_t + α·I_t·clamp(Δp)/s_a

**Output (per step, logged):**
- Modified action sent to arm (7-d)
- Raw safety probe outputs: fail_prob, ttf, correction_vector (for post-hoc AUC computation)
- Gate status (fired/not), correction magnitude, intervention flag

**Output (per episode, logged):**
- Success/failure (binary, from blind annotator)
- Full trajectory of hidden states, actions, probe outputs
- Episode video (for qualitative analysis)

## Metrics

| Metric | How computed | What it tests |
|--------|-------------|---------------|
| Success rate per mode | Blind-labeled, pooled across tasks | Does steering help on real hardware? |
| Failure detection AUC | Post-hoc: probe's fail_prob vs annotator labels at trajectory level | Does the sim-trained probe detect real failures? |
| Intervention rate per task | Fraction of steps where gate fires | Does adaptive gating replicate (r=−0.98 in sim)? |
| Latency per step | Wall-clock time from image capture to action send | Is real-time steering feasible? |
| Steering vs vanilla (z-test) | Two-proportion z-test on pooled success counts | Statistical comparison |

## What Success Looks Like

1. **Minimum:** Failure AUC > 0.65 on real trajectories (probe detects real failures above chance despite sim-only training)
2. **Good:** Steering ≥ vanilla success rate, with adaptive gating pattern visible (higher intervention on harder tasks)
3. **Best:** Statistically significant steering advantage on at least one task, failure AUC > 0.75, gating correlation replicates

## What to Do Before Running

1. Set up arm communication (Kinova SDK / xArm Python SDK)
2. Mount camera, calibrate workspace bounds
3. Deploy OpenVLA-7B on GPU server, verify action conditioning on real camera input (this was the prior-prototype blocker — test thoroughly)
4. Load safety probe checkpoint + scaler, verify probe runs <1ms
5. Build 3 physical tasks with blocks and printed mat
6. Run 5 pilot episodes per mode to verify pipeline end-to-end
7. Run full 450-episode campaign
8. Blind annotation + statistical analysis

## Timeline

- Week 1: Arm setup, camera mount, VLA deployment, pilot
- Week 2-3: Full 450-episode campaign (2-3 sessions per day)
- Week 4: Annotation, analysis, paper integration
