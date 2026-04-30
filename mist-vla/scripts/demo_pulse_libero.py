#!/usr/bin/env python3
"""
PULSE Live Demo — LIBERO + OpenVLA + Real Safety Probe
=======================================================

This is the REAL demo. It runs:
  1. LIBERO-Spatial environment with MuJoCo GLFW viewer (GUI window)
  2. OpenVLA-7B (or ACT) generating actions from the camera image
  3. The actual trained safety probe reading real VLA hidden states
  4. Double-gated steering corrections applied in real-time

The probe outputs are REAL — trained on LIBERO OpenVLA hidden states,
evaluated on the same environment. This is exactly what the paper reports.

Usage:
  # With OpenVLA (needs ~14GB VRAM across GPUs):
  MUJOCO_GL=glfw python3 scripts/demo_pulse_libero.py

  # Specific task:
  MUJOCO_GL=glfw python3 scripts/demo_pulse_libero.py --task-id 4

  # Different mode:
  MUJOCO_GL=glfw python3 scripts/demo_pulse_libero.py --mode vanilla

Requires: MUJOCO_GL=glfw and a display ($DISPLAY set)
"""

from __future__ import annotations
import argparse, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT.parent / "openvla-oft"))

# Stub data-loading modules not needed for inference
import types
for _mod_name in ['dlimp']:
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        _m.DLataset = type('DLataset', (), {})  # stub class
        sys.modules[_mod_name] = _m

from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from PIL import Image


# ─── Safety Probe (exact architecture from training) ─────────────

class EEFCorrectionMLP(nn.Module):
    HIDDEN_DIM = 256
    def __init__(self, input_dim=4096):
        super().__init__()
        h = self.HIDDEN_DIM
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.LayerNorm(h//2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h//2, h//4), nn.LayerNorm(h//4), nn.GELU(), nn.Dropout(0.3),
        )
        feat = h // 4
        self.fail_head = nn.Linear(feat, 1)
        self.ttf_head = nn.Linear(feat, 1)
        self.correction_head = nn.Linear(feat, 3)

    def forward(self, x):
        feat = self.encoder(self.input_norm(x))
        return {
            "will_fail": self.fail_head(feat).squeeze(-1),
            "ttf": self.ttf_head(feat).squeeze(-1),
            "correction": self.correction_head(feat),
        }


class ProbeRuntime:
    def __init__(self, ckpt_path, device="cuda:0"):
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.input_dim = int(ckpt["input_dim"])
        self.model = EEFCorrectionMLP(input_dim=self.input_dim)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.scaler_mean = np.asarray(ckpt.get("scaler_mean", np.zeros(self.input_dim)), dtype=np.float32)
        self.scaler_scale = np.asarray(ckpt.get("scaler_scale", np.ones(self.input_dim)), dtype=np.float32)

    def predict(self, hidden_state):
        """Takes raw hidden state numpy array, returns (fail_prob, ttf, correction)."""
        if isinstance(hidden_state, torch.Tensor):
            hidden_state = hidden_state.detach().float().cpu().numpy()
        x = np.asarray(hidden_state, dtype=np.float32).flatten()[:self.input_dim]
        if x.size < self.input_dim:
            x = np.pad(x, (0, self.input_dim - x.size))
        # Scale
        x = (x - self.scaler_mean) / np.clip(self.scaler_scale, 1e-8, None)
        # Forward
        t = torch.from_numpy(x).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.model(t)
        raw_logit = float(out["will_fail"].item())
        fail_prob = float(torch.sigmoid(out["will_fail"]).item())
        ttf = float(out["ttf"].item())
        correction = out["correction"].cpu().numpy().flatten()[:3]
        return fail_prob, ttf, correction, raw_logit


# ─── VLA Loading ──────────────────────────────────────────────────

def load_vla(device_map="auto"):
    """Load OpenVLA-OFT or base OpenVLA."""
    try:
        from models.openvla_oft_wrapper import OpenVLAOFTWrapper
        ckpt = "moojink/openvla-7b-oft-finetuned-libero-spatial"
        print(f"Loading OpenVLA-OFT: {ckpt}")
        print("This takes ~60 seconds...")
        vla = OpenVLAOFTWrapper(pretrained_checkpoint=ckpt, device_map=device_map)
        print("OpenVLA-OFT loaded.")
        return vla, "OpenVLA-OFT"
    except Exception as e:
        print(f"OFT failed: {e}")
    try:
        from models.vla_wrapper import OpenVLAWrapper
        print("Loading base OpenVLA-7B...")
        vla = OpenVLAWrapper(
            model_name="openvla/openvla-7b",
            device_map=device_map,
            enable_hidden_state_hooks=True,
        )
        print("OpenVLA loaded.")
        return vla, "OpenVLA-7B"
    except Exception as e:
        print(f"Base OpenVLA failed: {e}")
    return None, None


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PULSE Live Demo — LIBERO + OpenVLA + Real Probe")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--mode", default="steering", choices=["vanilla", "steering", "mppi", "latent_stop"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    # Steering params (best config from calibration)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ema-beta", type=float, default=0.3)
    parser.add_argument("--max-correction", type=float, default=0.0035)
    parser.add_argument("--correction-threshold", type=float, default=0.0025)
    parser.add_argument("--fail-threshold", type=float, default=16.56,
                        help="Threshold on raw logit (not sigmoid). Calibrated: success=16.09, failure=17.03, midpoint=16.56")
    parser.add_argument("--action-scale", type=float, default=0.05)
    args = parser.parse_args()

    # ── Load safety probe ──
    probe_ckpt = REPO_ROOT / "hpc_mirror" / "checkpoints" / "eef_correction_mlp" / "best_model.pt"
    if not probe_ckpt.exists():
        print(f"ERROR: No probe checkpoint at {probe_ckpt}")
        sys.exit(1)
    probe = ProbeRuntime(probe_ckpt, device="cuda:0")
    print(f"Safety probe loaded: {probe.input_dim}-d input")

    # ── Load VLA ──
    vla, vla_name = load_vla()
    if vla is None:
        print("ERROR: Could not load any VLA model. Cannot run demo.")
        sys.exit(1)

    # ── Setup LIBERO environment ──
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    bddl = task_suite.get_task_bddl_file_path(args.task_id)
    instruction = task_suite.get_task(args.task_id).language
    print(f"\nTask {args.task_id}: {instruction}")
    print(f"Mode: {args.mode} | VLA: {vla_name} | Episodes: {args.episodes}")
    print(f"Loading LIBERO environment with GLFW viewer...")

    env = ControlEnv(
        bddl_file_name=bddl,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        render_camera="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview"],
    )

    # ── Steering state ──
    ema_correction = np.zeros(3)
    log = []

    for ep in range(args.episodes):
        obs = env.reset()
        ema_correction = np.zeros(3)
        success = False
        ep_log = {"episode": ep, "mode": args.mode, "steps": []}

        print(f"\n── Episode {ep+1}/{args.episodes} ({args.mode}) ──")

        for step in range(args.steps):
            # Get camera image for VLA
            img_array = obs.get("agentview_image", obs.get("image", None))
            if img_array is None:
                # Try any key with 'image' in it
                for k, v in obs.items():
                    if "image" in k and isinstance(v, np.ndarray):
                        img_array = v
                        break
            if img_array is None:
                print(f"  WARNING: No image in obs (keys: {list(obs.keys())})")
                action = np.zeros(7)
            else:
                pil_img = Image.fromarray(img_array.astype(np.uint8))

                # ── VLA forward pass → action + hidden state ──
                try:
                    action, hidden_state = vla.get_action_with_features(pil_img, instruction)
                    if isinstance(action, torch.Tensor):
                        action = action.detach().float().cpu().numpy()
                    action = np.asarray(action, dtype=np.float32)
                except Exception as e:
                    if step == 0:
                        print(f"  VLA error: {e}")
                    action = np.zeros(7)
                    hidden_state = None

                # ── Safety probe ──
                if hidden_state is not None:
                    fail_prob, ttf, raw_correction, raw_logit = probe.predict(hidden_state)
                else:
                    fail_prob, ttf, raw_correction, raw_logit = 0.0, 0.0, np.zeros(3), 0.0

                # EMA smoothing
                ema_correction = args.ema_beta * ema_correction + (1 - args.ema_beta) * raw_correction

                # Clamp
                mag = np.linalg.norm(ema_correction)
                if mag > args.max_correction:
                    ema_correction *= args.max_correction / mag
                correction = ema_correction.copy()

                # Double gate
                # Gate on raw logit, not sigmoid (logits are in range 15-18, sigmoid saturates at 1.0)
                gate_fired = (np.linalg.norm(correction) > args.correction_threshold) and (raw_logit > args.fail_threshold)

                # Apply based on mode
                if args.mode == "steering" and gate_fired:
                    action[:3] = action[:3] + args.alpha * correction / args.action_scale
                elif args.mode == "mppi" and gate_fired:
                    samples = correction[None,:] + np.random.randn(64, 3) * 0.003
                    best = samples[np.argmin(np.abs(samples).sum(1))]
                    action[:3] = action[:3] + args.alpha * best / args.action_scale
                elif args.mode == "latent_stop" and raw_logit > args.fail_threshold:
                    action[:3] = 0.0

                # Log
                step_info = {
                    "step": step,
                    "fail_prob": float(fail_prob),
                    "ttf": float(ttf),
                    "correction_mag": float(np.linalg.norm(correction)),
                    "gate_fired": bool(gate_fired),
                }
                ep_log["steps"].append(step_info)

                # Print status every 30 steps
                if step % 30 == 0:
                    gate_str = "INTERVENING" if gate_fired else "safe"
                    print(f"  step {step:3d}: risk={fail_prob:.2f} ttf={ttf:.1f} |c|={np.linalg.norm(correction):.4f} [{gate_str}]")

            # Step environment
            obs, reward, done, info = env.step(action)

            # Render GUI (ControlEnv renders automatically when has_renderer=True)
            if hasattr(env, 'render'):
                try:
                    env.render()
                except Exception:
                    pass

            if done:
                success = bool(info.get("success", reward > 0))
                break

        status = "SUCCESS" if success else "FAILED"
        print(f"  → {status} at step {step}")
        ep_log["success"] = success
        ep_log["final_step"] = step
        log.append(ep_log)

    # Summary
    n_success = sum(1 for e in log if e["success"])
    print(f"\n{'='*50}")
    print(f"RESULTS: {n_success}/{len(log)} successes ({100*n_success/len(log):.0f}%)")
    print(f"Mode: {args.mode} | VLA: {vla_name} | Task: {args.task_id}")

    # Count interventions
    total_steps = sum(len(e["steps"]) for e in log)
    total_gates = sum(1 for e in log for s in e["steps"] if s["gate_fired"])
    if total_steps > 0:
        print(f"Intervention rate: {total_gates}/{total_steps} ({100*total_gates/total_steps:.1f}%)")
        avg_risk = np.mean([s["fail_prob"] for e in log for s in e["steps"]])
        print(f"Mean risk: {avg_risk:.3f}")
    print(f"{'='*50}")

    # Save log
    out_path = REPO_ROOT / "results" / f"demo_{args.mode}_task{args.task_id}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2, default=float)
    print(f"Log saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
