#!/usr/bin/env python3
"""
PULSE Live Demo — xArm6 pick-and-place with OpenVLA + safety probe overlay
===========================================================================
Runs the ACTUAL OpenVLA-7B (sharded across GPUs) in a ManiSkill xArm6 sim
with the real PULSE safety probe providing live risk overlay.

Usage:
  python3 scripts/demo_pulse_live.py
  python3 scripts/demo_pulse_live.py --mode vanilla --steps 2000
  python3 scripts/demo_pulse_live.py --no-vla   # fallback heuristic controller

Keys: q=quit, r=reset, SPACE=pause, 1=vanilla, 2=steering, 3=mppi, 4=stop
"""

from __future__ import annotations
import argparse, sys, time, os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
_MS = REPO_ROOT.parent / "FailSafe_code" / "ManiSkill"
if _MS.is_dir():
    sys.path.insert(0, str(_MS))

import gymnasium as gym
import mani_skill.envs
import pygame


# ─── Safety Probe ─────────────────────────────────────────────────

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


def load_probe(ckpt_path, device="cuda:0"):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    dim = int(ckpt["input_dim"])
    model = EEFCorrectionMLP(input_dim=dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    scaler_mean = np.asarray(ckpt.get("scaler_mean", np.zeros(dim)), dtype=np.float32)
    scaler_scale = np.asarray(ckpt.get("scaler_scale", np.ones(dim)), dtype=np.float32)
    return model, scaler_mean, scaler_scale, dim


def physics_to_latent(env_unwrapped, dim=4096):
    """Surrogate latent from physics state (used when VLA is not loaded)."""
    tcp = np.asarray(env_unwrapped.agent.tcp.pose.p).flatten()[:3]
    cube_p = np.asarray(env_unwrapped.cube.pose.p).flatten()[:3]
    goal_p = np.asarray(env_unwrapped.goal_site.pose.p).flatten()[:3]
    vec = np.concatenate([tcp, cube_p, goal_p, tcp-cube_p, cube_p-goal_p, tcp-goal_p])
    return np.tile(vec, int(np.ceil(dim/vec.size)))[:dim].astype(np.float32)


# ─── VLA Loading ──────────────────────────────────────────────────

def load_vla():
    """Load OpenVLA-7B sharded across all GPUs. Returns wrapper or None."""
    # Try OFT first (better actions), then base OpenVLA
    try:
        from models.openvla_oft_wrapper import OpenVLAOFTWrapper
        ckpt = "moojink/openvla-7b-oft-finetuned-libero-spatial"
        print(f"Loading OpenVLA-OFT ({ckpt})... ~60s")
        vla = OpenVLAOFTWrapper(pretrained_checkpoint=ckpt, device_map="auto")
        print("OpenVLA-OFT loaded.")
        return vla
    except Exception as e1:
        print(f"OFT failed: {e1}")
        try:
            from models.vla_wrapper import OpenVLAWrapper
            model_name = "openvla/openvla-7b"
            print(f"Loading base OpenVLA ({model_name})...")
            vla = OpenVLAWrapper(
                model_name=model_name,
                device_map="auto",
                enable_hidden_state_hooks=True,
            )
            print("OpenVLA loaded.")
            return vla
        except Exception as e2:
            print(f"Base OpenVLA also failed: {e2}")
            print("Falling back to heuristic controller")
            return None


class HeuristicController:
    """Reach-grasp-lift-place state machine."""
    def __init__(self):
        self.phase = "approach"  # approach → descend → grasp → lift → place
        self.grasp_timer = 0

    def reset(self):
        self.phase = "approach"
        self.grasp_timer = 0

    def step(self, env_unwrapped):
        tcp = np.asarray(env_unwrapped.agent.tcp.pose.p).flatten()[:3]
        cube = np.asarray(env_unwrapped.cube.pose.p).flatten()[:3]
        goal = np.asarray(env_unwrapped.goal_site.pose.p).flatten()[:3]

        above_cube = cube.copy(); above_cube[2] += 0.08
        above_goal = goal.copy(); above_goal[2] += 0.08
        dist_xy = np.linalg.norm(tcp[:2] - cube[:2])
        dist_z = tcp[2] - cube[2]

        if self.phase == "approach":
            xyz = (above_cube - tcp) * 8.0
            gripper = 1.0  # open
            if np.linalg.norm(tcp - above_cube) < 0.015:
                self.phase = "descend"
        elif self.phase == "descend":
            target = cube.copy(); target[2] += 0.01
            xyz = (target - tcp) * 8.0
            gripper = 1.0
            if dist_z < 0.025 and dist_xy < 0.02:
                self.phase = "grasp"
                self.grasp_timer = 0
        elif self.phase == "grasp":
            xyz = np.zeros(3)
            gripper = -1.0  # close
            self.grasp_timer += 1
            if self.grasp_timer > 10:
                self.phase = "lift"
        elif self.phase == "lift":
            xyz = np.array([0, 0, 0.5])
            gripper = -1.0
            if tcp[2] > cube[2] + 0.08:
                self.phase = "place"
        elif self.phase == "place":
            xyz = (above_goal - tcp) * 5.0
            gripper = -1.0
            if np.linalg.norm(tcp[:2] - goal[:2]) < 0.03:
                gripper = 1.0  # release

        return np.clip(xyz, -1, 1).astype(np.float32), np.array([gripper])


# ─── Drawing ──────────────────────────────────────────────────────

def draw_overlay(surface, step, fail_prob, ttf, correction, gate_fired, mode,
                 fps, episode, successes, using_vla):
    w, h = surface.get_size()
    font = pygame.font.SysFont("monospace", 16, bold=True)
    small = pygame.font.SysFont("monospace", 13)

    # Top bar
    pygame.draw.rect(surface, (20, 20, 20), (0, 0, w, 75))

    policy_str = "OpenVLA-7B" if using_vla else "Heuristic"
    title = font.render(
        f"PULSE | {policy_str} | Mode: {mode.upper()} | Ep: {episode} | OK: {successes}",
        True, (255, 255, 255))
    surface.blit(title, (10, 8))

    info = small.render(f"Step: {step}  FPS: {fps:.0f}  TTF: {ttf:.1f}  Robot: xArm 6", True, (170, 170, 170))
    surface.blit(info, (10, 30))

    # Risk bar
    bar_x, bar_y, bar_w, bar_h = 10, 52, 250, 14
    pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
    color = (0,200,0) if fail_prob < 0.3 else (200,200,0) if fail_prob < 0.6 else (220,0,0)
    pygame.draw.rect(surface, color, (bar_x, bar_y, int(bar_w*min(fail_prob,1)), bar_h))
    surface.blit(small.render(f"Risk: {fail_prob:.2f}", True, (200,200,200)), (bar_x+bar_w+10, bar_y-1))

    # Gate
    if gate_fired:
        surface.blit(font.render("INTERVENING", True, (255,50,50)), (w-180, 8))
    else:
        surface.blit(font.render("SAFE", True, (50,255,50)), (w-80, 8))

    # Border
    pygame.draw.rect(surface, color, (0,0,w,h), 6 if gate_fired else 2)

    # Correction arrow
    cx, cy = 60, h-50
    pygame.draw.circle(surface, (40,40,40), (cx,cy), 30)
    s = 2000
    dx, dy = int(correction[0]*s), int(-correction[1]*s)
    end = (cx+dx, cy+dy)
    pygame.draw.line(surface, (0,255,255), (cx,cy), end, 2)
    if abs(dx)+abs(dy) > 3:
        pygame.draw.circle(surface, (0,255,255), end, 4)
    mag = np.linalg.norm(correction)
    surface.blit(small.render(f"|c|={mag:.4f}m", True, (150,150,150)), (cx+35, cy-8))

    # Legend
    surface.blit(small.render("1=vanilla 2=steering 3=mppi 4=stop  r=reset  q=quit", True, (120,120,120)), (w-420, h-22))


# ─── Main ─────────────────────────────────────────────────────────

def run_demo(args):
    probe_device = "cuda:0"

    # Load probe
    ckpt_path = REPO_ROOT / "hpc_mirror" / "checkpoints" / "eef_correction_mlp" / "best_model.pt"
    if ckpt_path.exists():
        probe, scaler_mean, scaler_scale, latent_dim = load_probe(ckpt_path, probe_device)
        print(f"Safety probe: {latent_dim}-d")
    else:
        print(f"No probe at {ckpt_path}")
        probe = None
        latent_dim = 4096

    # Load VLA
    vla = None
    if not args.no_vla:
        vla = load_vla()
    using_vla = vla is not None

    # Create env
    env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array",
                    robot_uids="xarm6_robotiq", sim_backend="cpu")
    obs, info = env.reset(seed=args.seed)
    print(f"Env: PickCube-v1 with xArm 6 (UFactory)")

    # Pygame
    pygame.init()
    W, H = args.width, args.height
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("PULSE Demo — xArm 6 + OpenVLA Safety Steering")
    clock = pygame.time.Clock()

    # Task instruction for VLA
    instruction = "pick up the red cube and place it at the green target"
    heuristic = HeuristicController()

    mode = args.mode
    step, episode, successes = 0, 1, 0
    paused = False
    ema_correction = np.zeros(3)
    running = True
    fps = 0

    policy_name = "OpenVLA" if using_vla else "heuristic"
    print(f"\nMode: {mode} | Policy: {policy_name}")
    print("Keys: 1-4=mode, r=reset, SPACE=pause, q=quit\n")

    while running and step < args.steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE): running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset(seed=args.seed+step)
                    ema_correction = np.zeros(3); heuristic.reset(); episode += 1
                    print(f"Reset → Ep {episode}")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_1: mode = "vanilla"; print("→ vanilla")
                elif event.key == pygame.K_2: mode = "steering"; print("→ steering")
                elif event.key == pygame.K_3: mode = "mppi"; print("→ mppi")
                elif event.key == pygame.K_4: mode = "latent_stop"; print("→ latent_stop")

        if paused:
            clock.tick(30); continue

        # ── Get action + hidden state ──
        hidden_state = None
        eu = env.unwrapped

        if using_vla:
            # Get camera image from env observation
            sensor_data = obs.get("sensor_data", {})
            cam_keys = [k for k in sensor_data.keys() if "rgb" in k.lower() or "color" in k.lower()]
            if not cam_keys:
                # Use rendered frame as observation
                frame_obs = env.render()
                if isinstance(frame_obs, torch.Tensor):
                    frame_obs = frame_obs[0].cpu().numpy().astype(np.uint8)
            else:
                frame_obs = sensor_data[cam_keys[0]]
                if isinstance(frame_obs, torch.Tensor):
                    frame_obs = frame_obs[0].cpu().numpy().astype(np.uint8)

            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(frame_obs).resize((224, 224))

            try:
                action_raw, hidden_state = vla.get_action_with_features(pil_img, instruction)
                action_xyz = np.asarray(action_raw[:3], dtype=np.float32)
                gripper = np.array([action_raw[6]]) if len(action_raw) > 6 else np.array([0.0])
            except Exception as e:
                print(f"VLA error: {e}, using heuristic")
                action_xyz, gripper = heuristic_controller(eu)
        else:
            action_xyz, gripper = heuristic.step(eu)

        # ── Probe ──
        fail_prob, ttf, correction, gate_fired = 0.0, 0.0, np.zeros(3), False
        if probe is not None:
            if hidden_state is not None:
                latent = np.asarray(hidden_state, dtype=np.float32).flatten()[:latent_dim]
                if latent.size < latent_dim:
                    latent = np.pad(latent, (0, latent_dim - latent.size))
            else:
                latent = physics_to_latent(eu, latent_dim)

            scaled = (latent - scaler_mean) / np.clip(scaler_scale, 1e-8, None)
            x = torch.from_numpy(scaled).unsqueeze(0).float().to(probe_device)
            with torch.no_grad():
                out = probe(x)
            fail_prob = float(torch.sigmoid(out["will_fail"]).item())
            ttf = float(out["ttf"].item())
            raw = out["correction"].cpu().numpy().flatten()[:3]
            ema_correction = 0.3 * ema_correction + 0.7 * raw
            mag = np.linalg.norm(ema_correction)
            if mag > args.max_correction:
                ema_correction *= args.max_correction / mag
            correction = ema_correction.copy()
            gate_fired = mag > args.correction_threshold and fail_prob > args.fail_threshold

            if mode == "steering" and gate_fired:
                action_xyz = action_xyz + args.alpha * correction
            elif mode == "mppi" and gate_fired:
                samples = correction[None,:] + np.random.randn(16, 3) * 0.003
                action_xyz = action_xyz + args.alpha * samples.mean(0)
            elif mode == "latent_stop" and fail_prob > args.fail_threshold:
                action_xyz = np.zeros(3)

        full_action = np.concatenate([np.clip(action_xyz, -1, 1), np.zeros(3), gripper]).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(full_action)

        # ── Render ──
        frame = env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame[0].cpu().numpy()
        frame = np.ascontiguousarray(frame.astype(np.uint8))
        import cv2
        frame = cv2.resize(frame, (W, H))
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        fps = clock.get_fps()
        draw_overlay(screen, step, fail_prob, ttf, correction, gate_fired, mode,
                     fps, episode, successes, using_vla)
        pygame.display.flip()
        step += 1
        clock.tick(args.fps)

        if terminated or truncated:
            success = info.get("success", False)
            if isinstance(success, torch.Tensor): success = bool(success.item())
            if success: successes += 1
            print(f"  Ep {episode}: {'OK' if success else 'FAIL'} (step {step})")
            obs, info = env.reset(seed=args.seed+step)
            ema_correction = np.zeros(3); heuristic.reset(); episode += 1

    env.close(); pygame.quit()
    print(f"\nDone. {episode-1} eps, {successes} successes.")


def main():
    parser = argparse.ArgumentParser(description="PULSE Live Demo — xArm 6 + OpenVLA")
    parser.add_argument("--mode", default="steering", choices=["vanilla","steering","mppi","latent_stop"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--fail-threshold", type=float, default=0.6)
    parser.add_argument("--correction-threshold", type=float, default=0.003)
    parser.add_argument("--max-correction", type=float, default=0.004)
    parser.add_argument("--no-vla", action="store_true", help="Skip VLA loading, use heuristic controller")
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
