#!/usr/bin/env python3
"""Sim ↔ real demo: **UFactory xArm6** pick-and-place + your **failure probe + MPPI** stack.

**Simulation:** ManiSkill ``PickCube-v1`` with ``robot_uids=xarm6_robotiq`` (UFactory xArm6 + Robotiq —
the closest match in-repo; Kinova Gen3 Lite is not shipped as a ManiSkill asset, but deployment uses the
same probe code paths via ``arm_server_kinova.py`` / ``arm_server_xarm.py`` + ``run_model_yahboom_loop.py``).

**Probe:** Loads ``hpc_mirror/checkpoints/eef_correction_mlp/best_model.pt`` (4096-d OpenVLA latent probe).
Because we are not running OpenVLA here, we build a **4096-d surrogate** by tiling physics state
(TCP, cube, goal, vectors). Outputs are **illustrative** (distribution shift vs LIBERO training) but the
**same PyTorch graph, scaler, gating, EMA, and MPPI** as paper/HW code run for real.

**Modes:** ``vanilla`` | ``steering`` | ``mppi`` | ``latent_stop`` — same structure as ``eval_tuning.py`` /
``eval_act_steering.py``.

Dependencies (sim):
  pip install mani_skill gymnasium
  # ManiSkill needs SAPIEN; follow https://github.com/haosulab/ManiSkill

Usage:
  cd mist-vla
  python3 scripts/demo_sim_vs_realworld.py --output figures/demo_sim_vs_realworld.mp4
  python3 scripts/demo_sim_vs_realworld.py --mode mppi --steps 160 --fps 12
  python3 scripts/demo_sim_vs_realworld.py --gui   # SAPIEN viewer (needs DISPLAY)

Also writes a **raw sim RGB still** (ManiSkill camera only), default ``<output_stem>_sim.png``,
for slides or the paper (use ``--no-sim-image`` to skip).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Prefer vendored ManiSkill (assets) if present
_MANISKILL_SRC = REPO_ROOT.parent / "FailSafe_code" / "ManiSkill"
if _MANISKILL_SRC.is_dir():
    sys.path.insert(0, str(_MANISKILL_SRC))


def _patch_torch_load():
    try:
        import torch

        try:
            import numpy._core.multiarray as _ma

            torch.serialization.add_safe_globals(
                [_ma._reconstruct, np.ndarray, np.dtype]
            )
        except Exception:
            try:
                from numpy.core import multiarray as _ma2

                torch.serialization.add_safe_globals(
                    [_ma2._reconstruct, np.ndarray, np.dtype]
                )
            except Exception:
                pass
        _o = torch.load

        def _w(*a, **k):
            if "weights_only" not in k:
                k["weights_only"] = False
            return _o(*a, **k)

        torch.load = _w
    except Exception:
        pass


# ── Model (must match train_eef_correction_mlp.EEFCorrectionMLP) ─────────────
import torch
import torch.nn as nn


class EEFCorrectionMLP(nn.Module):
    HIDDEN_DIM = 256

    def __init__(self, input_dim: int = 4096):
        super().__init__()
        h = self.HIDDEN_DIM
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(h // 2, h // 4),
            nn.LayerNorm(h // 4),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        feat = h // 4
        self.fail_head = nn.Linear(feat, 1)
        self.ttf_head = nn.Linear(feat, 1)
        self.correction_head = nn.Linear(feat, 3)

    def forward(self, x: torch.Tensor):
        x = self.input_norm(x)
        feat = self.encoder(x)
        return {
            "will_fail": self.fail_head(feat).squeeze(-1),
            "ttf": self.ttf_head(feat).squeeze(-1),
            "correction": self.correction_head(feat),
        }


def load_probe(ckpt_path: Path, device: str = "cpu"):
    from sklearn.preprocessing import StandardScaler

    _patch_torch_load()
    ckpt = torch.load(str(ckpt_path), map_location=device)
    dim = int(ckpt["input_dim"])
    m = EEFCorrectionMLP(input_dim=dim).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    sc = StandardScaler()
    sc.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    sc.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    sc.var_ = sc.scale_**2
    sc.n_features_in_ = dim
    sc.n_samples_seen_ = 1
    return m, sc, dim


def tile_to_dim(v: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size == 0:
        return np.zeros(dim, dtype=np.float64)
    reps = int(np.ceil(dim / v.size))
    return np.resize(np.tile(v, reps), (dim,))


def physics_surrogate_latent(env_unwrapped, dim: int = 4096) -> np.ndarray:
    """TCP / cube / goal geometry → tiled pseudo-latent (NOT OpenVLA hidden states)."""
    tcp = env_unwrapped.agent.tcp.pose.p
    cube_p = env_unwrapped.cube.pose.p
    goal_p = env_unwrapped.goal_site.pose.p
    if hasattr(tcp, "cpu"):
        tcp = tcp[0].detach().cpu().numpy()
        cube_p = cube_p[0].detach().cpu().numpy()
        goal_p = goal_p[0].detach().cpu().numpy()
    else:
        tcp, cube_p, goal_p = map(np.asarray, (tcp, cube_p, goal_p))
    d1 = tcp - cube_p
    d2 = cube_p - goal_p
    # Keep all pieces 1-D so concatenate is stable across NumPy versions.
    raw = np.concatenate(
        [tcp, cube_p, goal_p, d1, d2, np.array([np.linalg.norm(d1), np.linalg.norm(d2)])]
    )
    return tile_to_dim(raw, dim)


# ── Controllers (same logic as eval_tuning / eval_act_steering) ───────────────
class SteeredAgent:
    def __init__(
        self,
        mlp,
        scaler,
        *,
        alpha=1.0,
        ema_beta=0.7,
        action_scale=0.05,
        correction_threshold=0.005,
        max_correction=0.02,
        use_fail_gate=True,
        fail_threshold=0.5,
        device="cpu",
    ):
        self.mlp = mlp
        self.scaler = scaler
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.action_scale = action_scale
        self.correction_threshold = correction_threshold
        self.max_correction = max_correction
        self.use_fail_gate = use_fail_gate
        self.fail_threshold = fail_threshold
        self.device = device
        self.prev_correction = None

    def reset(self):
        self.prev_correction = None

    def apply(self, action: np.ndarray, features: np.ndarray):
        if features is None or np.prod(features.shape) < 2:
            return action, False, 0.0
        scaled = self.scaler.transform(features.reshape(1, -1))
        x = torch.FloatTensor(scaled).to(self.device)
        with torch.no_grad():
            out = self.mlp(x)
        fail_prob = torch.sigmoid(out["will_fail"]).item()
        raw = out["correction"].cpu().numpy()[0]
        if self.prev_correction is not None:
            smoothed = self.ema_beta * self.prev_correction + (1.0 - self.ema_beta) * raw
        else:
            smoothed = raw.copy()
        self.prev_correction = smoothed.copy()
        mag = float(np.linalg.norm(smoothed))
        if mag > self.max_correction and mag > 1e-8:
            smoothed = smoothed * (self.max_correction / mag)
            mag = self.max_correction
        intervene = mag > self.correction_threshold
        if self.use_fail_gate:
            intervene = intervene and (fail_prob >= self.fail_threshold)
        if intervene:
            action = action.copy()
            action[:3] = action[:3] + (self.alpha * smoothed / self.action_scale)
            return action, True, fail_prob
        return action, False, fail_prob


class MPPIController:
    def __init__(
        self,
        mlp,
        scaler,
        *,
        n_samples=16,
        temperature=5.0,
        correction_std=0.005,
        max_correction=0.02,
        action_scale=0.05,
        device="cpu",
    ):
        self.mlp = mlp
        self.scaler = scaler
        self.n_samples = n_samples
        self.temperature = temperature
        self.correction_std = correction_std
        self.max_correction = max_correction
        self.action_scale = action_scale
        self.device = device

    def reset(self):
        pass

    def apply(self, action: np.ndarray, features: np.ndarray):
        if features is None or np.prod(features.shape) < 2:
            return action, False, 0.0
        scaled = self.scaler.transform(features.reshape(1, -1))
        x = torch.FloatTensor(scaled).to(self.device)
        with torch.no_grad():
            out = self.mlp(x)
        fail_prob = torch.sigmoid(out["will_fail"]).item()
        if fail_prob < 0.5:
            return action, False, fail_prob
        candidates = np.random.normal(0, self.correction_std, size=(self.n_samples, 3)).astype(
            np.float32
        )
        scores = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            feat_p = scaled.copy() + np.random.normal(0, 0.01, scaled.shape)
            xp = torch.FloatTensor(feat_p).to(self.device)
            with torch.no_grad():
                op = self.mlp(xp)
            scores[i] = -torch.sigmoid(op["will_fail"]).item()
        w = np.exp(self.temperature * (scores - scores.max()))
        w /= w.sum()
        correction = (candidates * w[:, None]).sum(axis=0)
        mag = float(np.linalg.norm(correction))
        if mag > self.max_correction and mag > 1e-8:
            correction = correction * (self.max_correction / mag)
        action = action.copy()
        action[:3] = action[:3] + correction / self.action_scale
        return action, True, fail_prob


class LatentStopAgent:
    def __init__(self, mlp, scaler, *, stop_threshold=0.85, device="cpu"):
        self.mlp = mlp
        self.scaler = scaler
        self.stop_threshold = stop_threshold
        self.device = device

    def reset(self):
        pass

    def apply(self, action: np.ndarray, features: np.ndarray):
        if features is None:
            return action, False, 0.0
        scaled = self.scaler.transform(features.reshape(1, -1))
        x = torch.FloatTensor(scaled).to(self.device)
        with torch.no_grad():
            out = self.mlp(x)
        fail_prob = torch.sigmoid(out["will_fail"]).item()
        if fail_prob >= self.stop_threshold:
            action = action.copy()
            action[:] = 0.0
            return action, True, fail_prob
        return action, False, fail_prob


def _to_np_img(obs) -> np.ndarray:
    """Extract RGB uint8 HxWx3 from ManiSkill observation dict."""
    if not isinstance(obs, dict):
        raise TypeError(type(obs))

    def _tensor_to_rgb(t) -> np.ndarray:
        if hasattr(t, "cpu"):
            arr = t[0].detach().cpu().numpy() if t.ndim >= 3 else t.detach().cpu().numpy()
        else:
            arr = np.asarray(t)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr[..., :3]
        raise ValueError(arr.shape)

    if "sensor_data" in obs:
        for _name, cam in obs["sensor_data"].items():
            if isinstance(cam, dict) and "rgb" in cam:
                return _tensor_to_rgb(cam["rgb"])
    for k in ("image", "rgb", "base_camera"):
        if k not in obs:
            continue
        v = obs[k]
        if isinstance(v, dict) and "rgb" in v:
            return _tensor_to_rgb(v["rgb"])
        return _tensor_to_rgb(v)
    raise KeyError("no rgb in obs keys=" + str(list(obs.keys())[:12]))


def _resize_hw3(img: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((w, h), Image.Resampling.BILINEAR))


def _hud_panel(
    h: int,
    w: int,
    mode: str,
    fail_prob: float,
    intervened: bool,
    step: int,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), (18, 22, 30))
    dr = ImageDraw.Draw(im)
    dr.text((8, 6), "PULSE stack (same code as HW eval)", fill=(140, 200, 255))
    lines = [
        f"mode: {mode}",
        f"fail_prob: {fail_prob:.3f}",
        f"intervene: {intervened}",
        f"step: {step}",
        "",
        "Sim: UFactory xArm6 + pick cube",
        "HW: xArm or Kinova + Jetson API",
        "Probe: eef_correction_mlp (4096)",
        "Surrogate latent = tiled state",
    ]
    y = 28
    for line in lines:
        dr.text((10, y), line, fill=(210, 215, 225))
        y += 16
    return np.asarray(im)


def _bridge_real_hw(h: int, w: int, highlight: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    steps = [
        "① Wrist RGB → GET /snapshot",
        "② Policy (OpenVLA/ACT)",
        "③ Probe → fail, Δp, MPPI opt",
        "④ Gate + EMA + clamp",
        "⑤ POST /action (xArm or Kinova)",
    ]
    im = Image.new("RGB", (w, h), (16, 18, 26))
    dr = ImageDraw.Draw(im)
    dr.text((8, 4), "Real lab (same software roles)", fill=(255, 180, 120))
    lh = (h - 36) // max(len(steps), 1)
    for i, s in enumerate(steps):
        y = 28 + i * lh
        if i == highlight % len(steps):
            dr.rounded_rectangle([4, y - 2, w - 6, y + lh - 6], radius=5, fill=(45, 55, 72))
            fill = (240, 248, 255)
        else:
            fill = (160, 168, 182)
        dr.text((12, y), s, fill=fill)
    return np.asarray(im)


def _compose_triptych(sim_rgb, hud, bridge, gap: int = 6) -> np.ndarray:
    from PIL import Image

    h = min(sim_rgb.shape[0], hud.shape[0], bridge.shape[0])
    sim_rgb = _resize_hw3(sim_rgb, h, sim_rgb.shape[1] * h // max(sim_rgb.shape[0], 1))
    hud = _resize_hw3(hud, h, hud.shape[1] * h // max(hud.shape[0], 1))
    bridge = _resize_hw3(bridge, h, bridge.shape[1] * h // max(bridge.shape[0], 1))
    w = sim_rgb.shape[1] + gap + hud.shape[1] + gap + bridge.shape[1]
    if w % 2:
        w += 1
    canvas = Image.new("RGB", (w, h + 32), (10, 12, 16))
    from PIL import ImageDraw

    ImageDraw.Draw(canvas).text((6, 4), "SIM (ManiSkill)  |  METRICS  |  DEPLOY (Jetson + arm)", fill=(200, 200, 210))
    canvas.paste(Image.fromarray(sim_rgb), (0, 28))
    canvas.paste(Image.fromarray(hud), (sim_rgb.shape[1] + gap, 28))
    canvas.paste(Image.fromarray(bridge), (sim_rgb.shape[1] + gap + hud.shape[1] + gap, 28))
    arr = np.asarray(canvas)
    if arr.shape[1] % 2:
        arr = np.pad(arr, ((0, 0), (0, 1), (0, 0)), mode="edge")
    if arr.shape[0] % 2:
        arr = np.pad(arr, ((0, 1), (0, 0), (0, 0)), mode="edge")
    return arr


def _write_video(path: Path, frames: list, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)
        return
    except Exception as e1:
        try:
            import imageio.v2 as imageio

            gif = path.with_suffix(".gif")
            imageio.mimsave(str(gif), frames, fps=fps)
            print(f"MP4 failed ({e1}); wrote {gif}", file=sys.stderr)
        except Exception as e2:
            from PIL import Image

            d = path.parent / (path.stem + "_frames")
            d.mkdir(parents=True, exist_ok=True)
            for i, fr in enumerate(frames):
                Image.fromarray(fr).save(d / f"frame_{i:04d}.png")
            print(f"Video failed ({e1}; {e2}); PNGs in {d}", file=sys.stderr)


def naive_policy_ee_delta(
    env,
    obs,
    *,
    move_gain: float = 0.45,
    gripper: float = 0.4,
    full_swing: bool = False,
) -> np.ndarray:
    """EE delta toward cube. ``full_swing`` uses gain 1.0 + random rot/grip (no 0.45 damp)."""
    u = env.unwrapped
    tcp = u.agent.tcp.pose.p[0].detach().cpu().numpy()
    cube = u.cube.pose.p[0].detach().cpu().numpy()
    d = cube - tcp
    n = np.linalg.norm(d) + 1e-6
    d = d / n
    g = 1.0 if full_swing else move_gain
    tmpl = env.action_space.sample()
    a = np.asarray(tmpl, dtype=np.float32).reshape(-1)
    if a.size < 3:
        raise RuntimeError(f"action_space too small: {a.shape}")
    a[:] = 0.0
    a[:3] = np.clip((d * g).astype(np.float32), -1.0, 1.0)
    if a.size > 3:
        if full_swing:
            a[3 : min(6, a.size)] = np.random.uniform(-1.0, 1.0, size=min(6, a.size) - 3).astype(
                np.float32
            )
        else:
            a[3 : min(6, a.size)] = 0.0
    if a.size > 6:
        a[6] = float(np.random.uniform(-1.0, 1.0)) if full_swing else gripper
    return a.reshape(np.asarray(tmpl).shape)


def run_maniskill_demo(args) -> None:
    import gymnasium as gym

    try:
        import mani_skill.envs  # noqa: F401 — register envs
    except ImportError as e:
        vendored = REPO_ROOT.parent / "FailSafe_code" / "ManiSkill"
        print(
            "ManiSkill failed to import (this script prepends the vendored tree if present).\n"
            f"  Vendored path: {vendored}\n"
            "  Typical missing pieces: dacite, sapien, pytorch_kinematics (package name is "
            "`pytorch_kinematics`, not pytorch_kinematics_ms).\n"
            "  Try: pip install -e "
            f'"{vendored}"\n'
            "  (Linux wheels for sapien / pins like mplib may require following:\n"
            "   https://github.com/haosulab/ManiSkill/blob/main/docs/source/user_guide/getting_started/installation.md )\n"
            "  For the MuJoCo LIBERO Franka viewer (different stack), use:\n"
            "    ./scripts/local_run_from_hpc.sh libero-gui\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # "human" opens the SAPIEN viewer (needs DISPLAY); "rgb_array" is headless-friendly.
    _rm = "human" if getattr(args, "gui", False) else "rgb_array"
    env = gym.make(
        "PickCube-v1",
        robot_uids="xarm6_robotiq",
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode=_rm,
        sim_backend=getattr(args, "sim_backend", "auto"),
    )
    obs, _ = env.reset(seed=args.seed)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        ckpt = REPO_ROOT / "hpc_mirror" / "checkpoints" / "eef_correction_mlp" / "best_model.pt"
    device = "cuda" if args.device == "cuda" else "cpu"
    mlp, scaler, _dim = load_probe(ckpt, device=device)

    mode = args.mode
    steer = SteeredAgent(
        mlp,
        scaler,
        action_scale=args.action_scale,
        max_correction=args.max_correction,
        use_fail_gate=not args.no_fail_gate,
        fail_threshold=args.fail_threshold,
        device=device,
    )
    mppi = MPPIController(
        mlp,
        scaler,
        n_samples=args.mppi_samples,
        temperature=args.mppi_temperature,
        correction_std=args.mppi_correction_std,
        max_correction=args.max_correction,
        action_scale=args.action_scale,
        device=device,
    )
    stop = LatentStopAgent(mlp, scaler, stop_threshold=args.stop_threshold, device=device)
    steer.reset()
    mppi.reset()
    stop.reset()

    out_video = Path(args.output)
    _sim_arg = (getattr(args, "sim_image", None) or "").strip()
    sim_png = Path(_sim_arg) if _sim_arg else out_video.with_name(out_video.stem + "_sim.png")
    save_sim_still = not getattr(args, "no_sim_image", False)
    sim_snap_step = max(0, min(args.steps // 2, args.steps - 1)) if args.steps > 0 else 0
    sim_still_saved = False
    last_rgb: np.ndarray | None = None

    frames = []
    fail_prob = 0.0
    for step in range(args.steps):
        feat = physics_surrogate_latent(env.unwrapped, _dim)
        a = naive_policy_ee_delta(env, obs, full_swing=getattr(args, "no_cap", False))
        intervened = False

        if mode == "steering":
            a, intervened, fail_prob = steer.apply(np.asarray(a).reshape(-1), feat)
            a = a.reshape(env.action_space.shape)
        elif mode == "mppi":
            a, intervened, fail_prob = mppi.apply(np.asarray(a).reshape(-1), feat)
            a = a.reshape(env.action_space.shape)
        elif mode == "latent_stop":
            a, intervened, fail_prob = stop.apply(np.asarray(a).reshape(-1), feat)
            a = a.reshape(env.action_space.shape)
        else:
            with torch.no_grad():
                sc = scaler.transform(feat.reshape(1, -1))
                x = torch.FloatTensor(sc).to(device)
                fail_prob = torch.sigmoid(mlp(x)["will_fail"]).item()

        obs, _r, term, trunc, info = env.step(a)

        def _as_bool(x):
            if hasattr(x, "cpu"):
                return bool(x.any().item())
            return bool(x)

        try:
            rgb = _to_np_img(obs)
        except Exception:
            rgb = np.asarray(env.render())
        last_rgb = rgb

        if save_sim_still and step == sim_snap_step:
            from PIL import Image

            sim_png.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(sim_png, optimize=True)
            sim_still_saved = True

        hud = _hud_panel(200, 280, mode, fail_prob, intervened, step)
        bridge = _bridge_real_hw(200, 300, step)
        frames.append(_compose_triptych(rgb, hud, bridge))

        if _as_bool(term) or _as_bool(trunc):
            obs, _ = env.reset(seed=args.seed + step + 1)

    env.close()
    out = Path(args.output)
    if save_sim_still and not sim_still_saved and last_rgb is not None:
        from PIL import Image

        sim_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(last_rgb).save(sim_png, optimize=True)
        sim_still_saved = True

    _write_video(out, frames, args.fps)
    if out.exists():
        print(f"Wrote: {out.resolve()}")
    if save_sim_still and sim_still_saved:
        print(f"Sim still (RGB): {sim_png.resolve()}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=str, default=str(REPO_ROOT / "figures" / "demo_sim_vs_realworld.mp4"))
    p.add_argument("--checkpoint", type=str, default="", help="Path to eef_correction_mlp best_model.pt")
    p.add_argument("--mode", type=str, default="steering", choices=("vanilla", "steering", "mppi", "latent_stop"))
    p.add_argument("--steps", type=int, default=140)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--sim-backend", type=str, default="auto")
    p.add_argument("--action-scale", type=float, default=0.05)
    p.add_argument("--max-correction", type=float, default=0.02)
    p.add_argument("--fail-threshold", type=float, default=0.5)
    p.add_argument("--stop-threshold", type=float, default=0.85)
    p.add_argument("--no-fail-gate", action="store_true")
    p.add_argument("--mppi-samples", type=int, default=16)
    p.add_argument("--mppi-temperature", type=float, default=5.0)
    p.add_argument("--mppi-correction-std", type=float, default=0.005)
    p.add_argument(
        "--sim-image",
        type=str,
        default="",
        help="Path for raw sim RGB PNG (default: <output_stem>_sim.png next to the video)",
    )
    p.add_argument(
        "--no-sim-image",
        action="store_true",
        help="Do not save a separate sim-only PNG",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Open ManiSkill interactive viewer (requires DISPLAY; uses render_mode=human)",
    )
    p.add_argument(
        "--no-cap",
        action="store_true",
        help="No dampening: full ±1 translation toward cube + random rot/grip (ManiSkill demo)",
    )
    args = p.parse_args()

    run_maniskill_demo(args)


if __name__ == "__main__":
    main()
