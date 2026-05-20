#!/usr/bin/env python3
"""Collect real-robot rollouts on the SO-101 with hidden-state logging.

Two modes:
  --mode teleop    Leader → follower; you teleoperate while a Pi0 forward pass runs
                   alongside so we capture hidden states *as if* the policy were driving.
                   Use this to collect clean "expert successes" + occasional teleop-failures.
  --mode policy    Pi0 drives the follower autonomously. Use this once you have a few
                   teleop demonstrations or are evaluating policy generalization.

The output pkl matches the sim schema, so all downstream training/eval scripts work:
    research_data/rollouts/so101/<run_tag>/{success_rollouts.pkl, failure_rollouts.pkl}

Workflow:
  1.  python scripts/so101/calibrate.py            (one-time per port assignment)
  2.  python scripts/so101/collect_rollouts.py --mode teleop --n-episodes 30
  3.  python scripts/so101/label_rollouts.py --run-dir research_data/rollouts/so101/<run>
  4.  python scripts/so101/train_probe.py --run-dir ...
  5.  python scripts/so101/eval_realtime.py --probe ...
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.so101.common import (
    HiddenStateHook,
    RolloutRecord,
    SO101RunConfig,
    estop_handler,
)


# ─── LeRobot setup helpers ──────────────────────────────────────────────────


def make_follower(cfg: SO101RunConfig):
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cams = {
        # Use 640x480 MJPG for both. The USB 2.0 Camera (scene) is capped at
        # 5 fps with the default YUYV codec; MJPG unlocks 30 fps at 640x480.
        # SmolVLA's preprocessor resizes internally to the policy's expected
        # size, so there's no benefit to matching its target resolution at capture.
        "wrist": OpenCVCameraConfig(index_or_path=cfg.wrist_cam_index, width=640, height=480, fps=cfg.fps, fourcc="MJPG"),
        "scene": OpenCVCameraConfig(index_or_path=cfg.scene_cam_index, width=640, height=480, fps=cfg.fps, fourcc="MJPG"),
    }
    fc = SO101FollowerConfig(
        port=cfg.follower_port,
        id=cfg.follower_id,
        calibration_dir=cfg.calibration_dir,
        disable_torque_on_disconnect=True,
        max_relative_target=cfg.max_relative_target_deg,
        cameras=cams,
        use_degrees=cfg.use_degrees,
    )
    return SO101Follower(fc)


def make_leader(cfg: SO101RunConfig):
    from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig

    lc = SO101LeaderConfig(
        port=cfg.leader_port,
        id=cfg.leader_id,
        calibration_dir=cfg.calibration_dir,
        use_degrees=cfg.use_degrees,
    )
    return SO101Leader(lc)


def make_policy(cfg: SO101RunConfig):
    """Load policy onto the requested device.

    Returns (policy, hook, pre_processor, post_processor).

    SmolVLA was trained with input/output normalisation against an SO-100 dataset.
    Building the pipeline with `dataset_stats={}` (as we did originally) produced
    *identity* normalisation — so action outputs stayed in the raw model space
    (mean ~0, std ~1, real motion ~0.1-0.5 normalized units) and never moved the
    arm meaningfully. Loading the saved processors from HF via `from_pretrained`
    restores the SO-100 stats and unlocks ~10-100x larger commands.

    pre_processor: raw obs dict -> policy-ready batch (rename, tokenise, normalise)
    post_processor: raw policy output tensor -> denormalised action tensor
    """
    name = cfg.policy_name.lower()
    pre_processor = None
    post_processor = None
    if name in ("pi0", "pi05"):
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(cfg.policy_repo)
    elif name in ("smolvla", "smolvla_pp"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        # Force-import to register SmolVLA-specific processor steps so the
        # DataProcessorPipeline.from_pretrained call below can resolve them.
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.processor.pipeline import DataProcessorPipeline
        from lerobot.processor.normalize_processor import hotswap_stats
        policy = SmolVLAPolicy.from_pretrained(cfg.policy_repo)
        # Fine-tuned community checkpoints (orsoromeo/smolvla_so101_pick_and_place,
        # lerobot-edinburgh-white-team/smolvla_svla_so101_pickplace) ship only the
        # model weights, NOT the preprocessor pipelines or dataset stats. So we
        # always load the pre/post processor pipelines from the base model
        # release and hot-swap in the so100 buffer stats. This produces correct
        # normalisation for the base model and an approximation for the fine-
        # tunes (SO-100 and SO-101 joint distributions overlap heavily).
        base_repo = "lerobot/smolvla_base"
        pre_processor = DataProcessorPipeline.from_pretrained(
            base_repo, config_filename="policy_preprocessor.json",
        )
        post_processor = DataProcessorPipeline.from_pretrained(
            base_repo, config_filename="policy_postprocessor.json",
        )
        raw_stats = post_processor.steps[0].stats  # all 3 buffers
        target_prefix = "so100.buffer."
        flat_stats = {
            key[len(target_prefix):]: val
            for key, val in raw_stats.items()
            if key.startswith(target_prefix)
        }
        pre_processor = hotswap_stats(pre_processor, flat_stats)
        post_processor = hotswap_stats(post_processor, flat_stats)
        print(f"[smolvla] hotswapped stats to {target_prefix}* ({len(flat_stats)} keys)")
    else:
        raise NotImplementedError(
            f"Policy '{cfg.policy_name}' not wired up. Add a branch here."
        )
    # SmolVLA's flow-matching sampler allocates noise in fp32 internally and
    # feeds it to model weights, so it must run in fp32 (it's only 500M params
    # and fits comfortably on a single 2080 Ti). Pi0/Pi0.5 are large and stay
    # in bf16 to fit on consumer GPUs.
    if name in ("smolvla", "smolvla_pp"):
        dt = torch.float32
    elif cfg.dtype == "bfloat16":
        dt = torch.bfloat16
    else:
        dt = torch.float16
    policy.to(cfg.device, dtype=dt)
    policy.eval()

    target_path = HiddenStateHook.DEFAULT_PATHS.get(name, "")
    hook = HiddenStateHook(policy.model, target_path, pool="last_token").attach()
    print(f"[policy] {cfg.policy_repo} on {cfg.device} ({dt}); hook → {target_path}")
    print(f"[processors] pre={'yes' if pre_processor else 'no'}, post={'yes' if post_processor else 'no'}")
    return policy, hook, pre_processor, post_processor


# ─── Observation packing ────────────────────────────────────────────────────


CAMERA_KEY_MAPS = {
    # raw observation keys (from camera grabber) -> policy-specific batch keys
    "pi0":     [("wrist", "observation.images.wrist"), ("scene", "observation.images.scene")],
    "pi05":    [("wrist", "observation.images.wrist"), ("scene", "observation.images.scene")],
    "smolvla": [("wrist", "observation.images.camera1"), ("scene", "observation.images.camera2")],
    # orsoromeo/smolvla_so101_pick_and_place — checkpoint config.json specifies
    # `observation.images.laptop` (external workspace view) + `observation.images.phone`
    # (closer view). Map our external scene cam to laptop, wrist cam to phone.
    "smolvla_pp": [("scene", "observation.images.laptop"), ("wrist", "observation.images.phone")],
}


def obs_to_batch(obs: dict, instruction: str, device: str, policy_name: str = "pi0",
                 pre_processor=None, target_dtype: torch.dtype | None = None) -> dict:
    """Convert a LeRobot observation dict to the batch dict expected by the policy.

    If pre_processor is provided (SmolVLA path), the manually-built batch is run
    through the LeRobot processor pipeline so it gains tokenised language inputs,
    normalisation, and any other steps the policy expects.

    If target_dtype is provided, all float tensors in the final batch are cast
    to that dtype so they match the policy's loaded precision (e.g. bf16)."""
    batch: dict[str, torch.Tensor | list] = {}
    mapping = CAMERA_KEY_MAPS.get(policy_name.lower(),
                                  [("wrist", "observation.images.wrist"),
                                   ("scene", "observation.images.scene")])
    for raw_key, batch_key in mapping:
        if raw_key in obs:
            img = obs[raw_key]  # numpy HWC uint8
            if img.dtype == np.uint8:
                t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            else:
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            batch[batch_key] = t.to(device)
    # State (joint angles → flat vector). LeRobot SO-101 obs uses "<name>.pos" keys.
    state = []
    for k in sorted(obs.keys()):
        if k.endswith(".pos") and isinstance(obs[k], (int, float, np.floating)):
            state.append(float(obs[k]))
    if state:
        batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)
    batch["task"] = [instruction]
    if pre_processor is not None:
        # SmolVLA: pipeline adds OBS_LANGUAGE_TOKENS / OBS_LANGUAGE_ATTENTION_MASK
        batch = pre_processor(batch)
    if target_dtype is not None:
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != target_dtype:
                batch[k] = v.to(target_dtype)
    return batch


def extract_qpos_eef(obs: dict, follower) -> tuple[np.ndarray, np.ndarray]:
    """Pull joint positions + an approximate EEF position from the observation.

    EEF position is computed via the follower's URDF / forward kinematics if the robot
    exposes it; otherwise we use the last joint position as a coarse proxy (the
    important thing for our pkl schema is consistency, not absolute accuracy)."""
    qpos = np.array(
        [float(obs[k]) for k in sorted(obs.keys()) if k.endswith(".pos")],
        dtype=np.float32,
    )
    if hasattr(follower, "get_end_effector_position"):
        eef = np.asarray(follower.get_end_effector_position(), dtype=np.float32)
    else:
        # Fall back to last-joint surrogate; we'll log a warning once.
        eef = np.array([qpos[-1] if len(qpos) else 0.0, 0.0, 0.0], dtype=np.float32)
    return qpos, eef


# ─── Main loop ──────────────────────────────────────────────────────────────


def run_teleop_episode(cfg, follower, leader, policy, hook, episode_idx: int,
                       pre_processor=None, policy_dtype=None) -> RolloutRecord:
    rec = RolloutRecord(
        instruction=cfg.instruction,
        task_id=cfg.task_id,
        model_tag=cfg.policy_name,
    )
    print(f"\n[ep {episode_idx}] TELEOP. Move the leader; recording {cfg.max_steps} steps.")
    period = 1.0 / cfg.fps
    for t in range(cfg.max_steps):
        t0 = time.time()
        # Leader → action target → follower
        action = leader.get_action()
        follower.send_action(action)
        # Observe + run policy forward pass (for hidden-state capture only — output unused)
        obs = follower.get_observation()
        batch = obs_to_batch(obs, cfg.instruction, cfg.device, cfg.policy_name, pre_processor, policy_dtype)
        # SmolVLA/Pi0 cache action chunks and replay from a queue between
        # forward passes. policy.reset() clears that queue so the model runs
        # (and our hook fires) on every step.
        if hasattr(policy, "reset"):
            policy.reset()
        hook.reset()
        with torch.no_grad():
            _ = policy.select_action(batch)
        h = hook.latest()
        qpos, eef = extract_qpos_eef(obs, follower)
        a_vec = np.array(
            [float(action[k]) for k in sorted(action.keys())], dtype=np.float32
        )
        rec.step(action=a_vec, hidden_state=h, eef_pos=eef, qpos=qpos)
        # Pace the loop
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)
    return rec


def run_policy_episode(cfg, follower, policy, hook, episode_idx: int,
                       pre_processor=None, policy_dtype=None, post_processor=None) -> RolloutRecord:
    rec = RolloutRecord(
        instruction=cfg.instruction,
        task_id=cfg.task_id,
        model_tag=cfg.policy_name,
    )
    print(f"\n[ep {episode_idx}] POLICY. {cfg.policy_name} driving follower; recording {cfg.max_steps} steps.")
    period = 1.0 / cfg.fps
    for t in range(cfg.max_steps):
        t0 = time.time()
        obs = follower.get_observation()
        batch = obs_to_batch(obs, cfg.instruction, cfg.device, cfg.policy_name, pre_processor, policy_dtype)
        if hasattr(policy, "reset"):
            policy.reset()
        hook.reset()
        with torch.no_grad():
            action_t = policy.select_action(batch)
        h = hook.latest()
        # Apply the trained denormaliser so the policy's raw (zero-mean / unit-
        # variance) prediction is mapped back into joint-space command units that
        # actually move the arm meaningfully. The post-processor pipeline expects
        # an EnvTransition-shaped dict, so wrap & unwrap.
        if post_processor is not None:
            action_t = post_processor({"action": action_t})["action"]
        action_np = action_t.detach().cpu().float().numpy().flatten()
        # Hard per-joint clamp in normalized [-100, 100] space to keep the arm
        # off its mechanical end stops even if SmolVLA outputs extreme values.
        action_np = np.clip(action_np, -cfg.joint_abs_limit, cfg.joint_abs_limit)
        # Tighter clamp on the gripper specifically — empty-grip stall against
        # the close-stop overloads the servo. Index = order in the action vector
        # which follows sorted(obs).pos keys; gripper.pos sorts first.
        names = [k.replace(".pos", "") for k in sorted(obs.keys()) if k.endswith(".pos")]
        if "gripper" in names:
            gi = names.index("gripper")
            action_np[gi] = float(np.clip(action_np[gi], -cfg.gripper_abs_limit, cfg.gripper_abs_limit))
        # Map vector back into LeRobot's dict-of-floats format. Skip the gripper
        # key when torque is disabled so the bus never tries to drive motor 6.
        if cfg.disable_gripper_torque:
            action_dict = {f"{name}.pos": float(v) for name, v in zip(names, action_np)
                           if name != "gripper"}
        else:
            action_dict = {f"{name}.pos": float(v) for name, v in zip(names, action_np)}
        follower.send_action(action_dict)
        qpos, eef = extract_qpos_eef(obs, follower)
        rec.step(action=action_np, hidden_state=h, eef_pos=eef, qpos=qpos)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["teleop", "policy"], default="teleop")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--task-id", default="so101_pick_bowl_place_plate")
    p.add_argument("--instruction", default="pick up the black bowl and place it on the plate")
    p.add_argument("--follower-port", default="/dev/ttyACM1")
    p.add_argument("--leader-port", default="/dev/ttyACM0")
    p.add_argument("--wrist-cam", type=int, default=0)
    p.add_argument("--scene-cam", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--policy", default="pi0")
    p.add_argument("--policy-repo", default="lerobot/pi0")
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load policy + hook, skip robot connection. For local logic testing.",
    )
    args = p.parse_args()

    cfg = SO101RunConfig(
        follower_port=args.follower_port,
        leader_port=args.leader_port,
        wrist_cam_index=args.wrist_cam,
        scene_cam_index=args.scene_cam,
        device=args.device,
        task_id=args.task_id,
        instruction=args.instruction,
        policy_name=args.policy,
        policy_repo=args.policy_repo,
        fps=args.fps,
        max_steps=args.max_steps,
        n_episodes=args.n_episodes,
    )
    out_dir = Path(args.output_dir) if args.output_dir else (cfg.output_dir / cfg.run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    policy, hook, pre_processor, post_processor = make_policy(cfg)
    policy_dtype = next(policy.parameters()).dtype

    if args.dry_run:
        print("[dry-run] skipping robot connect; verifying policy + hook only.")
        dummy_obs = {
            "wrist": np.zeros((240, 320, 3), dtype=np.uint8),
            "scene": np.zeros((480, 640, 3), dtype=np.uint8),
            "shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0, "wrist_roll.pos": 0.0, "gripper.pos": 0.0,
        }
        batch = obs_to_batch(dummy_obs, cfg.instruction, cfg.device, cfg.policy_name, pre_processor, policy_dtype)
        hook.reset()
        with torch.no_grad():
            a = policy.select_action(batch)
        h = hook.latest()
        print(f"[dry-run] action shape={tuple(a.shape)}, hidden dim={None if h is None else h.shape}")
        hook.detach()
        return

    follower = make_follower(cfg)
    leader = make_leader(cfg) if args.mode == "teleop" else None
    follower.connect()
    if leader is not None:
        leader.connect()
    print("[connected] follower + leader" if leader else "[connected] follower")
    if cfg.disable_gripper_torque:
        try:
            follower.bus.disable_torque(motors=["gripper"])
            print("[safety] gripper torque disabled — motor 6 hangs limp for this run")
        except Exception as e:
            print(f"[safety] WARNING: could not disable gripper torque: {e}")

    all_rollouts = []
    try:
        with estop_handler(follower, "follower"):
            for ep in range(cfg.n_episodes):
                input(f"\n>> Press ENTER to start episode {ep+1}/{cfg.n_episodes} (Ctrl+C to stop)... ")
                if args.mode == "teleop":
                    rec = run_teleop_episode(cfg, follower, leader, policy, hook, ep + 1,
                                              pre_processor=pre_processor, policy_dtype=policy_dtype)
                else:
                    rec = run_policy_episode(cfg, follower, policy, hook, ep + 1,
                                              pre_processor=pre_processor, policy_dtype=policy_dtype,
                                              post_processor=post_processor)
                all_rollouts.append(rec.finalize(success=False))  # blind label later
                # Save incrementally to disk so a crash doesn't lose data.
                with (out_dir / f"raw_ep{ep:03d}.pkl").open("wb") as f:
                    pickle.dump(all_rollouts[-1], f)
                print(f"[ep {ep+1}] {len(rec.steps)} steps recorded → raw_ep{ep:03d}.pkl")
    finally:
        try:
            follower.disconnect()
        except Exception:
            pass
        if leader is not None:
            try:
                leader.disconnect()
            except Exception:
                pass
        hook.detach()

    # Stash a single unlabeled.pkl for downstream labeling
    with (out_dir / "unlabeled_rollouts.pkl").open("wb") as f:
        pickle.dump(all_rollouts, f)
    print(f"\n[done] {len(all_rollouts)} episodes → {out_dir / 'unlabeled_rollouts.pkl'}")
    print(f"[next] python scripts/so101/label_rollouts.py --run-dir {out_dir}")


if __name__ == "__main__":
    main()
