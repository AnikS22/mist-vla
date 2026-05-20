#!/usr/bin/env python3
"""Render success vs failure rollout panels for the paper failure-definition figure.

Strategy:
  1. Pick a LIBERO-Spatial task with both success and failure trajectories in our dataset.
  2. Iterate through LIBERO's preset init_states for that task and find one for which a
     recorded success-pkl trajectory actually completes when replayed (env.set_init_state +
     open-loop action replay). This makes the success row land the bowl on the plate.
  3. From the SAME init_state, replay a recorded failure trajectory to produce the failure row.
  4. Stitch into a side-by-side filmstrip and save as
     paper/figures/28_success_failure_rollout.png.

Usage:
    cd mist-vla && python3 scripts/render_success_failure_panels.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

NUM_FRAMES = 7
FRAME_SIZE = 192
BORDER = 6
PAD = 12
LABEL_H = 36

GREEN = (62, 168, 89)
RED = (211, 68, 68)
BG = (255, 255, 255)

ROLLOUTS_DIR = REPO / "research_data" / "rollouts" / "merged_all"
OUT = REPO / "paper" / "figures" / "28_success_failure_rollout.png"

# How many init_states to try when searching for a successful replay.
MAX_INIT_STATES_TO_TRY = 30
# How many success trajectories to try per init_state.
MAX_SUCC_TRAJS_TO_TRY = 5


def load_pair_pool(task_id_filter: str | None = None):
    with (ROLLOUTS_DIR / "success_rollouts.pkl").open("rb") as f:
        succ = pickle.load(f)
    with (ROLLOUTS_DIR / "failure_rollouts.pkl").open("rb") as f:
        fail = pickle.load(f)
    if task_id_filter:
        succ = [r for r in succ if r["task_id"] == task_id_filter]
        fail = [r for r in fail if r["task_id"] == task_id_filter]
    return succ, fail


def make_env(suite: str, task_idx: int):
    bm = benchmark.get_benchmark_dict()[suite]()
    bddl = bm.get_task_bddl_file_path(task_idx)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        render_camera="agentview",
        camera_heights=FRAME_SIZE,
        camera_widths=FRAME_SIZE,
    )
    return env, bm


def grab(env, obs):
    img = obs.get("agentview_image")
    if img is None:
        img = env.sim.render(camera_name="agentview", height=FRAME_SIZE, width=FRAME_SIZE)
    img = np.flipud(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img)


def replay(env, init_state, actions, n_frames):
    """Replay an action sequence from a known init_state. Returns (frames, idxs, succeeded, fail_step)."""
    obs = env.reset()
    obs = env.set_init_state(init_state)
    T = len(actions)
    idxs = np.linspace(0, T - 1, n_frames).round().astype(int)
    frames = {}
    last = grab(env, obs)
    succeeded = False
    success_step = None
    for t in range(T):
        if t in idxs:
            frames[int(t)] = last
        a = np.asarray(actions[t], dtype=np.float32)
        obs, r, done, _ = env.step(a)
        last = grab(env, obs)
        if r and r > 0 and success_step is None:
            success_step = t
            succeeded = True
        if done and t < idxs.max():
            for k in idxs:
                if int(k) > t and int(k) not in frames:
                    frames[int(k)] = last
            break
    if int(idxs[-1]) not in frames:
        frames[int(idxs[-1])] = last
    fail_step = None if succeeded else int(idxs[-1])
    return [frames[int(i)] for i in idxs], idxs, succeeded, fail_step, success_step


def stamp_panel(frames, color, label, mark_step=None, mark_color=None, mark_text=None, idxs=None):
    n = len(frames)
    w = n * FRAME_SIZE + (n + 1) * PAD
    h = FRAME_SIZE + LABEL_H + 2 * PAD + 2 * BORDER
    canvas = Image.new("RGB", (w, h + 2 * BORDER), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w - 1, h + 2 * BORDER - 1], outline=color, width=BORDER)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small = font
    draw.text((PAD + BORDER, BORDER + 8), label, fill=color, font=font)
    for i, fr in enumerate(frames):
        x = PAD + i * (FRAME_SIZE + PAD) + BORDER
        y = LABEL_H + PAD + BORDER
        canvas.paste(Image.fromarray(fr), (x, y))
        if idxs is not None:
            tick = f"t={int(idxs[i])}"
            tw = draw.textlength(tick, font=small)
            tx0, ty0 = x + 4, y + FRAME_SIZE - 18
            draw.rectangle([tx0 - 2, ty0 - 1, tx0 + int(tw) + 4, ty0 + 14], fill=(0, 0, 0))
            draw.text((tx0, ty0), tick, fill=(255, 255, 255), font=small)
        is_mark_frame = (
            mark_step is not None
            and idxs is not None
            and (
                (i > 0 and int(idxs[i]) >= mark_step and int(idxs[i - 1]) < mark_step)
                or (i == 0 and int(idxs[i]) >= mark_step)
                or (i == len(frames) - 1 and mark_step > int(idxs[-1]))
            )
        )
        if is_mark_frame:
            mc = mark_color or (255, 60, 60)
            draw.rectangle([x - 2, y - 2, x + FRAME_SIZE + 2, y + FRAME_SIZE + 2], outline=mc, width=4)
            if mark_text:
                bw = draw.textlength(mark_text, font=font)
                bx0, by0 = x + 6, y + 6
                bx1, by1 = bx0 + int(bw) + 12, by0 + 24
                draw.rectangle([bx0, by0, bx1, by1], fill=mc)
                draw.text((bx0 + 6, by0 + 2), mark_text, fill=(255, 255, 255), font=font)
    return canvas


def state_to_flat(state_dict):
    """Convert a rollout's per-step state dict into a flat [time, qpos, qvel] vector
    that set_state_from_flattened expects."""
    qpos = np.asarray(state_dict["qpos"], dtype=np.float64)
    qvel = np.asarray(state_dict["qvel"], dtype=np.float64)
    return np.concatenate([[0.0], qpos, qvel])


def find_compatible_init(env, init_states, succ_pool, fail_pool, n_frames):
    """Search for an init_state and a success-pkl trajectory pair that produces a real success.
    Returns (init_idx, success_frames_pack, failure_frames_pack) or None."""
    for init_idx in range(min(len(init_states), MAX_INIT_STATES_TO_TRY)):
        init_state = init_states[init_idx]
        for succ_idx in range(min(len(succ_pool), MAX_SUCC_TRAJS_TO_TRY)):
            cand = succ_pool[succ_idx]
            frames, idxs, ok, _, success_step = replay(env, init_state, cand["actions"], n_frames)
            if ok:
                print(f"  init_state[{init_idx}] + success-pkl[{succ_idx}] -> SUCCESS at step {success_step}")
                # Now replay a failure trajectory from the SAME init_state.
                for fail_idx in range(len(fail_pool)):
                    cand_f = fail_pool[fail_idx]
                    ff, fi, ok_f, fs, _ = replay(env, init_state, cand_f["actions"], n_frames)
                    if not ok_f:
                        print(f"  init_state[{init_idx}] + failure-pkl[{fail_idx}] -> FAIL (clean failure trajectory found)")
                        return (
                            init_idx,
                            (frames, idxs, success_step, cand["instruction"]),
                            (ff, fi, fs, cand_f["instruction"]),
                        )
                # If every failure-pkl trajectory accidentally succeeds from this init,
                # fall back to using the success trajectory truncated for the failure row.
                print(f"  init_state[{init_idx}]: all failure-pkl trajectories also succeeded; trying next init.")
        # Try the next init_state
    return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Find a task with both success and failure trajectories.
    with (ROLLOUTS_DIR / "success_rollouts.pkl").open("rb") as f:
        s_all = pickle.load(f)
    with (ROLLOUTS_DIR / "failure_rollouts.pkl").open("rb") as f:
        f_all = pickle.load(f)
    s_tasks = {r["task_id"] for r in s_all}
    f_tasks = {r["task_id"] for r in f_all}
    common = sorted(s_tasks & f_tasks)
    if not common:
        raise RuntimeError("no task with both success and failure rollouts")
    task_id = common[0]
    suite, task_idx = task_id.split("__")
    task_idx = int(task_idx)
    print(f"task {task_id}")

    env, bm = make_env(suite, task_idx)
    succ_pool, fail_pool = load_pair_pool(task_id_filter=task_id)
    print(f"  {len(succ_pool)} success / {len(fail_pool)} failure trajectories for this task")

    # Pick the first success rollout whose own init reproduces a success when replayed.
    s_pack = None
    for si, cand in enumerate(succ_pool):
        try:
            init = state_to_flat(cand["robot_states"][0])
        except (KeyError, IndexError, TypeError):
            continue
        frames, idxs, ok, _, success_step = replay(env, init, cand["actions"], NUM_FRAMES)
        if ok:
            print(f"  success rollout {si}: reproduced success at step {success_step}")
            s_pack = (frames, idxs, success_step, cand["instruction"])
            break

    # Pick the first failure rollout whose own init reproduces a non-success.
    f_pack = None
    for fi, cand in enumerate(fail_pool):
        try:
            init = state_to_flat(cand["robot_states"][0])
        except (KeyError, IndexError, TypeError):
            continue
        frames, idxs, ok, fs, _ = replay(env, init, cand["actions"], NUM_FRAMES)
        if not ok:
            print(f"  failure rollout {fi}: reproduced failure (last sampled step {fs})")
            f_pack = (frames, idxs, fs, cand["instruction"])
            break
    env.close()

    if s_pack is None or f_pack is None:
        print("  could not reproduce both a success and a failure; aborting.")
        return

    s_frames, s_idxs, success_step, s_instr = s_pack
    f_frames, f_idxs, fail_step, f_instr = f_pack

    succ_label = f"Successful rollout: \"{s_instr}\""
    fail_label = f"Failed rollout: \"{s_instr}\""

    succ_panel = stamp_panel(
        s_frames, GREEN, succ_label,
        mark_step=success_step, mark_color=GREEN, mark_text="Success",
        idxs=s_idxs,
    )
    fail_panel = stamp_panel(
        f_frames, RED, fail_label,
        mark_step=fail_step, mark_color=RED, mark_text="Failure",
        idxs=f_idxs,
    )

    W = max(succ_panel.width, fail_panel.width)
    H = succ_panel.height + fail_panel.height + PAD
    fig = Image.new("RGB", (W, H), BG)
    fig.paste(succ_panel, (0, 0))
    fig.paste(fail_panel, (0, succ_panel.height + PAD))
    fig.save(OUT, dpi=(200, 200))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
