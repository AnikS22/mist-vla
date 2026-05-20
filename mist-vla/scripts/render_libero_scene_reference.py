#!/usr/bin/env python3
"""Render the LIBERO-Spatial Task 0 scene with object labels for the paper.

Projects MuJoCo body world positions into agentview pixel coordinates so that
leader lines land exactly on each object. Labels are placed in the margin and
connected to objects with thin leader lines.

Output: paper/figures/29_libero_scene_reference.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

RENDER = 1024
MARGIN_T = 56
BG = (255, 255, 255)
TITLE_COLOR = (30, 30, 30)

OUT = REPO / "paper" / "figures" / "29_libero_scene_reference.png"

# (body_name, display_label, role_color, dx, dy) — only role-annotated objects.
# dx/dy nudge the label off the object's projected center.
LABELS = [
    ("akita_black_bowl_1_main",   "akita_black_bowl (target)", (220, 60, 60),  -10, -42),
    ("plate_1_main",              "plate (goal)",              (60, 140, 220),  10,  34),
    ("cookies_1_main",            "cookies (decoy)",           (200, 150, 40), -10,  44),
    ("akita_black_bowl_2_main",   "akita_black_bowl (decoy)",  (200, 150, 40),  10, -42),
]

TITLE = "LIBERO-Spatial Task 0:  “pick up the black bowl between the plate and the ramekin and place it on the plate”"


def make_env():
    bm = benchmark.get_benchmark_dict()["libero_spatial"]()
    bddl = bm.get_task_bddl_file_path(0)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        render_camera="agentview",
        camera_heights=RENDER,
        camera_widths=RENDER,
    )
    return env, bm


def render_scene(env, bm):
    import torch as _torch
    task = bm.get_task(0)
    init_file = (
        Path("/home/mpcr/Desktop/SalusV5/LIBERO/libero/libero/init_files")
        / task.problem_folder
        / task.bddl_file.replace(".bddl", ".pruned_init")
    )
    if not init_file.exists():
        init_file = init_file.with_suffix(".init")
    init_states = _torch.load(init_file, weights_only=False)
    obs = env.reset()
    obs = env.set_init_state(init_states[0])
    img = obs.get("agentview_image")
    if img is None:
        img = env.sim.render(camera_name="agentview", height=RENDER, width=RENDER)
    img = np.flipud(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img)


def project_points(sim, body_names, width, height):
    """World -> image pixel coords for agentview, matching np.flipud-ed render."""
    cam_id = sim.model.camera_name2id("agentview")
    fovy_deg = float(sim.model.cam_fovy[cam_id])
    f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    cam_pos = np.array(sim.data.cam_xpos[cam_id])
    cam_mat = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    pts = {}
    for name in body_names:
        bid = sim.model.body_name2id(name)
        wp = np.array(sim.data.body_xpos[bid])
        rel = wp - cam_pos
        cam_xyz = cam_mat.T @ rel
        x_c, y_c, z_c = cam_xyz
        depth = -z_c
        if depth <= 1e-6:
            pts[name] = None
            continue
        u = width / 2.0 + f * x_c / depth
        v = height / 2.0 - f * y_c / depth
        pts[name] = (float(u), float(v))
    return pts


def draw_inline_label(draw, cx, cy, text, color, font):
    """Filled rectangle centered on (cx, cy) with white text."""
    tw = draw.textlength(text, font=font)
    th = font.size + 4
    pad_x, pad_y = 8, 4
    box = [
        cx - tw / 2 - pad_x,
        cy - th / 2 - pad_y,
        cx + tw / 2 + pad_x,
        cy + th / 2 + pad_y,
    ]
    draw.rectangle(box, fill=color)
    draw.text((cx - tw / 2, cy - th / 2), text, fill=(255, 255, 255), font=font)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    env, bm = make_env()
    scene = render_scene(env, bm)
    pts = project_points(env.sim, [name for name, _, _, _, _ in LABELS], RENDER, RENDER)
    env.close()

    H, W = scene.shape[:2]
    canvas_w = W
    canvas_h = MARGIN_T + H
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    canvas.paste(Image.fromarray(scene), (0, MARGIN_T))
    draw = ImageDraw.Draw(canvas)

    label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    title_font = None
    for size in range(22, 11, -1):
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        if draw.textlength(TITLE, font=f) <= canvas_w - 24:
            title_font = f
            break
    if title_font is None:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)

    tw = draw.textlength(TITLE, font=title_font)
    draw.text(((canvas_w - tw) / 2, 16), TITLE, fill=TITLE_COLOR, font=title_font)

    for name, label, color, dx, dy in LABELS:
        p = pts.get(name)
        if p is None:
            continue
        cx = p[0] + dx
        cy = MARGIN_T + p[1] + dy
        draw_inline_label(draw, cx, cy, label, color, label_font)

    canvas.save(OUT, dpi=(200, 200))
    print(f"wrote {OUT}  ({canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
