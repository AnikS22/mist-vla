#!/usr/bin/env python3
"""Render a gallery of all 10 LIBERO-Spatial tasks for the paper benchmark figure.

Each tile shows the agentview render of the task's preset init_state with a
caption above the image. Output: paper/figures/30_libero_spatial_gallery.png
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import torch as _torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

SUITE = "libero_spatial"
TILE = 256
COLS = 5
ROWS = 2
GAP = 24
PAD = 28
CAPTION_H = 64
CAPTION_FILL = (218, 226, 246)
CAPTION_OUTLINE = (170, 184, 222)
CAPTION_TEXT = (40, 40, 80)
BG = (255, 255, 255)
TITLE_COLOR = (30, 30, 30)
INIT_FILE_ROOT = Path("/home/mpcr/Desktop/SalusV5/LIBERO/libero/libero/init_files")
OUT = REPO / "paper" / "figures" / "30_libero_spatial_gallery.png"


def render_task(task_idx: int) -> tuple[np.ndarray, str]:
    bm = benchmark.get_benchmark_dict()[SUITE]()
    task = bm.get_task(task_idx)
    bddl = bm.get_task_bddl_file_path(task_idx)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        render_camera="agentview",
        camera_heights=TILE,
        camera_widths=TILE,
    )
    init_file = INIT_FILE_ROOT / task.problem_folder / task.bddl_file.replace(".bddl", ".pruned_init")
    if not init_file.exists():
        init_file = init_file.with_suffix(".init")
    init_states = _torch.load(init_file, weights_only=False)
    obs = env.reset()
    obs = env.set_init_state(init_states[0])
    img = obs.get("agentview_image")
    if img is None:
        img = env.sim.render(camera_name="agentview", height=TILE, width=TILE)
    img = np.flipud(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    env.close()
    return np.ascontiguousarray(img), task.language


def wrap_caption(draw, text: str, font, max_width: int) -> list[str]:
    """Word-wrap text so each line fits inside max_width pixels."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        candidate = (cur + " " + w).strip()
        if draw.textlength(candidate, font=font) <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_tile(scene: np.ndarray, caption: str, caption_font, idx: int) -> Image.Image:
    tile_w = TILE
    tile_h = CAPTION_H + 8 + TILE
    tile = Image.new("RGB", (tile_w, tile_h), BG)
    draw = ImageDraw.Draw(tile)
    # Caption rounded box.
    cap_box = [0, 0, tile_w - 1, CAPTION_H - 1]
    draw.rounded_rectangle(cap_box, radius=10, fill=CAPTION_FILL, outline=CAPTION_OUTLINE, width=2)
    lines = wrap_caption(draw, caption, caption_font, tile_w - 16)
    # Truncate to 3 lines for box height.
    if len(lines) > 3:
        lines = lines[:3]
        lines[-1] = lines[-1].rstrip(".,;:") + "..."
    line_h = caption_font.size + 4
    total_h = line_h * len(lines)
    y0 = (CAPTION_H - total_h) // 2
    for k, line in enumerate(lines):
        tw = draw.textlength(line, font=caption_font)
        draw.text(((tile_w - tw) / 2, y0 + k * line_h), line, fill=CAPTION_TEXT, font=caption_font)
    # Scene.
    tile.paste(Image.fromarray(scene), (0, CAPTION_H + 8))
    # Task index badge in the bottom-left of the scene.
    badge_font = caption_font
    txt = f"T{idx}"
    tw = draw.textlength(txt, font=badge_font)
    bx, by = 6, tile_h - 24
    draw.rectangle([bx, by, bx + int(tw) + 8, by + 18], fill=(0, 0, 0))
    draw.text((bx + 4, by + 1), txt, fill=(255, 255, 255), font=badge_font)
    return tile


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        cap_font = ImageFont.truetype("DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        cap_font = ImageFont.load_default()
        title_font = cap_font

    print("rendering 10 LIBERO-Spatial tiles...")
    tiles = []
    for i in range(COLS * ROWS):
        scene, lang = render_task(i)
        print(f"  T{i}: {lang}")
        tiles.append((scene, lang))

    # Build a dummy ImageDraw to measure tile sizes without rendering.
    sample = draw_tile(tiles[0][0], tiles[0][1], cap_font, 0)
    tile_w, tile_h = sample.size

    title = f"LIBERO-Spatial benchmark — {COLS * ROWS} tabletop pick-and-place tasks (agentview)"
    title_h = title_font.size + 32

    canvas_w = PAD * 2 + COLS * tile_w + (COLS - 1) * GAP
    canvas_h = title_h + PAD + ROWS * tile_h + (ROWS - 1) * GAP + PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    # Title.
    tw_title = draw.textlength(title, font=title_font)
    draw.text(((canvas_w - tw_title) / 2, 18), title, fill=TITLE_COLOR, font=title_font)

    # Paste tiles.
    for k, (scene, lang) in enumerate(tiles):
        col = k % COLS
        row = k // COLS
        tile_img = draw_tile(scene, lang, cap_font, k)
        x = PAD + col * (tile_w + GAP)
        y = title_h + PAD + row * (tile_h + GAP)
        canvas.paste(tile_img, (x, y))

    canvas.save(OUT, dpi=(200, 200))
    print(f"wrote {OUT}  ({canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
