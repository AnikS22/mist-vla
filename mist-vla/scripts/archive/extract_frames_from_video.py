#!/usr/bin/env python3
"""
Extract frames from an MP4 into numbered PNG files.
"""
import argparse
from pathlib import Path
import imageio.v2 as imageio


def extract(video_path: Path, out_dir: Path, stride: int = 1):
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(video_path))
    idx = 0
    out_idx = 0
    for frame in reader:
        if idx % stride == 0:
            imageio.imwrite(out_dir / f"frame_{out_idx:06d}.png", frame)
            out_idx += 1
        idx += 1
    reader.close()
    print(f"Saved {out_idx} frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to MP4")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    args = parser.parse_args()
    extract(Path(args.video), Path(args.out_dir), args.stride)


if __name__ == "__main__":
    main()
