#!/usr/bin/env python3
"""One-command paper artifact build for reproducible camera-ready workflow."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
SCRIPTS = PAPER / "scripts"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd), check=False)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    if not SCRIPTS.exists():
        raise SystemExit(f"Missing scripts dir: {SCRIPTS}")

    preferred = Path("/home/mpcr/miniconda/bin/python3")
    py = str(preferred) if preferred.exists() else (shutil.which("python3") or sys.executable)

    # 1) statistics + tables
    run([py, "run_stat_tests.py"], cwd=SCRIPTS)
    run([py, "generate_tables.py"], cwd=SCRIPTS)

    # 2) visuals
    run([py, "generate_visuals.py"], cwd=SCRIPTS)
    run([py, "generate_additional_visuals.py"], cwd=SCRIPTS)
    run([py, "generate_latent_embeddings.py"], cwd=SCRIPTS)
    run([py, "generate_steering_visuals.py"], cwd=SCRIPTS)

    # 3) latex build
    run(["tectonic", "--keep-logs", "main.tex"], cwd=PAPER)

    print("\n[ok] Paper artifacts and PDF rebuilt successfully.")


if __name__ == "__main__":
    main()

