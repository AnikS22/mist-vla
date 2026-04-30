#!/usr/bin/env python3
"""PULSE / Yahboom demo: what the software does and what the arm will run.

This is a **walk-through + optional figure** (no Jetson required). It matches the
physical protocol in `paper/robot_proposal.pdf` and the real loop in
`scripts/run_model_yahboom_loop.py` + UI in `scripts/yahboom_command_ui.py`.

Examples:
  python3 scripts/demo_pulse_robot.py
  python3 scripts/demo_pulse_robot.py --figure figures/25_demo_pulse_timeline.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _banner(title: str) -> None:
    line = "═" * min(72, max(12, len(title) + 8))
    print(f"\n{line}\n  {title}\n{line}\n")


def print_software_stack() -> None:
    _banner("Software stack (what runs where)")
    print(
        """\
  ┌─────────────────────────────────────────────────────────────────────┐
  │  GPU server (your machine)                                           │
  │  • OpenVLA / OpenVLA-OFT / SmolVLA → 7-DoF action + hidden state h_t │
  │  • Safety MLP probe (sim-trained, frozen policy):                    │
  │      h_t → fail_logit, time-to-fail, Δp (Cartesian correction)     │
  │  • Modes: vanilla | steering | mppi | latent_stop | latent_jiggle   │
  │           | heuristic (see proposal table)                          │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │ HTTP / JSON
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Jetson on arm (Yahboom controller)                                  │
  │  • GET /snapshot  → wrist RGB 640×480                              │
  │  • GET /status    → joint pose / coords                             │
  │  • POST /action   → staged move_to + gripper                        │
  └─────────────────────────────────────────────────────────────────────┘

  Repo entrypoints:
    • mist-vla/scripts/run_model_yahboom_loop.py  — closed loop to Jetson
    • mist-vla/scripts/yahboom_command_ui.py     — local web UI → same loop
"""
    )


def print_control_loop() -> None:
    _banner("Per-step control loop (~5 Hz, proposal)")
    rows = [
        ("1", "Capture RGB", "<5 ms", "640×480×3 → policy"),
        ("2", "Policy forward", "~200 ms", "image + instruction → h_t, a_t"),
        ("3", "Safety probe", "<1 ms", "h_t → fail_prob, TTF, Δp"),
        ("4", "EMA + clamp + double gate", "<0.5 ms", "fire if ‖Δp‖>τ_c AND σ(fail)>τ_f"),
        ("5", "Action modify + checks", "<0.5 ms", "‖ΔEEF‖ cap, ‖a‖∞ clamp"),
        ("6", "Send to arm", "~5 ms", "7-DoF command → Jetson /action"),
    ]
    w = [4, 22, 10, 42]
    hdr = f"{'#':<{w[0]}} {'Step':<{w[1]}} {'Latency':<{w[2]}} Detail"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:<{w[0]}} {r[1]:<{w[1]}} {r[2]:<{w[2]}} {r[3]}")
    print(
        "\n  Logged each step: h_t, raw/modified action, fail_prob, TTF, Δp, gates, EEF, time."
        "\n  Logged each episode: MP4, blind success, trajectory, intervention stats."
    )


def print_robot_protocol() -> None:
    _banner("What the robot will do (physical phase)")
    print(
        """\
  Setup
    • Wrist camera 640×480 @ 30 fps; Ethernet from GPU server to Jetson.
    • Load policies (OpenVLA-7B, ACT) + **sim-trained probe** (no real fine-tune).
    • Workspace calibration + acceptance checks.

  Tasks (tabletop blocks, ≤300 steps, success = block within 2 cm of goal)
    T1  Pick–place with obstacle
    T2  Pick near table edge  (risky EEF)
    T3  Pick from clutter      (probe should see rising fail risk)

  Study design (proposal)
    50 episodes × 6 modes × 3 tasks × 2 policies → 1,800 episodes (blind labels).

  How you show interventions help (not just fire)
    • Success when probe intervened vs silent (same mode).
    • Counterfactual replay: vanilla from same init when steering saved run.
    • Failure timing: when fail_prob crossed 0.5 vs when failure happened.
"""
    )


def print_cli_hints() -> None:
    _banner("Run the real stack (when hardware is up)")
    yahboom = REPO_ROOT / "scripts" / "run_model_yahboom_loop.py"
    ui = REPO_ROOT / "scripts" / "yahboom_command_ui.py"
    print(f"  Policy + arm loop:\n    python3 {yahboom} --help\n")
    print(f"  Browser prompt UI (starts server, calls loop script):\n    python3 {ui}\n")
    print(
        "  In the UI: leave “Disable steering” unchecked to exercise the probe path;\n"
        "  check it for vanilla OpenVLA/OFT without corrections.\n"
    )


def _synthetic_timeline(n: int = 120):
    """Toy signals for visualization only (not from a real run)."""
    t = [i / 5.0 for i in range(n)]  # 5 Hz → seconds
    fail = []
    dist = []
    intervene = []
    base = 0.12
    for i in range(n):
        u = i / max(1, n - 1)
        # Risk rises mid-episode (e.g. clutter / edge), then decays if “recovered”
        risk = 0.15 + 0.62 * math.exp(-((u - 0.45) ** 2) / 0.012)
        risk = min(0.97, risk + 0.05 * math.sin(u * 18.0))
        fail.append(risk)
        gate = risk > 0.52 and abs(math.sin(u * 11.0)) > 0.35
        intervene.append(gate)
        step = -0.9 if gate else -0.55
        d = base + step * u + (0.02 if gate else 0.0) * math.sin(u * 25.0)
        dist.append(max(0.0, d))
    return t, fail, dist, intervene


def write_timeline_figure(out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("matplotlib is required for --figure:", e, file=sys.stderr)
        sys.exit(1)

    t, fail, dist, iv = _synthetic_timeline()
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, constrained_layout=True)

    ax[0].plot(t, fail, color="#6ea8fe", lw=1.8, label="fail_prob (probe)")
    ax[0].axhline(0.5, color="#888", ls="--", lw=1, label="0.5 threshold (illustrative)")
    ax[0].fill_between(t, 0, 1, where=iv, color="#3fb950", alpha=0.12, label="steering gate ON")
    ax[0].set_ylabel("Probe output")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("PULSE demo — synthetic timeline (for slides; not real robot data)")

    ax[1].plot(t, dist, color="#ff7b72", lw=1.8, label="distance-to-goal (arb. units)")
    ax[1].fill_between(t, 0, max(dist) * 1.1, where=iv, color="#3fb950", alpha=0.12)
    ax[1].set_xlabel("Time (s) @ 5 Hz")
    ax[1].set_ylabel("Task progress proxy")
    ax[1].legend(loc="upper right", fontsize=8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"\n  Wrote figure: {out.resolve()}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--figure",
        nargs="?",
        const=str(REPO_ROOT / "figures" / "25_demo_pulse_timeline.png"),
        default=None,
        help="Save timeline PNG (default: figures/25_demo_pulse_timeline.png)",
    )
    args = p.parse_args()

    print_software_stack()
    print_control_loop()
    print_robot_protocol()
    print_cli_hints()

    if args.figure is not None:
        write_timeline_figure(Path(args.figure))


if __name__ == "__main__":
    main()
