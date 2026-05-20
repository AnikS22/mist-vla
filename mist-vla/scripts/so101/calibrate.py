#!/usr/bin/env python3
"""Calibrate the SO-101 follower (and optionally leader).

Wraps LeRobot's interactive calibration with a clear prompt sequence. Run once per
port assignment / motor reseat. Calibration files are saved under ~/.cache/huggingface/lerobot.

Usage:
    python scripts/so101/calibrate.py --follower-port /dev/ttyACM0
    python scripts/so101/calibrate.py --leader-port  /dev/ttyACM1
    python scripts/so101/calibrate.py --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def calibrate_follower(port: str, cal_dir: str | None):
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    cfg = SO101FollowerConfig(
        port=port, id="follower", calibration_dir=cal_dir,
        disable_torque_on_disconnect=True, cameras={}, use_degrees=False,
    )
    robot = SO101Follower(cfg)
    print(f"\n[follower] connecting on {port} ...")
    robot.connect(calibrate=True)
    print("[follower] calibration complete; disconnecting.")
    robot.disconnect()


def calibrate_leader(port: str, cal_dir: str | None):
    from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
    cfg = SO101LeaderConfig(port=port, id="leader", calibration_dir=cal_dir, use_degrees=False)
    tel = SO101Leader(cfg)
    print(f"\n[leader] connecting on {port} ...")
    tel.connect(calibrate=True)
    print("[leader] calibration complete; disconnecting.")
    tel.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follower-port")
    ap.add_argument("--leader-port")
    ap.add_argument("--calibration-dir", default=None)
    args = ap.parse_args()
    if not args.follower_port and not args.leader_port:
        ap.error("specify --follower-port and/or --leader-port")
    if args.follower_port:
        calibrate_follower(args.follower_port, args.calibration_dir)
    if args.leader_port:
        calibrate_leader(args.leader_port, args.calibration_dir)


if __name__ == "__main__":
    main()
