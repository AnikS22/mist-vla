#!/usr/bin/env python3
"""Host-side bridge for controlling Yahboom arm via Jetson arm_server API."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List
from urllib import request


def post(base_url: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode()
    req = request.Request(
        f"{base_url}/action",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def clamp_workspace(coords: List[float]) -> List[float]:
    # Conservative workspace envelope; adjust after calibration.
    x, y, z, rx, ry, rz = coords
    x = max(-220.0, min(220.0, x))
    y = max(-220.0, min(220.0, y))
    z = max(40.0, min(260.0, z))
    rx = max(-180.0, min(180.0, rx))
    ry = max(-180.0, min(180.0, ry))
    rz = max(-180.0, min(180.0, rz))
    return [x, y, z, rx, ry, rz]


def run_smoke(base_url: str) -> int:
    print("[smoke] status")
    st = post(base_url, {"action": "status"})
    print(" status:", st.get("ok"), "angles:", st.get("angles"))
    if not st.get("ok"):
        print("[smoke] status failed")
        return 2

    angles = st.get("angles") or [0, 0, 0, 0, 0, -45]
    print("[smoke] send_angles noop")
    r0 = post(base_url, {"action": "send_angles", "angles": angles, "speed": 20, "wait": 1.0})
    print("  ->", r0.get("ok"), "elapsed_ms:", r0.get("elapsed_ms"))

    a2 = list(angles)
    a2[4] = float(a2[4]) + 4.0
    print("[smoke] small J5 perturbation")
    r1 = post(base_url, {"action": "send_angles", "angles": a2, "speed": 15, "wait": 1.2})
    print("  ->", r1.get("ok"), "elapsed_ms:", r1.get("elapsed_ms"))
    r2 = post(base_url, {"action": "send_angles", "angles": angles, "speed": 15, "wait": 1.2})
    print("  -> return:", r2.get("ok"), "elapsed_ms:", r2.get("elapsed_ms"))

    print("[smoke] gripper pulse")
    ro = post(base_url, {"action": "set_gripper", "value": 80, "speed": 40})
    rc = post(base_url, {"action": "set_gripper", "value": 20, "speed": 40})
    print("  -> open:", ro.get("ok"), "close:", rc.get("ok"))
    return 0


def run_pick_place(base_url: str, pick: List[float], place: List[float], dry_run: bool) -> int:
    pick = clamp_workspace(pick)
    place = clamp_workspace(place)
    print("[pick_place] pick:", pick)
    print("[pick_place] place:", place)
    if dry_run:
        print("[pick_place] dry-run enabled; not sending motion.")
        return 0

    cmds = [
        {"action": "watch"},
        {"action": "pick_at", "coords": pick, "gripper_open": 80, "gripper_close": 10, "approach": "top"},
        {"action": "place_at", "coords": place, "gripper_open": 80, "approach": "top"},
        {"action": "home"},
    ]
    for c in cmds:
        print("[pick_place] ->", c["action"])
        out = post(base_url, c, timeout=45.0)
        print("   ok=", out.get("ok"), "elapsed_ms=", out.get("elapsed_ms"), "err=", out.get("error"))
        if not out.get("ok"):
            return 3
    return 0


def parse_coord(arg: str) -> List[float]:
    vals = [float(x.strip()) for x in arg.split(",")]
    if len(vals) != 6:
        raise ValueError("coord must be 6 comma-separated values: x,y,z,rx,ry,rz")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="Yahboom arm bridge over Jetson API")
    ap.add_argument("--host", default="192.168.55.1", help="Jetson host/IP")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--mode", choices=["smoke", "pick_place"], default="smoke")
    ap.add_argument("--pick", type=parse_coord, default=parse_coord("120,0,90,-175,0,-45"))
    ap.add_argument("--place", type=parse_coord, default=parse_coord("180,60,95,-175,0,-45"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print("[bridge] base_url:", base_url)

    # quick connectivity preflight
    try:
        st = post(base_url, {"action": "status"}, timeout=8.0)
        print("[bridge] status preflight ok:", bool(st.get("ok")))
    except Exception as e:
        print("[bridge] preflight failed:", e)
        return 1

    if args.mode == "smoke":
        return run_smoke(base_url)

    if args.mode == "pick_place":
        return run_pick_place(base_url, args.pick, args.place, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
