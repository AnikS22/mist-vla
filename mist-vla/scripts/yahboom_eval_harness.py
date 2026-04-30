#!/usr/bin/env python3
"""Paper-style evaluation harness for Yahboom real-robot studies.

Runs multiple methods over repeated trials with fixed reset protocol,
collects per-trial outcomes, and reports success rates with Wilson CIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_LOOP = REPO_ROOT / "scripts" / "run_model_yahboom_loop.py"
DEFAULT_HOME = [130.0, -25.0, 350.0, -92.0, -45.0, -87.0]


@dataclass
class MethodSpec:
    name: str
    extra_args: List[str]


def api_get_json(base: str, path: str, timeout_s: float = 8.0) -> Dict[str, Any]:
    with request.urlopen(f"{base}{path}", timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_post_json(base: str, payload: Dict[str, Any], timeout_s: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base}/action",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_get_bytes(base: str, path: str, timeout_s: float = 8.0) -> bytes:
    with request.urlopen(f"{base}{path}", timeout=timeout_s) as resp:
        return resp.read()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_coords(coords: List[float]) -> List[float]:
    x, y, z, rx, ry, rz = coords
    return [
        clamp(float(x), -220.0, 220.0),
        clamp(float(y), -220.0, 220.0),
        clamp(float(z), 40.0, 450.0),
        clamp(float(rx), -180.0, 180.0),
        clamp(float(ry), -180.0, 180.0),
        clamp(float(rz), -180.0, 180.0),
    ]


def reset_robot(base: str, home: List[float], speed: int, wait: bool) -> Dict[str, Any]:
    return api_post_json(
        base,
        {
            "action": "move_to",
            "coords": clamp_coords(home),
            "speed": int(speed),
            "wait": bool(wait),
        },
        timeout_s=45.0,
    )


def coord_l2_mm(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if not (isinstance(a, list) and isinstance(b, list) and len(a) >= 3 and len(b) >= 3):
        return None
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def get_git_commit() -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return "unknown"


def build_trial_schedule(methods: List[MethodSpec], trials_per_method: int, order_mode: str, rng: random.Random):
    items: List[Tuple[str, int]] = []
    if order_mode == "grouped":
        for m in methods:
            for rep in range(trials_per_method):
                items.append((m.name, rep))
        return items
    if order_mode == "interleaved":
        for rep in range(trials_per_method):
            round_methods = [m.name for m in methods]
            rng.shuffle(round_methods)
            for name in round_methods:
                items.append((name, rep))
        return items
    # random global order
    for m in methods:
        for rep in range(trials_per_method):
            items.append((m.name, rep))
    rng.shuffle(items)
    return items


def prompt_manual_label(label_name: str, trial_idx: int, rc: int) -> Optional[bool]:
    while True:
        raw = input(
            f"[label] method={label_name} trial={trial_idx} rc={rc} => success? [y/n/s(skip)]: "
        ).strip().lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        if raw in {"s", "skip", ""}:
            return None
        print("  enter y, n, or s")


def save_trial_snapshot(base: str, path: Path) -> bool:
    try:
        img = api_get_bytes(base, "/snapshot", timeout_s=8.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(img)
        return True
    except Exception:
        return False


def build_method_specs(args: argparse.Namespace) -> List[MethodSpec]:
    presets: Dict[str, MethodSpec] = {
        "raw_unsafe": MethodSpec(
            name="raw_unsafe",
            extra_args=[
                "--unsafe-no-constraints",
                "--disable-steering",
                "--skip-redundant-gripper",
            ],
        ),
        "vanilla_constrained": MethodSpec(
            name="vanilla_constrained",
            extra_args=[
                "--disable-steering",
                "--skip-redundant-gripper",
            ],
        ),
        "pick_profile": MethodSpec(
            name="pick_profile",
            extra_args=[
                "--task-profile",
                "pick_yellow_cube",
                "--disable-steering",
                "--wait-for-motion",
                "--skip-redundant-gripper",
            ],
        ),
    }

    names = [x.strip() for x in args.methods.split(",") if x.strip()]
    if not names:
        raise ValueError("No methods selected")
    out = []
    for name in names:
        if name not in presets:
            raise ValueError(f"Unknown method '{name}'. Available: {', '.join(sorted(presets.keys()))}")
        out.append(presets[name])
    return out


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def auto_label_lift(trial: Dict[str, Any], lift_threshold_mm: float, grasp_z_max: float) -> Optional[bool]:
    start = trial.get("start_coords")
    end = trial.get("end_coords")
    if not (isinstance(start, list) and isinstance(end, list) and len(start) >= 3 and len(end) >= 3):
        return None
    # Heuristic for tabletop pick tasks:
    # - robot reached near tabletop (start/end below grasp_z_max),
    # - then ended higher by at least lift_threshold_mm.
    start_z = float(start[2])
    end_z = float(end[2])
    if min(start_z, end_z) > float(grasp_z_max):
        return False
    return (end_z - start_z) >= float(lift_threshold_mm)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Yahboom paper-style evaluation harness")
    ap.add_argument("--jetson-host", default="192.168.55.1")
    ap.add_argument("--jetson-port", type=int, default=5000)
    ap.add_argument("--policy", default="openvla", choices=["openvla", "openvla_oft", "smolvla", "random"])
    ap.add_argument("--model-name", default="openvla/openvla-7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--instruction", default="pick up yellow cube")
    ap.add_argument("--methods", default="raw_unsafe,vanilla_constrained,pick_profile")
    ap.add_argument("--trials-per-method", type=int, default=20)
    ap.add_argument("--order-mode", choices=["grouped", "interleaved", "random"], default="interleaved")
    ap.add_argument("--blind-labels", action="store_true", help="hide method identities during manual labeling")
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--speed", type=int, default=30)
    ap.add_argument("--xyz-gain", type=float, default=0.2)
    ap.add_argument("--rot-gain", type=float, default=0.3)
    ap.add_argument("--reset-home", default="130,-25,350,-92,-45,-87", help="x,y,z,rx,ry,rz")
    ap.add_argument("--reset-speed", type=int, default=20)
    ap.add_argument("--reset-wait", action="store_true", help="block until reset move completes")
    ap.add_argument("--reset-max-retries", type=int, default=3)
    ap.add_argument("--reset-tol-mm", type=float, default=8.0, help="max allowed L2 distance from home after reset")
    ap.add_argument("--label-mode", choices=["manual", "auto_lift"], default="manual")
    ap.add_argument("--lift-threshold-mm", type=float, default=18.0, help="auto_lift success threshold")
    ap.add_argument("--grasp-z-max", type=float, default=345.0, help="auto_lift expected grasp-zone z bound")
    ap.add_argument("--inter-trial-sleep", type=float, default=0.5)
    ap.add_argument("--capture-snapshots", action="store_true", help="save pre/post trial camera snapshots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", default="mist-vla/research_data/results/yahboom_eval")
    args = ap.parse_args()

    base = f"http://{args.jetson_host}:{args.jetson_port}"
    methods = build_method_specs(args)
    method_by_name = {m.name: m for m in methods}
    rng = random.Random(args.seed)

    try:
        home = [float(x.strip()) for x in args.reset_home.split(",") if x.strip()]
        if len(home) != 6:
            raise ValueError("reset-home must have 6 comma-separated values")
    except Exception as e:
        raise ValueError(f"invalid --reset-home: {e}") from e

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.save_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": stamp,
        "jetson_host": args.jetson_host,
        "jetson_port": args.jetson_port,
        "policy": args.policy,
        "model_name": args.model_name,
        "instruction": args.instruction,
        "methods": [m.name for m in methods],
        "trials_per_method": args.trials_per_method,
        "order_mode": args.order_mode,
        "blind_labels": bool(args.blind_labels),
        "label_mode": args.label_mode,
        "steps": args.steps,
        "dt": args.dt,
        "speed": args.speed,
        "xyz_gain": args.xyz_gain,
        "rot_gain": args.rot_gain,
        "reset_home": home,
        "reset_max_retries": args.reset_max_retries,
        "reset_tol_mm": args.reset_tol_mm,
        "capture_snapshots": bool(args.capture_snapshots),
        "seed": args.seed,
        "git_commit": get_git_commit(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cwd": str(REPO_ROOT.parent),
        "env_conda_prefix": os.environ.get("CONDA_PREFIX", ""),
    }
    config_blob = json.dumps(config, sort_keys=True)
    config["config_sha256"] = hashlib.sha256(config_blob.encode("utf-8")).hexdigest()
    write_json(out_dir / "config.json", config)

    trials: List[Dict[str, Any]] = []
    schedule = build_trial_schedule(methods, args.trials_per_method, args.order_mode, rng)
    method_alias = {m.name: f"M{i+1}" for i, m in enumerate(methods)}
    write_json(out_dir / "schedule.json", [{"method": m, "rep": r + 1} for m, r in schedule])

    print("\n=== Evaluation Schedule ===")
    print(", ".join([f"{method_alias[m]}:{r+1}" for m, r in schedule]))

    trial_counter = 0
    for method_name, rep_idx in schedule:
        trial_counter += 1
        method = method_by_name[method_name]
        print(f"\n[trial {trial_counter}] method={method.name} rep={rep_idx+1}/{args.trials_per_method}")

        # Reset protocol with tolerance/retries.
        reset_attempts: List[Dict[str, Any]] = []
        reset_ok = False
        reset_resp: Dict[str, Any] = {"ok": False}
        start_coords = None
        for reset_try in range(args.reset_max_retries):
            reset_resp = reset_robot(base, home, speed=args.reset_speed, wait=args.reset_wait)
            st = api_get_json(base, "/status")
            coords = st.get("final_coords") or st.get("coords")
            drift = coord_l2_mm(coords, home)
            attempt_info = {
                "attempt": reset_try + 1,
                "reset_resp": reset_resp,
                "coords": coords,
                "drift_mm": drift,
            }
            reset_attempts.append(attempt_info)
            if bool(reset_resp.get("ok")) and (drift is None or drift <= float(args.reset_tol_mm)):
                reset_ok = True
                start_coords = coords
                break
            time.sleep(0.2)
        if start_coords is None:
            st = api_get_json(base, "/status")
            start_coords = st.get("final_coords") or st.get("coords")

        trial_log = out_dir / "trials" / method.name / f"trial_{rep_idx+1:03d}.json"
        trial_root = out_dir / "trials" / method.name
        pre_img = trial_root / f"trial_{rep_idx+1:03d}.pre.jpg"
        post_img = trial_root / f"trial_{rep_idx+1:03d}.post.jpg"
        if args.capture_snapshots:
            _ = save_trial_snapshot(base, pre_img)
            trial_log.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(RUN_LOOP),
            "--policy",
            args.policy,
            "--model-name",
            args.model_name,
            "--device",
            args.device,
            "--device-map",
            args.device_map,
            "--instruction",
            args.instruction,
            "--steps",
            str(args.steps),
            "--dt",
            str(args.dt),
            "--jetson-host",
            args.jetson_host,
            "--jetson-port",
            str(args.jetson_port),
            "--action-space",
            "meters",
            "--xyz-gain",
            str(args.xyz_gain),
            "--rot-gain",
            str(args.rot_gain),
            "--speed",
            str(args.speed),
            "--execute",
            "--seed",
            str(args.seed + rep_idx),
            "--log-json",
            str(trial_log),
        ] + method.extra_args

        t0 = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT.parent),
            capture_output=True,
            text=True,
        )
        elapsed_s = time.time() - t0

        end_st = api_get_json(base, "/status")
        end_coords = end_st.get("final_coords") or end_st.get("coords")
        if args.capture_snapshots:
            _ = save_trial_snapshot(base, post_img)

        stdout_path = out_dir / "trials" / method.name / f"trial_{rep_idx+1:03d}.stdout.txt"
        stderr_path = out_dir / "trials" / method.name / f"trial_{rep_idx+1:03d}.stderr.txt"
        stdout_path.write_text(proc.stdout or "")
        stderr_path.write_text(proc.stderr or "")

        trial = {
            "global_trial_index": trial_counter,
            "method": method.name,
            "method_alias": method_alias[method.name],
            "trial_index": rep_idx + 1,
            "reset_ok": bool(reset_ok),
            "reset_attempts": reset_attempts,
            "start_coords": start_coords,
            "end_coords": end_coords,
            "return_code": int(proc.returncode),
            "elapsed_s": float(elapsed_s),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "run_log_path": str(trial_log),
            "snapshot_pre_path": str(pre_img) if args.capture_snapshots else "",
            "snapshot_post_path": str(post_img) if args.capture_snapshots else "",
            "label_mode": args.label_mode,
            "success": None,
        }

        if args.label_mode == "manual":
            label_name = method_alias[method.name] if args.blind_labels else method.name
            trial["success"] = prompt_manual_label(label_name, rep_idx + 1, proc.returncode)
        else:
            trial["success"] = auto_label_lift(
                trial,
                lift_threshold_mm=args.lift_threshold_mm,
                grasp_z_max=args.grasp_z_max,
            )

        trials.append(trial)
        print(
            f"  rc={trial['return_code']} elapsed={trial['elapsed_s']:.2f}s "
            f"success={trial['success']} reset_ok={trial['reset_ok']} "
            f"start={trial['start_coords']} end={trial['end_coords']}"
        )
        time.sleep(max(0.0, float(args.inter_trial_sleep)))

    # Aggregate.
    summary_rows: List[Dict[str, Any]] = []
    for m in methods:
        mt = [t for t in trials if t["method"] == m.name]
        labeled = [t for t in mt if isinstance(t.get("success"), bool)]
        n = len(labeled)
        s = sum(1 for t in labeled if t["success"] is True)
        p = (s / n) if n > 0 else 0.0
        lo, hi = wilson_ci(s, n) if n > 0 else (0.0, 0.0)
        mean_elapsed = sum(float(t["elapsed_s"]) for t in mt) / max(len(mt), 1)
        row = {
            "method": m.name,
            "trials_total": len(mt),
            "trials_labeled": n,
            "successes": s,
            "success_rate": p,
            "success_rate_pct": 100.0 * p,
            "wilson_low_pct": 100.0 * lo,
            "wilson_high_pct": 100.0 * hi,
            "mean_trial_seconds": mean_elapsed,
        }
        summary_rows.append(row)

    # Write outputs.
    write_json(out_dir / "trials.json", trials)
    write_json(
        out_dir / "summary.json",
        {
            "config": config,
            "method_alias": method_alias,
            "summary": summary_rows,
        },
    )

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "trials_total",
                "trials_labeled",
                "successes",
                "success_rate",
                "success_rate_pct",
                "wilson_low_pct",
                "wilson_high_pct",
                "mean_trial_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Summary ===")
    for r in summary_rows:
        print(
            f"{r['method']}: {r['successes']}/{r['trials_labeled']} "
            f"({r['success_rate_pct']:.1f}% | CI {r['wilson_low_pct']:.1f}-{r['wilson_high_pct']:.1f}) "
            f"mean_trial={r['mean_trial_seconds']:.2f}s"
        )
    print(f"\nSaved results to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

