#!/usr/bin/env python3
"""
Build a comprehensive data index of all research data for statistical analysis.

Scans research_data/ and produces:
  - research_data/DATA_INDEX.json  — machine-readable index of every file
  - research_data/DATA_INDEX.csv   — spreadsheet-friendly summary of rollouts

This allows instant lookup of:
  - What models have data
  - How many success/failure rollouts per model per task
  - Feature dimensions
  - MLP training results
  - Evaluation results
"""

import json
import os
import pickle
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent / "research_data"


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


def inspect_pkl(path):
    """Load a pickle file and extract summary statistics."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return {"error": str(e)}

    if isinstance(data, list):
        n = len(data)
        if n == 0:
            return {"count": 0}

        info = {"count": n}

        # Check first rollout for structure
        sample = data[0]
        if isinstance(sample, dict):
            info["keys"] = list(sample.keys())

            # Feature dimension
            if "features" in sample and len(sample["features"]) > 0:
                feat = sample["features"][0]
                if hasattr(feat, "shape"):
                    info["feature_dim"] = list(feat.shape)
                elif hasattr(feat, "__len__"):
                    info["feature_dim"] = len(feat)

            # Steps count
            if "steps" in sample:
                steps = sample["steps"]
                if isinstance(steps, list):
                    step_counts = [len(r.get("steps", [])) for r in data
                                   if isinstance(r, dict)]
                    if step_counts:
                        info["avg_steps"] = round(sum(step_counts) / len(step_counts), 1)
                        info["min_steps"] = min(step_counts)
                        info["max_steps"] = max(step_counts)

            # Task distribution
            if "task_id" in sample:
                from collections import Counter
                task_counts = Counter(r.get("task_id") for r in data
                                      if isinstance(r, dict))
                info["tasks"] = dict(sorted(task_counts.items()))

            # Success distribution
            if "success" in sample:
                n_success = sum(1 for r in data
                                if isinstance(r, dict) and r.get("success"))
                info["n_success"] = n_success
                info["n_failure"] = n - n_success

        return info
    elif isinstance(data, dict):
        return {"type": "dict", "keys": list(data.keys()), "count": len(data)}
    else:
        return {"type": type(data).__name__}


def scan_rollouts(rollout_dir):
    """Scan all rollout directories and build rollout summaries."""
    entries = []

    for pkl_path in sorted(rollout_dir.rglob("*.pkl")):
        rel = pkl_path.relative_to(ROOT)
        size = pkl_path.stat().st_size

        # Skip partial files for the index summary
        if "partial" in pkl_path.name:
            continue

        info = inspect_pkl(pkl_path)
        entry = {
            "path": str(rel),
            "filename": pkl_path.name,
            "size_bytes": size,
            "size_human": sizeof_fmt(size),
            "parent_dir": str(pkl_path.parent.relative_to(ROOT)),
        }
        entry.update(info)
        entries.append(entry)

    return entries


def scan_checkpoints(ckpt_dir):
    """Scan checkpoint directory for models and results."""
    entries = []

    for json_path in sorted(ckpt_dir.rglob("*.json")):
        rel = json_path.relative_to(ROOT)
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            data = {"error": "failed to parse"}

        entries.append({
            "path": str(rel),
            "parent": str(json_path.parent.relative_to(ROOT)),
            "data": data,
        })

    for pt_path in sorted(ckpt_dir.rglob("*.pt")):
        rel = pt_path.relative_to(ROOT)
        entries.append({
            "path": str(rel),
            "parent": str(pt_path.parent.relative_to(ROOT)),
            "size_bytes": pt_path.stat().st_size,
            "size_human": sizeof_fmt(pt_path.stat().st_size),
        })

    return entries


def scan_results(results_dir):
    """Scan evaluation results."""
    entries = []

    for json_path in sorted(results_dir.rglob("*.json")):
        rel = json_path.relative_to(ROOT)
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            data = {"error": "failed to parse"}

        entries.append({
            "path": str(rel),
            "data": data,
        })

    return entries


def build_rollout_csv(rollout_entries, out_path):
    """Build a CSV summary of all rollout data."""
    rows = []
    for entry in rollout_entries:
        if entry.get("count", 0) == 0:
            continue

        # Determine model name from path
        parent = entry.get("parent_dir", "")
        if "act_spatial" in parent:
            model = "ACT"
        elif "dp_spatial" in parent:
            model = "Diffusion Policy"
        elif "octo_spatial" in parent:
            model = "Octo-Base"
        elif "openvla_oft" in parent:
            model = "OpenVLA-OFT"
        elif "openvla_spatial" in parent or "seed0" in parent:
            model = "OpenVLA-OFT (spatial)"
        elif "merged" in parent:
            model = "Merged (all models)"
        else:
            model = parent.split("/")[-1] if "/" in parent else parent

        # Determine type from filename
        is_success = "success" in entry["filename"]

        row = {
            "model": model,
            "type": "success" if is_success else "failure",
            "count": entry.get("count", 0),
            "feature_dim": str(entry.get("feature_dim", "?")),
            "avg_steps": entry.get("avg_steps", ""),
            "size": entry.get("size_human", ""),
            "path": entry.get("path", ""),
            "tasks": json.dumps(entry.get("tasks", {})),
        }
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "type", "count", "feature_dim",
                           "avg_steps", "size", "path", "tasks"])
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (x["model"], x["type"])):
            writer.writerow(row)


def main():
    print("=" * 70)
    print("  BUILDING COMPREHENSIVE DATA INDEX")
    print("=" * 70)
    print(f"  Root: {ROOT}")
    print(f"  Time: {datetime.now().isoformat()}")
    print()

    index = {
        "generated_at": datetime.now().isoformat(),
        "root": str(ROOT),
    }

    # 1. Scan rollouts
    rollout_dir = ROOT / "rollouts"
    if rollout_dir.exists():
        print("[1/3] Scanning rollout data...", flush=True)
        rollout_entries = scan_rollouts(rollout_dir)
        index["rollouts"] = rollout_entries
        print(f"  Found {len(rollout_entries)} rollout files", flush=True)

        # Build CSV
        csv_path = ROOT / "DATA_INDEX.csv"
        build_rollout_csv(rollout_entries, csv_path)
        print(f"  CSV: {csv_path}", flush=True)
    else:
        rollout_entries = []
        print("[1/3] No rollout directory found", flush=True)

    # 2. Scan checkpoints
    ckpt_dir = ROOT / "checkpoints"
    if ckpt_dir.exists():
        print("[2/3] Scanning checkpoints...", flush=True)
        ckpt_entries = scan_checkpoints(ckpt_dir)
        index["checkpoints"] = ckpt_entries
        print(f"  Found {len(ckpt_entries)} checkpoint files", flush=True)
    else:
        print("[2/3] No checkpoint directory found", flush=True)

    # 3. Scan results
    results_dir = ROOT / "results"
    if results_dir.exists():
        print("[3/3] Scanning evaluation results...", flush=True)
        result_entries = scan_results(results_dir)
        index["evaluation_results"] = result_entries
        print(f"  Found {len(result_entries)} result files", flush=True)
    else:
        print("[3/3] No results directory found", flush=True)

    # Summary
    print()
    print("=" * 70)
    print("  DATA SUMMARY")
    print("=" * 70)

    # Count by model
    model_counts = {}
    for entry in rollout_entries:
        parent = entry.get("parent_dir", "unknown")
        key = parent.split("/")[-1] if "/" in parent else parent
        if key not in model_counts:
            model_counts[key] = {"success": 0, "failure": 0, "feature_dim": "?"}
        if "success" in entry.get("filename", ""):
            model_counts[key]["success"] = entry.get("count", 0)
        else:
            model_counts[key]["failure"] = entry.get("count", 0)
        if entry.get("feature_dim"):
            model_counts[key]["feature_dim"] = entry["feature_dim"]

    for model, counts in sorted(model_counts.items()):
        s, f = counts["success"], counts["failure"]
        total = s + f
        sr = (s / total * 100) if total > 0 else 0
        print(f"  {model:45s}  {s:>5}S + {f:>5}F = {total:>5}  "
              f"(SR: {sr:.1f}%)  feat_dim={counts['feature_dim']}")

    # Save index
    json_path = ROOT / "DATA_INDEX.json"
    with open(json_path, "w") as f:
        json.dump(index, f, indent=2, default=str)
    print(f"\n  Full index: {json_path}")
    print(f"  CSV summary: {ROOT / 'DATA_INDEX.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
