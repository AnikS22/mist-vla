"""
Organize and inventory ALL collected data across models and suites.

This script:
1. Scans all data directories
2. Tags rollouts with model/suite metadata
3. Produces a comprehensive inventory
4. Optionally merges into unified training datasets

Directory structure (standardized):
    data/
    ├── by_model/                          ← Organized by model
    │   ├── openvla_oft/
    │   │   ├── libero_spatial/            ← 724+ rollouts (original + topup)
    │   │   ├── libero_object/             ← 188 old + new collection
    │   │   ├── libero_goal/               ← 203 old + new collection
    │   │   └── libero_10/                 ← 87 old + new collection
    │   ├── openvla_oft_allsuite/
    │   │   └── libero_spatial/            ← New: combined-model on spatial
    │   └── octo_base/
    │       └── libero_spatial/            ← New: JAX diffusion model
    ├── combined/                          ← Original 724 rollouts (spatial only)
    ├── multi_suite/                       ← Old collection (246+188+203+87)
    ├── multi_model/                       ← New collection (in progress)
    └── merged_all/                        ← Final merged dataset for training

Usage:
    python scripts/organize_data.py            # Inventory only
    python scripts/organize_data.py --merge    # Merge all into data/merged_all/
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import sys


def load_pkl_safe(path):
    """Load a pickle file safely."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  ⚠ Failed to load {path}: {e}")
        return []


def inspect_rollouts(rollouts, label=""):
    """Get stats about a list of rollouts."""
    if not rollouts:
        return {"count": 0, "label": label}

    n_success = sum(1 for r in rollouts if r.get("success", False))
    n_failure = len(rollouts) - n_success
    tasks = set(r.get("task_id", -1) for r in rollouts)

    # Check feature dimension
    feat_dim = 0
    for r in rollouts[:3]:
        feats = r.get("features", [])
        if feats and len(feats) > 0:
            f0 = feats[0]
            feat_dim = f0.shape[-1] if hasattr(f0, "shape") else len(f0)
            break

    # Check EEF availability
    has_eef = any("robot_states" in r and len(r.get("robot_states", [])) > 0
                   for r in rollouts[:3])

    def _get_steps(r):
        s = r.get("steps", 0)
        if isinstance(s, (int, float)):
            return int(s)
        feats = r.get("features", [])
        return len(feats) if isinstance(feats, list) else 0
    total_steps = sum(_get_steps(r) for r in rollouts)

    return {
        "label": label,
        "count": len(rollouts),
        "success": n_success,
        "failure": n_failure,
        "tasks": sorted(tasks),
        "n_tasks": len(tasks),
        "feat_dim": feat_dim,
        "has_eef": has_eef,
        "total_steps": total_steps,
    }


def scan_directory(base_dir):
    """Scan a directory for rollout pickle files."""
    results = []
    base = Path(base_dir)
    if not base.exists():
        return results

    for pkl in sorted(base.rglob("*.pkl")):
        if "partial" in pkl.name:
            continue  # Skip partial checkpoints
        results.append(pkl)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data",
                        help="Root data directory")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all data into data/merged_all/")
    parser.add_argument("--output", default="data/DATA_INVENTORY.json")
    args = parser.parse_args()

    root = Path(args.data_root)

    print("=" * 75)
    print("  DATA INVENTORY — Universal Latent Safety Steering")
    print("=" * 75)
    print()

    # ─── Define all known data sources ───
    sources = [
        # (path, model_tag, suite_tag, description)
        ("combined",
         "openvla-oft", "libero_spatial", "Original 724-rollout dataset"),

        ("multi_suite/libero_spatial",
         "openvla-oft", "libero_spatial", "Earlier spatial collection"),
        ("multi_suite/libero_object",
         "openvla-oft", "libero_object", "Earlier object collection"),
        ("multi_suite/libero_goal",
         "openvla-oft", "libero_goal", "Earlier goal collection"),
        ("multi_suite/libero_10",
         "openvla-oft", "libero_10", "Earlier LIBERO-10 collection"),

        ("multi_model/openvla_oft__libero_object",
         "openvla-oft", "libero_object", "New A100 collection"),
        ("multi_model/openvla_oft__libero_goal",
         "openvla-oft", "libero_goal", "New A100 collection"),
        ("multi_model/openvla_oft__libero_10",
         "openvla-oft", "libero_10", "New A100 collection"),
        ("multi_model/openvla_oft__libero_spatial_topup",
         "openvla-oft", "libero_spatial", "Spatial top-up (A100)"),
        ("multi_model/openvla_oft_allsuite__libero_spatial",
         "openvla-oft-allsuite", "libero_spatial", "All-suite model on spatial"),
        ("multi_model/octo_spatial",
         "octo-base", "libero_spatial", "Octo diffusion model"),
    ]

    inventory = []
    all_rollouts_by_model_suite = defaultdict(list)

    for rel_path, model_tag, suite_tag, description in sources:
        d = root / rel_path
        if not d.exists():
            continue

        s_path = d / "success_rollouts.pkl"
        f_path = d / "failure_rollouts.pkl"

        successes = load_pkl_safe(s_path) if s_path.exists() else []
        failures = load_pkl_safe(f_path) if f_path.exists() else []

        if not successes and not failures:
            continue

        # Tag rollouts
        for r in successes + failures:
            r["_model_tag"] = model_tag
            r["_suite_tag"] = suite_tag
            r["_source"] = str(rel_path)

        stats = inspect_rollouts(successes + failures, label=rel_path)
        stats["model"] = model_tag
        stats["suite"] = suite_tag
        stats["description"] = description
        inventory.append(stats)

        key = (model_tag, suite_tag)
        all_rollouts_by_model_suite[key].extend(successes + failures)

        print(f"  📁 {rel_path}")
        print(f"     {stats['success']:4d}S + {stats['failure']:4d}F = "
              f"{stats['count']:5d}  |  {stats['n_tasks']} tasks  |  "
              f"dim={stats['feat_dim']}  |  {description}")

    print()

    # ─── Per-Model Summary ───
    print("─" * 75)
    print("  PER-MODEL TOTALS")
    print("─" * 75)

    model_totals = defaultdict(lambda: {"success": 0, "failure": 0, "total": 0, "suites": set()})
    for (model, suite), rollouts in all_rollouts_by_model_suite.items():
        n_s = sum(1 for r in rollouts if r.get("success", False))
        n_f = len(rollouts) - n_s
        model_totals[model]["success"] += n_s
        model_totals[model]["failure"] += n_f
        model_totals[model]["total"] += len(rollouts)
        model_totals[model]["suites"].add(suite)

    for model, t in sorted(model_totals.items()):
        suites_str = ", ".join(sorted(t["suites"]))
        print(f"  {model:30s}  {t['success']:5d}S + {t['failure']:5d}F = "
              f"{t['total']:5d}  |  suites: {suites_str}")

    grand_total = sum(t["total"] for t in model_totals.values())
    print(f"  {'GRAND TOTAL':30s}  {' ':>5s}   {' ':>5s}   {grand_total:5d}")
    print()

    # ─── Per-Suite Summary ───
    print("─" * 75)
    print("  PER-SUITE TOTALS")
    print("─" * 75)

    suite_totals = defaultdict(lambda: {"success": 0, "failure": 0, "total": 0, "models": set()})
    for (model, suite), rollouts in all_rollouts_by_model_suite.items():
        n_s = sum(1 for r in rollouts if r.get("success", False))
        n_f = len(rollouts) - n_s
        suite_totals[suite]["success"] += n_s
        suite_totals[suite]["failure"] += n_f
        suite_totals[suite]["total"] += len(rollouts)
        suite_totals[suite]["models"].add(model)

    for suite, t in sorted(suite_totals.items()):
        models_str = ", ".join(sorted(t["models"]))
        print(f"  {suite:25s}  {t['success']:5d}S + {t['failure']:5d}F = "
              f"{t['total']:5d}  |  models: {models_str}")
    print()

    # ─── Models Testing List ───
    print("─" * 75)
    print("  MODELS BEING TESTED")
    print("─" * 75)
    models_info = {
        "openvla-oft": {
            "full_name": "OpenVLA-7B-OFT",
            "type": "Autoregressive VLA (LLaMA-2 backbone)",
            "embedding_dim": 4096,
            "framework": "PyTorch / HuggingFace",
            "checkpoints": [
                "moojink/openvla-7b-oft-finetuned-libero-spatial",
                "moojink/openvla-7b-oft-finetuned-libero-object",
                "moojink/openvla-7b-oft-finetuned-libero-goal",
                "moojink/openvla-7b-oft-finetuned-libero-10",
            ]
        },
        "openvla-oft-allsuite": {
            "full_name": "OpenVLA-7B-OFT (All Suites)",
            "type": "Autoregressive VLA (LLaMA-2 backbone, multi-task)",
            "embedding_dim": 4096,
            "framework": "PyTorch / HuggingFace",
            "checkpoints": [
                "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
            ]
        },
        "octo-base": {
            "full_name": "Octo-Base-1.5",
            "type": "Diffusion Transformer VLA",
            "embedding_dim": 512,
            "framework": "JAX / Flax",
            "checkpoints": [
                "hf://rail-berkeley/octo-base-1.5",
            ]
        },
    }

    for i, (tag, info) in enumerate(models_info.items(), 1):
        status = "✅ DATA" if tag in model_totals and model_totals[tag]["total"] > 0 else "⏳ PENDING"
        count = model_totals.get(tag, {}).get("total", 0)
        print(f"  {i}. {info['full_name']:40s}  [{status}]  ({count} rollouts)")
        print(f"     Type: {info['type']}")
        print(f"     Embedding: {info['embedding_dim']}-dim  |  Framework: {info['framework']}")
        print(f"     Checkpoints: {', '.join(c.split('/')[-1] for c in info['checkpoints'])}")
        print()

    # ─── Save inventory ───
    output = {
        "timestamp": str(Path("/proc/version").read_text().strip()[:50]) if Path("/proc/version").exists() else "unknown",
        "grand_total": grand_total,
        "sources": [{k: (list(v) if isinstance(v, set) else v) for k, v in s.items()} for s in inventory],
        "model_totals": {k: {**v, "suites": list(v["suites"])} for k, v in model_totals.items()},
        "suite_totals": {k: {**v, "models": list(v["models"])} for k, v in suite_totals.items()},
        "models_info": models_info,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Inventory saved: {out_path}")

    # ─── Optional merge ───
    if args.merge:
        print()
        print("=" * 75)
        print("  MERGING ALL DATA → data/merged_all/")
        print("=" * 75)

        merged_dir = root / "merged_all"
        merged_dir.mkdir(parents=True, exist_ok=True)

        all_s = []
        all_f = []
        for (model, suite), rollouts in all_rollouts_by_model_suite.items():
            for r in rollouts:
                if r.get("success", False):
                    all_s.append(r)
                else:
                    all_f.append(r)

        with open(merged_dir / "success_rollouts.pkl", "wb") as f:
            pickle.dump(all_s, f)
        with open(merged_dir / "failure_rollouts.pkl", "wb") as f:
            pickle.dump(all_f, f)

        print(f"  Merged: {len(all_s)}S + {len(all_f)}F = {len(all_s)+len(all_f)} total")
        print(f"  Saved: {merged_dir}/")

    print()
    print("=" * 75)
    print("  DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()
