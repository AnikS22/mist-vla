"""
Merge multi-model data into a unified training dataset.

Merges rollouts from ALL collected models (OpenVLA-OFT, ACT, DP, Octo)
across LIBERO suites. Namespaces task_ids to prevent cross-suite/model
matching during training.

Original task_id=3 from ACT on libero_spatial → "act__libero_spatial__3"

Usage:
    python scripts/merge_multi_model_data.py --output data/merged_all
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def load_and_tag(directory, model_tag, suite_tag):
    """Load rollouts, tag with model/suite, and namespace task_ids.

    Namespace format: "{model_tag}__{suite_tag}__{original_task_id}"
    This ensures task_ids are unique across models AND suites.
    """
    d = Path(directory)
    successes, failures = [], []

    for name, container in [("success_rollouts.pkl", successes),
                            ("failure_rollouts.pkl", failures)]:
        path = d / name
        if path.exists():
            with open(path, "rb") as f:
                rollouts = pickle.load(f)
            for r in rollouts:
                r["model_tag"] = model_tag
                r["suite_tag"] = suite_tag
                r["source_dir"] = str(directory)
                # Namespace task_id to prevent cross-model/suite matching
                raw_tid = r.get("task_id", 0)
                r["original_task_id"] = raw_tid
                r["task_id"] = f"{model_tag}__{suite_tag}__{raw_tid}"
            container.extend(rollouts)
        else:
            print(f"  ⚠ {path} not found, skipping")

    return successes, failures


def deduplicate_rollouts(rollouts):
    """Remove exact duplicates based on (instruction, n_features, success)."""
    seen = set()
    unique = []
    for r in rollouts:
        steps_val = r.get("steps", 0)
        if isinstance(steps_val, (list, dict)):
            steps_val = len(steps_val) if isinstance(steps_val, list) else 0
        fp = (
            r.get("instruction", "")[:50],
            int(steps_val),
            bool(r.get("success", False)),
            len(r.get("features", [])),
        )
        if fp not in seen:
            seen.add(fp)
            unique.append(r)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/merged_all",
                        help="Output directory for merged data")
    parser.add_argument("--deduplicate", action="store_true",
                        help="Remove duplicate rollouts (same instruction+steps)")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_successes = []
    all_failures = []

    # ─── All existing data sources ───
    # Each tuple: (directory, model_tag, suite_tag)
    # model_tag is used for:
    #   1. Namespacing task_ids
    #   2. Selecting per-model input projection in the universal MLP
    #   3. Paper table rows
    sources = [
        # ── OpenVLA-OFT (4096-dim features) ──
        ("data/combined",                                         "openvla-oft", "libero_spatial"),
        ("data/multi_model/openvla_oft_allsuite__libero_spatial", "openvla-oft", "libero_spatial"),
        ("data/multi_model/openvla_oft__libero_object",           "openvla-oft", "libero_object"),
        # ── ACT (256-dim features) ──
        ("data/multi_model/act_spatial",                          "act",         "libero_spatial"),
        # ── Diffusion Policy (256-dim features) ──
        ("data/multi_model/dp_spatial",                           "dp",          "libero_spatial"),
        # ── Octo (if/when collected) ──
        ("data/multi_model/octo_spatial",                         "octo-base",   "libero_spatial"),
    ]

    print("=" * 70)
    print("MERGING ALL DATA — Namespaced Task IDs")
    print("=" * 70)

    for src_dir, model, suite in sources:
        if not Path(src_dir).exists():
            continue
        s, f = load_and_tag(src_dir, model, suite)
        if s or f:
            print(f"  {src_dir:55s} → {len(s):4d}S + {len(f):4d}F  [{model}/{suite}]")
            all_successes.extend(s)
            all_failures.extend(f)

    # ─── Dedup if requested ───
    if args.deduplicate:
        n_before_s, n_before_f = len(all_successes), len(all_failures)
        all_successes = deduplicate_rollouts(all_successes)
        all_failures = deduplicate_rollouts(all_failures)
        print(f"\n  Dedup: {n_before_s}S → {len(all_successes)}S  "
              f"{n_before_f}F → {len(all_failures)}F")

    total = len(all_successes) + len(all_failures)
    print(f"\n  TOTAL: {len(all_successes)}S + {len(all_failures)}F = {total} rollouts")

    # ─── Per-suite/model breakdown ───
    print("\n  Per-suite breakdown:")
    suite_counts = defaultdict(lambda: {"S": 0, "F": 0})
    for r in all_successes:
        suite_counts[r["suite_tag"]]["S"] += 1
    for r in all_failures:
        suite_counts[r["suite_tag"]]["F"] += 1
    for suite, c in sorted(suite_counts.items()):
        print(f"    {suite:20s}: {c['S']:4d}S + {c['F']:4d}F = {c['S']+c['F']}")

    print("\n  Per-model breakdown:")
    model_counts = defaultdict(lambda: {"S": 0, "F": 0})
    for r in all_successes:
        model_counts[r["model_tag"]]["S"] += 1
    for r in all_failures:
        model_counts[r["model_tag"]]["F"] += 1
    for model, c in sorted(model_counts.items()):
        print(f"    {model:25s}: {c['S']:4d}S + {c['F']:4d}F = {c['S']+c['F']}")

    # ─── Namespaced task_id summary ───
    all_task_ids = sorted(set(r["task_id"] for r in all_successes + all_failures))
    print(f"\n  Unique namespaced task_ids: {len(all_task_ids)}")
    for tid in all_task_ids:
        ns = sum(1 for r in all_successes if r["task_id"] == tid)
        nf = sum(1 for r in all_failures if r["task_id"] == tid)
        print(f"    {tid:35s}: {ns:3d}S + {nf:3d}F")

    # ─── Embedding dimensions ───
    print("\n  Embedding dimensions per model:")
    for model_tag in sorted(set(r.get("model_tag", "?") for r in all_successes + all_failures)):
        sample = next((r for r in all_successes + all_failures
                       if r.get("model_tag") == model_tag and len(r.get("features", [])) > 0), None)
        if sample:
            feat = sample["features"][0]
            dim = feat.shape[-1] if hasattr(feat, "shape") else len(feat)
            print(f"    {model_tag}: {dim}-dim")

    # ─── Save ───
    with open(out / "success_rollouts.pkl", "wb") as fp:
        pickle.dump(all_successes, fp)
    with open(out / "failure_rollouts.pkl", "wb") as fp:
        pickle.dump(all_failures, fp)

    meta = {
        "total_success": len(all_successes),
        "total_failure": len(all_failures),
        "total": total,
        "n_unique_tasks": len(all_task_ids),
        "task_ids": all_task_ids,
        "models": sorted(set(r.get("model_tag", "?") for r in all_successes + all_failures)),
        "suites": sorted(set(r.get("suite_tag", "?") for r in all_successes + all_failures)),
        "sources": [str(s[0]) for s in sources if Path(s[0]).exists()],
    }
    with open(out / "metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    print(f"\n  ✅ Saved to: {out}/")
    print(f"     success_rollouts.pkl  ({len(all_successes)} rollouts)")
    print(f"     failure_rollouts.pkl  ({len(all_failures)} rollouts)")
    print(f"     metadata.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
