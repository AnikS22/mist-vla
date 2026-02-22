#!/usr/bin/env python3
"""Quick inspection: feature dims and format across all collected data."""
import pickle, os, glob

def inspect(path, label):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"{label}: FILE NOT FOUND"); return
    rollouts = data if isinstance(data, list) else data.get("rollouts", [])
    if not rollouts:
        print(f"{label}: EMPTY"); return
    r = rollouts[0]
    print(f"{label}: {len(rollouts)} rollouts")
    print(f"  Keys: {sorted(r.keys())}")
    for key in ["features", "hidden_states", "embeddings", "latent"]:
        if key in r:
            val = r[key]
            if isinstance(val, list) and len(val) > 0:
                v0 = val[0]
                if hasattr(v0, "shape"):
                    print(f"  {key}: list[{len(val)}] x shape {v0.shape} dtype={v0.dtype}")
                else:
                    print(f"  {key}: list[{len(val)}] x type {type(v0).__name__}")
            elif hasattr(val, "shape"):
                print(f"  {key}: shape {val.shape} dtype={val.dtype}")
    for key in ["eef_pos", "robot_states", "actions"]:
        if key in r:
            val = r[key]
            if isinstance(val, list) and len(val) > 0:
                v0 = val[0]
                if hasattr(v0, "shape"):
                    print(f"  {key}: list[{len(val)}] x shape {v0.shape}")
                elif isinstance(v0, dict):
                    print(f"  {key}: list[{len(val)}] x dict keys={list(v0.keys())[:5]}")
                else:
                    print(f"  {key}: list[{len(val)}] x type {type(v0).__name__}")
            elif hasattr(val, "shape"):
                print(f"  {key}: shape {val.shape}")
    print(f"  success: {r.get('success', 'N/A')}")
    print(f"  task_id: {r.get('task_id', 'N/A')}")
    # Check robot_states for eef_pos
    if "robot_states" in r:
        rs = r["robot_states"]
        if isinstance(rs, list) and len(rs) > 0 and isinstance(rs[0], dict):
            print(f"  robot_states[0] keys: {list(rs[0].keys())[:8]}")
            if "eef_pos" in rs[0]:
                import numpy as np
                print(f"  robot_states[0]['eef_pos']: {np.array(rs[0]['eef_pos'])}")
    print()

base = os.environ.get("DATA_ROOT", "/mnt/onefs/home/asahai2024/mist-vla/data")

pairs = [
    ("ACT", "multi_model/act_spatial"),
    ("DP", "multi_model/dp_spatial"),
    ("OpenVLA-OFT spatial", "multi_model/openvla_oft_allsuite__libero_spatial"),
    ("OpenVLA-OFT object", "multi_model/openvla_oft__libero_object"),
    ("Combined (original)", "combined"),
]

for label, subdir in pairs:
    d = os.path.join(base, subdir)
    for name in ["success_rollouts.pkl", "failure_rollouts.pkl",
                  "success_rollouts_partial.pkl", "failure_rollouts_partial.pkl"]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            inspect(p, f"{label} / {name}")
            break  # just inspect one file per source
