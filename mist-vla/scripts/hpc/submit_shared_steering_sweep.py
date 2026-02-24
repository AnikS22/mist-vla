#!/usr/bin/env python3
"""Submit paired OpenVLA+ACT shared-parameter steering sweeps on SLURM.

This script submits *paired* jobs so every parameter set is tested on both
models with the exact same steering policy knobs.
"""

import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _run(cmd: List[str]) -> str:
    out = subprocess.check_output(cmd, universal_newlines=True).strip()
    return out


def _submit(job_script: str, export_vars: Dict[str, str], dependency: Optional[str]) -> str:
    export_blob = ",".join([f"{k}={v}" for k, v in export_vars.items()])
    cmd = ["sbatch", "--parsable", f"--export=ALL,{export_blob}"]
    if dependency:
        cmd.append(f"--dependency=afterany:{dependency}")
    cmd.append(job_script)
    return _run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit shared steering sweep jobs (OpenVLA + ACT).")
    ap.add_argument("--n-trials", type=int, default=6, help="How many parameter sets to submit.")
    ap.add_argument("--episodes-per-task", type=int, default=16,
                    help="Episodes per task for sweep runs (use 20 for final).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-after-job", default="",
                    help="Optional job id; each sweep trial starts after this job completes.")
    ap.add_argument("--ledger", default="results/hpc/shared_steering_sweep_jobs.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    alphas = [0.10, 0.12, 0.15, 0.18]
    max_corrs = [0.0030, 0.0035, 0.0040]
    corr_thresholds = [0.0020, 0.0025, 0.0030]
    fail_thresholds = [0.55, 0.60, 0.65]
    ema_beta = 0.9

    for i in range(args.n_trials):
        alpha = random.choice(alphas)
        max_corr = random.choice(max_corrs)
        corr_th = random.choice(corr_thresholds)
        fail_th = random.choice(fail_thresholds)

        # Keep physically valid gate: correction threshold should not exceed clamp.
        if corr_th > max_corr:
            corr_th = max_corr

        run_tag = (
            f"sweep_{ts}_t{i+1:02d}_"
            f"a{str(alpha).replace('.', 'p')}_"
            f"mc{str(max_corr).replace('.', 'p')}_"
            f"ct{str(corr_th).replace('.', 'p')}_"
            f"ft{str(fail_th).replace('.', 'p')}"
        )

        exports = {
            "RUN_TAG": run_tag,
            "N_EPISODES": str(args.episodes_per_task),
            "ALPHA": str(alpha),
            "EMA_BETA": str(ema_beta),
            "MAX_CORR": str(max_corr),
            "CORR_THRESH": str(corr_th),
            "FAIL_THRESH": str(fail_th),
        }

        dep = args.start_after_job or None
        paper_job = _submit("scripts/hpc/eval_paper_table.slurm", exports, dep)
        act_job = _submit("scripts/hpc/eval_act_steering.slurm", exports, dep)

        rec = {
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "run_tag": run_tag,
            "params": exports,
            "paper_job_id": paper_job,
            "act_job_id": act_job,
            "dependency": dep or "",
            "paper_results": f"results/paper_table/category1_{run_tag}/eval_results.json",
            "act_results": f"results/eval_act_steering_{run_tag}/eval_results.json",
        }
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        print(
            f"[{i+1}/{args.n_trials}] {run_tag}\n"
            f"  paper_job={paper_job} act_job={act_job}\n"
            f"  alpha={alpha} max_corr={max_corr} corr_th={corr_th} fail_th={fail_th} eps={args.episodes_per_task}"
        )

    print(f"\nLedger: {ledger_path}")


if __name__ == "__main__":
    main()
