# LIBERO simulation evaluation protocol (paper data)

## Data source

- **Frozen results** live under [`paper/data/`](data/) as `*eval_results*.json`.
- Each file includes `per_task` with modes (`vanilla`, `mppi`, `steering`, ablations, …), `n_successes`, `n_episodes`, and optional `mean_apply_ms`.

## Family grouping (for pooling)

Scripts [`paper/scripts/generate_tables.py`](scripts/generate_tables.py) and [`paper/scripts/run_stat_tests.py`](scripts/run_stat_tests.py) assign files to families by filename prefix:

| Prefix | Family |
|--------|--------|
| `category1_sweep_*` | `openvla_sweep` |
| `category1_ovla_ood_*` | `openvla_ood` |
| `eval_act_steering_sweep_*` | `act_sweep` |
| `eval_act_steering_act_ood*` | `act_ood_baselines` |
| `eval_act_zero_shot_*` | `act_zero_shot_ood` |

**Paper-curated pool** = sum of the five families above (used for main tables and primary statistics).

## Baseline modes

Per [`sections/experimental_setup.tex`](../sections/experimental_setup.tex): `vanilla`, `latent_stop`, `noise`, `ema_only`, `latent_jiggle`, `mppi`, `steering` (where present in each run JSON).

## Checklist for new runs

1. Export eval JSON with the same `per_task` schema as existing files.  
2. Place under `paper/data/` with a name matching one of the prefixes above **or** extend the `family()` logic in both scripts consistently.  
3. Run `python3 run_stat_tests.py` then `python3 generate_tables.py`.  
4. Commit the new JSON and regenerated `STAT_TESTS_REPORT.md`, `stat_tests_summary.json`, and `tables/tab_*.tex`.

## Failure / episode taxonomy

For structured failure labels, prefer logging `termination_reason`, success bit, and per-task stats in the eval script (see `scripts/eval_closed_loop_study.py` patterns); aggregate into JSON for post-hoc stratification.
