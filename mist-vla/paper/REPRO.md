# Reproducibility: paper tables and statistics

## Environment

Use the project conda/venv with [`../requirements.txt`](../requirements.txt) (Python 3.10+ recommended). Minimum for **paper stats only**: `numpy`, `scipy`.

Pinned stack for VLA inference (see repo root): `torch==2.2.0`, `transformers==4.40.1` (and `src/models/vla_wrapper.py` attention note).

## One-command regeneration (simulation)

From repository root:

```bash
cd mist-vla/paper/scripts
python3 run_stat_tests.py    # STAT_TESTS_REPORT.md, stat_tests_summary.json, tab_stat_tests.tex
python3 generate_tables.py   # tab_final_pooled_results.tex, tab_*_pooled.tex, tab_wilson_paper_curated.tex, ...
```

Full rebuild (stats + all visuals + PDF):

```bash
cd mist-vla/paper/scripts
python3 build_paper_artifacts.py
```

Sanity-check consistency after rebuild:

```bash
cd mist-vla/paper/scripts
python3 check_paper_consistency.py
```

Build PDF (optional):

```bash
cd mist-vla/paper
tectonic --keep-logs main.tex
```

## Artifact map

| Artifact | Role |
|----------|------|
| `paper/data/*eval_results*.json` | Frozen per-run LIBERO metrics |
| `paper/data/stat_tests_summary.json` | Machine-readable tests + Wilson + Holm |
| `paper/STAT_TESTS_REPORT.md` | Human-readable statistical report |
| `paper/tables/tab_*.tex` | LaTeX inputs for the manuscript |

## Model checkpoints

Safety MLP and base policy checkpoints are referenced in eval configs and HPC scripts under `mist-vla/` (see `docs/` and `hpc_mirror/`). For exact SHA256 of a checkpoint used in a specific run, record it in the eval job metadata when launching sweeps.

## Git provenance

`run_stat_tests.py` stores `meta` with file counts; optional: run from a clean git checkout and note `git rev-parse HEAD` in supplementary material for the paper revision.
