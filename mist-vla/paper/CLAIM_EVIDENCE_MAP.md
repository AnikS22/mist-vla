# Claim-Evidence Map (submission consistency)

This file maps high-level manuscript claims to concrete artifacts in this repository.

## Primary quantitative claims

- **Pooled success parity for Vanilla/MPPI/Steering**
  - Evidence: `tables/tab_final_pooled_results.tex`
  - Source data: `data/*eval_results*.json`
  - Regeneration: `scripts/generate_tables.py`

- **Primary contrast significance status (z-test + Holm)**
  - Evidence: `tables/tab_stat_tests.tex`
  - Source data: `data/stat_tests_summary.json`
  - Regeneration: `scripts/run_stat_tests.py`

- **Wilson confidence intervals**
  - Evidence: `tables/tab_wilson_paper_curated.tex`
  - Source data: `data/stat_tests_summary.json`
  - Regeneration: `scripts/run_stat_tests.py` + `scripts/generate_tables.py`

- **Required sample sizes (effect-size grid)**
  - Evidence: `STAT_TESTS_REPORT.md`, `figures/14_required_n_by_effect_size.png`
  - Source data: `data/stat_tests_summary.json`
  - Regeneration: `scripts/run_stat_tests.py`, `scripts/generate_visuals.py`

- **Latency advantage (controller apply-time)**
  - Evidence: `tables/tab_final_pooled_results.tex`, `figures/09_latency_speedup_ood.png`
  - Source data: `data/*eval_results*.json` (`mean_apply_ms`)
  - Regeneration: `scripts/generate_tables.py`, `scripts/generate_visuals.py`

## Qualitative/context claims

- **LIBERO benchmark scene context**
  - Evidence: `figures/16_libero_annotated_panels.png`
  - Source generation: `scripts/generate_additional_visuals.py`
  - Note: qualitative context only, not an outcome claim.

- **Latent-space structure visualization**
  - Evidence: `figures/17_latent_pca_success_failure.png`, `figures/18_latent_pca_task_colored.png`
  - Source data: `research_data/rollouts/merged_all/{success_rollouts.pkl,failure_rollouts.pkl}`
  - Generation: `scripts/generate_latent_embeddings.py`
  - Note: PCA projections are descriptive visualizations, not hypothesis tests.

## Embodied scope boundary

- **No autonomous physical-arm completion claims in this version**
  - Evidence text: `sections/results.tex` (Embodied Bring-Up subsection),
    `sections/reproducibility_ethics.tex` (limitations).

## Reproducibility commands

From `paper/scripts`:

```bash
python3 run_stat_tests.py
python3 generate_tables.py
python3 generate_visuals.py
python3 generate_additional_visuals.py
python3 generate_latent_embeddings.py
```

