# Camera-Ready Checklist (CoRL-oriented)

Use this list before submission freeze.

## Data and statistics

- [ ] `paper/data/*eval_results*.json` are frozen for the submitted version.
- [ ] `python3 scripts/run_stat_tests.py` has been rerun on frozen JSONs.
- [ ] `python3 scripts/generate_tables.py` has been rerun.
- [ ] `python3 scripts/check_paper_consistency.py` passes.
- [ ] Primary contrast claims match `tables/tab_stat_tests.tex`.

## Figures

- [ ] All figures used in LaTeX exist under `paper/figures/`.
- [ ] Figure captions explicitly mark qualitative/context-only plots.
- [ ] Latent-space visualizations are described as descriptive, not significance tests.
- [ ] Steering timeline/dashboard figures are referenced in `sections/results.tex`.

## Text integrity

- [ ] No claim in abstract/introduction/conclusion exceeds evidence in current tables.
- [ ] Hardware claims remain bring-up-only unless hardware benchmark protocol is completed.
- [ ] Any archived/non-primary analyses are labeled as exploratory.
- [ ] Claim-evidence mapping updated in `CLAIM_EVIDENCE_MAP.md`.

## Build and formatting

- [ ] `tectonic --keep-logs main.tex` builds successfully.
- [ ] Overfull/underfull warnings are reviewed and acceptable.
- [ ] Bibliography and references resolve.
- [ ] Final PDF visually inspected for figure placement and readability.

