# Camera-Ready Checklist (CoRL-oriented)

Use this list before submission freeze.

## Data and statistics

- [x] `paper/data/*eval_results*.json` are frozen for the submitted version. (Verified 2026-05-15.)
- [x] `python3 scripts/run_stat_tests.py` has been rerun on frozen JSONs.
- [x] `python3 scripts/generate_tables.py` has been rerun.
- [x] `python3 scripts/check_paper_consistency.py` passes (tables ↔ `stat_tests_summary.json`; abstract + zero-shot prose; stranded-tables check added in this pass).
- [x] Primary contrast claims match `tables/tab_stat_tests.tex`.
- [x] Temporal baseline (`scripts/eval_temporal_baseline.py`) has been run; `tables/tab_temporal_baseline.tex` is `\input`'d in `app:temporal_baseline`.
- [x] Intervention-rate aggregator (`scripts/aggregate_intervention_rate.py`) has been run; `tab_intervention_rate.tex` is `\input`'d in `app:intervention_rate`.
- [x] OpenVLA OOD seed table (`scripts/build_openvla_ood_table.py`) has been regenerated from the complete s42/s43/s44 JSONs; `tab_openvla_ood_status.tex` is `\input`'d in `app:openvla_ood`.

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

- [x] `tectonic --keep-logs main.tex` builds successfully.
- [ ] Overfull/underfull warnings are reviewed and acceptable. (Current build has only underfull \hbox/\vbox warnings, no overfull issues; review before camera-ready.)
- [x] Bibliography and references resolve.
- [ ] Final PDF visually inspected for figure placement and readability.

## Anonymization and acknowledgments (camera-ready)

- [ ] Switch `\usepackage{corl_2026}` to `\usepackage[final]{corl_2026}` in [`main.tex`](main.tex) for the camera-ready compile so author block and affiliations are revealed.
- [ ] Uncomment `\acknowledgments{...}` in [`main.tex`](main.tex) and fill it in.
- [ ] Re-run anonymization grep on the *initial submission* PDF (`pdftotext main.pdf - | grep -iE "sahai|nojoumian|hahn|fau\.edu"`) to confirm no author info leaks for the anonymous version.
- [ ] Verify keywords line in [`main.tex`](main.tex) still reads "failure detection, latency-efficient safety probes, inference-time steering, vision-language-action models" (was changed from "robot safety" in the 2026-05 remediation pass).

