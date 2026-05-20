# Archived tables (not in the compiled paper)

These `.tex` files exist for historical reference but are **not** `\input{}`'d
anywhere in `main.tex` or `sections/*.tex`. They were moved here in the
2026-05 paper-remediation pass.

| File | Reason archived |
|------|------------------|
| `tab_act_final_pooled.tex` | Subset of `tab_final_pooled_results.tex`; the pooled table already covers both architectures. |
| `tab_openvla_final_pooled.tex` | Same reason as the ACT-only variant. |
| `tab_detection_vs_correction.tex` | Earlier framing of the same comparison; superseded by `tab_final_pooled_results.tex` and the §6 mechanism analysis. |
| `tab_paired_clean_runs.tex` | Per-tag paired clean-sweep deltas; informative but not load-bearing in the manuscript. |
| `tab_act_key_results.tex` | Three-config ACT comparison; superseded by the pooled table + zero-shot OOD appendix. |
| `tab_so101_plan_placeholder.tex` | Post-submission SO-101 evaluation plan; will become the real physical-robot table once data arrives. Replaces the earlier Yahboom placeholder (deleted 2026-05; the SO-101 pipeline under `scripts/so101/` is the path forward). |
| `tab_so101_bringup_status.tex` | Post-submission SO-101 hardware bring-up status; replaces the earlier Yahboom bring-up table (deleted 2026-05). |

Reviving any of these requires reading the data sources first and verifying
they still match the current paper-curated pool definitions in
`scripts/run_stat_tests.py` and `scripts/generate_tables.py`.
