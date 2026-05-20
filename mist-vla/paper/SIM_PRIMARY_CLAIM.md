# Primary claim (simulation-first, CoRL track)

This document locks the **pre-specified primary analysis** for the LIBERO simulation results. It aligns with `paper/scripts/run_stat_tests.py` (three primary contrasts on the **paper_curated** pool with Holm adjustment).

## Narrative

**Simulation-first latent safety steering:** We validate inference-time steering, MPPI, and ablations on standardized LIBERO evaluation JSONs aggregated across OpenVLA and ACT campaign families. The **primary statistical family** is:

1. Steering vs MPPI (success rate difference, two-proportion z-test)  
2. Steering vs Vanilla  
3. MPPI vs Vanilla  

Holm-adjusted *p*-values are reported for these three z-tests only (see `STAT_TESTS_REPORT.md`).

## What we do **not** claim from sim alone

- We do **not** claim statistically significant success-rate **lift** of steering over vanilla/MPPI unless the pooled tests reject the null at the stated α after multiplicity control.  
- We **do** report runtime (apply-time) advantages and regime-stratified qualitative patterns elsewhere in the paper where supported by data.

## Embodied / SO-101 (post-submission)

Physical-robot **task-completion** claims are **out of scope** for this submission track. Post-submission validation will use the SO-101 pipeline under `mist-vla/scripts/so101/` (Pi0 backbone); the prior Yahboom prototype is retired. Any hardware text in this submission is limited to infrastructure notes or deferred to future work (see plan: hardware deferred, conclusion item (i)).

## Regeneration

After adding or updating `paper/data/*eval_results*.json`:

```bash
cd paper/scripts
python3 run_stat_tests.py
python3 generate_tables.py
```
