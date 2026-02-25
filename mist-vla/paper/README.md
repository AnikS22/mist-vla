# CoRL Paper Draft (LaTeX)

This directory contains the full manuscript draft for the latent safety steering paper.

## Structure
- `main.tex` - entrypoint
- `sections/` - modular paper sections
- `tables/` - generated LaTeX tables from JSON outputs
- `figures/` - manuscript figures
- `data/` - copied result JSONs used in this draft
- `scripts/generate_tables.py` - regenerates result tables

## Regenerate Tables
```bash
python3 scripts/generate_tables.py
```

## Build (example)
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Author Block (as requested)
- Anik Sahai — Praxis Labs, MPCR Labs, FAU
- Merhdad Nojoumian — FAU
- William Hahn — MPCR Labs, FAU

Emails are currently marked TODO and should be replaced before submission.

## Placeholders
Pending long-queue experiments are explicitly marked as TODO placeholders in `sections/results.tex` and `sections/appendix.tex`.
