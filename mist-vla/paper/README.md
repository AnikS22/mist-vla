# CoRL Paper Draft (Complete Outline + Placeholders)

This folder contains the full CoRL-oriented paper structure with:
- finalized methods and experimental design,
- generated tables from completed JSON outputs,
- explicit placeholders for pending results and camera-ready fill-ins.

## What This Version Is
- A **comprehensive outline manuscript** that reads like a full paper.
- Numeric claims are intentionally conservative.
- Pending experiments are marked as `[TODO-*]` placeholders.

## Structure
- `main.tex` — top-level manuscript
- `sections/*.tex` — all paper sections
- `tables/*.tex` — auto-generated tables
- `scripts/generate_tables.py` — regenerates tables from `paper/data/*.json`
- `figures/` — current visual assets
- `data/` — synced run JSONs used for table generation

## Regenerate Tables
```bash
python3 scripts/generate_tables.py
```

## Generate Visuals
```bash
python3 scripts/generate_visuals.py
```

Notes:
- Some visuals are generated from completed run JSONs.
- Some are intentionally synthetic placeholders and are labeled as such in captions.

## Build PDF (tectonic)
```bash
cd /home/mpcr/Desktop/SalusV5/mist-vla/paper
tectonic --keep-logs --keep-intermediates main.tex
```

## Author Block
- Anik Sahai — Praxis Labs, MPCR Labs, FAU
- Merhdad Nojoumian — FAU
- William Hahn — MPCR Labs, FAU

## Camera-Ready Checklist
Search for `TODO-` in `sections/` and resolve each item only with completed evidence.
