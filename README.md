# MIST-VLA

Per-dimension collision forecasting and targeted activation steering for safer
Vision-Language-Action models on LIBERO.

This repo contains the minimal code needed to:
- collect rollouts (success + failure) with internal signals
- label failures (SAFE-style + per-dim risk)
- train a per-dimension risk predictor
- apply targeted activation steering

## Structure

- `mist-vla/` main package
  - `scripts/` runnable entry points
  - `src/` model wrappers, data collection, training, steering, evaluation
  - `configs/` basic config
  - `requirements.txt` environment deps
- `claude.md` project notes

## Quick start (local)

```
cd mist-vla
pip install -r requirements.txt
PYTHONPATH=$PWD:../openvla-oft:$PYTHONPATH \
python scripts/collect_failure_data_oft_eval.py \
  --env libero_spatial \
  --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
  --n_success 0 --n_failure 1 --max-attempts-per-task 1 \
  --camera-res 256 --save_dir data/rollouts_oft_eval --seed 0
```

## Key scripts

- `scripts/collect_failure_data_oft_eval.py`  
  Uses the official OpenVLA-OFT eval pipeline and logs MIST-VLA signals
  (actions, hidden states, collisions, robot states).
- `scripts/collect_failure_data.py`  
  Custom collector (no perturbations by default; can be enabled).
- `scripts/collect_phase1_data.py`  
  Phase 1 data collection with collision labels.
- `scripts/train_risk_predictor.py`  
  Train per-dimension failure predictor.
- `scripts/extract_steering_vectors.py`  
  Build steering vectors for targeted mitigation.
- `scripts/run_evaluation.py`  
  Evaluate success, collisions, recovery rate.

## Notes

- LIBERO uses a config file under `~/.libero/config.yaml`. Make sure paths exist.
- For OpenVLA-OFT, use `openvla-oft` on `PYTHONPATH`.
- GPU runs are recommended for large-scale rollouts.
