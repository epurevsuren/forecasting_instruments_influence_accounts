# Gemma-3-4B + custom NLP → XGBoost (standalone)

Full, self-contained port of the pipeline with **Gemma 3 4B** as the text
encoder. FinBERT was retired 2026-07-17: the NLP-only challenger matched the
FinBERT-embedding models on all 23 instruments, so its embeddings added
nothing this pipeline could use.

## Contents

- `gemma_embedder.py` — Gemma-3-4B local encoder. 4-bit
  (`unsloth/gemma-3-4b-it-unsloth-bnb-4bit`, fits 8 GB VRAM; fallback
  `google/gemma-3-4b-it`; override with env `GEMMA_MODEL_ID`). Pooling =
  mean + last-token concat (5120 dims) of the final hidden layer — one
  forward pass per post, no generation. Cache: `gemma3_embeddings_v1`.
- `train_gemma3_nlp_xgb.py` — trainer. Keeps every honest-eval upgrade:
  strict 4-way chronological split, trade-region magnitude calibration +
  TP quantile, vol-regime weights, PCA 5120→128 (fit on train slice),
  NLP-only challenger diagnostic. Models → `gemma3_nlp_xgb_models/`.
- `predict_gemma3_nlp_xgb.py` — all gates (NLP sigmoid, temporal,
  endorsement, self-news, chain, reiteration, commentary, event-window
  domain gating) + calibration.
- `backtest_simulator.py`, `simulate_tp_sl.py`, `simulate_ctrader.py`,
  `sweep_tp_sl.py`, `run_backtest.bat` — identical mechanics, this folder's
  models/results dirs.
- `db.py`, `signal_scorer.py` — shims to the canonical DP versions.

## Why encoder, not generator

The earlier experiment (Unsloth LoRA-SFT emitting 23-impact JSON) failed on
raw data and costs ~1000× more per post. Gemma reads; XGBoost forecasts.
Phase 2 (only if this baseline shows OOS edge): stack the fine-tuned
analyst's JSON predictions as 23 extra features for HIGH_SIGNAL posts.

## Run order (from this folder, gemma venv)

```
python train_gemma3_nlp_xgb.py      # 1st run Gemma-embeds ~190k posts (hours, one-time)
python backtest_simulator.py 202607101200 --from 201611010000
python simulate_tp_sl.py
python simulate_ctrader.py
```

Judge on the honest yardsticks only: OOS window (post ~2025-06), NLP-only
challenger gap, magnitude k≈1, Layer-2/3 P&L.
