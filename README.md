# Forecasting Instrument Influence from Political Accounts

Predicts **1-hour market impact** across 23 instruments from posts by high-influence
political accounts (Trump on Truth Social + geopolitical X/Twitter accounts), then
simulates the resulting trades through take-profit/stop-loss and a real broker
account model.

The pipeline is deliberately layered so each stage can be validated on its own:

```
posts ──► NLP scoring ──► labelling ──► Gemma-3 embed + XGBoost ──► backtest
                                                    │
                                                    ├─► Layer 2  TP/SL simulation
                                                    └─► Layer 3  cTrader account sim
```

---

## Table of contents

- [Architecture](#architecture)
- [Results](#results)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [DP/ — data pipeline](#dp--data-pipeline)
- [IBKR/ — market data](#ibkr--market-data)
- [Gemma-3-4b_NLP_XGBoost/ — model, backtest, simulation](#gemma-3-4b_nlp_xgboost--model-backtest-simulation)
- [Data stores](#data-stores)
- [Configuration files](#configuration-files)
- [Project conventions](#project-conventions)
- [Known issues](#known-issues)

---

## Architecture

| Layer | Question it answers | Entry point |
|---|---|---|
| Data | What was said, and what did the market do next? | `DP/*.py`, `IBKR/*.py` |
| Model | How big a move, and in which direction? | `train_gemma3_nlp_xgb.py` |
| Layer 1 | Would we have called it right? | `backtest_simulator.py` |
| Layer 2 | Does it survive TP/SL mechanics? | `simulate_tp_sl.py` |
| Layer 3 | Does it survive margin, lots and stop-outs? | `simulate_ctrader.py` |

### Three prediction heads

Magnitude regression on signed returns was abandoned (R² ≈ 0) because
`y = sign × size` lets the unpredictable factor dominate squared error. The model
now decomposes the problem:

| Head | Target | Output |
|---|---|---|
| **move** | binarised against a rolling VaR quantile | `P(abnormal move)` |
| **direction** | sign, on event rows only | `P(up \| move)` |
| **size** | `log1p(\|move\|)` | expected magnitude |

Trade gate = `P(move) × P(direction)`, and the TP/SL distance is sized from the
size head. Consumers **must** apply `expm1()` to the size prediction before use —
see [Known issues](#known-issues).

---

## Results

Backtest window 2016-11 to 2026-08, 23 instruments, walk-forward validated with
strict point-in-time loading. Direction and magnitude are scored separately
because they are predicted by separate heads.

### Direction

| population | n | accuracy |
|---|---|---|
| TRADE — gate fires | 15,691 | **60.3%** |
| TRADE and `\|actual\| >= 0.1%` | 10,449 | **66.4%** |
| SKIP — gate declines | 32,563 | 49.1% |

SKIP sitting at chance is the load-bearing result: the gate **separates**
tradeable from untradeable rather than riding market state. Macro accuracy
(mean over instruments) 64.0%; mean move-head AUC 0.665; 21 of 23 instruments
clear the tradeable threshold.

Strongest instruments on the filtered population: USD_CNY 85.5%, USD_JPY 77.0%,
USD_CHF 75.9%, EUR_USD 74.7%, USD_CAD 74.2%, GBP_USD 71.4%, GOLD 71.4%.

### Magnitude — size head

`mean corr +0.760`, `mean scale 0.70x`, 20 of 23 instruments at MdAPE 32–49%.

| instrument | n | corr | scale | MdAPE |
|---|---|---|---|---|
| XLE | 1155 | 0.863 | 0.76 | 39% |
| AUD_USD | 564 | 0.860 | 0.68 | 32% |
| OIL | 1296 | 0.850 | 0.83 | 42% |
| DIA | 893 | 0.849 | 0.71 | 37% |
| USD_MXN | 701 | 0.836 | 0.71 | 33% |
| SPY | 916 | 0.814 | 0.64 | 35% |
| XLF | 1034 | 0.813 | 0.69 | 35% |
| VIX | 1460 | 0.789 | 0.75 | 41% |
| GOLD | 936 | 0.698 | 0.65 | 34% |

For reference, the signed regressor this replaced scored **corr 0.031 at scale
0.003x, MdAPE 100%** — the decomposition, not tuning, is what moved the number.

### Feature contribution — move head

| group | gain share |
|---|---|
| cross-instrument TA | 41.5% |
| Gemma embeddings | 33.5% |
| NLP composite scores | 14.9% |
| own TA | 7.7% |
| policy flags | 2.4% |

`atr_pct` is the single strongest feature. Volatility clustering, not post
content, drives most of *whether* a move happens; the post content contributes
mainly to direction and size.

### Measured and rejected

Kept here so they are not retried:

| idea | result |
|---|---|
| Warm-start walk-forward | loses 8/8 folds vs cold refit |
| kNN semantic memory | +0.001 (median precedent 291 days old) |
| Stream-state memory | −0.004 (only 3% of posts are bursts) |
| Gemma as direction predictor | 50.5% encoder / 50.9% analyst |
| VIX-up volatility bet | AUC 0.50–0.52 |
| Tail tightening `MOVE_Q` 70→90 | +0.2% |
| Conditioning on `sample_weight` | −1.0% |
| Stacking per AVM paper | 5.24% → 5.17%, marginal |

Confirmed wins: intraday TA over daily (corr 0.006 → 0.283), OLS-slope momentum
over 2-point difference (better at every noise level), `log1p` target (MdAPE
57.4% → 53.3%), cross-instrument features (+0.034 on 8/10), walk-forward over a
fixed split (0.231 vs 0.203).

---

## Quick start

```bash
# daily incremental run (Windows)
DP\run_daily.bat

# or stage by stage
uv run python DP/daily_truths_retriever.py          # fetch Truth Social
uv run python DP/x_tweets_retriever.py              # fetch X/Twitter
uv run python DP/sync_unified_feed.py               # merge into unified_feed
uv run python DP/signal_scorer.py                   # NLP scoring -> posts_scored
uv run python DP/build_indicators.py                # 1-min TA -> post_indicators
uv run python DP/build_final_training_set.py        # labels -> training_set_FINAL
uv run python Gemma-3-4b_NLP_XGBoost/train_gemma3_nlp_xgb.py
Gemma-3-4b_NLP_XGBoost\run_backtest.bat
```

Full rebuild (after a scoring or config change) — order matters, because each
stage reads the previous stage's table:

```bash
uv run python DP/signal_scorer.py --full
uv run python DP/build_indicators.py --rebuild
uv run python DP/build_final_training_set.py --full
uv run python Gemma-3-4b_NLP_XGBoost/train_gemma3_nlp_xgb.py
```

---

## Repository layout

```
DP/                        data pipeline: retrieval, scoring, labelling
IBKR/                      market data: fetch, cache, resample, backfill
Gemma-3-4b_NLP_XGBoost/    embeddings, training, prediction, simulation
  gemma3_nlp_xgb_models/   trained boosters + config.json
  backtest_results/        Layer 1 output CSV
  tp_sl_results/           Layer 2 trades + summary
  ctrader_results/         Layer 3 ledger + summary
IBKR/market_data_cache/    {INSTRUMENT}_{1min,5min,15min,30min}.csv
database.db                DuckDB — posts, scores, indicators, labels
```

---

## DP/ — data pipeline

### `daily_truths_retriever.py`
Fetches Truth Social posts.

```bash
uv run python DP/daily_truths_retriever.py [--since DATE] [--account HANDLE]
                                           [--lookback-days N] [--dry-run]
```

### `x_tweets_retriever.py`
Playwright-based X/Twitter retrieval with cookie rotation and gap detection.
Parses the GraphQL response across 7 operation names with recursive instruction
fallback, so a Twitter UI change degrades rather than returns zero.

```bash
uv run python DP/x_tweets_retriever.py [--handles A,B] [--since D] [--until D]
                                       [--max-per-account N] [--gap-days N]
                                       [--refetch] [--debug] [--dry-run]
```

- `--handles` — restrict to specific accounts; the only way to fetch *archive*
  accounts (e.g. `@POTUS46Archive`)
- `--gap-days` (default 3) — day-level coverage check; a month with a hole larger
  than this is re-scanned
- `--debug` — prints `_DIAG` accounting (ops seen, tweets parsed, parse failures)

Empty results are treated as a throttling signal rather than "no tweets", which
triggers cookie rotation.

### `sync_unified_feed.py`
Merges both sources into `unified_feed`, normalises the schema, adds `text_clean`
and derives `is_primary` (rank-0 account **within its active window**).

```bash
uv run python DP/sync_unified_feed.py [--full]
```

### `signal_scorer.py`
**The NLP core.** Seven scoring layers over each post, writing `posts_scored`.

| Layer | Produces |
|---|---|
| 1 Policy flags | `flag_*` (regex, from `scorer_config.json`) |
| 2 NER | `num_gpe`, `num_org`, `num_money`, … (spaCy) |
| 3 Composites | `policy_intensity_score`, `hawkish_risk_score`, `macro_risk_score` |
| 4 Embedding | `score_embedding` (SBERT similarity to policy anchors) |
| 5 Novelty | `score_novelty` vs a rolling window |
| 6 Burst | `score_burst` — dampens rapid-fire posting |
| 7 Relative | `score_relative`, then `sample_weight` |

Key functions:

- `score_batch(batch, ctx_*)` — all seven layers, context-aware for incremental runs
- `score_single_post(text, …, account=…)` — one post, used by the prediction path;
  **kept behaviourally identical to the batch path**
- `country_of(handle)` / `countries_mentioned(text)` — speaker vs subject country
- `country_context_factor(text, cc, feats, is_primary)` — damps self-referential
  posts (an embassy talking only about its own country) by `country_self_damp`.
  Rank-0 primary speakers are **never** damped, and any post firing a
  `country_policy_guard` flag is protected
- `is_endorsement(text)` / `endorsement_damp(texts)` — zeroes ceremonial and
  endorsement posts at scoring time so they never reach training or embeddings
- `prepare_feed(feed)` — normalises dates, adds `text_clean` and `is_primary`

```bash
uv run python DP/signal_scorer.py [--full]     # --full = rescore everything
```

Run `--full` after **any** change to `scorer_config.json`, or the new flag columns
won't exist in `posts_scored` and the trainer will 0-fill them with a warning.

### `build_indicators.py`
Intraday technical indicators computed on **1-minute** bars and synced to post
timestamps, written to `post_indicators`.

Seven columns: `mom5`, `mom20`, `sma_rat`, `rsi14`, `macd_h`, `bb_pos`, `atr_pct`
(windows are in *minutes*: 15/60/60/70/60/70).

- `bar_cutoff(when)` — floors to the minute. IBKR stamps bars by their **start**,
  so a 14:03:30 post must not see the 14:03 bar
- `_indicator_frame(b)` — single vectorised pass; momentum uses an **OLS slope**
  rather than a 2-point difference (measurably better at every noise level)
- Resampling uses `label='right', closed='left'` to stop a bin leaking future bars

```bash
uv run python DP/build_indicators.py [--rebuild] [--instrument SPY]
                                     [--table posts_scored] [--limit N] [--at TS]
```

Incremental by default. `--instrument` with `--rebuild` deletes only that
instrument's rows.

### `build_final_training_set.py`
Joins posts, scores, indicators and market bars into `training_set_FINAL` —
one row per post with `{INSTRUMENT}_Impact` and `{INSTRUMENT}_quality` columns.

- `compute_impacts()` — 1-hour forward moves via **DuckDB ASOF joins** over the
  30-min CSVs (replaced 4.4M pandas lookups)
- `load_cached_intraday(name, suffix)` — reads the IBKR cache or the yfinance
  extension (`{name}_30min_yf.csv`)
- `apply_rag()` — optional confounder down-weighting (`RAG_CONFOUNDER_DAMP`)
- `train_columns()` — enforces the **label-side firewall**: any `rag_*` column
  reaching the feature set raises

```bash
uv run python DP/build_final_training_set.py [--full] [--relabel-since DATE]
                                             [--use-gemini] [--use-rag]
```

### Annotation and config

| Script | Purpose | Key flags |
|---|---|---|
| `gemini_impact_annotator.py` | Optional LLM impact labels | `--mode --table --limit --since --cost-cap --rpm --model --list-models --yes` |
| `rag_confounder_annotator.py` | GDELT + local Gemma confounder labels; `assert_label_side_only()` keeps them off the feature side | `--source --table --limit --window-min --selftest --pause --out` |
| `update_configs_llm.py` | Additive-only, validated config deltas | `--days --provider --dry-run` |
| `event_manager.py` | Event windows and per-event weight multipliers | `--account --event --update --status --priority --end-date --notes --multipliers` |

### Utilities

| Script | Purpose |
|---|---|
| `db.py` | DuckDB helpers — `read_table`, `append_table`, schema evolution via `ALTER … DEFAULT` |
| `account_aliases.py` | Handle → canonical account mapping across renames |
| `filter_english_tweets.py` | Language filtering |
| `sort_truths.py` | Chronological ordering / de-duplication |
| `tweet_truth_impact_analyzer.py` | Ad-hoc impact inspection |
| `_measure_memory_signal.py`, `_test_direction_source.py`, `_test_analyst_direction.py` | Diagnostics for memory, direction source and analyst contribution |

---

## IBKR/ — market data

### `fetch_ibkr_history.py`
Primary fetcher (ib_insync). Resumes by **date**, not month, so interior holes are
filled rather than skipped.

```bash
uv run python IBKR/fetch_ibkr_history.py [--bar-size 30min] [--from D] [--until D]
                                         [--instruments A,B] [--core-only]
                                         [--refetch] [--host H] [--port P] [--client-id N]
```

- `--refetch` clears and replaces a window — the repair path for distorted data
- `IBKR_FRESH_DAYS` (env, default 1) controls the freshness cutoff

### `bars.py`
DuckDB engine over the CSV cache — read, resample, merge, write. Files stay CSV.

```python
bars.path(name, bar)      # market_data_cache/{name}_{bar}.csv
```

### Backfill and repair

| Script | Purpose | Key flags |
|---|---|---|
| `backfill_fx_history.py` | Fills GOLD/OIL/FX from HistData; rebuilds 15/30-min | `--instruments --overwrite --agg-only --dry-run` |
| `backfill_intraday_history.py` | General intraday backfill | `--instruments --overwrite --agg-only --dry-run` |
| `backfill_oil_5min_gap.py` | Targeted 5-min gap repair | `--min-gap-days --between --keep-weekends --overwrite --dry-run` |
| `fetch_binance_history.py` | Crypto bars | `--bar-size --from --until --instruments --refetch` |
| `recompute_30min_vwap_barcount.py` | Rebuilds `wap`/`barCount` on 30-min bars | `--instruments --force --dry-run` |
| `verify_fx_bid_ask.py` | Sanity-checks FX bid/ask handling | — |

---

## Gemma-3-4b_NLP_XGBoost/ — model, backtest, simulation

### `gemma_embedder.py`
Gemma-3-4B as a frozen encoder (4-bit, mean + last-token pooling). Embeddings are
cached once per post in `gemma3_embeddings_v1` and never recomputed.

### `gemma_analyst.py`
Generates a per-post 23-instrument impact JSON, cached in `gemma3_analyst_v1`.
`analyst_features(ids, texts, accounts, dates)` returns the matrix appended to the
NLP block. `train_gemma3_analyst_lora.py` fine-tunes the analyst (LoRA/unsloth).

### `train_gemma3_nlp_xgb.py`
Trains all three heads per instrument.

```bash
uv run python Gemma-3-4b_NLP_XGBoost/train_gemma3_nlp_xgb.py
```

Feature matrix (264 columns): `[128 Gemma embedding | 80 NLP | 7 own TA | 49 cross-instrument TA]`.
The direction head instead uses a **lean 78-column matrix** (`dir_prior` + flags +
own TA) — the full block measurably hurt it.

Environment knobs:

| Variable | Default | Meaning |
|---|---|---|
| `MOVE_Q` | 90 | move-label quantile |
| `SIZE_LOG` | 1 | fit size head on `log1p(\|move\|)` |
| `SIZE_MIN_CORR` | 0.10 | below this the size head is marked unreliable |
| `TARGET_PRECISION` | 0.58 | gate sweep target |
| `DIR_MIN_ACC` | 0.52 | direction head must beat this to be tradeable |
| `WF_FOLDS` | 8 | walk-forward folds |

Notable internals:

- `safe_corr(x, y, min_n)` — Pearson r that cannot emit a `RuntimeWarning`;
  guards **both** inputs' variance and returns `nan` when r is undefined
- `_wf_oos()` — walk-forward expanding-window refit (cold, never warm-start)
- Final refit on all rows, with `best_iteration` captured *before* the refit
- Writes `gemma3_nlp_xgb_models/config.json`: `move_gate` per instrument
  (`size_k`, `size_log`, `size_reliable`, `median_abs_move`, `tradeable`, …)

### `predict_gemma3_nlp_xgb.py`
Single-post inference.

```bash
uv run python Gemma-3-4b_NLP_XGBoost/predict_gemma3_nlp_xgb.py "post text" \
    [--account HANDLE] [--account-name NAME] [--geo-account] \
    [--entity-weight W] [--event-weight W] [--time TS] [--model-dir DIR]
```

Loads all three heads, rebuilds the lean direction matrix, reverts `size_log`,
applies the per-instrument `edge_min`, and surfaces **post memory** — the nearest
historical precedents via `find_precedents()` / `show_precedents()`.

### `backtest_simulator.py` — Layer 1

```bash
uv run python Gemma-3-4b_NLP_XGBoost/backtest_simulator.py \
    [--from D] [--to D] [--dir-threshold X] [--trade-threshold X] [--edge-min X]
    [--trade-all] [--model-dir DIR] [--csv-out F] [--near-window-min N]
    [--fine-tune] [--fine-tune-rounds N] [--fine-tune-out DIR]
```

- `_assert_consumer_matches_models(cfg)` — **fails fast on trainer/consumer skew**;
  hard-exits if the config says `size_log` and this consumer can't revert it
- `print_size_summary()` — the table that matters: `corr`, `scale`, `MdAPE` per
  instrument on magnitude alone
- `print_trade_skip_summary()` / `print_filtered_accuracy()` — whether the gate
  actually selects (TRADE accuracy vs SKIP accuracy)

### `simulate_tp_sl.py` — Layer 2

```
TP = entry * (1 ± |pred| * tp_mult / 100)
SL = entry * (1 ∓ |pred| * sl_mult / 100)
```

```bash
uv run python Gemma-3-4b_NLP_XGBoost/simulate_tp_sl.py \
    [--csv F] [--tp-mult 1.0] [--sl-mult 2.0] [--tp-pct X] [--sl-pct X]
    [--max-hold-min 60] [--min-pred 0.3] [--pred-scale 1.0]
    [--min-barcount 10] [--sl-noise-mult X] [--out F]
```

TP/SL distances are **proportional to the predicted magnitude**, which is why a
mis-scaled size head corrupts the risk calculation and not just the P&L.
`sweep_tp_sl.py` (`--csv --rows --out-prefix --min-barcount`) explores the grid.

### `simulate_ctrader.py` — Layer 3
Real broker mechanics: contract sizes, lot steps, leverage, swaps, margin levels
and stop-outs, from the verified specs in `instruments.json`.

```bash
uv run python Gemma-3-4b_NLP_XGBoost/simulate_ctrader.py \
    [--csv F] [--balance 2500] [--leverage 30] [--sizing margin|risk]
    [--margin-per-trade 25] [--max-margin-total 80] [--stopout-buffer 20]
    [--risk-pct 2] [--no-sl] [--demo-caps] [--out F]
    [--no-drawdown-guard] [--dd-trigger 4] [--dd-recover 2] [--dd-size 0.40]
```

The drawdown guard is a two-state FSM (NORMAL ⇄ DEFENSIVE) that shrinks position
size after consecutive losses. Compare with `--no-drawdown-guard` to verify it
earns its place.

### `signal_scorer.py` (this folder)
A **shim**. The real implementation is `DP/signal_scorer.py`; this re-exports it so
training-data scoring and prediction cannot drift apart. **Edit the scorer in one
place: `DP/`.**

---

## Data stores

### `database.db` (DuckDB)

| Table | Contents |
|---|---|
| `unified_feed` | merged raw posts |
| `posts_scored` | all NLP features (`flag_*`, `score_*`, `raw_score`, `sample_weight`, `is_self_country`, `is_endorsement`) |
| `post_indicators` | 7 intraday TA columns per post × instrument |
| `training_set_FINAL` | labels: `{INST}_Impact`, `{INST}_quality` |
| `training_set_HIGH_SIGNAL` | high-weight subset |
| `gemma3_embeddings_v1` | cached Gemma embeddings |
| `gemma3_analyst_v1` | cached analyst JSON |
| `eval_report_gemma3` | per-instrument evaluation |

### `IBKR/market_data_cache/`
CSV, never a database. Canonical schema (IBKR is the standard):

```
date,open,high,low,close,volume,wap,barCount
```

yfinance rows add `adjClose` and **never** fabricate `wap`/`barCount`.
FX `volume = -1` is expected, not a bug.

---

## Configuration files

| File | Role |
|---|---|
| `DP/instruments.json` | **Master instrument list.** Every script loads from it — never hardcode tickers. Holds `yf`, `market` (`us`/`24h`), `impact_cap`, `ibkr{}` and verified `ctrader{}` specs |
| `DP/scorer_config.json` | Policy flag regexes, country terms, endorsement patterns, `country_self_damp`, `country_policy_guard`, norm divisors |
| `DP/influence_accounts.json` | Accounts, institutions, archives, active windows, country per handle |
| `DP/chain_thresholds.json` | Self-calibrated semantic chain-damp percentiles |
| `Gemma-3-4b_NLP_XGBoost/gemma3_nlp_xgb_models/config.json` | Shipped model contract — feature lists, calibration, `move_gate` |

---

## Project conventions

1. **Timezone.** UTC is the storage standard — bars and `unified_feed` are UTC.
   **Display is always New York.** Do not mix timezones inside the database.
2. **Bars stay CSV.** DuckDB reads and resamples them; nothing migrates to a table.
3. **`instruments.json` is the single source of truth** for tickers and specs.
4. **Never drop original bar columns.** Add, don't replace.
5. **Two-layer bars.** 30-min drives signals and features; 1-min drives TP/SL
   simulation and indicators.
6. **No look-ahead.** `bar_cutoff()` floors to the minute; resampling is
   `label='right', closed='left'`; the label-side firewall keeps `rag_*` out of
   features.
7. **Long-running scripts** print every step, save incrementally, and keep RAM low —
   the server runs ML models alongside.
8. **Run `--full` rescoring after any `scorer_config.json` change.**

---

## Known issues

**Bar coverage gaps.** `NATGAS_30min.csv` and `US10Y_30min.csv` have **no data for
2021–2022**, and the 1-min files are missing 2021–2025. `build_final_training_set`
turns a missing bar into a literal `0.0` move via `fillna`, so those instruments
carry ~48–51% fabricated zero labels. This is the root cause of their `nan` size
correlations and unstable MdAPE. yfinance cannot fill it (intraday limits are 7
days at 1-min, 60 days at 30-min, 730 days at 1-hour) — it needs IBKR dated
contracts via `--refetch`.

**Fabricated zeros vs. flat markets.** A missing label should be `NULL` and excluded,
not a zero move. Until that is fixed, any metric computed over all rows is optimistic.

**Trainer/consumer version skew.** The size head is fitted on `log1p(|move|)`;
every consumer must `expm1()` before applying `size_k`. Running a stale
`backtest_simulator.py` against a current model silently mis-scales every TP/SL
distance — which corrupts margin calculations, not just returns.
`_assert_consumer_matches_models()` now catches this at startup.

**Policy flags are weak and mostly unsigned.** Measured on 15,394 trainable rows,
the 24 flags contribute ~2.4% of move-head gain, and most correlate *negatively*
with movement. `flag_public_health` / `flag_pandemic_relief` are strongly positive
but act as 2020-era proxies rather than causal signal. Because every flag is
unsigned — "this is about tariffs", never "this is bullish" — the direction head
stays near a coin flip while the size head works.

**Live TA is zero-filled.** `predict` does not yet call `indicators_at()`, so the
TA block is empty at live inference.
