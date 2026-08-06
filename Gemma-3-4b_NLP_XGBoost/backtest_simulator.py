"""
backtest_simulator.py
======================
HIGH-SIGNAL backtest simulator with a time-range window, built on top of the
EXISTING training-set tables instead of rebuilding a point-in-time training
set from scratch every run (that's what build_test_training_set.py is for —
it is left completely untouched, see below).

WHY THIS EXISTS
---------------
training_set_HIGH_SIGNAL / training_set_FINAL already contain, for
every historical post, the REAL realized 1-hour market move (computed once by
build_final_training_set.py from IBKR + yfinance history). For any post whose
post-time + 60min already lies in the past relative to the simulated "now"
(--until), that stored *_Impact value IS the correct point-in-time "actual" —
re-deriving it via a fresh point-in-time rebuild (like
build_test_training_set.py does for the WHOLE dataset) is wasted work.

The ONLY posts whose stored label can leak "future" information relative to
--until are the ones tweeted within the last `--near-window-min` (default 60,
matching the 1h reaction window) minutes BEFORE --until — for those, this
script throws away the stored *_Impact and recomputes a PARTIAL reaction
(post -> min(post+60min, --until)) using fine-grained yfinance bars
(2m/5m/15m/30m, auto-picked the same way build_test_training_set.py does),
truncated at --until. This is the "alter the past 1 hour data" step the user
asked for — a handful of lightweight yfinance calls instead of a full rebuild.

WHAT IT DOES
------------
For every HIGH_SIGNAL post with post_time in [--from, --until]:
  1. Build the model input: cached gemma embedding (gemma3_embeddings_v1)
     + NLP scorer features (posts_scored), exactly as in training.
  2. Predict per-instrument 1h impact with the trained XGBoost models
     (gemma_nlp_xgb_models/ by default, or --model-dir), applying the SAME
     NLP gate + temporal gate used by predict_gemma_nlp_xgb.py.
  3. Determine "actual": stored *_Impact, or recomputed near-cutoff partial
     reaction (see above).
  4. Print a full per-post comparison (predicted vs actual, per instrument,
     direction match yes/no).

After the loop, prints a per-instrument + overall DIRECTION-MATCH ACCURACY
summary (sign(predicted) == sign(actual), restricted to "meaningful" actual
moves, |actual| >= --dir-threshold, same convention as the training eval's
dir_acc).

FINE-TUNING (--fine-tune)
--------------------------
Yes, dynamic fine-tuning on a DB time range is possible: XGBoost's sklearn API
supports CONTINUED boosting via `xgb_model=<path or Booster>` — it appends
`n_estimators` new trees on top of the existing ones instead of retraining
from scratch. With --fine-tune, after the backtest loop this script takes the
SAME window's (X, y, sample_weight) and continues training each instrument's
model with --fine-tune-rounds extra trees, writing the result to
gemma_nlp_xgb_models_live/ (created as a copy of --model-dir on first use;
the daily full retrain in train_gemma_nlp_xgb.py is untouched — it still
writes to gemma_nlp_xgb_models/). Each fine-tune run backs up the previous
live model as <inst>_Impact.json.bak_<timestamp> before overwriting.

Suggested cadence:
  - Daily:  uv run python train_gemma_nlp_xgb.py            (full retrain, baseline dir)
  - Hourly: uv run python backtest_simulator.py <now> --from <now-1h> --fine-tune --fine-tune-rounds 8
            (small continued-boosting nudge on the live dir using whatever
             HIGH_SIGNAL posts appeared in the last hour)

This script does NOT modify build_test_training_set.py, train_gemma_nlp_xgb.py,
or predict_gemma_nlp_xgb.py — the existing manual point-in-time testing
workflow (build_test_training_set.py -> train_gemma_nlp_xgb.py ->
predict_gemma_nlp_xgb.py) remains exactly as before.

USAGE
-----
  uv run python backtest_simulator.py 202606140900
  uv run python backtest_simulator.py 202606140900 --from 202605010000
  uv run python backtest_simulator.py 202606140900 --from 202606130900 --fine-tune
"""
import os
# FA2 install is broken in this venv — tell unsloth to go straight to
# Xformers instead of probing FA2 and warning (same speed either way).
os.environ.setdefault("UNSLOTH_FORCE_XFORMERS", "1")
try:
    import unsloth  # noqa: F401 — must import BEFORE transformers for its patches
except ImportError:
    pass
import sys
import json
import shutil
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
_DP   = os.path.normpath(os.path.join(_HERE, "..", "DP"))
for _p in (_HERE, _DP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import db                              # DuckDB helper (shim) -> ../database.db
import build_final_training_set as B   # TICKERS
import predict_gemma3_nlp_xgb as PR    # LABELS, temporal_factor, GATE_*

NY = 'America/New_York'

HS_TABLE     = "training_set_HIGH_SIGNAL"
SCORED_TABLE = "posts_scored"
EMB_TABLE    = "gemma3_embeddings_v1"

OUT_DIR  = os.path.join(_HERE, "gemma3_nlp_xgb_models")        # daily-retrained baseline
LIVE_DIR = os.path.join(_HERE, "gemma3_nlp_xgb_models_live")   # fine-tuned, updated by this script

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

_ENTITIES_FILE = os.path.join(_DP, "influence_accounts.json")

def _rank0_handle(when=None) -> str:
    """Return the rank-0 TruthSocial primary handle ACTIVE at `when` (default now).

    primary_accounts now holds MULTIPLE rank-0 entries across eras/platforms
    (realDonaldTrump on X 2017-2021, POTUS on X 2021-2025, realDonaldTrump on
    TruthSocial 2024-2028). Used here only as the default `account` for a scored
    row that lacks one, so pick the TruthSocial rank-0 primary whose
    [active_from, expiration_date] window contains `when`; fall back to the first
    TruthSocial rank-0, then 'us_president'. Mirrors predict_gemma_nlp_xgb.
    """
    try:
        import json
        with open(_ENTITIES_FILE, encoding="utf-8") as f:
            accounts = json.load(f).get("primary_accounts", [])
    except Exception:
        return "us_president"

    def _p(x):
        if x is None or str(x).strip().upper() in ("", "N/A", "NONE", "NULL"):
            return None
        try:
            return pd.Timestamp(x, tz="UTC")
        except Exception:
            return None

    now = pd.Timestamp(when, tz="UTC") if when is not None else pd.Timestamp.now(tz="UTC")
    ts = []
    for a in accounts:
        try:
            if int(a.get("rank", 99)) != 0:
                continue
        except (TypeError, ValueError):
            continue
        if a.get("platform") == "truthsocial":
            ts.append(a)

    for a in ts:
        lo = _p(a.get("active_from"))
        hi = _p(a.get("active_to")) or _p(a.get("expiration_date"))
        if (lo is None or now >= lo) and (hi is None or now <= hi):
            return str(a.get("account", "")).lstrip("@") or "us_president"
    if ts:
        return str(ts[0].get("account", "")).lstrip("@") or "us_president"
    return "us_president"


# ============================================================ yfinance helpers ----
# Inlined from the former build_test_training_set.py (deleted — no longer a separate file).
# ==========================================================================
# NEAR-CUTOFF PARTIAL REACTION  —  OFFSET-SAFE (no bar may extend past --until)
# ==========================================================================
# For posts tweeted within --near-window-min of --until, the stored 1-hour
# *_Impact can encode price action AFTER --until, which would leak the future
# into the backtest. We recompute a PARTIAL reaction post -> min(post+60, until)
# from the LOCAL market_data_cache bars, read with DuckDB (on-disk filtered so
# the huge 1-min CSVs never fully load), STRICTLY truncated at --until:
#   * a bar is usable ONLY if its WHOLE interval finishes at/before --until
#     (bar_start + interval <= until) — a bar STRADDLING the cutoff is dropped,
#     otherwise its close would reveal price after the offset (a hint) and the
#     backtest would be unrealistic;
#   * intervals: 1min preferred, then 15min, then 30min — never 5min;
#   * yfinance is used ONLY as a fallback for the recent tail the cache lacks.
FINE_REACH_DAYS = 59   # yfinance intraday (<=30m) reaches back ~60 calendar days
REACT_MIN       = 60   # the 1-hour reaction window baked into *_Impact
_CACHE_IVS = (("1min", 1), ("15min", 15), ("30min", 30))   # 1min first; NO 5min


def _to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _ts_lit(ts) -> str:
    """UTC datetime -> a DuckDB TIMESTAMPTZ literal."""
    return "TIMESTAMPTZ '" + _to_utc(ts).strftime("%Y-%m-%d %H:%M:%S") + "+00'"


def _col(d, name):
    """yfinance column -> list of floats (handles the single-symbol MultiIndex)."""
    s = d[name]
    try:
        if getattr(s, "ndim", 1) > 1:
            s = s.iloc[:, 0]
    except Exception:
        pass
    return [float(x) for x in s.values]


def _reaction_from_slice(slice_iv, post_utc, until_utc, react_end_utc):
    """Leak-free partial reaction % from a small {iv: [(t_utc, open, close)...]}
    slice (bars sorted ascending). initial = OPEN of the bar containing the post
    (matches build_final_training_set); final = CLOSE of the LAST bar whose whole
    interval ends at/before `until` (and after the post bar, within the reaction
    window). Prefers 1min. Returns None if not measurable."""
    for iv, mins in _CACHE_IVS:
        rows = slice_iv.get(iv)
        if not rows:
            continue
        w = pd.Timedelta(minutes=mins)
        init = init_t = None
        for t, o, c in rows:                       # last bar with t <= post
            if t <= post_utc:
                init, init_t = o, t
            else:
                break
        if init is None or init == 0:
            continue
        fin = None
        for t, o, c in rows:                       # last bar fully <= until, after post bar
            if t > init_t and (t + w) <= until_utc and t <= react_end_utc:
                fin = c
        if fin is None:
            continue
        return (float(fin) - float(init)) / float(init) * 100.0
    return None


def _cache_slice(name, lo, until):
    """{iv: [(t_utc, open, close)...]} from market_data_cache for [lo, until],
    read via DuckDB — only the tiny matching window is materialised, so the big
    CSVs never fully load. {} when the cache has no bars in that window."""
    import duckdb
    con = duckdb.connect(); con.execute("SET TimeZone='UTC'")
    out = {}
    try:
        for iv, _ in _CACHE_IVS:
            path = os.path.join(B.CACHE_DIR, f"{name}_{iv}.csv")
            if not os.path.exists(path):
                continue
            rows = con.execute(
                f"SELECT date::TIMESTAMPTZ, open, close "
                f"FROM read_csv_auto('{path}', null_padding=true) "
                f"WHERE date::TIMESTAMPTZ >= {_ts_lit(lo)} AND date::TIMESTAMPTZ <= {_ts_lit(until)} "
                f"ORDER BY 1"
            ).fetchall()
            if rows:
                out[iv] = [(_to_utc(t), o, c) for t, o, c in rows]
    finally:
        con.close()
    return out


def _yf_slice(name, lo, until):
    """Fallback slice from yfinance for the recent tail the cache lacks. 1-min
    bars when within yfinance's ~7-day 1m reach, else 30-min. Same tuple format
    (UTC). yfinance bar index is the bar START, so the same bar-end <= until
    cutoff in _reaction_from_slice keeps it offset-safe."""
    import yfinance as yf
    spec = B.TICKERS.get(name)
    if not spec:
        return {}
    days_back = (pd.Timestamp.now(tz="UTC") - _to_utc(until)).days
    iv, key = ("1m", "1min") if days_back <= 6 else ("30m", "30min")
    try:
        d = yf.download(spec[0], start=(_to_utc(lo) - pd.Timedelta(days=1)),
                        end=(_to_utc(until) + pd.Timedelta(hours=1)),
                        interval=iv, auto_adjust=True, progress=False)
        if d.empty:
            return {}
        idx = d.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        rows = [(pd.Timestamp(t), o, c)
                for t, o, c in zip(idx, _col(d, "Open"), _col(d, "Close"))
                if _to_utc(lo) <= pd.Timestamp(t) <= _to_utc(until)]
        return {key: rows} if rows else {}
    except Exception:
        return {}

DIR_THRESHOLD_DEFAULT = 0.1   # |actual| %% move must exceed this to count toward direction accuracy
NEAR_WINDOW_DEFAULT   = 60    # minutes — matches the 1h reaction window baked into *_Impact


# ============================================================================
# helpers
# ============================================================================
def parse_stamp(s, name="time"):
    try:
        return pd.Timestamp(datetime.datetime.strptime(s.strip(), "%Y%m%d%H%M"), tz=NY)
    except ValueError:
        sys.exit(f"❌ {name} must be yyyymmddhhmm (NY local), got: {s!r}")


def embed_texts(texts):
    """Gemma-3-4B mean + last-token pooling — must match train_gemma3_nlp_xgb
    (used only for posts missing cached embeddings in gemma3_embeddings_v1)."""
    from gemma_embedder import embed_texts as _ge
    return _ge(texts)


def gate_multiplier(row, emb=None):
    """Same NLP-gate + temporal-gate formula as predict_gemma_nlp_xgb.predict().
    `emb` = this post's Gemma vector, forwarded to the semantic chain guard so
    backtest and live make the identical restatement-vs-new-info call."""
    import math
    # DOMAIN-WEIGHTED signal — SAME formula as predict_gemma_nlp_xgb.predict()
    pis = min(float(row.get('policy_intensity_score') or 0.0) / 8.0, 1.0) \
        if pd.notna(row.get('policy_intensity_score')) else 0.0
    _h = float(row.get('hawkish_risk_score') or 0.0) if pd.notna(row.get('hawkish_risk_score')) else 0.0
    # macro term EVENT-WINDOW GATED at the post's date (same as build + predict)
    _m = PR.gated_macro_score(row, row['date_ny'].date())
    dom = max(min(_h / 5.0, 1.0), min(_m / 5.0, 1.0))
    sw = float(row.get('sample_weight') or 0.0) if pd.notna(row.get('sample_weight')) else 0.0
    signal = float(0.25 * pis + 0.45 * dom + 0.30 * sw)
    if PR.GATE_ENABLED:
        gate = 1.0 / (1.0 + math.exp(-PR.GATE_K * (signal - PR.GATE_MID)))
    else:
        gate = 1.0
    mult = 1.0 if gate >= 0.5 else gate
    _account = str(row.get('account', '') or '')
    _country = PR._country_for(_account)
    tfactor, tlabel = PR.temporal_factor(str(row['text']),
                                         post_hour=row['date_ny'].hour,
                                         post_ts=row['date_ny'],
                                         country=_country)
    efactor, elabel = PR.endorsement_factor(str(row['text']))
    if efactor < tfactor:            # endorsement gate overrides temporal
        tfactor, tlabel = efactor, elabel
    sfactor, slabel = PR.self_news_share_factor(
        str(row['text']), account=_account,
        account_name=str(row.get('account_name') or ''))
    if sfactor < tfactor:            # self-news-share gate overrides both
        tfactor, tlabel = sfactor, slabel
    # chain guard: SAME production code path as live prediction
    # (predict_gemma_nlp_xgb.chain_factor — in-process state, chronological)
    cfactor, clabel = PR.chain_factor(_account, row['date_ny'], signal, emb=emb)
    if cfactor < tfactor:            # chain guard overrides everything
        tfactor, tlabel = cfactor, clabel
    # reiteration + commentary damps (production code path) — reiteration uses
    # the REAL context-computed score_novelty; both only for temporally-neutral
    if tfactor == 1.0 and tlabel == "neutral":
        rfactor, rlabel = PR.reiteration_factor(
            str(row['text']), account=_account, post_ts=row['date_ny'],
            novelty=row.get('score_novelty'))
        if rfactor < tfactor:
            tfactor, tlabel = rfactor, rlabel
        mfactor, mlabel = PR.commentary_factor(str(row['text']))
        if mfactor < tfactor:
            tfactor, tlabel = mfactor, mlabel
    mult *= tfactor
    return signal, gate, tfactor, tlabel, mult


# ============================================================================
# data loading
# ============================================================================
def load_window(since, until):
    print(f"\n📂 Loading {HS_TABLE} ...")
    hs = db.read_table(HS_TABLE)
    if hs is None or hs.empty:
        sys.exit(f"❌ {HS_TABLE} not found / empty — run build_final_training_set.py first.")
    hs['date'] = pd.to_datetime(hs['date'], utc=True)
    hs['date_ny'] = hs['date'].dt.tz_convert(NY)

    mask = (hs['date_ny'] >= since) & (hs['date_ny'] <= until)
    hs = hs[mask].sort_values('date_ny').reset_index(drop=True)
    print(f"  HIGH_SIGNAL posts in [{since} -> {until}]: {len(hs)} "
          f"(of {mask.shape[0]} total HIGH_SIGNAL rows)")
    if hs.empty:
        sys.exit("❌ No HIGH_SIGNAL posts in this window — nothing to backtest.")

    print(f"📂 Loading {SCORED_TABLE} for NLP features + id ...")
    scored = db.read_table(SCORED_TABLE)
    if scored is None:
        sys.exit(f"❌ {SCORED_TABLE} not found — run signal_scorer.py first.")
    scored['id'] = pd.to_numeric(scored['id'], errors='coerce').astype('Int64')
    hs['id']     = pd.to_numeric(hs['id'],     errors='coerce').astype('Int64')
    scored['date'] = pd.to_datetime(scored['date'], format='mixed', utc=True)
    scored = scored.drop_duplicates(subset=['platform', 'id'])

    df = hs.merge(scored, on=['platform', 'id'], how='left', suffixes=('', '_sc'))
    n_missing = df['account'].isna().sum()
    if n_missing:
        print(f"  ⚠️  {n_missing} HIGH_SIGNAL post(s) had no NLP-feature match in "
              f"{SCORED_TABLE} ((platform, id) mismatch) — will use 0-filled features.")
    return df


def load_embeddings(df):
    print(f"📂 Loading cached embeddings ({EMB_TABLE}) ...")
    emb_cached = db.read_table(EMB_TABLE)
    cache = {}
    if emb_cached is not None and 'platform_id' in emb_cached.columns:
        emb_cached = emb_cached.set_index(emb_cached['platform_id'].astype(str))
        cache = {i: np.array(v, dtype=np.float32) for i, v in emb_cached['embedding'].items()}

    platform_ids = (df['platform'] + '_' + df['id'].astype(str)).tolist()
    missing_idx = [i for i, pid in enumerate(platform_ids) if pid not in cache]
    if missing_idx:
        print(f"  🔢 {len(missing_idx)}/{len(platform_ids)} posts missing cached embeddings — "
              f"computing live with gemma...")
        fresh = embed_texts(df.loc[missing_idx, 'text'].tolist())
        for j, i in enumerate(missing_idx):
            cache[platform_ids[i]] = fresh[j]
    else:
        print(f"  ✅ All {len(platform_ids)} embeddings found in cache")

    return np.vstack([cache[pid] for pid in platform_ids])


def _assert_consumer_matches_models(cfg):
    """Fail fast on trainer/consumer version skew.

    The size head may be fitted on log1p(|move|) (config: size_log) and MUST be
    reverted with expm1 before size_k is applied. A stale consumer that lacks
    the expm1 revert reads log-space numbers as percent moves, silently
    compressing every large prediction — TP/SL distances are derived from
    |pred|, so the whole risk calculation is then wrong (this is how a run
    ended up with a 54% worst-case margin level against a 70% guard floor).
    Likewise a trainer with the SIZE_MIN_CORR guard emits size_reliable; if it
    is absent, no-skill heads silently keep their size_k amplification.
    """
    gate = cfg.get("move_gate", {}) or {}
    per  = [(k, v) for k, v in gate.items() if isinstance(v, dict)]
    if not per:
        return
    log_insts = [k for k, v in per if v.get("size_log")]
    if log_insts and not _CONSUMER_REVERTS_SIZE_LOG:
        sys.exit(
            "❌ MODEL/CONSUMER MISMATCH: config.json marks size_log=true for "
            f"{len(log_insts)} instrument(s) ({', '.join(log_insts[:4])}...), but this "
            "backtest_simulator.py has no expm1 revert. Predictions would be read "
            "in log space and every TP/SL distance would be wrong.\n"
            "   Fix: use the backtest_simulator.py that matches your trainer.")
    missing = [k for k, v in per if "size_reliable" not in v]
    if missing:
        print(f"  ⚠️  {len(missing)} instrument(s) have no 'size_reliable' in config "
              f"({', '.join(missing[:4])}...) — trained by a pre-SIZE_MIN_CORR trainer. "
              "No-skill size heads keep their size_k amplification; NATGAS/US10Y/US2Y "
              "are the usual casualties. Retrain to restore the guard.")


# this module reverts log1p size predictions with expm1 in _size_pred()
_CONSUMER_REVERTS_SIZE_LOG = True


def load_models(model_dir, instruments):
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"❌ {cfg_path} not found — train models first (train_gemma_nlp_xgb.py).")
    cfg = json.load(open(cfg_path))
    _assert_consumer_matches_models(cfg)
    # attach the training-time embedding PCA (see PR.project_emb) if present
    if cfg.get("emb_pca"):
        _z = np.load(os.path.join(model_dir, "emb_pca.npz"))
        cfg["_emb_pca_mats"] = (_z["mean"], _z["components"])
    models = {}
    for inst in instruments:
        p = os.path.join(model_dir, f"{inst}_Impact.json")
        if os.path.exists(p):
            m = xgb.XGBRegressor()
            m.load_model(p)
            models[inst] = m
    print(f"  ✅ Loaded {len(models)}/{len(instruments)} models from {model_dir}")

    # TWO-HEAD CLASSIFIERS (2026-07-31). Optional: an older model dir without
    # them still runs, falling back to the |pred| >= threshold gate.
    clf_move, clf_dir, reg_size = {}, {}, {}
    for inst in instruments:
        pm = os.path.join(model_dir, f"{inst}_Impact__move.json")
        pd_ = os.path.join(model_dir, f"{inst}_Impact__dir.json")
        ps = os.path.join(model_dir, f"{inst}_Impact__size.json")
        if os.path.exists(pm):
            _c = xgb.XGBClassifier(); _c.load_model(pm); clf_move[inst] = _c
        if os.path.exists(pd_):
            _c = xgb.XGBClassifier(); _c.load_model(pd_); clf_dir[inst] = _c
        if os.path.exists(ps):
            _r = xgb.XGBRegressor(); _r.load_model(ps); reg_size[inst] = _r
    cfg["_clf_move"], cfg["_clf_dir"], cfg["_reg_size"] = clf_move, clf_dir, reg_size
    if clf_move:
        print(f"  🎯 Heads loaded: {len(clf_move)} move + {len(clf_dir)} direction"
              f" + {len(reg_size)} size — gate = P(move) x P(direction), "
              f"TP sized from the SIZE head")
    else:
        print("  ⚠️  No classifier heads in this model dir — falling back to the "
              "|pred| magnitude gate (retrain to enable the two-head gate)")
    return cfg, models


# ============================================================================
# near-cutoff "actual" recompute (lightweight, yfinance only — no full rebuild)
# ============================================================================
def recompute_near_cutoff_actuals(df, until, impact_cols, near_window_min):
    """For posts within `near_window_min` minutes BEFORE `until`, replace the
    stored *_Impact (which may encode info past `until`) with the partial
    reaction post -> min(post+60min, until), via fine yfinance bars truncated
    at `until`. Returns a copy of df with those cells overwritten plus an
    'is_near' column.
    """
    df = df.copy()
    df['is_near'] = False
    elapsed_min = (until - df['date_ny']).dt.total_seconds() / 60.0
    near_mask = elapsed_min < near_window_min
    if not near_mask.any():
        print(f"\n⏱️  No posts within {near_window_min}min of --until — "
              f"all 'actual' values reused from {HS_TABLE} as-is.")
        return df

    near = df[near_mask]
    df.loc[near_mask, 'is_near'] = True
    print(f"\n⏱️  {near_mask.sum()} post(s) within {near_window_min}min of --until — "
          f"recomputing PARTIAL reaction from cached bars (DuckDB, 1min→15min→30min), "
          f"strictly truncated at --until (no bar allowed past the offset)...")

    until_utc = _to_utc(until)
    # One tiny window covers every near post's containing bar + its reaction window.
    lo    = _to_utc(near['date_ny'].min()) - pd.Timedelta(minutes=35)
    names = [c.replace('_Impact', '') for c in impact_cols]

    # Read each instrument's bars for [lo, until] ONCE (cache first; yfinance only
    # for the recent tail the cache doesn't cover).
    slices, n_yf = {}, 0
    for name in names:
        s = _cache_slice(name, lo, until)
        if not s:
            s = _yf_slice(name, lo, until)
            if s:
                n_yf += 1
        slices[name] = s
    if n_yf:
        print(f"   ℹ️  {n_yf} instrument(s) not in the cache for this window — "
              f"used yfinance tail bars (same --until bar cutoff).")

    for idx in near.index:
        post_utc  = _to_utc(df.loc[idx, 'date_ny'])
        react_end = min(post_utc + pd.Timedelta(minutes=REACT_MIN), until_utc)
        for col in impact_cols:
            name = col.replace('_Impact', '')
            move = _reaction_from_slice(slices.get(name, {}), post_utc, until_utc, react_end)
            df.at[idx, col] = move if move is not None else 0.0
    return df


# ============================================================================
# main simulation loop
# ============================================================================
def run_backtest(df, X, cfg, models, impact_cols, dir_threshold, trade_threshold,
                 edge_min=0.05, require_tradeable=True):
    instruments = cfg['instruments']
    label_meta = {inst: (emoji, name) for inst, emoji, name in PR.LABELS}

    # per-instrument magnitude calibration (fitted on the train holdout;
    # actual = k x raw_pred — fixes squared-error shrinkage so predicted
    # PERCENTAGES track actual moves, not just direction)
    _cal = cfg.get("calibration", {})
    # PER-INSTRUMENT FEATURE MATRIX (2026-07-29): models are now trained on
    # [gemma-PCA | NLP | own technical indicators | global VIX/SPY block],
    # so inference must rebuild the SAME layout per instrument. The recipe
    # lives in config.json ("tech_cols" / "global_tech"); when it is absent
    # (older model sets) we fall back to the shared X unchanged.
    _tcols_r = cfg.get("tech_cols") or []
    _gtech_r = cfg.get("global_tech") or []

    def _X_for(inst):
        if not _tcols_r:
            return X
        cols = [f'{inst}_{c}' for c in _tcols_r] + list(_gtech_r)
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return X
        blk = df[cols].apply(pd.to_numeric, errors='coerce')
        blk = blk.fillna(blk.median()).fillna(0.0).values.astype(np.float32)
        return np.hstack([X, blk])

    if _tcols_r:
        _miss = [c for c in ([f'{list(models)[0]}_{t}' for t in _tcols_r] + list(_gtech_r))
                 if c not in df.columns]
        print(f"  📈 TA block per instrument: {len(_tcols_r)} own + {len(_gtech_r)} global"
              + (f"  ⚠️ {len(_miss)} column(s) missing -> 0-filled" if _miss else ""))

    preds = {inst: models[inst].predict(_X_for(inst)) * _cal.get(inst, 1.0)
             for inst in models}
    if _cal:
        print(f"  🎚️  Magnitude calibration applied "
              f"(k per instrument, e.g. " +
              ", ".join(f"{i}×{_cal[i]:.2f}" for i in list(_cal)[:5]) + " ...)")

    # ---- TWO-HEAD CLASSIFIER PROBABILITIES ------------------------------
    # p_move = P(this post produces an abnormal 1h move for this instrument)
    # p_up   = P(that move is upward)
    # The trade decision is (p_move >= gate) AND (|p_up - 0.5| >= edge), and
    # the DIRECTION comes from p_up — not from sign(pred). That is the whole
    # point: the regressor's sign carries almost no information (backtest
    # 20260731 micro-accuracy 48.2%), while direction conditioned on a real
    # move is where the signal actually lives.
    _clf_mv, _clf_dr = cfg.get("_clf_move", {}), cfg.get("_clf_dir", {})
    _gate = cfg.get("move_gate", {})
    p_move = {inst: _clf_mv[inst].predict_proba(_X_for(inst))[:, 1]
              for inst in _clf_mv}
    # ---- head B runs on a LEAN matrix, not the full feature block ---------
    # The direction head is trained on [dir_prior | flags | own TA] because the
    # 128 embedding dims and 59 unsigned NLP scores DILUTE the one signed
    # feature (measured: full NLP 50.4%, prior+flags 53.0%). Feeding it
    # _X_for() here would be a feature-count mismatch — XGBoost either raises
    # or silently scores garbage. Rebuild it exactly as the trainer did.
    def _X_dir_for(inst):
        g = _gate.get(inst, {})
        spec = g.get("dir_features")
        if not spec:
            return _X_for(inst)          # pre-2026-08-05 model dir
        lift = g.get("dir_prior_lift", {}) or {}
        fl = spec.get("flags", []) or []
        blocks = []
        # 1) directional prior = sum of the train-fitted lift of each fired flag
        if spec.get("prior"):
            pr = np.zeros((len(df), 1), dtype=np.float32)
            for f, L in lift.items():
                if f in df.columns:
                    pr[:, 0] += (pd.to_numeric(df[f], errors='coerce')
                                 .fillna(0.0).values.astype(np.float32) > 0) * float(L)
            blocks.append(pr)
        # 2) raw flags, in the trainer's order
        if fl:
            fb = np.zeros((len(df), len(fl)), dtype=np.float32)
            for j, f in enumerate(fl):
                if f in df.columns:
                    fb[:, j] = pd.to_numeric(df[f], errors='coerce').fillna(0.0).values
            blocks.append(fb)
        # 3) this instrument's TA block, same columns the trainer used
        tcols = spec.get("tech", []) or []
        if tcols:
            tb = np.zeros((len(df), len(tcols)), dtype=np.float32)
            for j, cname in enumerate(tcols):
                if cname in df.columns:
                    tb[:, j] = pd.to_numeric(df[cname], errors='coerce').fillna(0.0).values
            blocks.append(tb)
        return np.hstack(blocks).astype(np.float32) if blocks else _X_for(inst)

    p_up = {}
    for inst in _clf_dr:
        Xd = _X_dir_for(inst)
        try:
            p_up[inst] = _clf_dr[inst].predict_proba(Xd)[:, 1]
        except Exception as e:                                # noqa: BLE001
            print(f"  ⚠️  {inst}: direction head expects "
                  f"{getattr(_clf_dr[inst], 'n_features_in_', '?')} features, "
                  f"got {Xd.shape[1]} — retrain or check config['dir_features'] "
                  f"({str(e)[:60]})")
    if p_up:
        _n = _X_dir_for(list(p_up)[0]).shape[1]
        print(f"  🧭 Direction head on lean matrix: {_n} cols "
              f"(prior + flags + own TA), not the {X.shape[1]}-col block")
    # SIZE head: per-post |move| in %, replacing the single per-instrument
    # median. This is what Layer 2 needs for a reachable TP — the simulate log
    # showed TIMEOUT on 148/158 QQQ and 112/115 XLE trades because every trade
    # carried the same median-sized target regardless of the actual setup.
    _reg_sz = cfg.get("_reg_size", {})
    # size_k = the out-of-sample scale correction fitted at train time. Skipping
    # it would systematically under-size every TP by ~20%.
    # size_log: the head was fitted on log1p(|move|) — revert BEFORE size_k.
    def _size_pred(inst):
        g = _gate.get(inst, {})
        # No measurable skill -> a per-post number is noise. Use the empirical
        # median |move| on real events instead (NATGAS shipped MdAPE 124% by
        # trusting a head with corr -0.081).
        if g.get("size_reliable") is False:
            flat = float(g.get("median_abs_move") or 0.0)
            return np.full(len(df), flat, dtype=float)
        raw = _reg_sz[inst].predict(_X_for(inst))
        if g.get("size_log"):
            raw = np.expm1(raw)
        return np.clip(raw, 0.0, None) * float(g.get("size_k", 1.0))

    p_size = {inst: _size_pred(inst) for inst in _reg_sz}
    if p_size:
        _sk = {i: _gate.get(i, {}).get("size_k", 1.0) for i in list(p_size)[:5]}
        print("  📏 Size head scale-corrected: " +
              ", ".join(f"{i}x{v:.2f}" for i, v in _sk.items()) + " ...")
    if p_move:
        _ex = list(p_move)[:4]
        print("  🎯 Classifier gate per instrument: " +
              ", ".join(f"{i} p>={_gate.get(i, {}).get('p_move_thr', 0.5):.2f}"
                        for i in _ex) + " ...")
        _tradeable = [i for i in p_move if _gate.get(i, {}).get("tradeable")]
        if require_tradeable:
            print(f"  ✅ TRADEABLE instruments ({len(_tradeable)}/{len(p_move)}): "
                  + (", ".join(_tradeable) if _tradeable else "NONE")
                  + "\n     (direction head beat chance out-of-sample; the rest are "
                    "skipped — use --trade-all to override)")

    # Chain handling lives in the PRODUCTION module (PR.chain_factor, called
    # by gate_multiplier) — the backtest only resets its in-process state so
    # each run starts clean. df is chronological, so the state replays
    # exactly what a live prediction loop would have seen.
    PR._CHAIN_STATE.clear()

    stats = {inst: {'n': 0, 'match': 0, 'abs_err': 0.0} for inst in instruments}
    # TRADE/SKIP decision based on the model's OWN |predicted move| vs trade_threshold —
    # i.e. "would we have acted on this prediction?" — independent of the gate/temporal
    # mult (which only scales the magnitude) and independent of what 'actual' turned out
    # to be.
    td_stats = {inst: {
        'trade_n': 0, 'trade_match': 0,
        'trade_meaningful_n': 0, 'trade_meaningful_match': 0,
        'skip_n': 0, 'skip_match': 0,
    } for inst in instruments}

    # Per-post rows for the CSV export — wide format, one row per post with
    # <INST>_pred / <INST>_actual / <INST>_match / <INST>_decision columns for
    # every instrument, so all instruments for a post are visible side-by-side.
    csv_rows = []

    _emb_raw = cfg.get("_emb_raw")
    for row_i, (idx, row) in enumerate(df.iterrows()):
        _e = (_emb_raw[row_i] if _emb_raw is not None
              and row_i < len(_emb_raw) else None)
        signal, gate, tfactor, tlabel, mult = gate_multiplier(row, emb=_e)
        near_tag = " [NEAR-CUTOFF, recomputed]" if row.get('is_near') else ""

        # Source / account metadata — pulled from the merged df (HS + scored columns)
        platform  = str(row.get('platform', 'truthsocial'))
        account   = str(row.get('account', _rank0_handle()))
        acct_name = str(row.get('account_name', account))
        # is_primary comes from posts_scored: rank-0 primary WITHIN its active
        # window, on EITHER platform (Trump T2 on TruthSocial, Trump T1 / Biden
        # @POTUS on X in their eras). Don't override with a platform check — a
        # secondary account (rank > 0) or an out-of-window post is not is_primary.
        is_primary = bool(row.get('is_primary', False))
        country = PR._country_for(account)
        ctry_str = f"  [{country}]" if country else ""
        src_emoji = "🇺🇸" if (platform == 'truthsocial') else "🌍"
        src_label = (f"{src_emoji} @{account}{ctry_str}  (TruthSocial)"
                     if platform == 'truthsocial'
                     else f"{src_emoji} @{account} · {acct_name}{ctry_str}  (X/Twitter)")

        print("\n" + "─" * 78)
        # platform:id printed so log entries can be matched 1:1 against the
        # results CSV (id column) and unified_feed / posts_scored rows.
        print(f"#{row_i+1}/{len(df)}  {row['date_ny']:%Y-%m-%d %H:%M %Z}  "
              f"[{platform}:{row['id']}]{near_tag}")
        print(f"  {src_label}")
        print(f"  📝 {str(row['text'])}")
        print(f"  NLP signal={signal:.3f}  gate×{gate:.2f}  temporal={tlabel} ×{tfactor:.2f}  "
              f"-> total mult ×{mult:.2f}")

        csv_row = {
            'id': row['id'],
            'platform': platform,
            'account': account,
            'account_name': acct_name,
            'country': country,
            'date': row['date_ny'].isoformat(),
            'text': row['text'],
            'nlp_signal': signal,
            'gate': gate,
            'temporal_label': tlabel,
            'total_mult': mult,
        }
        damped = mult != 1.0

        print(f"  {'instrument':<10} {'pred':>9} {'actual':>9}  {'match':<6}  {'|err|':>7}  {'decision':<5}")
        for inst in instruments:
            if inst not in models:
                continue
            col = f"{inst}_Impact"
            if col not in df.columns:
                continue
            orig_pred = float(preds[inst][row_i])
            raw_pred = orig_pred * mult
            actual = float(row[col]) if pd.notna(row[col]) else 0.0
            abs_err = abs(raw_pred - actual)

            # ---- direction source: classifier head B when available -------
            # sign(raw_pred) scored 48.2% micro in backtest 20260731 — i.e.
            # worse than a coin. p_up is trained only on rows that really
            # moved, so it is the honest directional call.
            _pm = float(p_move[inst][row_i]) if inst in p_move else None
            _pu = float(p_up[inst][row_i]) if inst in p_up else None
            if _pu is not None:
                trade_dir = 1.0 if _pu > 0.5 else -1.0
                dir_edge = abs(_pu - 0.5)
            else:
                trade_dir = np.sign(raw_pred)
                dir_edge = None

            # Damped posts (chain-follower / endorsement / reiteration / stale)
            # were DELIBERATELY not traded — grading their zeroed prediction
            # against the market's move would pollute accuracy with fake
            # misses. They are excluded from direction stats entirely.
            meaningful = abs(actual) >= dir_threshold and not damped
            if meaningful:
                match = trade_dir == np.sign(actual)
                stats[inst]['n'] += 1
                stats[inst]['match'] += int(match)
                stats[inst]['abs_err'] += abs_err
                match_str = "✅" if match else "❌"
            else:
                match_str = "·"

            # TRADE/SKIP: would we have acted on this prediction at all?
            # A damped post (gate/temporal mult != 1.0) is by definition
            # low-confidence/stale -- never trade it, even if the signal
            # still happens to clear the gate.
            #
            # PRIMARY GATE = two-head classifier: the instrument must be
            # likely to MOVE abnormally *and* the direction head must have a
            # real edge. Falls back to the old magnitude bar only when the
            # model dir predates the classifier heads.
            if _pm is not None and dir_edge is not None:
                _g = _gate.get(inst, {})
                _thr = float(_g.get("p_move_thr", 0.5))
                # An instrument whose direction head failed to beat chance on
                # the held-out calibration slice is not traded at all. Trading
                # a coin flip with a spread is a guaranteed loss; better to run
                # 4 instruments that work than 23 that don't.
                _ok = _g.get("tradeable", True) or not require_tradeable
                # per-instrument edge from the WF sweep (it was tuned jointly
                # with p_move_thr); --edge-min only acts as a floor.
                _em = max(float(_g.get("edge_min", edge_min)), edge_min)
                is_trade = (_ok and _pm >= _thr and dir_edge >= _em
                            and not damped)
            else:
                is_trade = (abs(raw_pred) >= trade_threshold) and not damped
            sign_match = (trade_dir == np.sign(actual)) if actual != 0 else False
            if is_trade:
                td_stats[inst]['trade_n'] += 1
                td_stats[inst]['trade_match'] += int(sign_match)
                if meaningful:
                    td_stats[inst]['trade_meaningful_n'] += 1
                    td_stats[inst]['trade_meaningful_match'] += int(sign_match)
            else:
                td_stats[inst]['skip_n'] += 1
                td_stats[inst]['skip_match'] += int(sign_match)
            decision = "TRADE" if is_trade else "SKIP"

            emoji, name = label_meta.get(inst, ("", inst))
            orig_str = f"  (orig {orig_pred:+.4f}%)" if damped else ""
            _prob_str = (f"  p_mv={_pm:.2f} p_up={_pu:.2f}"
                         if (_pm is not None and _pu is not None) else "")
            print(f"  {emoji} {inst:<8} {raw_pred:>+8.4f}% {actual:>+8.4f}%  "
                  f"{match_str:<6}  {abs_err:>6.4f}  {decision:<5}{_prob_str}{orig_str}")

            csv_row[f'{inst}_pred'] = raw_pred
            csv_row[f'{inst}_actual'] = actual
            csv_row[f'{inst}_match'] = (bool(match) if meaningful else None)
            csv_row[f'{inst}_decision'] = decision
            # Layer 2 reads these: direction to trade, and the expected move
            # size (empirical median |move| given an event) for TP distance —
            # the regressor's collapsed |pred| must NOT size trades any more.
            csv_row[f'{inst}_p_move'] = _pm
            csv_row[f'{inst}_p_up'] = _pu
            csv_row[f'{inst}_dir'] = int(trade_dir) if trade_dir else 0
            # PER-POST predicted size when the size head exists; the flat
            # per-instrument median only as a fallback.
            csv_row[f'{inst}_exp_move'] = (
                float(p_size[inst][row_i]) if inst in p_size
                else _gate.get(inst, {}).get("median_abs_move"))
            csv_row[f'{inst}_size_flat'] = _gate.get(inst, {}).get("median_abs_move")

        csv_rows.append(csv_row)

    return stats, td_stats, csv_rows


def print_size_summary(csv_rows, instruments, dir_threshold):
    """SIZE ACCURACY — the honest answer to "the predicted number is nowhere
    near the actual". The old MAGNITUDE table scored the SIGNED regressor,
    which reads MdAPE 100% on every instrument because it predicts ~0. This
    table scores the SIZE head against |actual|: how big will the move be,
    with the sign supplied separately by the direction head.
    """
    import numpy as _np
    rows = []
    for inst in instruments:
        p, a = [], []
        for r in csv_rows:
            if r.get('total_mult', 1.0) != 1.0:
                continue
            pv, av = r.get(f'{inst}_exp_move'), r.get(f'{inst}_actual')
            if pv is None or av is None or abs(av) < dir_threshold:
                continue
            p.append(float(pv)); a.append(abs(float(av)))
        if len(p) < 8:
            continue
        p, a = _np.array(p), _np.array(a)
        corr = (float(_np.corrcoef(p, a)[0, 1])
                if (p.std() > 1e-12 and a.std() > 1e-12) else float('nan'))
        rows.append((inst, len(p), corr, p.mean() / max(a.mean(), 1e-9),
                     float(_np.median(_np.abs(p - a) / _np.maximum(a, 1e-9))),
                     p.mean(), a.mean()))
    if not rows:
        return
    print("\n" + "=" * 78)
    print("  SIZE ACCURACY  (|predicted move| vs |actual move|) — the number "
          "that matters")
    print("  Direction is supplied separately; this scores magnitude ALONE.")
    print("=" * 78)
    print(f"  {'instrument':<10} {'n':>5} {'corr':>7} {'scale':>7} {'MdAPE':>7} "
          f"{'mean|p|':>9} {'mean|a|':>9}")
    for inst, n, corr, sc, md, mp, ma in sorted(rows, key=lambda r: -r[2]):
        mk = "✅" if corr >= 0.15 else ("🟡" if corr >= 0.08 else "🔴")
        print(f"  {mk}{inst:<9} {n:>5} {corr:>7.3f} {sc:>7.2f} {md:>7.0%} "
              f"{mp:>9.3f} {ma:>9.3f}")
    _c = [r[2] for r in rows if r[2] == r[2]]
    _s = [r[3] for r in rows]
    print("-" * 78)
    print(f"  OVERALL  mean corr={_np.mean(_c):+.3f}  mean scale="
          f"{_np.mean(_s):.2f}x (want ~1.0)")
    print("  Reference: the SIGNED regressor scores corr 0.031 at scale 0.003x "
          "(MdAPE 100%)")
    print("=" * 78)


def print_magnitude_summary(csv_rows, instruments, dir_threshold):
    """
    MAGNITUDE ACCURACY — the second scoreboard Peter asked for: how well do
    predicted PERCENTAGES track actual moves (not just direction)? Per
    instrument, over undamped rows with meaningful actuals:
      k     = actual-vs-pred slope (regression through origin; 1.0 = calibrated)
      corr  = Pearson correlation (direction+shape quality)
      MdAPE = median |pred-actual| / |actual| (typical % error of the magnitude)
    """
    import numpy as _np
    print("\n" + "=" * 78)
    print(f"  MAGNITUDE ACCURACY  (predicted %% vs actual %%, undamped rows, "
          f"|actual| >= {dir_threshold}%)")
    print("=" * 78)
    print(f"  {'instrument':<10} {'n':>5} {'k=act/pred':>10} {'corr':>6} "
          f"{'mean|p|':>8} {'mean|a|':>8} {'MdAPE':>7}")
    agg_p, agg_a = [], []
    for inst in instruments:
        p, a = [], []
        for r in csv_rows:
            if r.get('total_mult', 1.0) != 1.0:
                continue
            pv, av = r.get(f'{inst}_pred'), r.get(f'{inst}_actual')
            if pv is None or av is None or abs(av) < dir_threshold:
                continue
            p.append(pv); a.append(av)
        if len(p) < 8:
            continue
        p, a = _np.array(p), _np.array(a)
        k = float((p * a).sum() / max((p * p).sum(), 1e-9))
        # both sides need variance, else corrcoef divides by zero -> nan
        corr = (float(_np.corrcoef(p, a)[0, 1])
                if (p.std() > 1e-12 and a.std() > 1e-12) else float('nan'))
        _den = _np.abs(a)
        _ok = _den > 1e-12          # zero actuals make MdAPE infinite
        mdape = (float(_np.median(_np.abs(p[_ok] - a[_ok]) / _den[_ok]))
                 if _ok.any() else float('nan'))
        agg_p.append(p); agg_a.append(a)
        flag = "✅" if 0.7 <= k <= 1.4 else ("🟡" if 0.5 <= k <= 2.0 else "🔴")
        print(f"  {flag} {inst:<8} {len(p):>5} {k:>10.2f} {corr:>6.2f} "
              f"{_np.abs(p).mean():>8.3f} {_np.abs(a).mean():>8.3f} {mdape:>6.0%}")
    if agg_p:
        P, A = _np.concatenate(agg_p), _np.concatenate(agg_a)
        K = float((P * A).sum() / max((P * P).sum(), 1e-9))
        print("-" * 78)
        _C = (float(_np.corrcoef(P, A)[0, 1])
              if (P.std() > 1e-12 and A.std() > 1e-12) else float('nan'))
        _D = _np.abs(A); _OK = _D > 1e-12
        _M = (float(_np.median(_np.abs(P[_OK] - A[_OK]) / _D[_OK]))
              if _OK.any() else float('nan'))
        print(f"  OVERALL   n={len(P)}  k={K:.2f} (want ~1.0)  "
              f"corr={_C:.3f}  MdAPE={_M:.0%}"
              + ("" if _OK.all() else
                 f"   [{(~_OK).sum()} zero-actual rows excluded from MdAPE]"))
    print("=" * 78)


def print_summary(stats, dir_threshold):
    print("\n" + "=" * 78)
    print(f"  DIRECTION-MATCH ACCURACY  (|actual| >= {dir_threshold}% counts as 'meaningful')")
    print("=" * 78)
    total_n, total_match = 0, 0
    rows = []
    for inst, s in stats.items():
        if s['n'] == 0:
            continue
        acc = s['match'] / s['n']
        mae = s['abs_err'] / s['n']
        total_n += s['n']
        total_match += s['match']
        rows.append((inst, s['n'], s['match'], acc, mae))

    rows.sort(key=lambda r: -r[3])
    for inst, n, match, acc, mae in rows:
        flag = "✅" if acc >= 0.6 else ("🟡" if acc >= 0.5 else "🔴")
        print(f"  {flag} {inst:<10} n={n:>4}  match={match:>4}  acc={acc:6.1%}  MAE={mae:.4f}")

    if total_n == 0:
        print("\n  ⚠️  No 'meaningful' actual moves in this window "
              "(lower --dir-threshold or widen --from/--until).")
        return

    macro = np.mean([r[3] for r in rows]) if rows else float('nan')
    micro = total_match / total_n
    print("-" * 78)
    print(f"  OVERALL  micro-accuracy={micro:6.1%}  ({total_match}/{total_n})   "
          f"macro-accuracy(mean of instruments)={macro:6.1%}")
    print("=" * 78)


def print_trade_skip_summary(td_stats, trade_threshold, dir_threshold, gate_on=False):
    print("\n" + "=" * 78)
    if gate_on:
        print("  TRADE vs SKIP  (TRADE = P(move) >= per-instrument gate AND the")
        print("                  direction head has an edge; direction from P(up))")
    else:
        print(f"  TRADE vs SKIP  (TRADE = |predicted move| >= {trade_threshold}%, "
              f"based on the model's own prediction)")
    print("=" * 78)

    tot = {'trade_n': 0, 'trade_match': 0,
           'trade_meaningful_n': 0, 'trade_meaningful_match': 0,
           'skip_n': 0, 'skip_match': 0}

    def acc(m, n):
        return m / n if n else float('nan')

    rows = []
    for inst, s in td_stats.items():
        if s['trade_n'] == 0 and s['skip_n'] == 0:
            continue
        for k in tot:
            tot[k] += s[k]
        rows.append((inst, s))

    rows.sort(key=lambda r: -acc(r[1]['trade_match'], r[1]['trade_n']) if r[1]['trade_n'] else 0)

    print(f"  {'instrument':<10} {'TRADE n':>8} {'acc':>7}  {'(meaningful n':>14} {'acc)':>7}  "
          f"{'SKIP n':>8} {'acc':>7}")
    for inst, s in rows:
        ta, tma = acc(s['trade_match'], s['trade_n']), acc(s['trade_meaningful_match'], s['trade_meaningful_n'])
        sa = acc(s['skip_match'], s['skip_n'])
        ta_s = f"{ta:6.1%}" if ta == ta else "   n/a"
        tma_s = f"{tma:6.1%}" if tma == tma else "   n/a"
        sa_s = f"{sa:6.1%}" if sa == sa else "   n/a"
        print(f"  {inst:<10} {s['trade_n']:>8} {ta_s:>7}  {s['trade_meaningful_n']:>13} {tma_s:>7}  "
              f"{s['skip_n']:>8} {sa_s:>7}")

    print("-" * 78)
    overall_trade = acc(tot['trade_match'], tot['trade_n'])
    overall_trade_m = acc(tot['trade_meaningful_match'], tot['trade_meaningful_n'])
    overall_skip = acc(tot['skip_match'], tot['skip_n'])
    print(f"  OVERALL  TRADE  n={tot['trade_n']:>5}  acc={overall_trade:6.1%}   "
          f"(|actual|>={dir_threshold}% subset: n={tot['trade_meaningful_n']:>5}  "
          f"acc={overall_trade_m:6.1%})")
    print(f"           SKIP   n={tot['skip_n']:>5}  acc={overall_skip:6.1%}")
    print("=" * 78)


def print_filtered_accuracy(td_stats, dir_threshold):
    """Dedicated, clearly-labeled table for the "filtered accuracy" -- TRADE
    calls whose actual move was also >= dir_threshold (i.e. excludes TRADE
    calls on noise-sized moves). This is the same trade_meaningful_n/acc data
    written to trade_accuracy.json, printed directly so you don't have to open
    the JSON file to see it.
    """
    def acc(m, n):
        return m / n if n else float('nan')

    rows = []
    tot_n, tot_m = 0, 0
    for inst, s in td_stats.items():
        n, m = s['trade_meaningful_n'], s['trade_meaningful_match']
        if n == 0:
            continue
        rows.append((inst, n, m, acc(m, n)))
        tot_n += n
        tot_m += m

    rows.sort(key=lambda r: -r[3])

    print("\n" + "=" * 78)
    print(f"  FILTERED ACCURACY  (TRADE calls AND |actual| >= {dir_threshold}% -- "
          f"excludes noise-sized moves)")
    print("=" * 78)
    print(f"  {'instrument':<10} {'n':>6} {'acc':>7}")
    for inst, n, m, a in rows:
        print(f"  {inst:<10} {n:>6} {a:6.1%}")
    print("-" * 78)
    overall = acc(tot_m, tot_n)
    print(f"  OVERALL    n={tot_n:>5}  acc={overall:6.1%}")
    print("=" * 78)


def write_trade_accuracy(td_stats, model_dir, trade_threshold, since, until):
    """Persist per-instrument TRADE/SKIP "filtered accuracy" (the same numbers
    print_trade_skip_summary prints) to <model_dir>/trade_accuracy.json, so
    predict_gemma_nlp_xgb.py can show, alongside a live prediction's
    TRADE/SKIP decision, how accurate that instrument's TRADE-flagged
    predictions have historically been in this backtest window.
    """
    def acc(m, n):
        return m / n if n else None

    out = {
        'window': {'since': since.isoformat(), 'until': until.isoformat()},
        'trade_threshold': trade_threshold,
        'instruments': {},
    }
    for inst, s in td_stats.items():
        if s['trade_n'] == 0 and s['skip_n'] == 0:
            continue
        out['instruments'][inst] = {
            'trade_n': s['trade_n'], 'trade_acc': acc(s['trade_match'], s['trade_n']),
            # "Filtered" accuracy: TRADE-flagged AND |actual| >= dir_threshold
            # (i.e. excludes TRADE calls on moves so small they're noise) --
            # this is the higher, more meaningful number shown by
            # print_trade_skip_summary's "(|actual|>=X% subset: ...)" column.
            'trade_meaningful_n': s['trade_meaningful_n'],
            'trade_meaningful_acc': acc(s['trade_meaningful_match'], s['trade_meaningful_n']),
            'skip_n': s['skip_n'], 'skip_acc': acc(s['skip_match'], s['skip_n']),
        }

    path = os.path.join(model_dir, "trade_accuracy.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"  💾 Wrote TRADE/SKIP filtered accuracy -> {path}")


def write_csv(csv_rows, csv_path, instruments):
    """Write the per-post backtest rows to a CSV file (wide format: one row
    per post, with <INST>_pred / <INST>_actual / <INST>_match / <INST>_decision
    columns for every instrument, side-by-side). This is for inspection /
    later charting only — NOT written to database.db, keeping the DB light.
    """
    import csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = ['id', 'platform', 'account', 'account_name', 'country',
                  'date', 'text',
                  'nlp_signal', 'gate', 'temporal_label', 'total_mult']
    for inst in instruments:
        fieldnames += [f'{inst}_pred', f'{inst}_actual', f'{inst}_match',
                       f'{inst}_decision',
                       # two-head classifier outputs (Layer 2 sizes from these)
                       f'{inst}_p_move', f'{inst}_p_up', f'{inst}_dir',
                       f'{inst}_exp_move', f'{inst}_size_flat']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  💾 Wrote {len(csv_rows)} row(s) to {csv_path}")


# ============================================================================
# fine-tuning (continued boosting on this window's data)
# ============================================================================
def fine_tune(df, X, cfg, model_dir, live_dir, rounds, impact_cols):
    print("\n" + "=" * 78)
    print(f"  FINE-TUNE: continued boosting (+{rounds} trees) on this window's "
          f"{len(df)} post(s) -> {live_dir}")
    print("=" * 78)
    os.makedirs(live_dir, exist_ok=True)

    if not os.path.exists(os.path.join(live_dir, "config.json")):
        shutil.copy2(os.path.join(model_dir, "config.json"), os.path.join(live_dir, "config.json"))
        print(f"  📋 Initialized {live_dir}/config.json from {model_dir}")

    w = df['sample_weight'].fillna(0.3).values if 'sample_weight' in df.columns else None
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for inst in cfg['instruments']:
        col = f"{inst}_Impact"
        if col not in df.columns:
            continue
        src = os.path.join(live_dir, f"{inst}_Impact.json")
        if not os.path.exists(src):
            src = os.path.join(model_dir, f"{inst}_Impact.json")
        if not os.path.exists(src):
            continue

        y = df[col].fillna(0.0).values
        if np.allclose(y, 0.0):
            print(f"  ⏭️  {inst:<10} skipped (all-zero labels in this window)")
            continue

        dst = os.path.join(live_dir, f"{inst}_Impact.json")
        if os.path.exists(dst):
            shutil.copy2(dst, dst + f".bak_{ts}")

        m = xgb.XGBRegressor(
            n_estimators=rounds, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.5, objective='reg:squarederror',
            n_jobs=-1, random_state=42)

        prev = xgb.XGBRegressor()
        prev.load_model(src)
        before = prev.predict(X)

        m.fit(X, y, sample_weight=w, xgb_model=src)
        m.save_model(dst)
        after = m.predict(X)

        shift = np.mean(np.abs(after - before))
        print(f"  🔧 {inst:<10} +{rounds} trees  mean|Δpred|={shift:.4f}  -> {dst}")

    print(f"\n  💾 Live model dir: {live_dir}")
    print(f"     Point predict_gemma_nlp_xgb.py / future backtests at this dir "
          f"with --model-dir {os.path.relpath(live_dir, _HERE)} to use the fine-tuned weights.")


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="HIGH_SIGNAL backtest simulator (time-ranged, label-reuse + near-cutoff recompute).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("until", help="yyyymmddhhmm, NY local time — simulated 'now' / end of window.")
    ap.add_argument("--from", dest="since", metavar="yyyymmddhhmm",
                    help="yyyymmddhhmm, NY local time — start of window. "
                         "Default: 30 days before --until.")
    ap.add_argument("--near-window-min", type=int, default=NEAR_WINDOW_DEFAULT,
                    help=f"Posts tweeted within this many minutes before --until get their "
                         f"'actual' recomputed point-in-time (default {NEAR_WINDOW_DEFAULT}, "
                         f"matching the 1h reaction window).")
    ap.add_argument("--dir-threshold", type=float, default=DIR_THRESHOLD_DEFAULT,
                    help=f"Minimum |actual %% move| to count toward direction accuracy "
                         f"(default {DIR_THRESHOLD_DEFAULT}, matches training eval).")
    ap.add_argument("--trade-threshold", type=float, default=None,
                    help=f"Minimum |predicted %% move| to count as a TRADE decision "
                         f"(vs SKIP) in the TRADE/SKIP summary. Default: same as "
                         f"--dir-threshold ({DIR_THRESHOLD_DEFAULT}).")
    ap.add_argument("--edge-min", type=float, default=0.05,
                    help="Two-head gate: minimum |P(up) - 0.5| from the direction "
                         "head for a TRADE. 0.05 = model must be at least 55/45. "
                         "Raise it to trade less but more confidently (default 0.05).")
    ap.add_argument("--trade-all", action="store_true",
                    help="Trade every instrument, including those whose direction "
                         "head failed to beat chance out-of-sample. Default is to "
                         "trade only instruments flagged tradeable in config.json.")
    ap.add_argument("--model-dir", default=OUT_DIR,
                    help=f"Directory with <INST>_Impact.json + config.json (default {OUT_DIR}).")
    ap.add_argument("--fine-tune", action="store_true",
                    help="After the backtest, continue-train each model on this window's data "
                         "and save to --fine-tune-out.")
    ap.add_argument("--fine-tune-rounds", type=int, default=50,
                    help="Extra boosting rounds (trees) added per instrument (default 50). "
                         "Use a SMALL number (e.g. 5-10) for short/hourly windows.")
    ap.add_argument("--fine-tune-out", default=LIVE_DIR,
                    help=f"Output dir for fine-tuned models (default {LIVE_DIR}).")
    ap.add_argument("--csv-out", default=None, metavar="PATH",
                    help="Write per-post/per-instrument backtest results (truth id, datetime, "
                         "text, predicted/actual, match, decision) to this CSV file for "
                         "inspection or charting. NOT written to database.db. Default: "
                         "backtest_results/backtest_<until>.csv. Pass an empty string '' "
                         "to skip CSV output.")
    args = ap.parse_args()

    until = parse_stamp(args.until, "until")
    since = parse_stamp(args.since, "--from") if args.since else until - pd.Timedelta(days=30)
    if until > pd.Timestamp.now(tz=NY):
        sys.exit("❌ --until is in the future — pick a past/present moment.")
    if since >= until:
        sys.exit("❌ --from must be before --until.")

    print("=" * 78)
    print("  HIGH-SIGNAL BACKTEST SIMULATOR")
    print(f"  Window: {since}  ->  {until}")
    print(f"  Near-cutoff recompute window: last {args.near_window_min} min before --until")
    print(f"  Direction-match threshold: |actual| >= {args.dir_threshold}%")
    print(f"  Model dir: {args.model_dir}")
    print("=" * 78)

    df = load_window(since, until)
    instruments = [n for n in B.TICKERS]
    impact_cols = [f"{n}_Impact" for n in instruments]

    df = recompute_near_cutoff_actuals(df, until, impact_cols, args.near_window_min)

    X_emb = load_embeddings(df)

    cfg, models = load_models(args.model_dir, instruments)
    use_nlp = [c for c in cfg['nlp_features'] if c in df.columns]
    missing_nlp = [c for c in cfg['nlp_features'] if c not in df.columns]
    if missing_nlp:
        print(f"  ⚠️  {len(missing_nlp)} NLP feature(s) absent from {SCORED_TABLE} "
              f"(0-filled): {missing_nlp[:5]}{'...' if len(missing_nlp) > 5 else ''}")
        for c in missing_nlp:
            df[c] = 0.0
        use_nlp = cfg['nlp_features']
    X_nlp = df[use_nlp].fillna(0.0).values.astype(np.float32)
    # gemma sentiment features are NOT in the DB — compute them from the
    # raw CLS vectors exactly as train did (must happen BEFORE projection)
    _sent = PR.gemma_sentiment_from_emb(X_emb, cfg)
    if _sent is not None:
        for _i, _n in enumerate(('gemma_pos', 'gemma_neg', 'gemma_sent')):
            if _n in use_nlp:
                X_nlp[:, use_nlp.index(_n)] = _sent[:, _i]
    # GEMMA ANALYST features are not stored in posts_scored — fill them from
    # the gemma3_analyst_v1 cache (generating any missing posts, checkpointed)
    _acols = [c for c in use_nlp if c.startswith('analyst_')]
    if _acols:
        import gemma_analyst as GA
        _pids = (df['platform'] + '_' + df['id'].astype(str)).tolist()
        _A = GA.analyst_features(
            _pids, df['text'].tolist(),
            accounts=df['account'].tolist() if 'account' in df.columns else None)
        for _c in _acols:
            X_nlp[:, use_nlp.index(_c)] = _A[:, GA.ANALYST_COLS.index(_c)]
    # Keep the RAW (pre-PCA) embeddings for the semantic chain guard. PCA drops
    # 31.3% of the variance, and the restatement test wants the full vector.
    cfg["_emb_raw"] = X_emb
    X_emb = PR.project_emb(X_emb, cfg)   # PCA projection if model was trained compressed
    X = np.hstack([X_emb, X_nlp])
    print(f"\n  Feature matrix: {X.shape} "
          f"({X_emb.shape[1]} gemma + {X_nlp.shape[1]} NLP)")

    trade_threshold = args.trade_threshold if args.trade_threshold is not None else args.dir_threshold

    stats, td_stats, csv_rows = run_backtest(df, X, cfg, models, impact_cols,
                                             args.dir_threshold, trade_threshold,
                                             edge_min=args.edge_min,
                                             require_tradeable=not args.trade_all)

    print_summary(stats, args.dir_threshold)
    print_magnitude_summary(csv_rows, [i for i in instruments if i in models],
                            args.dir_threshold)
    print_size_summary(csv_rows, [i for i in instruments if i in models],
                       args.dir_threshold)
    print_trade_skip_summary(td_stats, trade_threshold, args.dir_threshold,
                             gate_on=bool(cfg.get("_clf_move")))
    print_filtered_accuracy(td_stats, args.dir_threshold)
    write_trade_accuracy(td_stats, args.model_dir, trade_threshold, since, until)

    if args.csv_out != "":
        since_str = since.strftime("%Y%m%d%H%M")
        until_str = until.strftime("%Y%m%d%H%M")
        csv_path = args.csv_out or os.path.join(
            _HERE, "backtest_results", f"backtest_{since_str}_to_{until_str}.csv")
        write_csv(csv_rows, csv_path, [inst for inst in instruments if inst in models])

    if args.fine_tune:
        fine_tune(df, X, cfg, args.model_dir, args.fine_tune_out,
                  args.fine_tune_rounds, impact_cols)


if __name__ == "__main__":
    main()