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
  1. Build the model input: cached FinBERT embedding (finbert_embeddings_v2)
     + NLP scorer features (posts_scored), exactly as in training.
  2. Predict per-instrument 1h impact with the trained XGBoost models
     (finbert_nlp_xgb_models/ by default, or --model-dir), applying the SAME
     NLP gate + temporal gate used by predict_finbert_nlp_xgb.py.
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
finbert_nlp_xgb_models_live/ (created as a copy of --model-dir on first use;
the daily full retrain in train_finbert_nlp_xgb.py is untouched — it still
writes to finbert_nlp_xgb_models/). Each fine-tune run backs up the previous
live model as <inst>_Impact.json.bak_<timestamp> before overwriting.

Suggested cadence:
  - Daily:  uv run python train_finbert_nlp_xgb.py            (full retrain, baseline dir)
  - Hourly: uv run python backtest_simulator.py <now> --from <now-1h> --fine-tune --fine-tune-rounds 8
            (small continued-boosting nudge on the live dir using whatever
             HIGH_SIGNAL posts appeared in the last hour)

This script does NOT modify build_test_training_set.py, train_finbert_nlp_xgb.py,
or predict_finbert_nlp_xgb.py — the existing manual point-in-time testing
workflow (build_test_training_set.py -> train_finbert_nlp_xgb.py ->
predict_finbert_nlp_xgb.py) remains exactly as before.

USAGE
-----
  uv run python backtest_simulator.py 202606140900
  uv run python backtest_simulator.py 202606140900 --from 202605010000
  uv run python backtest_simulator.py 202606140900 --from 202606130900 --fine-tune
"""
import os
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
import predict_finbert_nlp_xgb as PR   # LABELS, temporal_factor, GATE_*

NY = 'America/New_York'

HS_TABLE     = "training_set_HIGH_SIGNAL"
SCORED_TABLE = "posts_scored"
EMB_TABLE    = "finbert_embeddings_v2"

OUT_DIR  = os.path.join(_HERE, "finbert_nlp_xgb_models")        # daily-retrained baseline
LIVE_DIR = os.path.join(_HERE, "finbert_nlp_xgb_models_live")   # fine-tuned, updated by this script

FINBERT = "ProsusAI/finbert"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

_ENTITIES_FILE = os.path.join(_DP, "geopolitical_entities.json")


def _rank0_handle() -> str:
    """Return the rank-0 TruthSocial account handle from geopolitical_entities.json."""
    try:
        import json
        with open(_ENTITIES_FILE, encoding="utf-8") as f:
            accounts = json.load(f).get("primary_accounts", [])
        ts = sorted([a for a in accounts if a.get("platform") == "truthsocial"],
                    key=lambda a: a.get("rank", 99))
        return ts[0]["account"] if ts else "us_president"
    except Exception:
        return "us_president"


# ============================================================ yfinance helpers ----
# Inlined from the former build_test_training_set.py (deleted — no longer a separate file).
# Used only by recompute_near_cutoff_actuals() to refresh partial reaction bars
# for posts within --near-window-min of the --until cutoff.

FINE_REACH_DAYS = 59   # yfinance fine intervals (≤30m) go back ~60 calendar days

BAR = pd.Timedelta('30min')   # default fallback bar width


def pick_interval(elapsed_min: float) -> str:
    """Choose the finest yfinance interval that covers elapsed_min minutes."""
    if elapsed_min <= 5:
        return '1m'
    if elapsed_min <= 15:
        return '5m'
    if elapsed_min <= 30:
        return '15m'
    return '30m'


def fetch_fine(interval: str, until: pd.Timestamp) -> dict:
    """
    Fetch intraday bars at `interval` for all B.TICKERS up to `until`.
    Returns {short_name: DataFrame} or {} on failure.
    """
    import yfinance as yf
    lookback = 7 if interval == '1m' else 60   # yfinance 1m limit = 7 days
    start = until - pd.Timedelta(days=lookback)
    result = {}
    for name, (yf_sym, _) in B.TICKERS.items():
        try:
            df = yf.download(yf_sym,
                             start=start,
                             end=until + pd.Timedelta(hours=1),
                             interval=interval,
                             auto_adjust=True, progress=False)
            if df.empty:
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert(NY)
            result[name] = df
        except Exception:
            pass
    return result


def short_reaction_move(bars, width: pd.Timedelta,
                        post_ts: pd.Timestamp, until: pd.Timestamp):
    """
    % price move starting at post_ts over `width`, truncated at `until`.
    Returns None if no bars fall in the window.
    """
    if bars is None or (hasattr(bars, 'empty') and bars.empty):
        return None
    end = min(post_ts + width, until)
    window = bars[(bars.index >= post_ts) & (bars.index <= end)]
    if len(window) < 1:
        return None
    open_price  = float(window['Close'].iloc[0])
    close_price = float(window['Close'].iloc[-1])
    if open_price == 0:
        return None
    return (close_price - open_price) / open_price * 100.0

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


_tok = _bert = None


def _load_finbert():
    global _tok, _bert
    if _bert is not None:
        return
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"📥 Loading FinBERT on {DEVICE} (for posts missing cached embeddings)...")
    _tok = AutoTokenizer.from_pretrained(FINBERT)
    _bert = AutoModelForSequenceClassification.from_pretrained(
        FINBERT, output_hidden_states=True).to(DEVICE).eval()


def embed_texts(texts):
    """CLS + mean pooling — must match training v2 (see train_finbert_nlp_xgb.py)."""
    _load_finbert()
    embs, batch = [], 32
    for i in range(0, len(texts), batch):
        chunk = [str(t)[:512] for t in texts[i:i + batch]]
        enc = _tok(chunk, return_tensors="pt", padding=True, truncation=True,
                   max_length=256).to(DEVICE)
        with torch.no_grad():
            out = _bert(**enc)
        last = out.hidden_states[-1]
        cls = last[:, 0, :]
        mask = enc['attention_mask'].unsqueeze(-1).float()
        mean = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs.append(torch.cat([cls, mean], dim=1).cpu().numpy())
    return np.vstack(embs)


def gate_multiplier(row):
    """Same NLP-gate + temporal-gate formula as predict_finbert_nlp_xgb.predict()."""
    import math
    parts = []
    if pd.notna(row.get('policy_intensity_score')):
        parts.append(min(float(row['policy_intensity_score']) / 8.0, 1.0))
    if pd.notna(row.get('hawkish_risk_score')):
        parts.append(min(float(row['hawkish_risk_score']) / 5.0, 1.0))
    if pd.notna(row.get('sample_weight')):
        parts.append(float(row['sample_weight']))
    signal = float(np.mean(parts)) if parts else 0.5
    if PR.GATE_ENABLED:
        gate = 1.0 / (1.0 + math.exp(-PR.GATE_K * (signal - PR.GATE_MID)))
    else:
        gate = 1.0
    mult = 1.0 if gate >= 0.5 else gate
    tfactor, tlabel = PR.temporal_factor(str(row['text']), post_hour=row['date_ny'].hour)
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
              f"computing live with FinBERT...")
        fresh = embed_texts(df.loc[missing_idx, 'text'].tolist())
        for j, i in enumerate(missing_idx):
            cache[platform_ids[i]] = fresh[j]
    else:
        print(f"  ✅ All {len(platform_ids)} embeddings found in cache")

    return np.vstack([cache[pid] for pid in platform_ids])


def load_models(model_dir, instruments):
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"❌ {cfg_path} not found — train models first (train_finbert_nlp_xgb.py).")
    cfg = json.load(open(cfg_path))
    models = {}
    for inst in instruments:
        p = os.path.join(model_dir, f"{inst}_Impact.json")
        if os.path.exists(p):
            m = xgb.XGBRegressor()
            m.load_model(p)
            models[inst] = m
    print(f"  ✅ Loaded {len(models)}/{len(instruments)} models from {model_dir}")
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
          f"recomputing PARTIAL reaction with fine yfinance bars (truncated at --until)...")

    days_back = (pd.Timestamp.now(tz=NY) - until).days
    fine_ok = days_back <= FINE_REACH_DAYS
    intervals_needed = sorted({pick_interval(min(m, 60)) for m in elapsed_min[near_mask]})

    fine = {}
    fallback30 = None
    if fine_ok:
        for iv in intervals_needed:
            fine[iv] = fetch_fine(iv, until)
    else:
        print(f"   ⚠️  --until is {days_back} days back — beyond yfinance's "
              f"~{FINE_REACH_DAYS}-day fine-interval reach; using 30m bars only.")
    # always have a 30m fallback ready (cheap single call per ticker, only fetched if needed)
    def get_fallback30():
        nonlocal fallback30
        if fallback30 is None:
            fallback30 = fetch_fine('30m', until)
        return fallback30

    for idx in near.index:
        row = df.loc[idx]
        elapsed = min(elapsed_min[idx], 60)
        iv = pick_interval(elapsed)
        width = pd.Timedelta(minutes=int(iv[:-1]))
        for col in impact_cols:
            name = col.replace('_Impact', '')
            bars = fine.get(iv, {}).get(name)
            move = short_reaction_move(bars, width, row['date_ny'], until)
            if move is None:
                fb = get_fallback30().get(name)
                move = short_reaction_move(fb, BAR, row['date_ny'], until)
            df.at[idx, col] = move if move is not None else 0.0
    return df


# ============================================================================
# main simulation loop
# ============================================================================
def run_backtest(df, X, cfg, models, impact_cols, dir_threshold, trade_threshold):
    instruments = cfg['instruments']
    label_meta = {inst: (emoji, name) for inst, emoji, name in PR.LABELS}

    preds = {inst: models[inst].predict(X) for inst in models}

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

    for row_i, (idx, row) in enumerate(df.iterrows()):
        signal, gate, tfactor, tlabel, mult = gate_multiplier(row)
        near_tag = " [NEAR-CUTOFF, recomputed]" if row.get('is_near') else ""

        # Source / account metadata — pulled from the merged df (HS + scored columns)
        platform  = str(row.get('platform', 'truthsocial'))
        account   = str(row.get('account', _rank0_handle()))
        acct_name = str(row.get('account_name', account))
        # is_primary comes from posts_scored (rank-0 TruthSocial only).
        # Don't override with platform check — a secondary TruthSocial account
        # (rank > 0, e.g. a future candidate) is not is_primary.
        is_primary = bool(row.get('is_primary', False))
        country = PR._country_for(account)
        ctry_str = f"  [{country}]" if country else ""
        src_emoji = "🇺🇸" if (platform == 'truthsocial') else "🌍"
        src_label = (f"{src_emoji} @{account}{ctry_str}  (TruthSocial)"
                     if platform == 'truthsocial'
                     else f"{src_emoji} @{account} · {acct_name}{ctry_str}  (X/Twitter)")

        print("\n" + "─" * 78)
        print(f"#{row_i+1}/{len(df)}  {row['date_ny']:%Y-%m-%d %H:%M %Z}{near_tag}")
        print(f"  {src_label}")
        print(f"  📝 {str(row['text'])[:140]}{'...' if len(str(row['text']))>140 else ''}")
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

            meaningful = abs(actual) >= dir_threshold
            if meaningful:
                match = np.sign(raw_pred) == np.sign(actual)
                stats[inst]['n'] += 1
                stats[inst]['match'] += int(match)
                stats[inst]['abs_err'] += abs_err
                match_str = "✅" if match else "❌"
            else:
                match_str = "·"

            # TRADE/SKIP: would we have acted on this prediction at all?
            # A damped post (gate/temporal mult != 1.0) is by definition
            # low-confidence/stale -- never trade it, even if the scaled
            # |raw_pred| still happens to clear trade_threshold.
            is_trade = (abs(raw_pred) >= trade_threshold) and not damped
            sign_match = (np.sign(raw_pred) == np.sign(actual)) if actual != 0 else False
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
            print(f"  {emoji} {inst:<8} {raw_pred:>+8.4f}% {actual:>+8.4f}%  "
                  f"{match_str:<6}  {abs_err:>6.4f}  {decision:<5}{orig_str}")

            csv_row[f'{inst}_pred'] = raw_pred
            csv_row[f'{inst}_actual'] = actual
            csv_row[f'{inst}_match'] = (bool(match) if meaningful else None)
            csv_row[f'{inst}_decision'] = decision

        csv_rows.append(csv_row)

    return stats, td_stats, csv_rows


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


def print_trade_skip_summary(td_stats, trade_threshold, dir_threshold):
    print("\n" + "=" * 78)
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
    predict_finbert_nlp_xgb.py can show, alongside a live prediction's
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
        fieldnames += [f'{inst}_pred', f'{inst}_actual', f'{inst}_match', f'{inst}_decision']
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
    print(f"     Point predict_finbert_nlp_xgb.py / future backtests at this dir "
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
    X = np.hstack([X_emb, X_nlp])
    print(f"\n  Feature matrix: {X.shape} "
          f"({X_emb.shape[1]} FinBERT + {X_nlp.shape[1]} NLP)")

    trade_threshold = args.trade_threshold if args.trade_threshold is not None else args.dir_threshold

    stats, td_stats, csv_rows = run_backtest(df, X, cfg, models, impact_cols, args.dir_threshold, trade_threshold)

    print_summary(stats, args.dir_threshold)
    print_trade_skip_summary(td_stats, trade_threshold, args.dir_threshold)
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