"""
build_final_training_set.py
----------------------------
Builds the final training CSV using EVENT STUDY methodology
from financial econometrics (Brown & Warner 1985, MacKinlay 1997).

Core principle — a post's impact is real ONLY when TWO things agree:
  1. NLP signal      — the post contains genuine market-moving content
  2. Abnormal return — the market moved beyond its normal noise band,
                       AND that move is statistically significant (|t| > threshold)

If they DON'T agree → impact ≈ 0 (honest: the post didn't move markets)

This fixes the confounding-events problem: a noise post (endorsement)
that happens to sit near a real market move caused by SOMETHING ELSE
no longer gets credit for that move — because its NLP signal is low.

Key concepts from research:
  - Abnormal Return (AR) = actual return − expected (baseline) return
  - Significance via t-stat = AR / std(returns in estimation window)
  - Confounding control: NLP signal acts as the attribution filter

Outputs: training_set_FINAL.csv        (no hard tiers, continuous honest labels)
         training_set_HIGH_SIGNAL.csv  (sample_weight > 0.5 subset)
         posts_labeled                        (full metadata; also the incremental dedup ledger)

CLI:
  python build_final_training_set.py          -> INCREMENTAL (daily): label only posts in
                                                 posts_scored not yet in
                                                 posts_labeled, APPEND to all 3 outputs
  python build_final_training_set.py --full   -> full rebuild (rewrites all 3 outputs)

DAILY PIPELINE: run_daily_truths.bat runs daily_truths_retriever.py, then
signal_scorer.py (incremental), then this script (incremental).

MATURITY RULE (incremental only): a post is labeled only when its market
reaction is actually measurable — otherwise it is DEFERRED to the next run
instead of being mislabeled 0:
  * posts < 2h old                      -> deferred (1h reaction window incomplete)
  * after-hours posts                   -> deferred until the NEXT trading day's
                                           SPY daily bar exists (their label is the
                                           next session's reaction)
  * all of SPY/VIX/OIL returned no data -> deferred
  * posts older than 7 days are labeled unconditionally (matches --full behavior)

INTRADAY DATA = IBKR cache + yfinance extension: the IBKR cache is static
(ends 2026-05-22 unless re-downloaded), and yfinance intraday only reaches back
~60 days. Each run therefore fetches the latest yfinance 30min bars and APPENDS
the ones past the cache end to {name}_30min_yf.csv in the same cache dir —
coverage accumulates daily with no gap, and the IBKR files are never modified.
A guard returns None when a post lands at/after the final cached bar, so labels
can never again be frozen at a stale bar.

  python build_final_training_set.py --relabel-since 2026-05-23
    -> one-time repair: drops output rows from that date on and re-labels them
       with the now-complete intraday data.
"""

import os
import sys
import datetime
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

# Windows: when stdout is redirected to a log file Python defaults to cp1252,
# which can't encode the emoji in our log lines. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ==========================================
# CONFIG / PATHS (resolve next to this file)
# ==========================================
_HERE       = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import db  # DuckDB helper (DP/db.py) -> ../database.db
import json

CACHE_DIR      = os.path.join(_HERE, "..", "IBKR", "market_data_cache")
_ENTITIES_FILE = os.path.join(_HERE, "geopolitical_entities.json")


def _rank0_handle() -> str:
    """Return the rank-0 TruthSocial account handle from geopolitical_entities.json.
    Falls back to 'us_president' if the file is missing or has no primary accounts."""
    try:
        with open(_ENTITIES_FILE, encoding="utf-8") as f:
            accounts = json.load(f).get("primary_accounts", [])
        ts = sorted([a for a in accounts if a.get("platform") == "truthsocial"],
                    key=lambda a: a.get("rank", 99))
        return ts[0]["account"] if ts else "us_president"
    except Exception:
        return "us_president"

# DuckDB table names
# posts_scored   unified NLP-scored posts (TruthSocial primary accounts + X/Twitter geo)
# posts_labeled  posts joined with market-impact labels for model training
SCORED_TABLE   = "posts_scored"
FINAL_TABLE    = "training_set_FINAL"
HS_TABLE       = "training_set_HIGH_SIGNAL"
LABELED_TABLE  = "posts_labeled"

# Baseline = simple average move of same instrument over prior 30 days.
# Subtracted from actual move to get the abnormal (US President-attributable) return.
BASELINE_DAYS        = 30    # prior 30 calendar days for expected-return baseline
DOMINANCE_WINDOW_HRS = 2
NLP_FLOOR_MULT       = 0.05

# ------------------------------------------------------------------------
# INSTRUMENTS — loaded DYNAMICALLY from DP/instruments.json (master registry:
# yf symbol, market type, impact_cap, core flag, IBKR/Binance specs, emoji).
# Add or remove an instrument THERE — no code edits needed anywhere.
# ------------------------------------------------------------------------
_INSTRUMENTS_FILE = os.path.join(_HERE, "instruments.json")
with open(_INSTRUMENTS_FILE, encoding="utf-8") as _f:
    INSTRUMENT_REGISTRY = json.load(_f)["instruments"]

# {name: (yfinance_symbol, market_type)} — 'us' = US session, '24h' = always on
TICKERS = {k: (v["yf"], v["market"]) for k, v in INSTRUMENT_REGISTRY.items()}

# Realistic 1hr move ceilings — caps daily-data noise on old posts.
IMPACT_CAP = {k: float(v["impact_cap"]) for k, v in INSTRUMENT_REGISTRY.items()}

# Pipeline core (labels deferred when ALL core instruments lack data)
CORE_INSTRUMENTS = [k for k, v in INSTRUMENT_REGISTRY.items() if v.get("core")]

# Market data containers — populated by fetch_market(), read by abnormal_* fns.
intraday, daily = {}, {}
baseline_30d, volatility_30d = {}, {}


# ==========================================
# LOAD SCORED DATA
# ==========================================
def market_session(dt):
    t = dt.time()
    if t < datetime.time(9, 30):   return 'pre_market'
    elif t <= datetime.time(16, 0): return 'during_market'
    else:                           return 'after_hours'


def compute_nlp_signal(row):
    """Normalized NLP signal per post (0-1): policy intensity + hawkish risk + sample weight."""
    parts = []
    if 'policy_intensity_score' in row.index and pd.notna(row['policy_intensity_score']):
        parts.append(min(row['policy_intensity_score'] / 8.0, 1.0))  # ~8 = strong
    if 'hawkish_risk_score' in row.index and pd.notna(row['hawkish_risk_score']):
        parts.append(min(row['hawkish_risk_score'] / 5.0, 1.0))
    if 'sample_weight' in row.index and pd.notna(row['sample_weight']):
        parts.append(row['sample_weight'])
    return float(np.mean(parts)) if parts else 0.0


def load_scored():
    print(f"\n📂 Loading {SCORED_TABLE} from {db.DB_PATH}...")
    scored = db.read_table(SCORED_TABLE)
    if scored is None:
        print(f"❌ {SCORED_TABLE} not found — run signal_scorer.py first")
        sys.exit(1)
    scored['id'] = pd.to_numeric(scored['id'], errors='coerce').astype('Int64')
    if 'platform' not in scored.columns:
        scored['platform'] = 'truthsocial'
    if 'is_primary' not in scored.columns:
        # rank-0 TruthSocial only — secondary TS accounts are NOT is_primary
        _ar = scored['account_rank'].fillna(99).astype(float) if 'account_rank' in scored.columns else pd.Series(99.0, index=scored.index)
        scored['is_primary'] = (scored['platform'] == 'truthsocial') & (_ar == 0)
    if 'entity_weight' not in scored.columns:
        scored['entity_weight'] = 1.0
    if 'event_weight' not in scored.columns:
        scored['event_weight'] = 1.0
    scored['date'] = pd.to_datetime(scored['date'], format='mixed', utc=True).dt.tz_convert('America/New_York')
    scored = scored.sort_values('date').reset_index(drop=True)
    n_primary = int(scored['is_primary'].sum())
    n_twitter = len(scored) - n_primary
    print(f"  Posts loaded: {len(scored)}  (TruthSocial: {n_primary} | X/Twitter: {n_twitter})")
    scored['session']    = scored['date'].apply(market_session)
    scored['nlp_signal'] = scored.apply(compute_nlp_signal, axis=1)
    print(f"  NLP signal: mean={scored['nlp_signal'].mean():.3f}, "
          f">0.3: {(scored['nlp_signal']>0.3).sum()} posts")
    return scored


# ==========================================
# FETCH DATA
# ==========================================
def load_cached_intraday(name, suffix="", after=None, before=None):
    """
    Load cached 30min bars for `name` if present (IBKR file, or the yfinance
    extension file when suffix='_yf').
    Returns dict {Open,High,Low,Close} as NYC tz-aware Series, or None.
    `after` / `before` are ISO date strings — when supplied, DuckDB filters
    the CSV on disk so only the needed rows are loaded into RAM.
    """
    import duckdb
    path = os.path.join(CACHE_DIR, f"{name}_30min{suffix}.csv")
    if not os.path.exists(path):
        return None
    try:
        where_clauses = []
        if after:
            where_clauses.append(f"date >= '{after}'")
        if before:
            where_clauses.append(f"date <= '{before}'")
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        df = duckdb.query(
            f"SELECT date, open, high, low, close FROM read_csv_auto('{path}') {where}"
        ).df()
        if 'date' not in df.columns or len(df) == 0:
            return None
        idx = pd.to_datetime(df['date'], utc=True).dt.tz_convert('America/New_York')
        out = {}
        for c_lower, c_title in [('open','Open'),('high','High'),
                                 ('low','Low'),('close','Close')]:
            if c_lower in df.columns:
                s = pd.Series(df[c_lower].values, index=idx)
                out[c_title] = s
        if 'Open' not in out:
            return None
        return out
    except Exception as e:
        print(f"     ⚠️  cache read failed for {name}: {str(e)[:50]}")
        return None


def fetch_30min(ticker, end_date):
    try:
        start = (pd.Timestamp.now(tz='UTC') - timedelta(days=57)).strftime('%Y-%m-%d')
        end   = (pd.Timestamp(end_date) + timedelta(days=2)).strftime('%Y-%m-%d')
        data  = yf.download(ticker, start=start, end=end, interval='30m',
                            progress=False, auto_adjust=True)
        if data.empty: return None
        out = {}
        for c in ['Open','High','Low','Close']:
            s = data[c]
            if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
            idx = pd.to_datetime(s.index)
            idx = idx.tz_localize('America/New_York') if idx.tz is None else idx.tz_convert('America/New_York')
            s.index = idx
            out[c] = s
        return out
    except: return None


def fetch_daily(ticker, start_date, end_date):
    try:
        start = (pd.Timestamp(start_date) - timedelta(days=40)).strftime('%Y-%m-%d')  # extra for estimation
        end   = (pd.Timestamp(end_date)   + timedelta(days=5)).strftime('%Y-%m-%d')
        data  = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty: return None
        out = {}
        for c in ['Open','High','Low','Close']:
            s = data[c]
            if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
            s.index = pd.to_datetime(s.index).date
            out[c] = s
        return out
    except: return None


def _merge_bars(parts):
    """Merge a list of {O,H,L,C} dicts chronologically, IBKR (first) wins on overlap."""
    out = {}
    for c in ['Open', 'High', 'Low', 'Close']:
        series = [p[c] for p in parts if p is not None and c in p]
        if not series:
            continue
        s = pd.concat(series)
        out[c] = s[~s.index.duplicated(keep='first')].sort_index()
    return out if 'Open' in out else None


def _persist_yf_extension(name, ext, after_ts):
    """
    Append yfinance 30min bars newer than `after_ts` to {name}_30min_yf.csv so
    intraday coverage accumulates daily and never falls outside yfinance's
    ~60-day window. The IBKR cache files are never touched. Returns rows added.
    """
    if ext is None:
        return 0
    idx = ext['Open'].index
    mask = idx > after_ts if after_ts is not None else pd.Series(True, index=idx).values
    if not mask.any():
        return 0
    rows = pd.DataFrame({
        'date':  idx[mask].tz_convert('UTC'),
        'open':  ext['Open'].values[mask],
        'high':  ext['High'].values[mask]  if 'High'  in ext else ext['Open'].values[mask],
        'low':   ext['Low'].values[mask]   if 'Low'   in ext else ext['Open'].values[mask],
        'close': ext['Close'].values[mask] if 'Close' in ext else ext['Open'].values[mask],
    }).dropna(subset=['open', 'close'])
    if rows.empty:
        return 0
    path = os.path.join(CACHE_DIR, f"{name}_30min_yf.csv")
    exists = os.path.exists(path)
    rows.to_csv(path, mode='a', header=not exists, index=False, lineterminator='\n')
    return len(rows)


def fetch_market(start_date, end_date):
    """Populate the module-level intraday/daily dicts for [start_date, end_date]."""
    intraday.clear(); daily.clear()
    print(f"\n📈 Fetching market data ({start_date} → {end_date})...")
    print(f"   Priority: IBKR cache ({CACHE_DIR}/) + yfinance 30min extension → yfinance daily")
    cache_hits = 0
    for name,(ticker,mtype) in TICKERS.items():
        if mtype in ('24h','us'):
            d = fetch_daily(ticker, start_date, end_date)
            if d: daily[name] = d

            # 1st choice: IBKR cached 30min (deep history), EXTENDED with
            # yfinance 30min bars past the cache end. yfinance intraday only
            # reaches back ~60 days, so the extension is persisted to
            # {name}_30min_yf.csv — coverage accumulates run by run and the
            # cache-end gap that froze labels at the 2026-05-22 bar is closed.
            # 5-day lookback buffer for estimation-window feature extraction
            _after  = (pd.Timestamp(start_date) - timedelta(days=5)).strftime('%Y-%m-%d')
            _before = (pd.Timestamp(end_date)   + timedelta(days=2)).strftime('%Y-%m-%d')
            cached = load_cached_intraday(name, after=_after, before=_before)
            if cached is not None:
                yfc = load_cached_intraday(name, suffix="_yf", after=_after, before=_before)
                last_ts = max(p['Open'].index.max() for p in (cached, yfc) if p is not None)
                ext = fetch_30min(ticker, end_date)
                n_added = _persist_yf_extension(name, ext, last_ts)
                yfc = load_cached_intraday(name, suffix="_yf", after=_after, before=_before)   # reload incl. new bars
                merged = _merge_bars([cached, yfc])
                intraday[name] = merged
                cache_hits += 1
                span = f"{merged['Open'].index.min().date()} → {merged['Open'].index.max().date()}"
                ext_note = f", +{n_added} new yf bars" if n_added else ""
                print(f"  💾 {name:<12} IBKR+yf cache ({len(merged['Open'])} bars, {span}{ext_note})")
            else:
                # 2nd choice: yfinance 30min (last 57 days only)
                b = fetch_30min(ticker, end_date)
                if b:
                    intraday[name] = b
                    print(f"  ✅ {name:<12} yfinance 30min (recent only)")
                elif d:
                    print(f"  ⚠️  {name:<12} daily only")
                else:
                    print(f"  ❌ {name:<12} failed")
        else:
            d = fetch_daily(ticker, start_date, end_date)
            if d:
                daily[name] = d
                print(f"  ✅ {name:<12} daily")
            else:
                print(f"  ❌ {name:<12} failed")
    print(f"\n  📊 {cache_hits} instruments using IBKR cache (deep intraday history)")


def compute_baselines():
    """30-day mean + std per instrument from the daily data already fetched."""
    baseline_30d.clear(); volatility_30d.clear()
    print("\n📏 Computing 30-day baseline (mean + std) per instrument...")
    for name in TICKERS:
        if name in daily:
            closes = daily[name]['Close']
            opens  = daily[name]['Open']
            # intraday-style move = (close-open)/open per day
            moves = ((closes - opens) / opens * 100).dropna()
            recent = moves.tail(BASELINE_DAYS)
            if len(recent) >= 5:
                baseline_30d[name]   = round(recent.mean(), 4)
                # std = how much this instrument NORMALLY swings.
                std = recent.std()
                volatility_30d[name] = round(std, 4) if std > 0 else 1.0
            else:
                baseline_30d[name]   = 0.0
                volatility_30d[name] = 1.0
    print(f"  Baseline computed for {len(baseline_30d)} instruments")
    print(f"  Example — SPY:  mean={baseline_30d.get('SPY',0):+.3f}% std={volatility_30d.get('SPY',1):.3f}%")
    print(f"            VIX:  mean={baseline_30d.get('VIX',0):+.3f}% std={volatility_30d.get('VIX',1):.3f}%")
    print(f"            EUR:  mean={baseline_30d.get('EUR_USD',0):+.3f}% std={volatility_30d.get('EUR_USD',1):.3f}%")


# ==========================================
# ABNORMAL RETURN (actual − 30d baseline)
# ==========================================
def abnormal_return_intraday(post_dt, name):
    """
    FAIR VALUE = raw % move from the post moment to ~1 hour later.

      initial  = OPEN of the bar containing the post (best price estimate AT
                 the post — captures the start of the reaction, including the
                 first 30 min that a 'next bar open' start would discard).
      reaction = price ~60 min after the post, found by timestamp:
                 the bar nearest to post + 60min. If the post is late in the
                 session and a full hour runs past the close, use the LAST
                 available bar of that session (captures the move up to close).
    No baseline, no std, no z-score, no weight multiplication.
    """
    if name not in intraday:
        return None, None
    b = intraday[name]
    opens = b['Open']
    closes = b['Close'] if 'Close' in b else b['Open']
    if len(opens) == 0 or post_dt < opens.index[0]:
        return None, None

    past = opens[opens.index <= post_dt]
    if len(past) == 0:
        return None, None
    pos = opens.index.get_loc(past.index[-1])
    if isinstance(pos, slice):
        pos = pos.stop - 1
    post_bar_t = opens.index[pos]

    # GUARD: the post landed in (or after) the FINAL cached bar — there is no
    # forward data to measure a reaction. Without this, every post past the
    # cache end was "measured" against the same stale last bar (constant
    # labels like SPY -0.0766 for weeks). Fall back to daily data instead.
    if pos >= len(opens) - 1:
        return None, None

    # initial = OPEN of the bar the post landed in.
    initial = opens.iloc[pos]
    if pd.isna(initial) or initial == 0:
        initial = closes.iloc[pos]
    if pd.isna(initial) or initial == 0:
        return None, None

    # reaction target = post bar + 60 min
    target_t = post_bar_t + pd.Timedelta(minutes=60)
    same_session_end = post_bar_t + pd.Timedelta(hours=8)  # rough session bound

    after = opens[opens.index >= target_t]
    if len(after) > 0 and (after.index[0] - target_t) <= pd.Timedelta(minutes=45):
        # found a bar within 45 min of the +60min target → use it
        rpos = opens.index.get_loc(after.index[0])
        if isinstance(rpos, slice):
            rpos = rpos.start
        reaction = closes.iloc[rpos]
    else:
        # late-day post: no full hour before close. Use the last bar of the
        # SAME session (the close-of-day price) to capture the move so far.
        session_bars = opens[(opens.index > post_bar_t) &
                             (opens.index <= same_session_end) &
                             (opens.index.date == post_bar_t.date() if hasattr(opens.index,'date') else True)]
        # fallback: just take a few bars forward, bounded
        end_pos = min(pos + 2, len(opens) - 1)
        reaction = closes.iloc[end_pos]

    if pd.isna(reaction) or reaction == 0:
        return None, None
    move = (reaction - initial) / initial * 100
    return round(move, 4), round(move, 4)


def abnormal_return_daily(post_dt, name, session):
    """Daily fallback: raw open→close % move for the relevant session. No std/baseline."""
    if name not in daily:
        return None, None
    d = daily[name]
    closes, opens = d['Close'], d['Open']
    dates = sorted(closes.index)
    post_date = post_dt.date()
    if session in ('pre_market','during_market'):
        day = next((x for x in reversed(dates) if x <= post_date), None)
    else:
        day = next((x for x in dates if x > post_date), None)
    if day is None:
        return None, None
    try:
        initial, close = opens[day], closes[day]
        if pd.isna(initial) or initial == 0:
            return None, None
        move = (close - initial) / initial * 100
        return round(move, 4), round(move, 4)
    except:
        return None, None


def abnormal_next_session(post_dt, name):
    """Asian/EU: raw next-session open→close % move. No std/baseline."""
    if name not in daily:
        return None, None
    d = daily[name]
    closes, opens = d['Close'], d['Open']
    dates = sorted(closes.index)
    nd = next((x for x in dates if x > post_dt.date()), None)
    if nd is None:
        return None, None
    try:
        o, c = opens[nd], closes[nd]
        if pd.isna(o) or o == 0:
            return None, None
        move = (c - o) / o * 100
        return round(move, 4), round(move, 4)
    except:
        return None, None


# ==========================================
# COMPUTE ABNORMAL RETURNS FOR ALL POSTS
# ==========================================
def compute_impacts(scored):
    """Add {NAME}_Impact / _zscore / _quality columns for every ticker. Returns (scored, impact_cols)."""
    print("\n🧮 Computing raw 1-hour moves (fair value) per instrument...")
    impact_cols = []
    new_columns = {}  # collect all columns, add at once to avoid fragmentation
    for name,(ticker,mtype) in TICKERS.items():
        col  = f'{name}_Impact'
        zcol = f'{name}_zscore'
        qcol = f'{name}_quality'   # 'intraday' (reliable) or 'daily' (noisy)
        ar_list, z_list, q_list = [], [], []
        for _, row in scored.iterrows():
            if mtype in ('24h','us'):
                ar, z = abnormal_return_intraday(row['date'], name)
                if ar is None:
                    ar, z = abnormal_return_daily(row['date'], name, row['session'])
                    q = 'daily'
                else:
                    q = 'intraday'
            else:
                ar, z = abnormal_next_session(row['date'], name)
                q = 'daily'   # Asian/EU always next-session daily
            ar_list.append(ar)
            z_list.append(z)
            q_list.append(q)
        new_columns[col]  = ar_list
        new_columns[zcol] = z_list
        new_columns[qcol] = q_list
        valid = sum(1 for x in ar_list if x is not None)
        print(f"  ✅ {col:<22} {valid:>4} valid [{mtype}]")
        impact_cols.append(col)

    # Add all columns at once (avoids DataFrame fragmentation warning)
    scored = pd.concat([scored, pd.DataFrame(new_columns, index=scored.index)], axis=1)
    return scored, impact_cols


def apply_caps(scored, impact_cols):
    """Label = real measured 1-hour move, clamped to a realistic ceiling. None → 0."""
    print("\n📐 Building labels = FAIR VALUE (raw 1-hour % move). No z-score, no std,")
    print("   no NLP deflation. NLP lives only in sample_weight (training trust).")
    for col in impact_cols:
        name = col.replace('_Impact', '')
        for i in scored.index:
            ar = scored.at[i, col]
            if ar is None or pd.isna(ar):
                scored.at[i, col] = 0.0
                continue
            cap = IMPACT_CAP.get(name, 5.0)
            scored.at[i, col] = round(max(min(ar, cap), -cap), 4)
    return scored


def finalize(scored, impact_cols):
    """fillna, drop core-NaN rows, sample_weight = nlp_signal."""
    for col in impact_cols:
        scored[col] = scored[col].fillna(0.0)
    core = [f'{n}_Impact' for n in CORE_INSTRUMENTS]
    scored = scored.dropna(subset=core)
    # Sample weight = NLP signal directly (continuous, no gate)
    scored['sample_weight'] = scored['nlp_signal'].round(4)
    return scored


def train_columns(impact_cols):
    # platform / is_primary / entity_weight / event_weight give XGB source-awareness.
    # sample_weight already incorporates them, but explicit columns let the model
    # learn non-linear interactions (e.g. geo tweet + high event_weight → high impact).
    # is_primary is a bool (1.0 = rank-0 TruthSocial only; 0.0 = secondary TruthSocial or X/Twitter geo).
    base = ['id', 'date', 'text', 'platform', 'is_primary',
            'account', 'account_rank', 'entity_weight', 'event_weight',
            'sample_weight', 'nlp_signal']
    return base + impact_cols


# ==========================================
# FULL REBUILD (original behaviour)
# ==========================================
def main_full():
    print("=" * 66)
    print("  BUILD FINAL TRAINING SET — EVENT STUDY (FULL REBUILD)")
    print("  Impact = f(NLP signal × statistically significant abnormal return)")
    print("=" * 66)

    scored = load_scored()
    fetch_market(scored['date'].dt.date.min(), scored['date'].dt.date.max())
    compute_baselines()
    scored, impact_cols = compute_impacts(scored)
    scored = apply_caps(scored, impact_cols)

    # Diagnostic counts (informational only — NLP is the training weight, not a gate)
    strong = (scored['nlp_signal'] >= 0.5).sum()
    medium = ((scored['nlp_signal'] >= 0.3) & (scored['nlp_signal'] < 0.5)).sum()
    weak   = (scored['nlp_signal'] < 0.3).sum()
    print(f"  High-weight posts (NLP≥0.5):   {strong}")
    print(f"  Medium-weight (0.3-0.5):       {medium}")
    print(f"  Low-weight (<0.3):             {weak}")

    # Verification on known posts
    print("\n🔬 Verification on known posts:")
    iran = scored[scored['text'].str.contains('Emir of Qatar', na=False)]
    if len(iran):
        r = iran.iloc[0]
        print(f"\n  IRAN/EMIR (high NLP → strong impact):")
        print(f"    NLP signal: {r['nlp_signal']:.3f}")
        print(f"    VIX impact: {r['VIX_Impact']:+.4f}%")
    endorse = scored[scored['text'].str.contains('Tremendous Champion', na=False, regex=False)]
    if len(endorse):
        r = endorse.iloc[0]
        print(f"\n  ENDORSEMENT (low NLP → damped to ~5%):")
        print(f"    NLP signal: {r['nlp_signal']:.3f}")
        print(f"    VIX impact: {r['VIX_Impact']:+.4f}%")

    scored = finalize(scored, impact_cols)

    print(f"\n📊 Final impact distribution:")
    for col in ['VIX_Impact','SPY_Impact','OIL_Impact']:
        s = scored[col]
        nonzero = (s.abs() > 0.05).sum()
        print(f"  {col}: {nonzero} meaningful (>0.05%), range [{s.min():.3f}, {s.max():.3f}]")

    train_cols = train_columns(impact_cols)
    db.write_table(FINAL_TABLE, scored[train_cols])
    print(f"\n💾 Saved {FINAL_TABLE} ({len(scored)} rows)")

    hs = scored[scored['sample_weight'] > 0.5][train_cols]
    db.write_table(HS_TABLE, hs)
    print(f"💾 Saved {HS_TABLE} ({len(hs)} rows)")

    db.write_table(LABELED_TABLE, scored)
    print(f"💾 Saved {LABELED_TABLE} (full metadata)")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║       EVENT-STUDY TRAINING SET COMPLETE                  ║
╠══════════════════════════════════════════════════════════╣
║  Total posts:                {len(scored):>6}                      ║
║  Strong NLP posts (≥0.5):    {strong:>6}                      ║
║  Weak NLP posts (<0.3):      {weak:>6}                      ║
╚══════════════════════════════════════════════════════════╝
""")


# ==========================================
# RELABEL — purge bad rows so incremental redoes them
# ==========================================
def relabel_since(since_str):
    """
    Drop all rows dated >= since (NY time) from the 3 output csvs. The next
    incremental pass then re-labels those posts from scratch. Use this once
    after fixing a data problem (e.g. the stale cache-end labels).
    """
    since = pd.Timestamp(since_str, tz='America/New_York')
    print(f"\n🧹 Relabel: purging rows dated >= {since} from outputs...")
    for table in (FINAL_TABLE, HS_TABLE, LABELED_TABLE):
        df = db.read_table(table)
        if df is None:
            continue
        dates = pd.to_datetime(df['date'], format='mixed', utc=True).dt.tz_convert('America/New_York')
        keep = df[dates < since]
        if len(keep) < len(df):
            db.write_table(table, keep)
        print(f"  {table}: removed {len(df) - len(keep)} rows ({len(keep)} kept)")


# ==========================================
# INCREMENTAL (daily) — label only NEW posts, APPEND
# ==========================================
def _spy_next_day_available(post_dt):
    """True if a SPY daily bar exists AFTER the post's date (needed to label after-hours posts)."""
    if 'SPY' not in daily:
        return False
    return any(x > post_dt.date() for x in daily['SPY']['Close'].index)


def main_incremental():
    print("=" * 66)
    print("  BUILD FINAL TRAINING SET — EVENT STUDY (INCREMENTAL, daily)")
    print("=" * 66)

    labeled_df = db.read_table(LABELED_TABLE)
    final_df = db.read_table(FINAL_TABLE)
    if labeled_df is None or final_df is None:
        print("⚠️  No existing labeled/FINAL table — running FULL rebuild instead.")
        return main_full()

    scored = load_scored()
    labeled_cols = list(labeled_df.columns)
    # Dedup on (platform, id) — composite key safe across TruthSocial and X/Twitter
    labeled_keys = set(zip(labeled_df['platform'], labeled_df['id'].astype(str)))
    final_cols = list(final_df.columns)

    scored_keys = list(zip(scored['platform'], scored['id'].astype(str)))
    new = scored[[k not in labeled_keys for k in scored_keys]].copy()
    print(f"\n  Already labeled: {len(labeled_keys)} | new to label: {len(new)}")
    if new.empty:
        print("✅ Nothing new to label — training set is up to date.")
        return 0

    now = pd.Timestamp.now(tz='America/New_York')

    # MATURITY 1: reaction window must have passed (1h reaction + bar granularity)
    too_fresh = new['date'] > (now - pd.Timedelta(hours=2))
    if too_fresh.any():
        print(f"  ⏳ Deferred {too_fresh.sum()} posts <2h old (reaction window incomplete)")
        new = new[~too_fresh]
    if new.empty:
        print("✅ All new posts deferred — will label on the next run.")
        return 0

    fetch_market(new['date'].dt.date.min(), new['date'].dt.date.max())
    compute_baselines()

    # MATURITY 2: after-hours posts are labeled with the NEXT session's reaction —
    # defer them until that session's data exists (unless >7 days old: give up waiting).
    mature = (now - new['date']) > pd.Timedelta(days=7)
    ah_not_ready = (new['session'] == 'after_hours') & ~mature \
                   & ~new['date'].apply(_spy_next_day_available)
    if ah_not_ready.any():
        print(f"  ⏳ Deferred {ah_not_ready.sum()} after-hours posts (next session data not yet available)")
        new = new[~ah_not_ready]
    if new.empty:
        print("✅ All new posts deferred — will label on the next run.")
        return 0

    new, impact_cols = compute_impacts(new)

    # MATURITY 3: if ALL core instruments returned no data for a recent post,
    # the market data simply isn't there yet — defer instead of writing zeros.
    core_cols = [f'{n}_Impact' for n in CORE_INSTRUMENTS]
    mature = (now - new['date']) > pd.Timedelta(days=7)
    core_none = new[core_cols].isna().all(axis=1)
    defer = core_none & ~mature
    if defer.any():
        print(f"  ⏳ Deferred {defer.sum()} posts (no core market data yet)")
        new = new[~defer]
    if new.empty:
        print("✅ All new posts deferred — will label on the next run.")
        return 0

    new = apply_caps(new, impact_cols)
    new = finalize(new, impact_cols)
    new = new.sort_values('date')

    # ---- APPEND to all three outputs, preserving each file's exact column order ----
    train_cols = train_columns(impact_cols)
    if set(final_cols) != set(train_cols):
        print(f"  ⚠️  Column mismatch vs existing {FINAL_TABLE} — "
              f"appending with the EXISTING header order (missing cols filled 0). "
              f"Run --full to rebuild with the new schema.")
    out_final = new.reindex(columns=final_cols, fill_value=0)
    db.append_table(FINAL_TABLE, out_final)
    print(f"\n💾 Appended {len(out_final)} rows -> {FINAL_TABLE}")

    hs = out_final[new['sample_weight'].values > 0.5]
    if len(hs):
        db.append_table(HS_TABLE, hs)
    print(f"💾 Appended {len(hs)} high-signal rows -> {HS_TABLE}")

    out_labeled = new.copy()
    for c in labeled_cols:
        if c not in out_labeled.columns:
            out_labeled[c] = 0
    out_labeled = out_labeled[labeled_cols]
    db.append_table(LABELED_TABLE, out_labeled)
    print(f"💾 Appended {len(out_labeled)} rows -> {LABELED_TABLE} (dedup ledger)")

    print(f"\n📊 New labels: VIX range [{new['VIX_Impact'].min():.3f}, {new['VIX_Impact'].max():.3f}], "
          f"high-signal: {len(hs)}")
    return len(new)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Build event-study training set (daily incremental by default).")
    ap.add_argument("--full", action="store_true",
                    help="Full rebuild: re-label ALL posts and rewrite all 3 output csvs.")
    ap.add_argument("--relabel-since", metavar="YYYY-MM-DD",
                    help="One-time repair: drop output rows dated >= this (NY time) "
                         "and re-label them in the same run. NOTE: yfinance intraday "
                         "only reaches back ~60 days — older posts fall back to daily data.")
    args = ap.parse_args()
    if args.full:
        main_full()
    else:
        if args.relabel_since:
            relabel_since(args.relabel_since)
        main_incremental()
