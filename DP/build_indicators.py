"""
build_indicators.py — INTRADAY TECHNICAL INDICATORS, SYNCED TO POST TIMES
==========================================================================
Replaces the daily-bar indicators that were computed inside
build_final_training_set.py and written straight into training_set_FINAL.

WHAT WAS WRONG WITH THAT
------------------------
1. DAILY BARS FOR A 1-HOUR HORIZON. `mom5` was 5 DAYS of momentum used to
   predict the next 60 minutes. The market state that actually sets the size
   of a reaction is intraday: where are we in today's range, is realised vol
   elevated right now, is the tape already moving.
2. NO LIVE PATH. The values existed ONLY as columns inside
   training_set_FINAL, keyed to posts. At prediction time there was nothing
   to read, so `cfg["_live_ta"]` was zero-filled — the model was trained on
   real indicators and served zeros. Since the indicators are the entire
   reason the SIZE head works (NLP alone: corr 0.007; NLP+TA: 0.218), that
   silently gutted the one part of the model that does work.
3. POSTS ARE NOT A CLOCK. Truths/tweets do not arrive on a bar schedule, so
   an indicator table keyed to posts cannot be sampled at an arbitrary "now".

THE FIX
-------
ONE function, `indicators_at()`, computes the 7 indicators from the trailing
1-minute bars at ANY timestamp. Training calls it in batch over every post
time; live prediction calls it for now. Identical code, so the two can never
diverge.

BARS STAY OUT OF THE DATABASE. The 1-min CSVs in IBKR/market_data_cache are
4.7 GB / ~66M rows across 23 instruments; they are read in place via DuckDB
(`read_csv_auto`) exactly like bars.py does. Only the DERIVED indicators are
stored, and only at the timestamps something actually consumes — post times —
which is ~190k rows instead of 66M.

LEAK GUARD
----------
Indicators at post time T use bars STRICTLY BEFORE T (`bar.date < post.date`,
ASOF). A post at 14:03:30 sees the bar closing 14:03:00 and everything before
it, never the bar it landed inside. That bar contains the reaction we are
trying to predict.

USAGE
-----
  python build_indicators.py --full          # all posts, all instruments
  python build_indicators.py --limit 5000    # pilot
  python build_indicators.py --instrument SPY --at "2026-07-01 14:30"   # probe
"""
import os
import sys
import json
import argparse
import time
import datetime as dt
import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import db

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.normpath(os.path.join(_HERE, "..", "IBKR", "market_data_cache"))
OUT_TABLE = "post_indicators"

# Same seven names the model already knows (config.json "tech_cols"), but the
# windows are now MINUTES, matched to the 1-hour prediction horizon instead of
# 5-20 trading days.
TECH_COLS = ['mom5', 'mom20', 'sma_rat', 'rsi14', 'macd_h', 'bb_pos', 'atr_pct']
W = {
    "mom5":    15,    # % return over the last 15 min   (was 5 days)
    "mom20":   60,    # % return over the last 60 min   (was 20 days)
    "sma_rat": 60,    # close / SMA(60 min)
    "rsi14":   70,    # Wilder RSI over 14 x 5-min      (was 14 days)
    "bb_pos":  60,    # %B within the 60-min band
    "atr_pct": 70,    # ATR(14 x 5-min) as % of price
}
RESAMPLE_MIN = 5      # rsi/macd/atr on 5-min bars: 1-min ticks are too noisy
LOOKBACK_MIN = 600    # trailing window pulled per timestamp (covers the longest)


# ---------------------------------------------------------------- loading ---
def _csv(inst):
    return os.path.join(CACHE_DIR, f"{inst}_1min.csv")


def has_bars(inst):
    return os.path.exists(_csv(inst))


def _load_window(inst, lo, hi):
    """1-min bars in [lo, hi) via DuckDB over the CSV — the file is never
    imported into the database."""
    import duckdb
    p = _csv(inst).replace("\\", "/")
    try:
        q = (f"SELECT date, open, high, low, close FROM read_csv_auto('{p}') "
             f"WHERE date >= '{lo:%Y-%m-%d %H:%M:%S}' "
             f"AND date < '{hi:%Y-%m-%d %H:%M:%S}' ORDER BY date")
        d = duckdb.query(q).df()
    except Exception:
        return None
    if d is None or d.empty:
        return None
    d['date'] = pd.to_datetime(d['date'], utc=True)
    return d.set_index('date')


# ------------------------------------------------------------- indicators ---
def _from_frame(b):
    """The 7 indicators from a trailing 1-min frame. Returns dict of floats.
    The LAST row of `b` must already be strictly before the target time —
    slicing is the caller's job so the leak guard lives in one place."""
    if b is None or len(b) < 30:
        return {c: np.nan for c in TECH_COLS}
    c, h, l = b['close'], b['high'], b['low']
    last = float(c.iloc[-1])
    out = {}

    def _ret(mins):
        ref = c.iloc[-min(mins, len(c) - 1) - 1] if len(c) > mins else c.iloc[0]
        return (last / float(ref) - 1.0) * 100.0 if ref else np.nan

    out['mom5'] = _ret(W['mom5'])
    out['mom20'] = _ret(W['mom20'])
    sma = c.tail(W['sma_rat']).mean()
    out['sma_rat'] = last / float(sma) if sma else np.nan

    # bollinger %B on the 60-min window
    win = c.tail(W['bb_pos'])
    m, s = float(win.mean()), float(win.std(ddof=0))
    # %B legitimately leaves [0,1] when price is outside the bands, but an
    # unbounded value becomes an outlier the trees split on; clip to [-1, 2].
    out['bb_pos'] = (float(np.clip((last - m) / (2 * s) + 0.5, -1.0, 2.0))
                     if s > 1e-12 else 0.5)

    # 5-min resample for the oscillators — 1-min ticks are mostly microstructure
    r = b.resample(f"{RESAMPLE_MIN}min").agg(
        {'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(r) < 20:
        out['rsi14'] = np.nan
        out['macd_h'] = np.nan
        out['atr_pct'] = np.nan
        return out

    d = r['close'].diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out['rsi14'] = float((100 - 100 / (1 + rs)).iloc[-1])

    e12 = r['close'].ewm(span=12, adjust=False).mean()
    e26 = r['close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    out['macd_h'] = float(hist.iloc[-1]) / last * 100.0 if last else np.nan

    pc = r['close'].shift(1)
    tr = pd.concat([r['high'] - r['low'], (r['high'] - pc).abs(),
                    (r['low'] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    out['atr_pct'] = float(atr) / last * 100.0 if last else np.nan
    return out


def bar_cutoff(when):
    """Last bar timestamp usable at `when`, EXCLUSIVE.

    IBKR stamps a bar with its START, so the bar labelled 14:03:00 covers
    14:03:00-14:04:00 and is NOT complete for a post at 14:03:30 — it
    contains up to 59 seconds of the very reaction we are predicting.
    Flooring to the minute and comparing strictly means a 14:03:30 post sees
    at most the 14:02:00 bar (which closed at 14:03:00, before the post).
    """
    return pd.Timestamp(when).floor("min")


def indicators_at(inst, when, bars=None):
    """THE shared entry point. Indicators for `inst` as of `when`, using only
    bars strictly BEFORE `when`. Training and live prediction both call this.

    `bars` — optional pre-loaded frame (batch path passes a day's slice to
    avoid one query per post).
    """
    when = pd.Timestamp(when)
    if when.tzinfo is None:
        when = when.tz_localize("UTC")
    when = when.tz_convert("UTC")
    cut = bar_cutoff(when)
    if bars is None:
        bars = _load_window(inst, cut - pd.Timedelta(minutes=LOOKBACK_MIN), cut)
    else:
        bars = bars[bars.index < cut]           # LEAK GUARD, strict
    return _from_frame(bars)


# ------------------------------------------------------- vectorised engine ---
def _indicator_frame(b):
    """All 7 indicators as SERIES over the whole 1-min frame, in ONE pass.

    This replaces a per-post `_from_frame()` call. The old loop ran
    `.tail()/.resample()/.ewm()` once PER POST PER INSTRUMENT — ~190k x 23 =
    4.4M pandas pipelines, which is why a rebuild took hours. Rolling/ewm are
    causal by construction (each row uses only rows at or before it), so
    computing the series once and reading the value at the post's cutoff gives
    an identical answer for a fraction of the work.
    """
    c, h, l = b['close'], b['high'], b['low']
    out = pd.DataFrame(index=b.index)
    out['mom5'] = (c / c.shift(W['mom5']) - 1.0) * 100.0
    out['mom20'] = (c / c.shift(W['mom20']) - 1.0) * 100.0
    out['sma_rat'] = c / c.rolling(W['sma_rat'], min_periods=10).mean()
    _m = c.rolling(W['bb_pos'], min_periods=10).mean()
    _s = c.rolling(W['bb_pos'], min_periods=10).std(ddof=0)
    out['bb_pos'] = (((c - _m) / (2 * _s)) + 0.5).clip(-1.0, 2.0)

    # 5-min oscillators — resampled ONCE for the whole series.
    #
    # LEAK GUARD: label='right' stamps each bin with its CLOSE time, so the bin
    # covering [10:00, 10:05) is labelled 10:05. A later ffill onto the 1-min
    # index then gives a post at 10:03 the bar that closed at 10:00 — strictly
    # past. With the default label='left' that bin is stamped 10:00 and ffill
    # hands back a bar containing 10:03 and 10:04, i.e. up to 4 minutes of the
    # reaction being predicted. (Caught by a diff against the per-post path,
    # which never saw this because it only ever held a partial trailing bin.)
    r = b.resample(f"{RESAMPLE_MIN}min", label='right', closed='left').agg(
        {'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(r) >= 20:
        d = r['close'].diff()
        up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        e12 = r['close'].ewm(span=12, adjust=False).mean()
        e26 = r['close'].ewm(span=26, adjust=False).mean()
        macd = e12 - e26
        hist = (macd - macd.ewm(span=9, adjust=False).mean()) / r['close'] * 100.0
        pc = r['close'].shift(1)
        tr = pd.concat([r['high'] - r['low'], (r['high'] - pc).abs(),
                        (r['low'] - pc).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean() / r['close'] * 100.0
        # ffill the 5-min values onto the 1-min index: at any minute the latest
        # COMPLETED 5-min bar is the correct reading, never a future one.
        out['rsi14'] = rsi.reindex(out.index, method='ffill')
        out['macd_h'] = hist.reindex(out.index, method='ffill')
        out['atr_pct'] = atr.reindex(out.index, method='ffill')
    else:
        out['rsi14'] = out['macd_h'] = out['atr_pct'] = np.nan
    return out


# ------------------------------------------------------------------ batch ---
def build(posts, instruments, verbose=True, on_instrument=None):
    """Indicators for every (post, instrument), vectorised + ASOF-joined.

    `on_instrument(df)` is called after EACH instrument so the caller can
    persist incrementally — a Ctrl+C then costs one instrument, not the run.
    """
    all_rows = []
    lo = posts['date'].min() - pd.Timedelta(minutes=LOOKBACK_MIN * 2)
    hi = posts['date'].max() + pd.Timedelta(minutes=1)
    for n_i, inst in enumerate(instruments, 1):
        if not has_bars(inst):
            if verbose:
                print(f"  [{n_i}/{len(instruments)}] ⏭️  {inst}: no 1-min cache",
                      flush=True)
            continue
        t0 = time.time()
        b = _load_window(inst, lo, hi)
        if b is None or b.empty:
            if verbose:
                print(f"  [{n_i}/{len(instruments)}] ⚠️  {inst}: no bars in span",
                      flush=True)
            continue
        ind = _indicator_frame(b)
        del b

        # ---- ASOF: last bar STRICTLY BEFORE each post's minute boundary ----
        cut = pd.DatetimeIndex([bar_cutoff(w) for w in posts['date']])
        j = ind.index.searchsorted(cut, side='left') - 1     # strict '<'
        ok = j >= 0
        if not ok.any():
            if verbose:
                print(f"  [{n_i}/{len(instruments)}] ⚠️  {inst}: no post is "
                      f"inside the bar range", flush=True)
            del ind
            continue
        vals = ind.iloc[j[ok]].reset_index(drop=True)
        df = pd.DataFrame({
            "platform_id": np.asarray(posts['platform_id'])[ok],
            "instrument": inst,
            "date": np.asarray(posts['date'])[ok],
        })
        for cc in TECH_COLS:
            df[cc] = vals[cc].values
        df = df[np.isfinite(df['atr_pct'].astype(float))]
        del ind, vals

        if verbose:
            print(f"  [{n_i}/{len(instruments)}] ✅ {inst}: {len(df)}/{len(posts)} "
                  f"posts ({len(df) / max(len(posts), 1):.0%})  "
                  f"{time.time() - t0:.1f}s", flush=True)
        if on_instrument is not None and len(df):
            on_instrument(df)          # persist NOW — Ctrl+C safe
        else:
            all_rows.append(df)
    return (pd.concat(all_rows, ignore_index=True) if all_rows
            else pd.DataFrame(columns=["platform_id", "instrument", "date"] + TECH_COLS))


def main():
    ap = argparse.ArgumentParser(
        description="Intraday (1-min) technical indicators synced to post times. "
                    "Bars stay in CSV; only indicators are stored.")
    ap.add_argument("--full", action="store_true", help="every post")
    ap.add_argument("--limit", type=int, default=None)
    # posts_scored, NOT training_set_FINAL. training_set_FINAL is written BY
    # build_final_training_set.py, which JOINS post_indicators — reading it here
    # is circular: new posts can never get indicators because they only enter
    # training_set_FINAL after the build that needs them. Measured 2026-08-03:
    # posts_scored 192,548 (to 08-03) vs training_set_FINAL 189,993 (to 07-11),
    # so 2,555 posts were permanently indicator-less. posts_scored is upstream
    # (signal_scorer, step 5) and always current at step 5.5.
    ap.add_argument("--table", default="posts_scored",
                    help="Source of post timestamps (default posts_scored — "
                         "upstream of build_final_training_set, so new posts "
                         "get indicators BEFORE the build that consumes them).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignore existing rows and rebuild the whole table "
                         "(default: INCREMENTAL — only posts not yet present).")
    ap.add_argument("--instrument", default=None,
                    help="Limit to ONE instrument (works for the build too, "
                         "not just --at). Was previously ignored during a "
                         "build, so --instrument SPY still ran all 23.")
    ap.add_argument("--at", default=None,
                    help="probe: indicators at this UTC timestamp, print and exit")
    args = ap.parse_args()

    reg = json.load(open(os.path.join(_HERE, "instruments.json"), encoding="utf-8"))
    instruments = list(reg["instruments"].keys())
    if args.instrument:
        _want = [i.strip() for i in args.instrument.split(",") if i.strip()]
        _bad = [i for i in _want if i not in instruments]
        if _bad:
            sys.exit(f"❌ unknown instrument(s): {', '.join(_bad)}\n"
                     f"   known: {', '.join(instruments)}")
        instruments = _want

    # ---- probe mode: prove the live path works, annotate nothing ----------
    if args.at:
        inst = args.instrument or instruments[0]
        v = indicators_at(inst, args.at)
        print(f"\n{inst} @ {args.at} UTC  (bars strictly before)")
        for c in TECH_COLS:
            print(f"   {c:<9} {v[c]:+.4f}" if np.isfinite(v[c])
                  else f"   {c:<9} n/a")
        return 0

    q = f'SELECT platform, id, date FROM "{args.table}" ORDER BY date'
    if args.limit:
        q += f" LIMIT {int(args.limit)}"
    posts = db.query(q)
    posts['date'] = pd.to_datetime(posts['date'], utc=True)
    posts['platform_id'] = posts['platform'].astype(str) + '_' + posts['id'].astype(str)

    # ---- INCREMENTAL: only posts not already in post_indicators -----------
    # A daily run has ~2.5k new posts, not 190k. Rebuilding everything each
    # time is what made this unusable (and a Ctrl+C mid-rebuild left the table
    # in an unknown state). --rebuild forces the full pass.
    n_all = len(posts)
    already = 0
    if not args.rebuild and db.table_exists(OUT_TABLE):
        try:
            done = db.query(f'SELECT DISTINCT platform_id FROM "{OUT_TABLE}"')
            if done is not None and len(done):
                have = set(done['platform_id'].astype(str))
                already = len(have)
                posts = posts[~posts['platform_id'].isin(have)].reset_index(drop=True)
        except Exception as e:                                # noqa: BLE001
            print(f"  ⚠️  could not read {OUT_TABLE} ({e}) — doing a full pass")
    if posts.empty:
        print(f"✅ nothing to do — all {n_all} posts already have indicators "
              f"({OUT_TABLE} covers {already}).")
        return 0

    print("=" * 74)
    print("  INTRADAY INDICATORS (1-min bars) SYNCED TO POST TIMES")
    print("=" * 74)
    print(f"  posts TODO  : {len(posts)} of {n_all}  "
          f"({posts['date'].min():%Y-%m-%d} -> {posts['date'].max():%Y-%m-%d})")
    print(f"  already done: {already}"
          + ("   [--rebuild: ignoring, full pass]" if args.rebuild else ""))
    print(f"  instruments : {len(instruments)}")
    print(f"  windows     : mom5={W['mom5']}m  mom20={W['mom20']}m  "
          f"sma={W['sma_rat']}m  rsi/atr={RESAMPLE_MIN}m x14  bb={W['bb_pos']}m")
    print(f"  leak guard  : bars STRICTLY before each post timestamp")
    print(f"  bars        : read in place from {CACHE_DIR} (never stored in the DB)")
    print("=" * 74, flush=True)

    # Persist per instrument so an interrupt costs one instrument, not the run.
    written = {"rows": 0, "insts": 0, "first": not db.table_exists(OUT_TABLE)}
    if args.rebuild and db.table_exists(OUT_TABLE):
        if args.instrument:
            # SCOPED rebuild: replacing the whole table would delete the other
            # 22 instruments. Drop only the rows for the instrument(s) asked
            # for, then append — so `--instrument SPY --rebuild` refreshes SPY
            # and leaves everything else intact.
            _lst = ",".join("'" + i.replace("'", "''") + "'" for i in instruments)
            db.query(f'DELETE FROM "{OUT_TABLE}" WHERE instrument IN ({_lst})')
            print(f"  🧹 --rebuild --instrument: dropped existing rows for "
                  f"{', '.join(instruments)} only (other instruments kept)")
            written["first"] = False   # append into the surviving table
        else:
            written["first"] = True    # full rebuild replaces the table

    def _persist(df):
        if written["first"]:
            db.write_table(OUT_TABLE, df)
            written["first"] = False
        else:
            db.append_table(OUT_TABLE, df)
        written["rows"] += len(df)
        written["insts"] += 1
        print(f"       💾 saved ({written['rows']} rows so far)", flush=True)

    t_all = time.time()
    build(posts, instruments, on_instrument=_persist)
    if not written["rows"]:
        print("\n❌ nothing produced — check the 1-min cache")
        return 1
    print(f"\n💾 {OUT_TABLE}: +{written['rows']} rows across "
          f"{written['insts']} instrument(s) in {time.time() - t_all:.0f}s")
    print(f"   next: build_final_training_set.py --full  (ASOF-joins this table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
