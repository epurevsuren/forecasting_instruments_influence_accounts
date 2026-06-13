"""
build_test_training_set.py
--------------------------
Builds truth_training_set_TEST.csv — a POINT-IN-TIME labeled dataset for
backtesting, using the same event-study labeling engine as
build_final_training_set.py but with a hard knowledge cutoff:

    ALL truths up to the cutoff are included. NOTHING after it is used.

Two labeling regimes, chosen automatically per post:

  STANDARD (post + 90min fits before the cutoff)
    Identical to the daily pipeline: 30-min bars, ~1-hour reaction.

  NEAR-CUTOFF (post is within 90min of the cutoff, e.g. a 14:10 post
  with a 15:00 cutoff)
    The post is still included — its reaction is measured with an
    AUTOMATICALLY chosen finer yfinance interval and runs from the post
    to min(post + 60min, cutoff):
        >= 45 min remaining -> 15m bars
        >= 20 min remaining ->  5m bars
         <  20 min remaining ->  2m bars
    These are honest "what was knowable at the cutoff" labels: a 14:40
    post gets a 20-minute reaction, because at 15:00 that's all there was.

YFINANCE REALITY CHECK (free intraday limits):
    1m -> ~7 days back | 2m/5m/15m/30m/90m -> ~60 days | 60m -> ~730 days
If the cutoff is older than ~59 days, the fine intervals are NOT
retrievable; near-cutoff posts then fall back to the 30-min IBKR/_yf cache
bars truncated at the cutoff (coarser but still point-in-time correct).

Other guarantees:
  * intraday bars are used only if the bar CLOSED by the cutoff
  * daily bars: only days complete before the cutoff's date
  * 30-day baselines computed from the truncated daily data only
  * after-hours posts labeled from intraday gap bars are kept; if an
    instrument's data simply didn't exist at the cutoff, the impact is 0
    (the honest "not knowable" label) rather than a future peek
  * this script never writes to the cache files (read-only market data)

USAGE
  python build_test_training_set.py 202605181500
  python build_test_training_set.py 202605181500 --from 202601010000

  until    yyyymmddhhmm, NY (America/New_York) local time — the cutoff
  --from   yyyymmddhhmm, optional: only include posts at/after this moment

Output: truth_training_set_TEST.csv (overwritten each run — one file per
backtest cutoff; rename it if you want to keep several).
"""

import os
import sys
import argparse
import datetime
import numpy as np
import pandas as pd

import build_final_training_set as B   # the labeling engine (shares UTF-8 fix)

_HERE    = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(_HERE, "truth_training_set_TEST.csv")

NY  = 'America/New_York'
BAR = pd.Timedelta(minutes=30)              # cache bar width
FINE_REACH_DAYS = 58                        # 2m/5m/15m reach ~60 days; stay safe


def parse_stamp(s, name):
    try:
        return pd.Timestamp(datetime.datetime.strptime(s.strip(), "%Y%m%d%H%M"), tz=NY)
    except ValueError:
        sys.exit(f"❌ {name} must be yyyymmddhhmm (e.g. 202605181500), got: {s!r}")


def pick_interval(remaining_min):
    """Finest sensible yfinance interval for the time left before the cutoff."""
    if remaining_min >= 45: return '15m'
    if remaining_min >= 20: return '5m'
    return '2m'


def truncate_intraday(bars, cutoff, width=BAR):
    """Keep only bars that fully CLOSED by the cutoff (start + width <= cutoff)."""
    if bars is None:
        return None
    limit = cutoff - width
    out = {c: s[s.index <= limit] for c, s in bars.items()}
    return out if len(out.get('Open', [])) else None


def truncate_daily(d, cutoff):
    """Daily bars: only days strictly BEFORE the cutoff's date — an intraday
    cutoff cannot know that day's close yet."""
    if d is None:
        return None
    out = {}
    for c, s in d.items():
        keep = [i for i in s.index if i < cutoff.date()]
        out[c] = s[s.index.isin(keep)] if hasattr(s.index, 'isin') else s.loc[keep]
    return out if len(out.get('Open', [])) else None


def _yf_intraday(ticker, interval, start, end):
    """One yfinance intraday download -> {O,H,L,C} NY-tz Series, or None."""
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, interval=interval,
                           progress=False, auto_adjust=True)
        if data.empty:
            return None
        out = {}
        for c in ['Open', 'High', 'Low', 'Close']:
            s = data[c]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            idx = pd.to_datetime(s.index)
            idx = idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)
            s.index = idx
            out[c] = s
        return out
    except Exception:
        return None


def fetch_market_pit(start_date, cutoff):
    """Populate B.intraday / B.daily with 30-min + daily data truncated at the
    cutoff. Read-only on the cache files."""
    B.intraday.clear(); B.daily.clear()
    days_back = (pd.Timestamp.now(tz=NY) - cutoff).days
    yf30_ok = days_back <= FINE_REACH_DAYS

    print(f"\n📈 Point-in-time market data (cutoff {cutoff})...")
    if not yf30_ok:
        print(f"   ⚠️  Cutoff is {days_back} days back — beyond yfinance's ~60-day intraday "
              f"reach. Using IBKR + _yf caches only (no fresh intraday).")

    cache_hits = 0
    for name, (ticker, mtype) in B.TICKERS.items():
        d = truncate_daily(B.fetch_daily(ticker, start_date, cutoff.date()), cutoff)
        if d:
            B.daily[name] = d

        parts = [B.load_cached_intraday(name),
                 B.load_cached_intraday(name, suffix="_yf")]
        if yf30_ok:
            start = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=FINE_REACH_DAYS)).strftime('%Y-%m-%d')
            end   = (cutoff + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
            parts.append(_yf_intraday(ticker, '30m', start, end))
        merged = truncate_intraday(B._merge_bars([p for p in parts if p is not None]), cutoff)
        if merged is not None:
            B.intraday[name] = merged
            cache_hits += 1
            span = f"{merged['Open'].index.min().date()} → {merged['Open'].index.max().date()}"
            print(f"  💾 {name:<12} intraday≤cutoff ({len(merged['Open'])} bars, {span})")
        elif d:
            print(f"  ⚠️  {name:<12} daily only (≤ {max(d['Open'].index)})")
        else:
            print(f"  ❌ {name:<12} no pre-cutoff data")
    print(f"\n  📊 {cache_hits} instruments with intraday data before the cutoff")


def fetch_fine(interval, cutoff):
    """Fetch `interval` bars for ALL instruments around the cutoff, truncated.
    Returns {name: bars} (may be empty if unavailable)."""
    width = pd.Timedelta(minutes=int(interval[:-1]))
    start = (cutoff - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    end   = (cutoff + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    out = {}
    for name, (ticker, _) in B.TICKERS.items():
        bars = truncate_intraday(_yf_intraday(ticker, interval, start, end), cutoff, width)
        if bars is not None:
            out[name] = bars
    print(f"  🔎 {interval} bars fetched for {len(out)}/{len(B.TICKERS)} instruments")
    return out


def short_reaction_move(bars, width, post_dt, cutoff):
    """% move from the post to min(post+60min, cutoff), using only bars that
    closed by the cutoff. Returns None if no usable bars."""
    if bars is None:
        return None
    opens = bars['Open']
    closes = bars['Close'] if 'Close' in bars else bars['Open']
    past = opens[opens.index <= post_dt]
    if len(past) == 0:
        return None
    pos = opens.index.get_loc(past.index[-1])
    if isinstance(pos, slice):
        pos = pos.stop - 1
    initial = opens.iloc[pos]
    if pd.isna(initial) or initial == 0:
        initial = closes.iloc[pos]
    if pd.isna(initial) or initial == 0:
        return None

    target = min(post_dt + pd.Timedelta(minutes=60), cutoff)
    idx = opens.index
    mask = (idx > idx[pos]) & (idx <= target)
    if mask.any():
        reaction = closes.loc[idx[mask][-1]]
    else:
        # no later bar fits — the post's own bar close is the latest knowable price
        # (bars are pre-truncated, so this close is <= cutoff)
        reaction = closes.iloc[pos]
    if pd.isna(reaction) or reaction == 0:
        return None
    move = (reaction - initial) / initial * 100
    return round(float(move), 4)


def label_near_cutoff(near, cutoff, impact_cols):
    """Label posts within 90min of the cutoff: auto-pick a fine interval per
    post, fetch each needed interval once, fall back to 30m cache bars."""
    near = near.copy()
    remaining = (cutoff - near['date']).dt.total_seconds() / 60
    near['_interval'] = remaining.apply(pick_interval)

    days_back = (pd.Timestamp.now(tz=NY) - cutoff).days
    fine_ok = days_back <= FINE_REACH_DAYS
    needed = sorted(near['_interval'].unique())
    print(f"\n⏱️  Near-cutoff posts: {len(near)} (within 90min of cutoff) — "
          f"auto intervals: {', '.join(needed)}")
    fine = {}
    if fine_ok:
        for iv in needed:
            fine[iv] = fetch_fine(iv, cutoff)
    else:
        print(f"   ⚠️  Cutoff {days_back} days back — fine intervals not retrievable from "
              f"yfinance; using 30m cache bars (coarser reaction, still no lookahead).")

    new_cols = {c: [] for c in impact_cols}
    for _, row in near.iterrows():
        iv = row['_interval']
        width = pd.Timedelta(minutes=int(iv[:-1]))
        for col in impact_cols:
            name = col.replace('_Impact', '')
            bars = fine.get(iv, {}).get(name)
            move = short_reaction_move(bars, width, row['date'], cutoff)
            if move is None:   # fall back to the truncated 30m merged cache
                move = short_reaction_move(B.intraday.get(name), BAR, row['date'], cutoff)
            new_cols[col].append(move)
    for col in impact_cols:
        near[col] = new_cols[col]
        name = col.replace('_Impact', '')
        near[f'{name}_zscore']  = near[col]
        near[f'{name}_quality'] = 'intraday_short'
    return near.drop(columns=['_interval'])


def main():
    ap = argparse.ArgumentParser(description="Point-in-time backtest training set builder.")
    ap.add_argument("until", help="Cutoff, yyyymmddhhmm in NY time. No data after this is used.")
    ap.add_argument("--from", dest="since", metavar="yyyymmddhhmm",
                    help="Only include posts at/after this moment.")
    args = ap.parse_args()

    cutoff = parse_stamp(args.until, "until")
    since = parse_stamp(args.since, "--from") if args.since else None
    if cutoff > pd.Timestamp.now(tz=NY):
        sys.exit("❌ Cutoff is in the future — pick a past moment.")

    print("=" * 66)
    print("  BACKTEST TRAINING SET — POINT-IN-TIME (no data after cutoff)")
    print(f"  Cutoff: {cutoff}" + (f" | From: {since}" if since is not None else ""))
    print("=" * 66)

    scored = B.load_scored()
    n0 = len(scored)
    scored = scored[scored['date'] <= cutoff]
    n_after = n0 - len(scored)
    if since is not None:
        scored = scored[scored['date'] >= since]
    print(f"\n  Posts in window: {len(scored)} | excluded after cutoff: {n_after}")
    if scored.empty:
        sys.exit("❌ No posts in the window — nothing to label.")

    fetch_market_pit(scored['date'].dt.date.min(), cutoff)
    B.compute_baselines()

    # split: STANDARD (full 1h reaction knowable: reaction bar closed by cutoff)
    #        NEAR-CUTOFF (within 90min of cutoff -> finer interval, shorter reaction)
    # floor in UTC: flooring NY-tz timestamps raises on the ambiguous DST
    # fall-back hour (e.g. 2024-11-03 01:00). NY offsets are whole hours, so
    # the :00/:30 bar alignment is identical in UTC.
    label_known_at = (scored['date'].dt.tz_convert('UTC').dt.floor('30min')
                      .dt.tz_convert(NY) + pd.Timedelta(minutes=90))
    is_near = label_known_at > cutoff
    std, near = scored[~is_near].reset_index(drop=True), scored[is_near].reset_index(drop=True)
    print(f"  standard posts: {len(std)} | near-cutoff posts: {len(near)}")

    parts = []
    impact_cols = [f'{n}_Impact' for n in B.TICKERS]
    if len(std):
        std, impact_cols = B.compute_impacts(std)
        parts.append(std)
    if len(near):
        parts.append(label_near_cutoff(near, cutoff, impact_cols))

    scored = pd.concat(parts, ignore_index=True)
    scored = B.apply_caps(scored, impact_cols)
    scored = B.finalize(scored, impact_cols)
    scored = scored.sort_values('date')

    train_cols = B.train_columns(impact_cols)
    scored[train_cols].to_csv(TEST_CSV, index=False, lineterminator='\n')

    print(f"\n💾 Saved {TEST_CSV} ({len(scored)} rows)")
    print(f"   window: {scored['date'].min()} → {scored['date'].max()}")
    print(f"   high-signal (sample_weight>0.5): {(scored['sample_weight'] > 0.5).sum()}")
    print(f"   VIX range [{scored['VIX_Impact'].min():.3f}, {scored['VIX_Impact'].max():.3f}]")
    print("\n   Point-in-time guarantee: every truth ≤ cutoff is included; every label")
    print("   uses only bars that CLOSED by the cutoff (near-cutoff posts via auto")
    print("   2m/5m/15m intervals with reaction capped at the cutoff).")


if __name__ == '__main__':
    main()
