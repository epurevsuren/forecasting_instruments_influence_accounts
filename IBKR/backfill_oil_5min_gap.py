"""
backfill_oil_5min_gap.py
========================
Fill the OIL 15-minute and 30-minute market_data_cache GAPS from the local
5-minute WTI candle dump, and (re)build 15/30-min bars from those 5-minute bars.

CONTEXT
-------
After backfill_fx_history.py, OIL's 15/30 caches run 2010->2026 but have ONE
large hole: 2023-12-01 -> 2024-07-28 (~240 days) — the window between where the
HistData WTI FX feed ends (2023-12) and the next available data begins. (A
smaller late-2025 hole also exists.) `WTI-Crude-Oil-5-Minute-OHLC-Candles.csv`
covers 2011->2026 and is used to fill those holes.

SOURCE  IBKR/WTI-Crude-Oil-5-Minute-OHLC-Candles.csv
  header: time,date,instrument,granularity,open,high,low,close
  * time  = ISO-8601 UTC (e.g. 2024-06-01T00:00:00.000Z) — the 'Z' means UTC, so
    it is parsed straight to UTC (confirmed: the source close matches the
    existing OIL bar at the same timestamp near the 2023-11 boundary, 81.3 vs
    81.4, i.e. aligned with no offset AND on the same price basis).
  * no volume column.
  * 24/7 calendar (it HAS Saturday bars), unlike the existing OIL series which
    follows the futures calendar (no Saturday, Sunday only from ~21:00 UTC).

CACHE  IBKR/market_data_cache/OIL_<bar>.csv — schema matched EXACTLY:
  * 15min = date,open,high,low,close,volume,average,barCount
  * 30min = date,open,high,low,close,volume
  date is tz-aware UTC. Filled rows carry the quote-only sentinels
  volume=-1.0, average=-1.0, barCount=-1 (the source has no volume/VWAP/count).

WHAT IT DOES
  * Resamples the 5-min source to 15/30 (IBKR convention label='left',
    closed='left'; open=first, high=max, low=min, close=last). 5 divides 15 and
    30 evenly, so each 15-min bar aggregates 3 source bars and each 30-min bar 6.
  * Fills ONLY inside the LARGE gaps of the existing series (> --min-gap-days,
    default 5) — this targets the 240-day hole (and the late-2025 hole) while
    NOT touching the ordinary ~2-day weekend gaps. Use --between to fill an
    explicit window instead.
  * Drops the source's weekend / closed-session bars by default (Saturday all
    day, Sunday before 21:00 UTC, Friday after 22:00 UTC) so the filled region
    matches the existing futures calendar. --keep-weekends disables that.
  * MERGE keeps existing bars on overlap (never overwrites real data);
    --overwrite flips that.

USAGE
  python backfill_oil_5min_gap.py                     # fill OIL 15/30 large gaps
  python backfill_oil_5min_gap.py --between 2023-12-01 2024-07-29
  python backfill_oil_5min_gap.py --keep-weekends
  python backfill_oil_5min_gap.py --dry-run
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths ----
_HERE            = os.path.dirname(os.path.abspath(__file__))
SRC_FILE         = os.path.join(_HERE, "WTI-Crude-Oil-5-Minute-OHLC-Candles.csv")
CACHE_DIR        = os.path.join(_HERE, "market_data_cache")
INSTRUMENTS_FILE = os.path.join(_HERE, "..", "DP", "instruments.json")

INSTRUMENT = "OIL"
SENT_VOL, SENT_AVG, SENT_BC = -1.0, -1.0, -1
OHLC = ["open", "high", "low", "close"]

# bar -> (resample rule, target schema)
TARGETS = {
    "15min": ("15min", ["date", *OHLC, "volume", "average", "barCount"]),
    "30min": ("30min", ["date", *OHLC, "volume"]),
}


# --------------------------------------------------------------- helpers -----
def load_digits(name: str) -> int:
    try:
        with open(INSTRUMENTS_FILE, encoding="utf-8") as f:
            spec = json.load(f)["instruments"][name]
        return int(spec.get("ctrader", {}).get("digits", 5))
    except Exception:
        return 5


def load_5min(drop_weekends: bool, digits: int) -> pd.DataFrame:
    """Read the 5-min WTI CSV -> [date(UTC), open, high, low, close], sorted."""
    df = pd.read_csv(SRC_FILE, usecols=["time", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["time"], utc=True)          # 'Z' -> UTC
    for c in OHLC:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(digits)
    df = (df[["date", *OHLC]]
            .dropna(subset=["date", "open", "close"])
            .sort_values("date")
            .drop_duplicates("date", keep="first")
            .reset_index(drop=True))
    print(f"   source 5-min: {len(df):,} rows  {df['date'].min()} .. {df['date'].max()}")
    if drop_weekends:
        dow, hr = df["date"].dt.dayofweek, df["date"].dt.hour
        closed = (dow == 5) | ((dow == 6) & (hr < 21)) | ((dow == 4) & (hr > 22))
        n = int(closed.sum())
        df = df[~closed].reset_index(drop=True)
        print(f"   dropped {n:,} weekend/closed-session bars (Sat, Sun<21:00, Fri>22:00 UTC) "
              f"-> {len(df):,} rows on the futures calendar")
    return df


def read_cache(path: str) -> "pd.DataFrame | None":
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def to_schema(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            continue
        df[c] = SENT_BC if c == "barCount" else (SENT_VOL if c == "volume" else SENT_AVG)
    if "barCount" in cols:
        df["barCount"] = df["barCount"].fillna(SENT_BC).astype(int)
    return df[cols]


def resample_ohlc(src: pd.DataFrame, rule: str, cols: list) -> pd.DataFrame:
    """5-min -> rule, IBKR label='left'/closed='left'; sentinel aggregates."""
    s = src.set_index("date").sort_index()
    agg = (s.resample(rule, label="left", closed="left")
             .agg(open=("open", "first"), high=("high", "max"),
                  low=("low", "min"), close=("close", "last"))
             .dropna(subset=["open"])
             .reset_index())
    return to_schema(agg, cols)


def large_gaps(existing: pd.DataFrame, min_days: float) -> list:
    """[(gap_start, gap_end)] for gaps in `existing` longer than min_days."""
    if existing is None or len(existing) < 2:
        return []
    dt = existing["date"].sort_values().reset_index(drop=True)
    prev = dt.shift(1)
    span = dt - prev
    out = []
    for i in dt.index[span > pd.Timedelta(days=min_days)]:
        out.append((prev[i], dt[i]))
    return out


def in_any(dates: pd.Series, intervals: list) -> pd.Series:
    m = pd.Series(False, index=dates.index)
    for lo, hi in intervals:
        m |= (dates > lo) & (dates < hi)
    return m


def merge_fill(existing, incoming, cols, prefer_existing=True) -> pd.DataFrame:
    incoming = to_schema(incoming, cols)
    if existing is None or len(existing) == 0:
        combined = incoming
    else:
        existing = to_schema(existing, cols)
        if prefer_existing:
            keep = incoming[~incoming["date"].isin(set(existing["date"]))]
            combined = pd.concat([existing, keep], ignore_index=True)
        else:
            keep = existing[~existing["date"].isin(set(incoming["date"]))]
            combined = pd.concat([incoming, keep], ignore_index=True)
    combined = (combined.sort_values("date")
                        .drop_duplicates("date", keep="first")
                        .reset_index(drop=True))
    if "barCount" in cols:
        combined["barCount"] = combined["barCount"].fillna(SENT_BC).astype(int)
    return combined


def write_cache(path, df, cols, dry) -> None:
    df = to_schema(df, cols)
    if dry:
        print(f"     [dry-run] would write {len(df):,} rows -> {os.path.relpath(path, _HERE)}")
        return
    df.to_csv(path, index=False)
    print(f"     ✅ wrote {len(df):,} rows -> {os.path.relpath(path, _HERE)}")


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(
        description="Fill OIL 15/30-min cache gaps from the 5-minute WTI candle file.")
    ap.add_argument("--min-gap-days", type=float, default=5.0,
                    help="Only fill existing gaps longer than this (default 5 -> the "
                         "240-day 2023-24 hole and the late-2025 hole; skips 2-day weekends).")
    ap.add_argument("--between", nargs=2, metavar=("START", "END"),
                    help="Fill this explicit UTC window (YYYY-MM-DD) instead of auto-detected gaps.")
    ap.add_argument("--keep-weekends", action="store_true",
                    help="Keep the source's Sat/Sun/off-hours bars (default drops them to match "
                         "the existing futures calendar).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Let resampled values win on overlapping timestamps (default keeps existing).")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    if not os.path.exists(SRC_FILE):
        sys.exit(f"❌ source not found: {SRC_FILE}")

    print("=" * 72)
    print("  OIL 5-min -> 15/30-min gap fill" + ("   [DRY RUN]" if args.dry_run else ""))
    print("=" * 72)

    digits = load_digits(INSTRUMENT)
    src = load_5min(drop_weekends=not args.keep_weekends, digits=digits)

    explicit = None
    if args.between:
        lo = pd.Timestamp(args.between[0], tz="UTC")
        hi = pd.Timestamp(args.between[1], tz="UTC")
        explicit = [(lo, hi)]
        print(f"   fill window (--between): {lo} .. {hi}")

    for bar, (rule, cols) in TARGETS.items():
        tgt = os.path.join(CACHE_DIR, f"{INSTRUMENT}_{bar}.csv")
        existing = read_cache(tgt)
        cols = list(existing.columns) if existing is not None else cols   # match real header
        res = resample_ohlc(src, rule, cols)

        intervals = explicit if explicit is not None else large_gaps(existing, args.min_gap_days)
        print(f"\n   [{bar}] existing rows: {0 if existing is None else len(existing):,}; "
              f"{len(intervals)} gap(s) to fill:")
        for lo, hi in intervals:
            n_in = int(((res['date'] > lo) & (res['date'] < hi)).sum())
            print(f"      {lo} -> {hi}  ({(hi - lo).days}d)  ->  {n_in:,} source bars available")

        cand = res[in_any(res["date"], intervals)] if intervals else res.iloc[0:0]
        before = 0 if existing is None else len(existing)
        merged = merge_fill(existing, cand, cols, prefer_existing=not args.overwrite)
        print(f"   [{bar}] filled {len(merged) - before:,} bars -> {len(merged):,} total")
        write_cache(tgt, merged, cols, args.dry_run)

    print("\n" + "=" * 72)
    print("  done." + ("  (dry run — nothing written)" if args.dry_run else ""))
    print("=" * 72)


if __name__ == "__main__":
    main()
