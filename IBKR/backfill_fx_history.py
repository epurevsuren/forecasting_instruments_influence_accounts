"""
backfill_fx_history.py
======================
Fill IBKR/market_data_cache 1-minute bars for the FX-backed commodity contracts
(GOLD, OIL) from the local HistData FX-1-Minute-Data dumps, then rebuild the
15-minute and 30-minute caches from the merged 1-minute series.

WHY
---
IBKR only returned ~1 recent month of 1-min GOLD/OIL bars (they are continuous
futures — GC/CL — with short intraday history). HistData's FX-1-Minute-Data
provides YEARS of 1-minute history (XAUUSD spot -> GOLD, WTIUSD spot -> OIL)
that we use to backfill the missing history, then rebuild 15/30-min from it.

SOURCE  (IBKR/FX-1-Minute-Data/<sym>/DAT_ASCII_<SYM>_M1_<YYYY|YYYYMM>.csv)
  * no header, ';'-separated
  * columns: DateTimeStamp;Bar OPEN;Bar HIGH;Bar LOW;Bar CLOSE;Volume
  * DateTimeStamp = 'YYYYMMDD HHMMSS' in AMERICA/NEW_YORK local time, DST-aware.
    Empirically confirmed against the IBKR overlap: May-2026 XAUUSD aligns with
    IBKR GOLD only at UTC-4 (1-min return-correlation 0.88 vs ~0.06 at other
    offsets), and the weekly Sunday session opens at a constant ~18:00 local in
    BOTH winter and summer -> the source observes US daylight saving.
  * Volume is always 0 for HistData FX/CFD data and is NOT used.

CACHE  (IBKR/market_data_cache/<NAME>_<bar>.csv) — schema matched EXACTLY:
  * 1min, 15min : date,open,high,low,close,volume,average,barCount   (8 cols)
  * 30min       : date,open,high,low,close,volume                    (6 cols)
  date is tz-aware UTC. FX-derived rows carry the same sentinels the rest of the
  cache uses for quote-only feeds: volume=-1.0, average=-1.0, barCount=-1 (spot
  FX has no real volume / VWAP / tick-count — identical convention to the
  EUR_USD / AUD_USD MIDPOINT caches).

POLICY  (confirmed with the user)
  * 1-min : MERGE — prepend FX history, KEEP existing IBKR bars on any
            overlapping timestamp (they carry real futures volume; OIL's only
            2026 data is IBKR). NOTE: FX is SPOT and sits below the IBKR FUTURES
            level by the cost-of-carry basis (~$33 for gold), so there is ONE
            small price seam at the spot->futures join — reported per instrument.
  * 15/30 : FILL GAPS — resample the merged 1-min and add ONLY timestamps that
            are missing from the existing 15/30 files; keep IBKR-native bars.
  --overwrite flips both to let the FX / resampled values win on overlap.

RESAMPLE  (matches IBKR's own bars — verified reproducing the native GOLD 15-min
from the 1-min at 78% exact with no systematic offset):
  label='left', closed='left'; open=first, high=max, low=min, close=last.

USAGE
  python backfill_fx_history.py                 # GOLD + OIL, merge + fill (default)
  python backfill_fx_history.py --instruments GOLD
  python backfill_fx_history.py --overwrite     # FX / resampled win on overlap
  python backfill_fx_history.py --agg-only      # skip 1-min fill; only rebuild 15/30
  python backfill_fx_history.py --dry-run       # compute + report, write nothing
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths ----
_HERE            = os.path.dirname(os.path.abspath(__file__))
FX_DIR           = os.path.join(_HERE, "FX-1-Minute-Data")
CACHE_DIR        = os.path.join(_HERE, "market_data_cache")
INSTRUMENTS_FILE = os.path.join(_HERE, "..", "DP", "instruments.json")

# Instrument -> HistData source sub-folder. Only these two ship an FX dump; add a
# line here (and drop the folder in FX-1-Minute-Data/) to backfill another one.
FX_SOURCE = {"GOLD": "xauusd", "OIL": "wtiusd"}

SRC_TZ = "America/New_York"          # HistData local time, DST-aware (see header)

# FX / quote-only sentinels used across market_data_cache
SENT_VOL, SENT_AVG, SENT_BC = -1.0, -1.0, -1

COLS_8 = ["date", "open", "high", "low", "close", "volume", "average", "barCount"]
COLS_6 = ["date", "open", "high", "low", "close", "volume"]

# bar -> (pandas resample rule, target schema).  1-min is COLS_8 (handled apart).
BAR_RULES = {"15min": ("15min", COLS_8), "30min": ("30min", COLS_6)}


# --------------------------------------------------------------- helpers -----
def load_digits(name: str) -> int:
    """Price decimals for `name` from the instruments registry (fallback 5)."""
    try:
        with open(INSTRUMENTS_FILE, encoding="utf-8") as f:
            spec = json.load(f)["instruments"][name]
        return int(spec.get("ctrader", {}).get("digits", 5))
    except Exception:
        return 5


def load_fx_1min(folder: str, digits: int) -> "pd.DataFrame | None":
    """Load + concat every DAT_ASCII_*_M1_*.csv in `folder`, convert the
    America/New_York timestamps to UTC, and return the canonical 8-col frame
    with FX sentinels for volume/average/barCount."""
    files = sorted(glob.glob(os.path.join(FX_DIR, folder, "DAT_ASCII_*_M1_*.csv")))
    if not files:
        return None

    frames = []
    for fp in files:
        # usecols drops the always-zero HistData volume column (index 5)
        df = pd.read_csv(fp, sep=";", header=None, usecols=[0, 1, 2, 3, 4],
                         names=["dt", "open", "high", "low", "close"])
        print(f"     • {os.path.basename(fp)}: {len(df):,} rows")
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    naive = pd.to_datetime(raw["dt"], format="%Y%m%d %H%M%S")

    # sort ascending so tz_localize(ambiguous='infer') can resolve the fall-back
    # (duplicated) DST hour from bar order
    order = np.argsort(naive.values, kind="stable")
    raw, naive = raw.iloc[order].reset_index(drop=True), naive.iloc[order].reset_index(drop=True)

    try:
        local = naive.dt.tz_localize(SRC_TZ, ambiguous="infer", nonexistent="shift_forward")
    except Exception as e:
        print(f"     ⚠️  DST-ambiguous timestamps could not be inferred ({type(e).__name__}); "
              f"assuming standard time for those (~<=60 bars/yr, ±1h). ")
        local = naive.dt.tz_localize(SRC_TZ, ambiguous=False, nonexistent="shift_forward")

    out = pd.DataFrame({
        "date":     local.dt.tz_convert("UTC"),
        "open":     raw["open"].round(digits),
        "high":     raw["high"].round(digits),
        "low":      raw["low"].round(digits),
        "close":    raw["close"].round(digits),
        "volume":   SENT_VOL,
        "average":  SENT_AVG,
        "barCount": SENT_BC,
    })
    out = (out.dropna(subset=["date"])
              .sort_values("date")
              .drop_duplicates("date", keep="first")
              .reset_index(drop=True))
    out["barCount"] = out["barCount"].astype(int)
    return out


def read_cache(path: str) -> "pd.DataFrame | None":
    """Read an existing cache CSV as a UTC-tz-aware frame, or None if absent."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def resample_ohlc(df1: pd.DataFrame, rule: str, cols: list) -> pd.DataFrame:
    """Resample 1-min bars to `rule` with IBKR's label='left', closed='left'
    convention. FX-derived aggregates carry the volume/average/barCount
    sentinels (spot FX has no real volume/VWAP/tick-count)."""
    s = df1.set_index("date").sort_index()
    agg = (s.resample(rule, label="left", closed="left")
             .agg(open=("open", "first"), high=("high", "max"),
                  low=("low", "min"), close=("close", "last"))
             .dropna(subset=["open"])
             .reset_index())
    agg["volume"] = SENT_VOL
    if "average" in cols:
        agg["average"] = SENT_AVG
    if "barCount" in cols:
        agg["barCount"] = SENT_BC
        agg["barCount"] = agg["barCount"].astype(int)
    return agg[cols]


def merge_fill(existing, incoming, cols, prefer_existing=True) -> pd.DataFrame:
    """Union of `existing` and `incoming` keyed on `date`. On overlap, keep the
    preferred side (existing IBKR bars by default; incoming when --overwrite)."""
    incoming = incoming.reindex(columns=cols)
    if existing is None or len(existing) == 0:
        combined = incoming
    else:
        existing = existing.reindex(columns=cols)
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


def write_cache(path: str, df: pd.DataFrame, cols: list, dry: bool) -> None:
    df = df.reindex(columns=cols)
    if dry:
        print(f"     [dry-run] would write {len(df):,} rows -> {os.path.relpath(path, _HERE)}")
        return
    df.to_csv(path, index=False)   # tz-aware date -> 'YYYY-MM-DD HH:MM:SS+00:00'
    print(f"     ✅ wrote {len(df):,} rows -> {os.path.relpath(path, _HERE)}")


def report_seam(existing1, fx, name: str) -> None:
    """Print the spot->futures price seam at the join of FX history and the
    existing IBKR bars (a heads-up, not an error)."""
    if existing1 is None or len(existing1) == 0:
        return
    boundary = existing1["date"].min()
    near = fx[fx["date"] <= boundary]
    if len(near):
        f_last  = near.iloc[-1]
        f_close = float(f_last["close"])
        i_close = float(existing1.sort_values("date").iloc[0]["close"])
        gap_days = (boundary - f_last["date"]).total_seconds() / 86400.0
        if gap_days <= 2:
            print(f"     ↔ basis seam @ {boundary}: IBKR futures {i_close} vs FX spot "
                  f"{f_close}  (Δ {i_close - f_close:+.2f}, FX/IBKR adjoin)")
        else:
            print(f"     ⚠️  data gap: FX ends {f_last['date']} ({f_close}), IBKR starts "
                  f"{boundary} ({i_close}) — {gap_days:.0f}-day hole with no source.")


# ---------------------------------------------------------------- per-inst ----
def process_instrument(name: str, args) -> None:
    if name not in FX_SOURCE:
        print(f"\n❌ {name}: no FX source folder mapped (known: {', '.join(FX_SOURCE)}) — skipping.")
        return
    print(f"\n=== {name}  (FX source: {FX_SOURCE[name]}) ===")
    digits = load_digits(name)
    one_path = os.path.join(CACHE_DIR, f"{name}_1min.csv")
    existing1 = read_cache(one_path)

    if args.agg_only:
        merged1 = existing1
        n = 0 if merged1 is None else len(merged1)
        print(f"   --agg-only: rebuilding 15/30 from existing {name}_1min.csv ({n:,} rows)")
        if merged1 is None or len(merged1) == 0:
            print("   ⚠️  no 1-min data available — skipping.")
            return
    else:
        print(f"   loading FX 1-min from {FX_SOURCE[name]}/ ...")
        fx = load_fx_1min(FX_SOURCE[name], digits)
        if fx is None:
            print(f"   ⚠️  no FX files under {FX_SOURCE[name]}/ — skipping {name}.")
            return
        print(f"   FX 1-min: {len(fx):,} rows  {fx['date'].min()} .. {fx['date'].max()}")
        before = 0 if existing1 is None else len(existing1)
        merged1 = merge_fill(existing1, fx, COLS_8, prefer_existing=not args.overwrite)
        print(f"   1-min: {before:,} existing + {len(merged1) - before:,} filled "
              f"= {len(merged1):,} rows  ({merged1['date'].min()} .. {merged1['date'].max()})")
        report_seam(existing1, fx, name)
        write_cache(one_path, merged1, COLS_8, args.dry_run)

    # rebuild 15/30 from the merged 1-min (fill gaps, keep native bars)
    for bar, (rule, cols) in BAR_RULES.items():
        tgt = os.path.join(CACHE_DIR, f"{name}_{bar}.csv")
        existing = read_cache(tgt)
        res = resample_ohlc(merged1, rule, cols)
        before = 0 if existing is None else len(existing)
        filled = merge_fill(existing, res, cols, prefer_existing=not args.overwrite)
        print(f"   {bar}: {before:,} existing + {len(filled) - before:,} filled = {len(filled):,} rows")
        write_cache(tgt, filled, cols, args.dry_run)


def main():
    ap = argparse.ArgumentParser(
        description="Backfill GOLD/OIL market_data_cache from local FX-1-Minute-Data and rebuild 15/30-min.")
    ap.add_argument("--instruments", nargs="+", default=list(FX_SOURCE),
                    metavar="NAME", help=f"Instruments to fill (default: {', '.join(FX_SOURCE)}).")
    ap.add_argument("--overwrite", action="store_true",
                    help="On overlapping timestamps let FX / resampled values win "
                         "(default keeps existing IBKR bars).")
    ap.add_argument("--agg-only", action="store_true",
                    help="Skip the 1-min FX fill; only rebuild 15/30 from the existing 1-min cache.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and report everything but write no files.")
    args = ap.parse_args()

    if not os.path.isdir(CACHE_DIR):
        sys.exit(f"❌ cache dir not found: {CACHE_DIR}")

    print("=" * 72)
    print("  FX -> market_data_cache backfill" + ("   [DRY RUN]" if args.dry_run else ""))
    print(f"  policy: {'OVERWRITE (FX wins)' if args.overwrite else 'MERGE (keep existing IBKR bars)'}"
          f"{' | agg-only' if args.agg_only else ''}")
    print("=" * 72)

    for name in args.instruments:
        process_instrument(name.upper(), args)

    print("\n" + "=" * 72)
    print("  done." + ("  (dry run — nothing written)" if args.dry_run else ""))
    print("=" * 72)


if __name__ == "__main__":
    main()
