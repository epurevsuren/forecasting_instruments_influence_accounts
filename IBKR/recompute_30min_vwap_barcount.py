"""
recompute_30min_vwap_barcount.py
================================
Add the missing `average` (VWAP) and `barCount` columns to the 30-minute
market_data_cache CSVs by deriving them from the matching 15-minute bars.

IS IT THEORETICALLY POSSIBLE?  YES — exactly.
  * barCount is a COUNT, so a 30-min bar's barCount = the sum of its two 15-min
    bars' barCounts.
  * IBKR's `average` is the bar VWAP. Because VWAP * volume = Σ(price*size) is
    additive, the 30-min VWAP is the volume-weighted blend of its two 15-min
    VWAPs:   avg30 = (avg_a*vol_a + avg_b*vol_b) / (vol_a + vol_b).
  Both need only each 15-min bar's own volume + average + barCount, which we
  have. (Validated against 1-min ground truth on SPY: VWAP mean |Δ| ~3e-4,
  barCount 99% exact — the residual is just coverage differences between the
  separately-fetched 1-min and 15-min files, not the method.)

QUOTE-ONLY FEEDS (no real volume) keep their existing convention:
  * FX MIDPOINT (volume=average=barCount = -1): no VWAP possible -> stays -1/-1.
  * VIX index    (volume=average=barCount = 0): no volume to weight -> average
    falls back to the plain mean of the 15-min averages (= 0), barCount = sum (= 0).

WHAT IT DOES (non-destructive to OHLCV)
  For every instrument whose <NAME>_30min.csv LACKS these columns and whose
  <NAME>_15min.csv HAS them:
    1. group the 15-min bars into 30-min bins (label='left', closed='left'),
    2. compute average (sentinel-aware VWAP) and barCount (sentinel-aware sum),
    3. LEFT-join those two columns onto the EXISTING 30-min rows by timestamp —
       OHLCV is untouched; 30-min bars with no 15-min data get -1 sentinels,
    4. rewrite the CSV in the 15-min column order
       date,open,high,low,close,volume,<average|wap>,barCount.
  Instruments whose 30-min already has the columns are skipped (use --force to
  recompute). BTC/ETH already ship 8-col 30-min and are skipped.

USAGE
  python recompute_30min_vwap_barcount.py                 # all eligible instruments
  python recompute_30min_vwap_barcount.py --instruments OIL GOLD SPY
  python recompute_30min_vwap_barcount.py --force         # recompute even if present
  python recompute_30min_vwap_barcount.py --dry-run
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import bars   # DuckDB engine over market_data_cache CSVs (read / resample / write)

_HERE     = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "market_data_cache")

BASE_COLS = ["date", "open", "high", "low", "close", "volume"]
SENT_BC   = -1


def process(name: str, force: bool, dry: bool) -> None:
    p30 = os.path.join(CACHE_DIR, f"{name}_30min.csv")
    p15 = os.path.join(CACHE_DIR, f"{name}_15min.csv")
    d30 = bars.read(p30)
    if d30 is None:
        print(f"  ⏭️  {name:<10} no 30-min file — skip"); return
    if not os.path.exists(p15):
        print(f"  ⏭️  {name:<10} no 15-min file — cannot derive — skip"); return

    # The 15-min VWAP column is whatever sits in slot 7 (average, or wap for
    # crypto). Read only the header (cheap); DuckDB streams the full 15-min file
    # in bars.resample — the big file never enters pandas.
    cols15  = list(pd.read_csv(p15, nrows=0).columns)
    extra15 = [c for c in cols15 if c not in BASE_COLS]
    if "barCount" not in cols15 or not extra15:
        print(f"  ⏭️  {name:<10} 15-min lacks average/barCount — skip"); return
    vwap_col = next((c for c in extra15 if c != "barCount"), None)
    if vwap_col is None:
        print(f"  ⏭️  {name:<10} 15-min has no VWAP column — skip"); return

    already = (vwap_col in d30.columns) and ("barCount" in d30.columns)
    if already and not force:
        print(f"  ✅ {name:<10} already has {vwap_col}+barCount — skip (use --force)"); return

    # DuckDB: 15-min CSV -> 30-min sentinel-aware VWAP + summed barCount.
    derived = bars.resample(p15, 30, ["date", vwap_col, "barCount"], vwap_col=vwap_col)

    # LEFT-join derived columns onto existing 30-min rows (OHLCV untouched)
    base = d30.drop(columns=[c for c in (vwap_col, "barCount") if c in d30.columns])
    out = base.merge(derived, on="date", how="left")

    # classify BEFORE filling NaNs (so we can tell "no 15-min bar" from "15-min
    # had no volume" i.e. an FX/spot -1 sentinel that legitimately yields -1)
    no_bin = int(out[vwap_col].isna().sum())
    real   = int((out[vwap_col] > 0).sum())
    zero   = int((out[vwap_col] == 0).sum())          # VIX / index
    neg15  = int((out[vwap_col] == -1.0).sum())        # matched, but 15-min = no-volume

    out[vwap_col] = out[vwap_col].fillna(-1.0)
    out["barCount"] = out["barCount"].fillna(SENT_BC).astype("int64")
    out = out[[*BASE_COLS, vwap_col, "barCount"]].sort_values("date").reset_index(drop=True)

    print(f"  {name:<10} {len(out):>7,} rows | real VWAP {real:,} | index/zero {zero:,} | "
          f"-1 (15-min no-volume) {neg15:,} | -1 (no 15-min bar) {no_bin:,}")

    if dry:
        print(f"     [dry-run] would write -> {os.path.relpath(p30, _HERE)}")
        return
    bars.write_csv(p30, out, list(out.columns))   # byte-identical cache format
    print(f"     ✅ wrote {os.path.relpath(p30, _HERE)}  (now {len(out.columns)} cols)")


def discover() -> list:
    names = []
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*_30min.csv"))):
        b = os.path.basename(f)
        if b.endswith("_30min_yf.csv"):
            continue
        names.append(b[:-len("_30min.csv")])
    return names


def main():
    ap = argparse.ArgumentParser(
        description="Add 30-min average(VWAP)+barCount derived from 15-min bars.")
    ap.add_argument("--instruments", nargs="+", metavar="NAME", default=None,
                    help="Instruments to process (default: all with a *_30min.csv).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if the 30-min file already has the columns.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    if not os.path.isdir(CACHE_DIR):
        sys.exit(f"❌ cache dir not found: {CACHE_DIR}")

    names = [n.upper() for n in args.instruments] if args.instruments else discover()
    print("=" * 66)
    print("  Recompute 30-min average(VWAP)+barCount from 15-min bars"
          + ("   [DRY RUN]" if args.dry_run else ""))
    print("=" * 66)
    for name in names:
        process(name, force=args.force, dry=args.dry_run)
    print("=" * 66)
    print("  done." + ("  (dry run — nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
