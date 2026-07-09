"""
backfill_intraday_history.py
============================
Fill IBKR/market_data_cache 1-minute bars for contracts whose IBKR history is
short/missing, using the local `stock-data-intraday-minute-bar` dataset
(pyfinancialdata layout), then rebuild the 15-min and 30-min caches from the
merged 1-minute series.

INSTRUMENTS  (instrument -> dataset sub-path, all values verified present)
  NATGAS -> currencies/oanda/NATGAS_USD    (2005-2020)   OANDA nat-gas CFD
  US10Y  -> currencies/oanda/USB10Y_USD    (2005-2020)   OANDA US 10Y T-note PRICE
  US2Y   -> currencies/oanda/USB02Y_USD    (2005-2020)   OANDA US 2Y  T-note PRICE
  BTC    -> cryptocurrencies/bitstamp/BTC_USD (2012-2018) — only the portion
            BEFORE the existing cache start (~2017-08-17) is used (the rest is
            already covered by the Binance cache).
  ETH    -> NO usable source in this dataset. The only ETH file is
            cryptocurrencies/kraken/ETH_EUR: it is EUR-denominated, covers 2018
            ONLY (after the 2017-08 cutoff) and carries BTC-scale values. ETH is
            therefore SKIPPED — supply a real pre-2017-08 ETH-USD source (Gemini
            / CryptoDataDownload / Kraken ETHUSD) to fill it.

TIMEZONE — both sources store UTC (empirically confirmed against the caches):
  * OANDA: its own oanda_prices.py builds timestamps at UTC offset 0; and OANDA
    XAU_USD vs the (NY-tz) HistData XAUUSD aligns only at offset 0 with 1-min
    return-correlation 0.98.
  * bitstamp BTC_USD vs the existing (UTC) BTC cache aligns only at offset 0,
    with matching price level (~$5.78k on the 2017 overlap).
  So timestamps are localized straight to UTC (no DST handling needed).

SOURCE FORMATS (no timezone in the strings; naive local == UTC here)
  * oanda    : header  time,close,high,low,open,volume   (one CSV per month)
  * bitstamp : header  date,open,high,low,close          (one CSV per year)
  (oanda's `volume` is a tick-count, NOT contract volume comparable to IBKR's,
   so it is intentionally NOT mapped — see sentinels below.)

CACHE  (IBKR/market_data_cache/<NAME>_<bar>.csv) — schema matched PER TARGET:
  * NATGAS/US10Y/US2Y : 1min,15min = date,open,high,low,close,volume,average,barCount
                        30min       = date,open,high,low,close,volume
  * BTC (crypto)      : 1min,15min,30min = date,open,high,low,close,volume,wap,barCount
  date is tz-aware UTC. Alt-source rows carry the cache's quote-only sentinels
  volume=-1.0, average/wap=-1.0, barCount=-1 (same convention as the FX / FX-
  MIDPOINT caches — no fabricated volume/VWAP/tick-count).

POLICY  (same as backfill_fx_history.py, user-confirmed)
  * 1-min : MERGE — add source history, KEEP existing IBKR/Binance bars on any
            overlapping timestamp (they carry real volume). Crypto sources are
            pre-cut to timestamps strictly BEFORE the existing cache start, so
            they only ever fill the leading gap.
  * 15/30 : FILL GAPS — resample the merged 1-min and add ONLY timestamps
            missing from the existing 15/30 files; keep native bars.
  --overwrite flips both to let the source / resampled values win on overlap.

RESAMPLE  (IBKR convention): label='left', closed='left';
  open=first, high=max, low=min, close=last.

USAGE
  python backfill_intraday_history.py                       # all 4 fillable
  python backfill_intraday_history.py --instruments NATGAS US10Y
  python backfill_intraday_history.py --overwrite
  python backfill_intraday_history.py --agg-only            # only rebuild 15/30
  python backfill_intraday_history.py --dry-run
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
SB_DIR           = os.path.join(_HERE, "stock-data-intraday-minute-bar",
                                "pyfinancialdata", "data")
CACHE_DIR        = os.path.join(_HERE, "market_data_cache")
INSTRUMENTS_FILE = os.path.join(_HERE, "..", "DP", "instruments.json")

# instrument -> dataset location + reader format + crypto flag.
# 'crypto' sources are pre-cut to BEFORE the existing cache start (gap-fill only).
SOURCE = {
    "NATGAS": {"path": "currencies/oanda/NATGAS_USD",       "fmt": "oanda",    "crypto": False},
    "US10Y":  {"path": "currencies/oanda/USB10Y_USD",       "fmt": "oanda",    "crypto": False},
    "US2Y":   {"path": "currencies/oanda/USB02Y_USD",       "fmt": "oanda",    "crypto": False},
    "BTC":    {"path": "cryptocurrencies/bitstamp/BTC_USD",  "fmt": "bitstamp", "crypto": True},
    # "ETH": no usable pre-2017-08 ETH-USD source in this dataset (see header).
}

SENT_VOL, SENT_AVG, SENT_BC = -1.0, -1.0, -1     # cache quote-only sentinels
OHLC = ["open", "high", "low", "close"]

# fallback schemas if an existing target file is somehow absent
FALLBACK_COLS = {
    ("oanda",  "1min"):  ["date", *OHLC, "volume", "average", "barCount"],
    ("oanda",  "15min"): ["date", *OHLC, "volume", "average", "barCount"],
    ("oanda",  "30min"): ["date", *OHLC, "volume"],
    ("crypto", "1min"):  ["date", *OHLC, "volume", "wap", "barCount"],
    ("crypto", "15min"): ["date", *OHLC, "volume", "wap", "barCount"],
    ("crypto", "30min"): ["date", *OHLC, "volume", "wap", "barCount"],
}
BAR_RULES = {"15min": "15min", "30min": "30min"}


# --------------------------------------------------------------- helpers -----
def load_digits(name: str) -> int:
    """Price decimals for `name` from the instruments registry (fallback 5)."""
    try:
        with open(INSTRUMENTS_FILE, encoding="utf-8") as f:
            spec = json.load(f)["instruments"][name]
        return int(spec.get("ctrader", {}).get("digits", 5))
    except Exception:
        return 5


def _read_one(fp: str, fmt: str) -> pd.DataFrame:
    """Read a single source CSV -> [date(str), open, high, low, close]."""
    if fmt == "oanda":            # time,close,high,low,open,volume
        df = pd.read_csv(fp, usecols=["time", "open", "high", "low", "close"])
        return df.rename(columns={"time": "date"})
    if fmt == "bitstamp":         # date,open,high,low,close
        return pd.read_csv(fp, usecols=["date", "open", "high", "low", "close"])
    if fmt == "kraken":           # time,close,high,low,open
        df = pd.read_csv(fp, usecols=["time", "open", "high", "low", "close"])
        return df.rename(columns={"time": "date"})
    raise ValueError(f"unknown source format: {fmt}")


def load_source_1min(cfg: dict, digits: int) -> "pd.DataFrame | None":
    """Load + concat every CSV under the instrument path, localize UTC, and
    return the canonical [date(UTC), o,h,l,c, volume=-1, <avg/wap>=-1, barCount=-1].
    The 7th column name is decided later per target; here we carry 'average'."""
    files = sorted(glob.glob(os.path.join(SB_DIR, cfg["path"], "**", "*.csv"), recursive=True))
    if not files:
        return None
    frames = [_read_one(fp, cfg["fmt"]) for fp in files]
    raw = pd.concat(frames, ignore_index=True)
    print(f"     loaded {len(files)} file(s), {len(raw):,} raw rows")

    date = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize("UTC")
    out = pd.DataFrame({
        "date":     date,
        "open":     pd.to_numeric(raw["open"],  errors="coerce").round(digits),
        "high":     pd.to_numeric(raw["high"],  errors="coerce").round(digits),
        "low":      pd.to_numeric(raw["low"],   errors="coerce").round(digits),
        "close":    pd.to_numeric(raw["close"], errors="coerce").round(digits),
        "volume":   SENT_VOL,
        "average":  SENT_AVG,     # placeholder; renamed to 'wap' for crypto targets
        "barCount": SENT_BC,
    })
    out = (out.dropna(subset=["date", "open", "close"])
              .sort_values("date")
              .drop_duplicates("date", keep="first")
              .reset_index(drop=True))
    out["barCount"] = out["barCount"].astype(int)
    return out


def read_cache(path: str) -> "pd.DataFrame | None":
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def to_schema(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return df with exactly `cols`. Missing sentinel columns are added:
    the 7th col may be 'average' OR 'wap' — whichever the target uses gets -1."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            continue
        if c in ("average", "wap"):
            df[c] = SENT_AVG
        elif c == "volume":
            df[c] = SENT_VOL
        elif c == "barCount":
            df[c] = SENT_BC
        else:
            df[c] = np.nan
    if "barCount" in cols:
        df["barCount"] = df["barCount"].fillna(SENT_BC).astype(int)
    return df[cols]


def resample_ohlc(df1: pd.DataFrame, rule: str, cols: list) -> pd.DataFrame:
    """Resample 1-min -> rule with IBKR's label='left', closed='left'. Aggregate
    columns (volume/average/wap/barCount) carry the quote-only sentinels."""
    s = df1.set_index("date").sort_index()
    agg = (s.resample(rule, label="left", closed="left")
             .agg(open=("open", "first"), high=("high", "max"),
                  low=("low", "min"), close=("close", "last"))
             .dropna(subset=["open"])
             .reset_index())
    return to_schema(agg, cols)


def merge_fill(existing, incoming, cols, prefer_existing=True) -> pd.DataFrame:
    """Union on `date`; on overlap keep the preferred side."""
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


def target_cols(existing, fmt_key, bar) -> list:
    """Exact target schema: prefer the existing file's header, else fallback."""
    if existing is not None:
        return list(existing.columns)
    return FALLBACK_COLS[(fmt_key, bar)]


def write_cache(path, df, cols, dry) -> None:
    df = to_schema(df, cols)
    if dry:
        print(f"     [dry-run] would write {len(df):,} rows -> {os.path.relpath(path, _HERE)}")
        return
    df.to_csv(path, index=False)
    print(f"     ✅ wrote {len(df):,} rows -> {os.path.relpath(path, _HERE)}")


# ---------------------------------------------------------------- per-inst ----
def process_instrument(name: str, args) -> None:
    if name not in SOURCE:
        if name == "ETH":
            print(f"\n=== ETH === ⏭️  SKIPPED: no usable pre-2017-08 ETH-USD source in this "
                  f"dataset (only kraken/ETH_EUR: EUR-priced, 2018-only, BTC-scale). "
                  f"Supply a real ETH-USD minute source to fill it.")
        else:
            print(f"\n❌ {name}: not mapped in SOURCE (known: {', '.join(SOURCE)}) — skipping.")
        return

    cfg = SOURCE[name]
    fmt_key = "crypto" if cfg["crypto"] else "oanda"
    print(f"\n=== {name}  (source: {cfg['path']}) ===")
    digits = load_digits(name)
    one_path = os.path.join(CACHE_DIR, f"{name}_1min.csv")
    existing1 = read_cache(one_path)
    cols1 = target_cols(existing1, fmt_key, "1min")

    if args.agg_only:
        merged1 = existing1
        n = 0 if merged1 is None else len(merged1)
        print(f"   --agg-only: rebuilding 15/30 from existing {name}_1min.csv ({n:,} rows)")
        if merged1 is None or len(merged1) == 0:
            print("   ⚠️  no 1-min data — skipping.")
            return
    else:
        print(f"   loading source 1-min ...")
        src = load_source_1min(cfg, digits)
        if src is None or len(src) == 0:
            print(f"   ⚠️  no source rows under {cfg['path']} — skipping {name}.")
            return
        # crypto: keep only bars BEFORE the existing cache start (gap-fill only)
        if cfg["crypto"] and existing1 is not None and len(existing1):
            cut = existing1["date"].min()
            before = len(src)
            src = src[src["date"] < cut]
            print(f"   crypto pre-cut @ {cut}: {before:,} -> {len(src):,} source rows kept "
                  f"(only bars before the existing cache start)")
        print(f"   source 1-min: {len(src):,} rows  "
              f"{src['date'].min()} .. {src['date'].max()}" if len(src) else "   source 1-min: 0 rows")
        if len(src) == 0:
            print("   nothing to fill (source fully covered by existing cache).")
            merged1 = existing1
        else:
            n0 = 0 if existing1 is None else len(existing1)
            merged1 = merge_fill(existing1, src, cols1, prefer_existing=not args.overwrite)
            print(f"   1-min: {n0:,} existing + {len(merged1) - n0:,} filled = {len(merged1):,} rows "
                  f"({merged1['date'].min()} .. {merged1['date'].max()})")
            write_cache(one_path, merged1, cols1, args.dry_run)

    # rebuild 15/30 from the merged 1-min (fill gaps, keep native bars)
    for bar, rule in BAR_RULES.items():
        tgt = os.path.join(CACHE_DIR, f"{name}_{bar}.csv")
        existing = read_cache(tgt)
        cols = target_cols(existing, fmt_key, bar)
        res = resample_ohlc(merged1, rule, cols)
        n0 = 0 if existing is None else len(existing)
        filled = merge_fill(existing, res, cols, prefer_existing=not args.overwrite)
        print(f"   {bar}: {n0:,} existing + {len(filled) - n0:,} filled = {len(filled):,} rows")
        write_cache(tgt, filled, cols, args.dry_run)


def main():
    ap = argparse.ArgumentParser(
        description="Backfill market_data_cache from the stock-data-intraday-minute-bar dataset "
                    "(NATGAS/US10Y/US2Y + BTC pre-2017-08) and rebuild 15/30-min.")
    ap.add_argument("--instruments", nargs="+", default=list(SOURCE),
                    metavar="NAME", help=f"Instruments to fill (default: {', '.join(SOURCE)}).")
    ap.add_argument("--overwrite", action="store_true",
                    help="On overlapping timestamps let the source / resampled values win "
                         "(default keeps existing IBKR/Binance bars).")
    ap.add_argument("--agg-only", action="store_true",
                    help="Skip the 1-min fill; only rebuild 15/30 from the existing 1-min cache.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and report everything but write no files.")
    args = ap.parse_args()

    if not os.path.isdir(SB_DIR):
        sys.exit(f"❌ dataset not found: {SB_DIR}")
    if not os.path.isdir(CACHE_DIR):
        sys.exit(f"❌ cache dir not found: {CACHE_DIR}")

    print("=" * 72)
    print("  stock-data-intraday-minute-bar -> market_data_cache backfill"
          + ("   [DRY RUN]" if args.dry_run else ""))
    print(f"  policy: {'OVERWRITE (source wins)' if args.overwrite else 'MERGE (keep existing bars)'}"
          f"{' | agg-only' if args.agg_only else ''}")
    print("=" * 72)

    for name in args.instruments:
        process_instrument(name.upper(), args)

    print("\n" + "=" * 72)
    print("  done." + ("  (dry run — nothing written)" if args.dry_run else ""))
    print("=" * 72)


if __name__ == "__main__":
    main()
