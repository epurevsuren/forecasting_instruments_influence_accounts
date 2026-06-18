"""
fetch_binance_history.py
------------------------
Downloads historical bars from Binance public REST API and caches them
locally as CSV files alongside the IBKR cache. Re-runs read the cache and only
fetch what is missing. No API key required.

Instruments : BTC (BTCUSDT), ETH (ETHUSDT)
Bar sizes   : 1m, 3m, 5m, 15m, 30m  (each gets its own CSV)
History     : ~2017-08-01 onwards for all sizes

CSV naming  : {name}_{N}min.csv  e.g. BTC_1min.csv, ETH_15min.csv

USAGE:
  python fetch_binance_history.py                                   # 30m, forward fill to now
  python fetch_binance_history.py --bar-size 1m --from 20170801    # 1m backfill
  python fetch_binance_history.py --bar-size 5m --instruments BTC
  python fetch_binance_history.py --from 20170801 --until 20241030 --bar-size 15m
"""

import argparse
import os
import json
import time
import datetime
import requests
import pandas as pd

# ==========================================
# CONFIG
# ==========================================
CACHE_DIR = "market_data_cache"
MAX_BARS  = 1000     # Binance max per single klines request

# Bar size -> (Binance interval string, CSV suffix, throttle seconds)
BAR_CONFIG = {
    "1m":  ("1m",  "1min",  0.3),
    "3m":  ("3m",  "3min",  0.3),
    "5m":  ("5m",  "5min",  0.3),
    "15m": ("15m", "15min", 0.3),
    "30m": ("30m", "30min", 0.5),
}

INSTRUMENTS = [
    ("BTC", "BTCUSDT"),
    ("ETH", "ETHUSDT"),
]

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


# ==========================================
# Helpers
# ==========================================
def safe_to_csv(df, path):
    """Atomic write: temp file -> rename. Ctrl+C retries to avoid corrupt file."""
    tmp = path + ".tmp"
    try:
        df.to_csv(tmp, index=False)
    except KeyboardInterrupt:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        raise
    os.replace(tmp, path)


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    mpath = os.path.join(CACHE_DIR, "_manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(os.path.join(CACHE_DIR, "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def month_starts(since, until):
    cur = pd.Timestamp(since).replace(day=1)
    end = pd.Timestamp(until) - pd.Timedelta(seconds=1)
    out = []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def parse_stamp(s):
    s = s.strip()
    if len(s) == 12:
        try:
            return pd.Timestamp(datetime.datetime.strptime(s, "%Y%m%d%H%M"), tz="UTC")
        except ValueError:
            pass
    if len(s) == 8:
        try:
            return pd.Timestamp(datetime.datetime.strptime(s, "%Y%m%d"), tz="UTC")
        except ValueError:
            pass
    raise SystemExit(f"date must be YYYYMMDDhhmm or YYYYMMDD, got: {s!r}")


def _csv_latest_date(name, suffix):
    path = os.path.join(CACHE_DIR, f"{name}_{suffix}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=["date"])
        if df.empty:
            return None
        return pd.to_datetime(df["date"], utc=True).max()
    except Exception:
        return None


# ==========================================
# Binance API call (with pagination)
# ==========================================
def fetch_klines(symbol, interval, start_ms, end_ms, throttle):
    """
    Fetch all klines for `symbol` / `interval` in [start_ms, end_ms) from Binance.
    Paginates automatically -- Binance caps each response at 1000 bars.
    Returns list of dicts: {date, open, high, low, close, volume}.
    """
    all_bars  = []
    cur_start = start_ms

    while cur_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": cur_start,
            "endTime":   end_ms - 1,
            "limit":     MAX_BARS,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        for row in data:
            all_bars.append({
                "date":   pd.Timestamp(row[0], unit="ms", tz="UTC"),
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            })

        cur_start = data[-1][0] + 1   # +1 ms to avoid overlap
        time.sleep(throttle)

        if len(data) < MAX_BARS:
            break   # no more data in this range

    return all_bars


# ==========================================
# Per-instrument fetcher
# ==========================================
def fetch_one(name, symbol, interval, suffix, throttle, manifest, since, until):
    cache_file   = os.path.join(CACHE_DIR, f"{name}_{suffix}.csv")
    manifest_key = f"{name}_{suffix}"

    # Load existing cache
    if os.path.exists(cache_file):
        existing = pd.read_csv(cache_file, parse_dates=["date"])
        existing["date"] = pd.to_datetime(existing["date"], utc=True)
        cached_months = set(
            m for m in existing["date"].dt.strftime("%Y-%m").unique()
            if isinstance(m, str)
        )
        if cached_months:
            cache_earliest = min(cached_months)
            cache_latest   = max(cached_months)

            full_end  = cache_latest
            full_all  = set()
            cur = since.replace(day=1)
            while cur.strftime("%Y-%m") <= full_end:
                full_all.add(cur.strftime("%Y-%m"))
                cur = cur + pd.offsets.MonthBegin(1)
            full_missing = sorted(full_all - cached_months)

            win_end = (until - pd.Timedelta(seconds=1)).strftime("%Y-%m")
            missing = sorted(m for m in full_missing if m <= win_end)

            if full_missing:
                gap_msg = f" | gap {full_missing[0]} -> {full_missing[-1]} ({len(full_missing)} mo)"
                if missing and missing != full_missing:
                    gap_msg += f", fetching {len(missing)} mo this run"
            elif missing:
                gap_msg = f" | gap {missing[0]} -> {missing[-1]} ({len(missing)} mo)"
            else:
                gap_msg = " | window fully cached"

            now_month   = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
            is_live_run = (
                (pd.Timestamp.now(tz="UTC") - until) < pd.Timedelta(days=2) and
                (pd.Timestamp.now(tz="UTC") - since) < pd.Timedelta(days=35)
            )
            if is_live_run and cache_latest == now_month:
                cached_months.discard(cache_latest)
                refresh_msg = f", refreshing {cache_latest}"
            else:
                refresh_msg = ""

            print(f"  [i] {name} ({suffix}): cache {cache_earliest} -> {cache_latest} "
                  f"({len(existing):,} bars){gap_msg}{refresh_msg}")
    else:
        existing      = pd.DataFrame()
        cached_months = set()

    # Month-by-month fetch
    months        = month_starts(since, until)
    fetched       = 0
    skipped       = 0
    failed_months = []
    combined      = existing.copy() if len(existing) else pd.DataFrame()

    for m in months:
        mkey = m.strftime("%Y-%m")
        if mkey in cached_months:
            skipped += 1
            continue

        next_month = m + pd.offsets.MonthBegin(1)
        start_ms   = int(m.timestamp() * 1000)
        end_ms     = int(min(next_month, until + pd.Timedelta(days=1)).timestamp() * 1000)

        print(f"     -> {name} ({suffix}) {mkey} ...", end="", flush=True)
        got_data = False
        for attempt in range(1, 4):
            try:
                bars = fetch_klines(symbol, interval, start_ms, end_ms, throttle)
                if bars:
                    dfm = pd.DataFrame(bars)
                    dfm["date"] = pd.to_datetime(dfm["date"], utc=True)
                    dfm = dfm[dfm["date"] <= until]
                    print(f" {len(dfm)} bars")
                    fetched  += 1
                    got_data  = True

                    combined = pd.concat([combined, dfm], ignore_index=True)
                    combined["date"] = pd.to_datetime(combined["date"], utc=True)
                    combined = (combined.drop_duplicates(subset=["date"])
                                .sort_values("date").reset_index(drop=True))
                    try:
                        safe_to_csv(combined, cache_file)
                        cached_months.add(mkey)
                    except KeyboardInterrupt:
                        cached_months.add(mkey)
                        print(f"\n[!] Ctrl+C -- saved {mkey}, stopping.")
                        raise
                    break
                else:
                    print(f" empty")
                    print(f"     ~  {name} ({suffix}) {mkey}: empty (attempt {attempt}/3)")
                    time.sleep(1)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"     [!] {name} ({suffix}) {mkey} attempt {attempt}/3: {str(e)[:60]}")
                time.sleep(2)

        if not got_data:
            failed_months.append(mkey)

    if failed_months:
        print(f"     [x] {name} ({suffix}): {len(failed_months)} months failed/empty")

    if fetched:
        manifest[manifest_key] = {"status": "ok", "rows": len(combined),
                                   "first": str(combined["date"].min()),
                                   "last":  str(combined["date"].max())}
        print(f"  [ok] {name} ({suffix})  {len(combined):>7} bars  "
              f"(fetched {fetched} mo, cached {skipped} mo)")
    elif len(existing):
        print(f"  [--] {name} ({suffix})  {len(existing):>7} bars  [cached, no change]")
        manifest[manifest_key] = {"status": "ok", "rows": len(existing)}
    else:
        print(f"  [x]  {name} ({suffix})  no data returned")
        manifest[manifest_key] = {"status": "no_data"}

    save_manifest(manifest)


# ==========================================
# CLI + main
# ==========================================
def main():
    ap = argparse.ArgumentParser(
        description="Fetch Binance historical bars into market_data_cache/.",
    )
    ap.add_argument("--bar-size", dest="bar_size", default="30m",
                    choices=list(BAR_CONFIG.keys()),
                    help="Bar size to fetch (default: 30m). Each size stored in its own CSV.")
    ap.add_argument("--from", dest="since", metavar="YYYYMMDD[hhmm]", default=None,
                    help="Start of fetch window. Default: per-instrument latest cached date "
                         "(or 2017-08-01 if no cache).")
    ap.add_argument("--until", metavar="YYYYMMDD[hhmm]", default=None,
                    help="End of fetch window. Default: now.")
    ap.add_argument("--instruments", nargs="+", metavar="NAME", default=None,
                    help="Fetch only these (e.g. --instruments BTC ETH).")
    args = ap.parse_args()

    interval, suffix, throttle = BAR_CONFIG[args.bar_size]
    global_until = parse_stamp(args.until) if args.until else pd.Timestamp.now(tz="UTC")
    global_since = parse_stamp(args.since) if args.since else None

    print("=" * 60)
    print("  BINANCE HISTORICAL DATA FETCHER")
    print("=" * 60)
    print(f"  Bar size : {args.bar_size}  ->  {suffix}.csv per instrument")
    print(f"  Until    : {global_until.strftime('%Y-%m-%d %H:%M UTC')}")
    if global_since:
        print(f"  Since    : {global_since.strftime('%Y-%m-%d')}  (--from, backfill mode)")
    else:
        print(f"  Since    : per-instrument latest cached date")
    print(f"  Source   : Binance public REST API (no API key required)")
    print("=" * 60)

    manifest = ensure_cache_dir()

    if args.instruments:
        names   = {n.upper() for n in args.instruments}
        todo    = [(n, s) for n, s in INSTRUMENTS if n in names]
        unknown = names - {n for n, _ in todo}
        if unknown:
            print(f"[!] Unknown instruments ignored: {', '.join(sorted(unknown))}")
    else:
        todo = INSTRUMENTS

    mode = ", ".join(n for n, _ in todo)
    print(f"\nFetching {mode}\n")

    t0 = time.time()
    for name, symbol in todo:
        if global_since is not None:
            since = global_since
        else:
            latest = _csv_latest_date(name, suffix)
            since  = (latest - pd.Timedelta(hours=1)) if latest is not None \
                     else pd.Timestamp("2017-08-01", tz="UTC")

        if since >= global_until:
            print(f"  [--] {name} ({suffix})  already up to date ({since.date()})")
            continue

        try:
            fetch_one(name, symbol, interval, suffix, throttle, manifest, since, global_until)
        except KeyboardInterrupt:
            print("\n[STOP] Stopped by user.")
            break

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min")
    print(f"Cache: {CACHE_DIR}/  -- reused automatically on next run")


if __name__ == "__main__":
    main()
