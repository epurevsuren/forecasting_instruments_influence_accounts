"""
fetch_binance_history.py
------------------------
Download historical bars from the Binance public REST API and cache them as CSVs
alongside the IBKR cache. Re-runs resume from the last cached bar. No API key.

Engine: DuckDB ONLY (no pandas). DuckDB reads the existing CSV, the freshly
fetched bars go into a DuckDB temp table, and the two are merged + DEDUPED by
date (freshest bar wins) and written back in the exact cache format. So fetching
can never write a duplicate timestamp.

Instruments : BTC (BTCUSDT), ETH (ETHUSDT) — from ../DP/instruments.json 'binance'.
Bar sizes   : DEFAULT 1m, 15m, 30m (3m / 5m available via --bar-size).
CSV schema  : date,open,high,low,close,volume,wap,barCount  (tz-aware UTC date)

DEFAULT BEHAVIOUR (no arguments):
  Fetches 1m, 15m AND 30m. For each instrument and size, RESUMES from the last
  cached bar's datetime and fetches forward to now (fills the tail). A brand-new
  instrument starts at 2017-08-01.

BACKFILL / REPAIR:
  --from YYYYMMDD           fetch that window (dedup keeps existing bars; fills holes)
  --refetch --from YYYYMMDD CLEAR existing bars in [--from,--until] then re-fetch
                            them fresh (repairs distorted/corrupt data)

USAGE:
  python fetch_binance_history.py                                  # 1m 15m 30m, resume to now
  python fetch_binance_history.py --bar-size 1m 30m
  python fetch_binance_history.py --from 20170801                  # backfill from history start
  python fetch_binance_history.py --refetch --from 20170801        # full clean re-download
  python fetch_binance_history.py --instruments BTC --bar-size 1m
"""
import argparse
import os
import json
import time
import datetime as dt

import requests
import duckdb

UTC = dt.timezone.utc

# ==========================================
# CONFIG
# ==========================================
CACHE_DIR   = "market_data_cache"
MAX_BARS    = 1000                                   # Binance cap per klines request
FLUSH_EVERY = 200_000                                # save-to-disk cadence (resumable)
DEFAULT_START = dt.datetime(2017, 8, 1, tzinfo=UTC)  # Binance history start for new caches
COLS = ["date", "open", "high", "low", "close", "volume", "wap", "barCount"]

# Bar size -> (Binance interval, CSV suffix, throttle seconds)
BAR_CONFIG = {
    "1m":  ("1m",  "1min",  0.3),
    "3m":  ("3m",  "3min",  0.3),
    "5m":  ("5m",  "5min",  0.3),
    "15m": ("15m", "15min", 0.3),
    "30m": ("30m", "30min", 0.5),
}
# Sizes fetched when --bar-size is not given.
DEFAULT_BAR_SIZES = ["1m", "15m", "30m"]

# Instruments loaded from ../DP/instruments.json (every entry with a 'binance' symbol).
_INSTRUMENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DP", "instruments.json")
with open(_INSTRUMENTS_FILE, encoding="utf-8") as _f:
    INSTRUMENTS = [(name, v["binance"])
                   for name, v in json.load(_f)["instruments"].items()
                   if v.get("binance")]

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


# ==========================================
# DuckDB engine (read / dedup-write cache CSVs — no pandas)
# ==========================================
def connect():
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    return con


def _ts_lit(d) -> str:
    """UTC datetime -> a DuckDB TIMESTAMPTZ literal."""
    return "TIMESTAMPTZ '" + d.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S") + "+00'"


def _csv(path) -> str:
    return f"read_csv_auto('{path}', null_padding=true)"


def coverage(con, cache_file):
    """(rows, min_date, max_date) for a cache CSV, or (0, None, None)."""
    if not os.path.exists(cache_file):
        return 0, None, None
    return con.execute(
        f"SELECT count(*), min(date::TIMESTAMPTZ), max(date::TIMESTAMPTZ) FROM {_csv(cache_file)}"
    ).fetchone()


def latest_date(con, cache_file):
    """Max date in a cache CSV (tz-aware datetime) or None — reads only max(date)."""
    if not os.path.exists(cache_file):
        return None
    return con.execute(f"SELECT max(date::TIMESTAMPTZ) FROM {_csv(cache_file)}").fetchone()[0]


def _select_cols(from_expr, priority):
    """SELECT the 8 canonical cols (date cast to TIMESTAMPTZ) + a priority tag."""
    return (f'SELECT date::TIMESTAMPTZ AS date, open, high, low, close, volume, wap, '
            f'"barCount", {priority} AS _p FROM {from_expr}')


def merge_write(con, cache_file):
    """Merge the `_new` temp table into `cache_file` (read via DuckDB), DEDUPED by
    date (the freshly-fetched row wins on overlap), written atomically in the
    exact cache format. Guarantees no duplicate timestamps."""
    tmp = cache_file + ".tmp"
    rest = 'open, high, low, close, volume, wap, "barCount"'
    if os.path.exists(cache_file):
        union = _select_cols(_csv(cache_file), 0) + " UNION ALL " + _select_cols("_new", 1)
    else:
        union = _select_cols("_new", 1)
    con.execute(
        f"COPY (WITH u AS ({union}), "
        f"d AS (SELECT *, row_number() OVER (PARTITION BY date ORDER BY _p DESC) AS rn FROM u) "
        f"SELECT strftime(date, '%Y-%m-%d %H:%M:%S') || '+00:00' AS date, {rest} "
        f"FROM d WHERE rn = 1 ORDER BY date) "
        f"TO '{tmp}' (HEADER, DELIMITER ',')"
    )
    os.replace(tmp, cache_file)


def clear_window(con, cache_file, lo, hi):
    """Rewrite `cache_file` WITHOUT the bars in [lo, hi] (for --refetch). Returns
    the number of bars dropped."""
    if not os.path.exists(cache_file):
        return 0
    tmp = cache_file + ".tmp"
    n = con.execute(
        f"SELECT count(*) FROM {_csv(cache_file)} "
        f"WHERE date::TIMESTAMPTZ >= {_ts_lit(lo)} AND date::TIMESTAMPTZ <= {_ts_lit(hi)}"
    ).fetchone()[0]
    con.execute(
        f"COPY (SELECT strftime(date::TIMESTAMPTZ, '%Y-%m-%d %H:%M:%S') || '+00:00' AS date, "
        f'open, high, low, close, volume, wap, "barCount" FROM {_csv(cache_file)} '
        f"WHERE NOT (date::TIMESTAMPTZ >= {_ts_lit(lo)} AND date::TIMESTAMPTZ <= {_ts_lit(hi)}) "
        f"ORDER BY date::TIMESTAMPTZ) TO '{tmp}' (HEADER, DELIMITER ',')"
    )
    os.replace(tmp, cache_file)
    return n


# ==========================================
# misc helpers
# ==========================================
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


def parse_stamp(s):
    s = s.strip()
    for fmt, ln in (("%Y%m%d%H%M", 12), ("%Y%m%d", 8)):
        if len(s) == ln:
            try:
                return dt.datetime.strptime(s, fmt).replace(tzinfo=UTC)
            except ValueError:
                pass
    raise SystemExit(f"date must be YYYYMMDDhhmm or YYYYMMDD, got: {s!r}")


# ==========================================
# Per-(instrument, size) fetcher
# ==========================================
def fetch_one(con, name, symbol, interval, suffix, throttle,
              since, until, refetch, default_mode, manifest):
    cache_file   = os.path.join(CACHE_DIR, f"{name}_{suffix}.csv")
    manifest_key = f"{name}_{suffix}"

    n0, cmin, cmax = coverage(con, cache_file)
    if n0:
        resume = ", resuming from last cached bar" if default_mode else ""
        print(f"  [i] {name} ({suffix}): cache {cmin} -> {cmax} ({n0:,} bars){resume}")

    if refetch and n0:
        dropped = clear_window(con, cache_file, since, until)
        print(f"     [refetch] cleared {dropped:,} {name} {suffix} bars in "
              f"[{since.date()} .. {until.date()}] — re-fetching fresh")

    # DuckDB temp table for freshly-fetched bars (no pandas)
    con.execute('CREATE OR REPLACE TEMP TABLE _new '
                '(date TIMESTAMPTZ, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, '
                'volume DOUBLE, wap DOUBLE, "barCount" BIGINT)')

    start_ms = int(since.timestamp() * 1000)
    end_ms   = int(until.timestamp() * 1000)
    cur      = start_ms
    total_new = 0
    pending   = 0
    print(f"     -> {name} ({suffix}) {since.date()} -> {until.date()} ...", flush=True)

    try:
        while cur < end_ms:
            params = {"symbol": symbol, "interval": interval,
                      "startTime": cur, "endTime": end_ms - 1, "limit": MAX_BARS}
            data = None
            for attempt in range(1, 4):
                try:
                    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"     [!] {name} ({suffix}) attempt {attempt}/3: {str(e)[:60]}")
                    time.sleep(2)
            if not data:
                break

            rows = []
            for r in data:
                if r[0] > end_ms - 1:
                    continue
                vol = float(r[5]); qv = float(r[7])
                rows.append((
                    dt.datetime.fromtimestamp(r[0] / 1000, UTC),
                    float(r[1]), float(r[2]), float(r[3]), float(r[4]), vol,
                    round(qv / vol, 8) if vol else float(r[4]), int(r[8]),
                ))
            if rows:
                con.executemany('INSERT INTO _new VALUES (?,?,?,?,?,?,?,?)', rows)
                total_new += len(rows)
                pending   += len(rows)

            cur = data[-1][0] + 1          # +1 ms, no overlap
            time.sleep(throttle)

            if pending >= FLUSH_EVERY:
                merge_write(con, cache_file)
                con.execute("DELETE FROM _new")
                print(f"        .. saved (+{total_new:,} new so far)")
                pending = 0
            if len(data) < MAX_BARS:
                break                       # reached the end of available data
    except KeyboardInterrupt:
        if pending:
            merge_write(con, cache_file)
            con.execute("DELETE FROM _new")
        print(f"\n[STOP] Ctrl+C — saved {total_new:,} new bars, stopping.")
        raise

    if pending:
        merge_write(con, cache_file)
        con.execute("DELETE FROM _new")

    n1, _, _ = coverage(con, cache_file)
    if total_new:
        print(f"  [ok] {name} ({suffix})  {n1:,} bars  (+{total_new:,} fetched, deduped)")
        manifest[manifest_key] = {"status": "ok", "rows": n1}
    elif n0:
        print(f"  [--] {name} ({suffix})  {n0:,} bars  [up to date, +0 new]")
        manifest[manifest_key] = {"status": "ok", "rows": n0}
    else:
        print(f"  [x]  {name} ({suffix})  no data returned")
        manifest[manifest_key] = {"status": "no_data"}
    save_manifest(manifest)


# ==========================================
# CLI + main
# ==========================================
def main():
    ap = argparse.ArgumentParser(
        description="Fetch Binance historical bars into market_data_cache/ (DuckDB, dedup).")
    ap.add_argument("--bar-size", dest="bar_sizes", nargs="+", default=None,
                    choices=list(BAR_CONFIG.keys()), metavar="SIZE",
                    help="Bar size(s), space-separated. Default: 1m 15m 30m (each its own CSV).")
    ap.add_argument("--from", dest="since", metavar="YYYYMMDD[hhmm]", default=None,
                    help="Start of fetch window. Default: per-instrument last cached bar "
                         "(or 2017-08-01 if no cache).")
    ap.add_argument("--until", metavar="YYYYMMDD[hhmm]", default=None,
                    help="End of fetch window. Default: now.")
    ap.add_argument("--instruments", nargs="+", metavar="COIN", default=None,
                    help="Fetch only these (default BTC ETH). Any other coin pairs with USDT "
                         "(SOL -> SOLUSDT).")
    ap.add_argument("--refetch", action="store_true",
                    help="Clear existing bars in [--from,--until] then re-fetch them fresh "
                         "(repairs distorted data). Pair with --from.")
    args = ap.parse_args()

    bar_sizes    = args.bar_sizes if args.bar_sizes else DEFAULT_BAR_SIZES
    sizes_str    = " ".join(bar_sizes)
    global_until = parse_stamp(args.until) if args.until else dt.datetime.now(UTC)
    global_since = parse_stamp(args.since) if args.since else None
    default_mode = global_since is None

    print("=" * 60)
    print(f"  BINANCE HISTORICAL DATA FETCHER -- {sizes_str} bars")
    print("=" * 60)
    print(f"  Bar sizes: {sizes_str}  ->  one CSV each per instrument")
    print(f"  Until    : {global_until.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Since    : {global_since.strftime('%Y-%m-%d') + '  (--from backfill)' if global_since else 'per-instrument last cached bar'}")
    print(f"  Engine   : DuckDB (dedup write) | Source: Binance public REST (no key)")
    print("=" * 60)

    manifest = ensure_cache_dir()
    con = connect()

    if args.instruments:
        registry = dict(INSTRUMENTS)
        todo = []
        for token in args.instruments:
            nm = token.upper()
            if nm in registry:
                todo.append((nm, registry[nm]))
            else:
                todo.append((nm, f"{nm}USDT"))
                print(f"[+] {nm} -> {nm}USDT")
    else:
        todo = INSTRUMENTS
    print(f"\nFetching {', '.join(n for n, _ in todo)}  ({sizes_str})\n")

    t0 = time.time()
    stopped = False
    for bar_size in bar_sizes:
        interval, suffix, throttle = BAR_CONFIG[bar_size]
        print("-" * 60)
        print(f"  > {bar_size} bars  ->  {suffix}.csv   (~{throttle}s between requests)")
        print("-" * 60)
        for name, symbol in todo:
            if global_since is not None:
                since = global_since
            else:
                latest = latest_date(con, os.path.join(CACHE_DIR, f"{name}_{suffix}.csv"))
                since  = (latest - dt.timedelta(hours=1)) if latest is not None else DEFAULT_START
            if since >= global_until:
                print(f"  [--] {name} ({suffix})  already up to date ({since.date()})")
                continue
            try:
                fetch_one(con, name, symbol, interval, suffix, throttle,
                          since, global_until, args.refetch, default_mode, manifest)
            except KeyboardInterrupt:
                print("\n[STOP] Stopped by user.")
                stopped = True
                break
        if stopped:
            break

    con.close()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print(f"Cache: {CACHE_DIR}/  -- reused automatically on next run")


if __name__ == "__main__":
    main()
