"""
fetch_ibkr_history.py
---------------------
Downloads historical bars from Interactive Brokers and caches them
locally as CSV files. Re-runs read the cache and only fetch what is missing.

Bar sizes : 1m, 5m, 15m, 30m  (each gets its own CSV)

DEFAULT BEHAVIOUR (no arguments):
  For each instrument, reads the latest date already in its 30-min CSV and fetches
  forward to now. New instruments with no CSV start from now - 2 years.

WITH ARGUMENTS:
  python fetch_ibkr_history.py --bar-size 15m --from 20241101 --until 20260101
  Fills only what is missing inside that window (already-cached months
  are skipped automatically).

CSV naming  : {name}_{N}min.csv  e.g. SPY_30min.csv, SPY_1min.csv

Edit the CONFIG block below to match your TWS/Gateway setup.
"""

import argparse
import os
import json
import time
import datetime
import pandas as pd
from ib_async import IB, Stock, Future, Index, Forex, Crypto, ContFuture

# Contract delivery schedules for expired-contract backfill.
# Maps base symbol -> set of valid delivery months (1-12).
# Dated Future contracts (includeExpired=True) allow explicit endDateTime,
# unlike ContFuture which is locked to endDateTime="" only (IBKR error 10339).
_DELIVERY_MONTHS = {
    "CL": set(range(1, 13)),        # crude oil: every month
    "NG": set(range(1, 13)),        # nat gas:   every month
    "GC": {2, 4, 6, 8, 10, 12},    # gold:      even months
    "SI": {3, 5, 7, 9, 12},        # silver:    Mar May Jul Sep Dec
    "HG": {3, 5, 7, 9, 12},        # copper:    Mar May Jul Sep Dec
    "ZN": {3, 6, 9, 12},           # 10Y note:  quarterly
    "ZT": {3, 6, 9, 12},           # 2Y note:   quarterly
}


def _expired_months(base_sym, start_ts, end_ts):
    """Return list of 'YYYYMM' strings for valid delivery months in [start, end]."""
    valid = _DELIVERY_MONTHS.get(base_sym)
    if not valid:
        return []
    out = []
    cur = start_ts.replace(day=1)
    while cur <= end_ts:
        if cur.month in valid:
            out.append(cur.strftime("%Y%m"))
        cur = cur + pd.offsets.MonthBegin(1)
    return out


# ==========================================
# CONFIG -- edit these
# ==========================================
PORT      = 7497   # TWS desktop PAPER  (7496=TWS live / 4002/4001=Gateway paper/live)
HOST      = "127.0.0.1"
CLIENT_ID = 11

CACHE_DIR = "market_data_cache"

# Bar size key -> (IBKR barSizeSetting, CSV suffix, chunk per request,
#                  throttle_sec, contfut_durations to try in order)
#
# Chunk notes:
#   30m / 15m / 5m : "1 M" -- one month fits easily in a single IBKR request
#   1m              : "1 W" -- one week per request; code sub-iterates weekly
#                              within each calendar month to stay under bar limits
#
# ContFuture durations:
#   IBKR caps bars-per-request; smaller bar sizes need shorter durations.
#   Durations tried in order; first one that returns data is used.
BAR_CONFIG = {
    # key: (ibkr_str,    suffix,   chunk,  throttle, contfut_durations)
    "1m":  ("1 min",   "1min",   "1 W",  11, ["30 D", "21 D", "14 D", "7 D"]),
    "5m":  ("5 mins",  "5min",   "1 M",  11, ["90 D", "60 D", "30 D"]),
    "15m": ("15 mins", "15min",  "1 M",  11, ["90 D", "120 D", "180 D", "270 D", "365 D"]),
    "30m": ("30 mins", "30min",  "1 M",  11, ["90 D", "120 D", "180 D", "270 D", "365 D"]),
}

# Set True to fetch ONLY the 5 core instruments (~20 min).
CORE_ONLY = False

# ==========================================
# INSTRUMENT -> IBKR CONTRACT MAPPING
# ==========================================
def C_stock(sym):              return Stock(sym, "SMART", "USD")
def C_index(sym, exch="CBOE"): return Index(sym, exch, "USD")
def C_contfut(sym, exch):      return ContFuture(sym, exch, currency="USD")
def C_fx(pair):                return Forex(pair)

INSTRUMENTS = [
    # ---- CORE 5 ----
    ("SPY",   C_stock("SPY"),               "TRADES"),
    ("VIX",   C_index("VIX", "CBOE"),       "TRADES"),
    ("OIL",   C_contfut("CL", "NYMEX"),     "TRADES"),
    ("GOLD",  C_contfut("GC", "COMEX"),     "TRADES"),
    ("BTC",   Crypto("BTC", "PAXOS", "USD"),"AGGTRADES"),

    # ---- US equity sectors ----
    ("QQQ",  C_stock("QQQ"), "TRADES"),
    ("DIA",  C_stock("DIA"), "TRADES"),
    ("XLI",  C_stock("XLI"), "TRADES"),
    ("XLF",  C_stock("XLF"), "TRADES"),
    ("XLE",  C_stock("XLE"), "TRADES"),

    # ---- commodities ----
    ("COPPER", C_contfut("HG", "COMEX"), "TRADES"),
    ("NATGAS", C_contfut("NG", "NYMEX"), "TRADES"),

    # ---- FX ----
    ("EUR_USD", C_fx("EURUSD"), "MIDPOINT"),
    ("USD_JPY", C_fx("USDJPY"), "MIDPOINT"),
    ("GBP_USD", C_fx("GBPUSD"), "MIDPOINT"),
    ("USD_CNY", C_fx("USDCNH"), "MIDPOINT"),
    ("USD_CAD", C_fx("USDCAD"), "MIDPOINT"),
    ("USD_MXN", C_fx("USDMXN"), "MIDPOINT"),
    ("USD_CHF", C_fx("USDCHF"), "MIDPOINT"),
    ("AUD_USD", C_fx("AUDUSD"), "MIDPOINT"),

    # ---- bonds ----
    ("US10Y", C_contfut("ZN", "CBOT"), "TRADES"),
    ("US2Y",  C_contfut("ZT", "CBOT"), "TRADES"),

    # ---- crypto ----
    ("ETH", Crypto("ETH", "PAXOS", "USD"), "AGGTRADES"),
]

CORE_NAMES = {"SPY", "VIX", "OIL", "GOLD", "BTC"}


# ==========================================
# Helpers
# ==========================================
def safe_to_csv(df, path):
    """Atomic write: temp file -> rename. If Ctrl+C hits mid-write, retries to
    ensure the file is complete before re-raising the interrupt."""
    tmp = path + ".tmp"
    try:
        df.to_csv(tmp, index=False)
    except KeyboardInterrupt:
        df.to_csv(tmp, index=False)   # complete the interrupted write
        os.replace(tmp, path)
        raise                          # re-raise so caller can clean up
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
    # Subtract 1s so until=2016-01-01 00:00 gives last month=2015-12.
    end = pd.Timestamp(until) - pd.Timedelta(seconds=1)
    out = []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def _csv_latest_date(name, suffix):
    """Return the latest date in the cache CSV for (name, suffix), or None."""
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
# Per-instrument fetcher
# ==========================================
def fetch_one(ib, name, contract, what_to_show, manifest,
              since, until, suffix, bar_size_str, chunk, throttle, contfut_durations):
    """
    Fetch bars for one instrument over [since, until], skipping cached months.
    `since` and `until` are UTC Timestamps.
    """
    cache_file   = os.path.join(CACHE_DIR, f"{name}_{suffix}.csv")
    manifest_key = f"{name}_{suffix}"

    # ── Load existing cache ───────────────────────────────────────────────────
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

            # Full gap: all missing months from since to the cache's own latest
            # (not limited by --until), so both runs show the same true gap.
            full_end  = cache_latest
            full_all  = set()
            cur = since.replace(day=1)
            while cur.strftime("%Y-%m") <= full_end:
                full_all.add(cur.strftime("%Y-%m"))
                cur = cur + pd.offsets.MonthBegin(1)
            full_missing = sorted(full_all - cached_months)

            # Window: what this run will actually fetch (limited by --until)
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

            # Refresh the current live month only in default mode (no --from),
            # detected by since being recent (<= 35 days ago).
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

            print(f"  ℹ️  {name} ({suffix}): cache {cache_earliest} → {cache_latest} "
                  f"({len(existing):,} bars){gap_msg}{refresh_msg}")
    else:
        existing      = pd.DataFrame()
        cached_months = set()

    # ── Qualify contract ──────────────────────────────────────────────────────
    try:
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            print(f"  ❌ {name:<12} could not qualify contract -- skipping")
            manifest[manifest_key] = {"status": "failed_qualify"}
            return
    except Exception as e:
        print(f"  ❌ {name:<12} qualify error: {str(e)[:60]} -- skipping")
        manifest[manifest_key] = {"status": "qualify_error"}
        return

    # ── CONTINUOUS FUTURES path ───────────────────────────────────────────────
    is_contfut = type(contract).__name__ == "ContFuture"
    if is_contfut:
        c = qualified[0] if isinstance(qualified, list) else contract

        # Live combined frame: every successful fetch is merged and written to
        # disk IMMEDIATELY (same incremental, resumable pattern as the
        # stocks/FX path) rather than accumulating all frames and writing once
        # at the end -- so a timeout / Ctrl+C never discards already-fetched bars.
        combined = existing.copy() if len(existing) else pd.DataFrame()
        got_new  = False

        def _flush(bar_list, before=None):
            """Merge fetched bars into `combined` and write to disk IMMEDIATELY
            (incremental + resumable). The caller does the printing.

            `before` (a UTC Timestamp) clips the new bars to dates strictly
            earlier than it -- used by the expired-contract backfill so the
            continuous-future bars always own their own region (no overlap)."""
            nonlocal combined, got_new
            df = pd.DataFrame([{
                "date": b.date, "open": b.open, "high": b.high,
                "low": b.low, "close": b.close, "volume": b.volume,
                "average": b.average, "barCount": b.barCount,
            } for b in bar_list])
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df[df["date"] <= until]
            if before is not None:
                df = df[df["date"] < before]
            if df.empty:
                return
            base = [combined] if len(combined) else []
            combined = pd.concat(base + [df], ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"], utc=True)
            combined = (combined.drop_duplicates(subset=["date"], keep="last")
                        .sort_values("date").reset_index(drop=True))
            try:
                safe_to_csv(combined, cache_file)   # write-on-each-fetch
            except KeyboardInterrupt:
                print(f"\n⚠️  Ctrl+C -- saved {len(combined):,} bars, stopping.")
                raise
            got_new = True

        # Path A: fetch recent data via ContFuture (endDateTime="" = now).
        # Skip if cache is already fresh (within 10 days of now).
        cache_is_fresh = (
            len(existing) > 0 and
            existing["date"].max() >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
        )
        if cache_is_fresh:
            print(f"     💾 {name}: cached up to {existing['date'].max().date()}, skipping recent fetch")
        else:
            for dur in contfut_durations:
                print(f"     → {name} ({suffix}) ContFuture {dur} ...", end="", flush=True)
                try:
                    bars = ib.reqHistoricalData(
                        c, endDateTime="", durationStr=dur,
                        barSizeSetting=bar_size_str, whatToShow=what_to_show,
                        useRTH=0, formatDate=2, timeout=180,
                    )
                    if bars:
                        earliest = pd.to_datetime(bars[0].date, utc=True)
                        _flush(bars)
                        print(f" {len(bars)} bars (earliest {earliest.date()})")
                        time.sleep(throttle)
                        break
                    print(" empty")
                    time.sleep(throttle)
                except Exception as e:
                    print(f"\n     ⚠️  {name} ContFuture {dur}: {str(e)[:55]}")
                    time.sleep(throttle)

        # Path B: backfill via dated expired contracts.
        # ContFuture forbids explicit endDateTime (IBKR error 10339), but plain
        # Future contracts with includeExpired=True DO allow it.
        if len(combined):
            cf_earliest = pd.to_datetime(combined["date"], utc=True).min()
        else:
            cf_earliest = until

        gap_end   = cf_earliest - pd.Timedelta(days=1)
        gap_start = since

        base_sym = contract.symbol
        exch     = contract.exchange
        expiries = _expired_months(base_sym, gap_start, gap_end)

        if expiries:
            print(f"     🔙 {name}: backfilling {len(expiries)} expired contracts "
                  f"({expiries[0]} -> {expiries[-1]})")

        # Smaller bar sizes need shorter backfill windows to stay under IBKR bar limits.
        backfill_dur = "30 D" if suffix in ("1min", "5min") else "90 D"

        for ym in expiries:
            ym_month = f"{ym[:4]}-{ym[4:6]}"
            if ym_month in cached_months:
                continue
            fut = Future(symbol=base_sym, exchange=exch, currency="USD",
                         lastTradeDateOrContractMonth=ym)
            fut.includeExpired = True
            try:
                q = ib.qualifyContracts(fut)
            except Exception:
                q = []
            if not q or q[0] is None:
                continue  # not found -- no sleep, just skip

            print(f"     → {name} ({suffix}) expired {ym} ...", end="", flush=True)
            try:
                end_dt = (pd.Timestamp(ym + "01") + pd.offsets.MonthEnd(0)
                          ).strftime("%Y%m%d %H:%M:%S")
                bars = ib.reqHistoricalData(
                    q[0], endDateTime=end_dt, durationStr=backfill_dur,
                    barSizeSetting=bar_size_str, whatToShow=what_to_show,
                    useRTH=0, formatDate=2, timeout=90,
                )
                if bars:
                    # clip to < cf_earliest so the continuous-future bars keep
                    # ownership of their region (preserves the old "ContFuture
                    # wins the overlap" behavior); written to disk immediately.
                    _flush(bars, before=cf_earliest)
                    print(f" {len(bars)} bars")
                else:
                    print(" empty")
                time.sleep(throttle)
            except Exception as e:
                print(f"\n     ⚠️  {name} expired {ym}: {str(e)[:55]}")
                time.sleep(throttle)

        # Data is already on disk (written incrementally by _flush). This block
        # only records the final manifest entry and prints the summary line.
        if got_new and len(combined):
            span = f"{combined['date'].min().date()} -> {combined['date'].max().date()}"
            manifest[manifest_key] = {"status": "ok", "rows": len(combined),
                                      "first": str(combined["date"].min()),
                                      "last":  str(combined["date"].max())}
            print(f"  ✅ {name:<12} {len(combined):>6} bars  ({span})  [fetched]")
        elif len(combined):
            span = f"{combined['date'].min().date()} -> {combined['date'].max().date()}"
            print(f"  💾 {name:<12} {len(combined):>6} bars  ({span})  [cached, no change]")
            manifest[manifest_key] = {"status": "ok", "rows": len(combined)}
        else:
            print(f"  ❌ {name:<12} no data")
            manifest[manifest_key] = {"status": "no_data"}

        save_manifest(manifest)
        return
    # ── end ContFuture path ───────────────────────────────────────────────────

    # ── Stocks / FX / Crypto / Index path ────────────────────────────────────
    months = month_starts(since, until)
    fetched, skipped, failed_months = 0, 0, []
    consecutive_empty = 0

    # Work from a live copy of existing so we can save incrementally.
    combined = existing.copy() if len(existing) else pd.DataFrame()

    # 1m bars use weekly sub-chunking: "1 W" chunk is too small for a full month
    # in one request, so we slice the month into weekly windows and combine.
    use_weekly_sub = chunk.endswith(" W")

    for m in months:
        mkey = m.strftime("%Y-%m")
        if mkey in cached_months:
            skipped += 1
            continue

        if consecutive_empty >= 2 and fetched == 0:
            print(f"     ⏭️  {name}: no data on first 2 months -- likely unsubscribed, skipping rest")
            failed_months.append("(bailed early)")
            break

        next_month = m + pd.offsets.MonthBegin(1)
        limit_ts   = min(next_month, until + pd.Timedelta(days=1))

        # Build list of (start, end) sub-windows within this month.
        if use_weekly_sub:
            sub_periods = []
            cur_w = m
            while cur_w < limit_ts:
                end_w = min(cur_w + pd.Timedelta(weeks=1), limit_ts)
                sub_periods.append(end_w)
                cur_w = end_w
        else:
            sub_periods = [limit_ts]

        print(f"     → {name} ({suffix}) {mkey} ...", end="", flush=True)
        month_bars = []

        for end_ts in sub_periods:
            end_dt = end_ts.strftime("%Y%m%d %H:%M:%S")
            for attempt in range(1, 4):
                try:
                    bars = ib.reqHistoricalData(
                        qualified[0] if isinstance(qualified, list) else contract,
                        endDateTime=end_dt, durationStr=chunk,
                        barSizeSetting=bar_size_str, whatToShow=what_to_show,
                        useRTH=0, formatDate=2, timeout=60,
                    )
                    if bars:
                        month_bars.extend(bars)
                        consecutive_empty = 0
                    time.sleep(throttle)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    if attempt < 3:
                        print(f"\n     ⚠️  {name} ({suffix}) {mkey} attempt {attempt}/3: {str(e)[:50]}")
                    time.sleep(throttle)

        if month_bars:
            dfm = pd.DataFrame([{
                "date": b.date, "open": b.open, "high": b.high,
                "low": b.low, "close": b.close, "volume": b.volume,
                "average": b.average, "barCount": b.barCount,
            } for b in month_bars])
            print(f" {len(dfm)} bars")
            fetched += 1
            dfm["date"] = pd.to_datetime(dfm["date"], utc=True)
            dfm = dfm[dfm["date"] <= until]
            combined = pd.concat([combined, dfm], ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"], utc=True)
            combined = (combined.drop_duplicates(subset=["date"])
                        .sort_values("date").reset_index(drop=True))
            try:
                safe_to_csv(combined, cache_file)
                cached_months.add(mkey)
            except KeyboardInterrupt:
                cached_months.add(mkey)
                print(f"\n⚠️  Ctrl+C -- saved {mkey}, stopping.")
                raise
        else:
            print(f" empty")
            failed_months.append(mkey)
            consecutive_empty += 1

    if failed_months:
        print(f"     ❌ {name}: {len(failed_months)} months failed/empty")

    if fetched:
        manifest[manifest_key] = {"status": "ok", "rows": len(combined),
                                  "first": str(combined["date"].min()),
                                  "last":  str(combined["date"].max())}
        print(f"  ✅ {name:<12} {len(combined):>6} bars  "
              f"(fetched {fetched} mo, cached {skipped} mo)")
    elif len(existing):
        print(f"  💾 {name:<12} {len(existing):>6} bars  [cached, no change]")
        manifest[manifest_key] = {"status": "ok", "rows": len(existing)}
    else:
        print(f"  ❌ {name:<12} no data returned")
        manifest[manifest_key] = {"status": "no_data"}

    save_manifest(manifest)


# ==========================================
# CLI + main
# ==========================================
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


def main():
    ap = argparse.ArgumentParser(
        description="Fetch IBKR historical bars into market_data_cache/.",
    )
    ap.add_argument("--bar-size", dest="bar_size", default="30m",
                    choices=list(BAR_CONFIG.keys()),
                    help="Bar size to fetch (default: 30m). Each size stored in its own CSV.")
    ap.add_argument("--from", dest="since", metavar="YYYYMMDD[hhmm]", default=None,
                    help="Start of fetch window. Default: per-instrument latest cached date.")
    ap.add_argument("--until", metavar="YYYYMMDD[hhmm]", default=None,
                    help="End of fetch window. Default: now.")
    ap.add_argument("--instruments", nargs="+", metavar="NAME", default=None,
                    help="Fetch only these instruments (e.g. --instruments OIL GOLD SPY).")
    ap.add_argument("--core-only", action="store_true",
                    help="Fetch only SPY VIX OIL GOLD BTC.")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--client-id", type=int, default=CLIENT_ID)
    args = ap.parse_args()

    bar_size_str, suffix, chunk, throttle, contfut_durations = BAR_CONFIG[args.bar_size]

    global_until = parse_stamp(args.until) if args.until else pd.Timestamp.now(tz="UTC")
    global_since = parse_stamp(args.since) if args.since else None

    print("=" * 60)
    print(f"  IBKR HISTORICAL DATA FETCHER -- {args.bar_size} bars")
    print("=" * 60)
    print(f"  Bar size : {args.bar_size}  ->  {suffix}.csv per instrument")
    print(f"  Until    : {global_until.strftime('%Y-%m-%d %H:%M UTC')}")
    if global_since:
        print(f"  Since    : {global_since.strftime('%Y-%m-%d')}  (--from, backfill mode)")
    else:
        print(f"  Since    : per-instrument latest cached date")
    print(f"  Port     : {args.port}   Host: {args.host}   ClientID: {args.client_id}")
    print("=" * 60)

    manifest = ensure_cache_dir()

    print(f"\n🔌 Connecting to IBKR at {args.host}:{args.port} (client {args.client_id})...")
    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("    Is TWS/Gateway running with API enabled on this port?")
        return
    print("✅ Connected\n")

    if args.instruments:
        names   = {n.upper() for n in args.instruments}
        todo    = [x for x in INSTRUMENTS if x[0] in names]
        unknown = names - {x[0] for x in todo}
        if unknown:
            print(f"⚠️  Unknown instruments ignored: {', '.join(sorted(unknown))}")
        mode = ", ".join(x[0] for x in todo)
    elif args.core_only or CORE_ONLY:
        todo = [x for x in INSTRUMENTS if x[0] in CORE_NAMES]
        mode = "5 CORE only"
    else:
        todo = INSTRUMENTS
        mode = f"all {len(INSTRUMENTS)}"
    print(f"📥 Fetching {mode}  ({args.bar_size} bars)")
    print(f"  ~{throttle}s between requests to respect pacing limits\n")

    t0 = time.time()
    for name, contract, wts in todo:
        if global_since is not None:
            since = global_since
        else:
            latest = _csv_latest_date(name, suffix)
            if latest is not None:
                since = latest - pd.Timedelta(hours=1)
            else:
                since = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=2)

        if since >= global_until:
            print(f"  💾 {name:<12} already up to date ({since.date()})")
            continue

        try:
            fetch_one(ib, name, contract, wts, manifest,
                      since, global_until,
                      suffix, bar_size_str, chunk, throttle, contfut_durations)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            break

    ib.disconnect()
    elapsed = (time.time() - t0) / 60
    print(f"\n✅ Done in {elapsed:.1f} min")
    print(f"💾 Cache: {CACHE_DIR}/  -- reused automatically on next run")

    ok     = sum(1 for v in manifest.values() if v.get("status") == "ok")
    failed = sum(1 for v in manifest.values() if v.get("status") != "ok")
    print(f"\n     {ok} instruments cached, {failed} failed/skipped")


if __name__ == "__main__":
    main()
