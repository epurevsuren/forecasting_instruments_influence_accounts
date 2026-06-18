"""
fetch_ibkr_history.py
---------------------
Downloads 30-min historical bars from Interactive Brokers and caches them
locally as CSV files. Re-runs read the cache and only fetch what is missing.

DEFAULT BEHAVIOUR (no arguments):
  For each instrument, reads the latest date already in its CSV and fetches
  forward to now. New instruments with no CSV start from now - 2 years
  (IBKR ContFuture retention limit).

WITH ARGUMENTS:
  python fetch_ibkr_history.py --from 20241101 --until 20260101
  Fills only what is missing inside that window (already-cached months
  are skipped automatically).

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

CACHE_DIR    = "market_data_cache"
BAR_SIZE     = "30 mins"
THROTTLE_SEC = 11      # safe: stays under 60 requests / 10 min
CHUNK        = "1 M"   # one month of 30-min bars per request (stocks/FX path)

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
    """Atomic write: temp file → rename. If Ctrl+C hits mid-write, retries to
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
    # Subtract 1s so until=2016-01-01 00:00 gives last month=2015-12,
    # same boundary logic as win_end in the cache info block.
    end = pd.Timestamp(until) - pd.Timedelta(seconds=1)
    out = []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def _csv_latest_date(name):
    """Return the latest date in the cache CSV, or None if no CSV exists."""
    path = os.path.join(CACHE_DIR, f"{name}_30min.csv")
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
def fetch_one(ib, name, contract, what_to_show, manifest, since, until):
    """
    Fetch bars for one instrument over [since, until], skipping cached months.
    `since` and `until` are UTC Timestamps.
    """
    cache_file = os.path.join(CACHE_DIR, f"{name}_30min.csv")

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

            # Full gap: all missing months from since to the cache's own latest
            # (not limited by --until), so both runs show the same true gap.
            full_end = cache_latest
            full_all = set()
            cur = since.replace(day=1)
            while cur.strftime("%Y-%m") <= full_end:
                full_all.add(cur.strftime("%Y-%m"))
                cur = cur + pd.offsets.MonthBegin(1)
            full_missing = sorted(full_all - cached_months)

            # Window: what this run will actually fetch (limited by --until)
            win_end = (until - pd.Timedelta(seconds=1)).strftime("%Y-%m")
            missing = sorted(m for m in full_missing if m <= win_end)

            if full_missing:
                gap_msg = f" | gap {full_missing[0]} → {full_missing[-1]} ({len(full_missing)} mo)"
                if missing and missing != full_missing:
                    gap_msg += f", fetching {len(missing)} mo this run"
            elif missing:
                gap_msg = f" | gap {missing[0]} → {missing[-1]} ({len(missing)} mo)"
            else:
                gap_msg = " | window fully cached"

            # Refresh the current live month only in default mode (no --from),
            # detected by since being recent (≤35 days ago). Explicit --from
            # backfills never refresh cached months, even when until ≈ now.
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

            print(f"  ℹ️  {name}: cache {cache_earliest} → {cache_latest} ({len(existing):,} bars){gap_msg}{refresh_msg}")
    else:
        existing = pd.DataFrame()
        cached_months = set()

    # Qualify contract
    try:
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            print(f"  ❌ {name:<12} could not qualify contract -- skipping")
            manifest[name] = {"status": "failed_qualify"}
            return
    except Exception as e:
        print(f"  ❌ {name:<12} qualify error: {str(e)[:60]} -- skipping")
        manifest[name] = {"status": "qualify_error"}
        return

    # ── CONTINUOUS FUTURES path ───────────────────────────────────────────────
    is_contfut = type(contract).__name__ == "ContFuture"
    if is_contfut:
        new_frames = []
        c = qualified[0] if isinstance(qualified, list) else contract

        # Path A: fetch the most recent chunk via ContFuture (endDateTime="" = now).
        # Skip if cache is already fresh (within 10 days of now).
        cache_is_fresh = (
            len(existing) > 0 and
            existing["date"].max() >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
        )
        if cache_is_fresh:
            print(f"     💾 {name}: cached up to {existing['date'].max().date()}, skipping recent fetch")
        else:
            # 365 D is the practical max for 30-min ContFuture; longer durations timeout.
            for dur in ["365 D", "270 D", "180 D", "120 D", "90 D"]:
                try:
                    bars = ib.reqHistoricalData(
                        c, endDateTime="", durationStr=dur,
                        barSizeSetting=BAR_SIZE, whatToShow=what_to_show,
                        useRTH=0, formatDate=2, timeout=60,
                    )
                    if bars:
                        new_frames.append(pd.DataFrame([{
                            "date": b.date, "open": b.open, "high": b.high,
                            "low": b.low, "close": b.close, "volume": b.volume,
                        } for b in bars]))
                        earliest = pd.to_datetime(bars[0].date, utc=True)
                        print(f"     ✓ {name}: {len(bars)} bars via ContFuture ({dur}) "
                              f"earliest={earliest.date()}")
                        time.sleep(THROTTLE_SEC)
                        break
                    time.sleep(THROTTLE_SEC)
                except Exception as e:
                    print(f"     ⚠️  {name} ContFuture {dur}: {str(e)[:55]}")
                    time.sleep(THROTTLE_SEC)

        # Path B: backfill via dated expired contracts.
        # ContFuture forbids explicit endDateTime (IBKR error 10339), but plain
        # Future contracts with includeExpired=True DO allow it. Use them to fill
        # the gap between `since` and the earliest data we have from path A.
        all_so_far = new_frames + ([existing] if len(existing) > 0 else [])
        if all_so_far:
            cf_earliest = min(pd.to_datetime(f["date"], utc=True).min() for f in all_so_far)
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

        expired_frames = []
        for ym in expiries:
            ym_month = f"{ym[:4]}-{ym[4:6]}"
            if ym_month in cached_months:
                continue
            # qualifyContracts as fast pre-filter (< 1 sec on failure).
            # Split from the data fetch so AttributeError on old contracts
            # just skips to next ym with no sleep — no 90-sec timeout wait.
            fut = Future(symbol=base_sym, exchange=exch, currency="USD",
                         lastTradeDateOrContractMonth=ym)
            fut.includeExpired = True
            try:
                q = ib.qualifyContracts(fut)
            except Exception:
                q = []
            if not q or q[0] is None:
                continue  # not found — no sleep, just skip

            try:
                end_dt = (pd.Timestamp(ym + "01") + pd.offsets.MonthEnd(0)
                          ).strftime("%Y%m%d %H:%M:%S")
                bars = ib.reqHistoricalData(
                    q[0], endDateTime=end_dt, durationStr="90 D",
                    barSizeSetting=BAR_SIZE, whatToShow=what_to_show,
                    useRTH=0, formatDate=2, timeout=90,
                )
                if bars:
                    expired_frames.append(pd.DataFrame([{
                        "date": b.date, "open": b.open, "high": b.high,
                        "low": b.low, "close": b.close, "volume": b.volume,
                    } for b in bars]))
                    print(f"     ✓ {name}: {len(bars)} bars from expired {ym}")
                time.sleep(THROTTLE_SEC)
            except Exception as e:
                print(f"     ⚠️  {name} expired {ym}: {str(e)[:55]}")
                time.sleep(THROTTLE_SEC)

        # ContFuture bars take priority over expired-contract bars for same timestamps.
        # expired_frames first, then new_frames; drop_duplicates keeps the first seen.
        new_frames = expired_frames + new_frames

        # Merge with existing and save
        if new_frames:
            new_df = pd.concat(new_frames, ignore_index=True)
            new_df["date"] = pd.to_datetime(new_df["date"], utc=True)
            new_df = new_df[new_df["date"] <= until]   # trim only NEW data, keep existing intact
            combined = pd.concat(
                ([existing] if len(existing) else []) + [new_df],
                ignore_index=True,
            )
            combined["date"] = pd.to_datetime(combined["date"], utc=True)
            combined = (combined.drop_duplicates(subset=["date"], keep="last")
                        .sort_values("date").reset_index(drop=True))
            safe_to_csv(combined, cache_file)
            span = f"{combined['date'].min().date()} -> {combined['date'].max().date()}"
            manifest[name] = {"status": "ok", "rows": len(combined),
                               "first": str(combined["date"].min()),
                               "last":  str(combined["date"].max())}
            print(f"  ✅ {name:<12} {len(combined):>6} bars  ({span})  [fetched]")
        elif len(existing):
            span = f"{existing['date'].min().date()} -> {existing['date'].max().date()}"
            print(f"  💾 {name:<12} {len(existing):>6} bars  ({span})  [cached, no change]")
            manifest[name] = {"status": "ok", "rows": len(existing)}
        else:
            print(f"  ❌ {name:<12} no data")
            manifest[name] = {"status": "no_data"}

        save_manifest(manifest)
        return
    # ── end ContFuture path ───────────────────────────────────────────────────

    # ── Stocks / FX / Crypto / Index path ────────────────────────────────────
    months = month_starts(since, until)
    fetched, skipped, failed_months = 0, 0, []
    consecutive_empty = 0

    # Work from a live copy of existing so we can save incrementally.
    combined = existing.copy() if len(existing) else pd.DataFrame()

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
        end_dt = min(next_month, until + pd.Timedelta(days=1)).strftime("%Y%m%d %H:%M:%S")

        print(f"     → {name} {mkey} ...", end="", flush=True)
        got_data = False
        for attempt in range(1, 4):
            try:
                bars = ib.reqHistoricalData(
                    qualified[0] if isinstance(qualified, list) else contract,
                    endDateTime=end_dt, durationStr=CHUNK,
                    barSizeSetting=BAR_SIZE, whatToShow=what_to_show,
                    useRTH=0, formatDate=2, timeout=60,
                )
                if bars:
                    dfm = pd.DataFrame([{
                        "date": b.date, "open": b.open, "high": b.high,
                        "low": b.low, "close": b.close, "volume": b.volume,
                    } for b in bars])
                    print(f" {len(bars)} bars")
                    fetched += 1
                    got_data = True
                    consecutive_empty = 0
                    # Merge and save immediately so progress survives Ctrl+C
                    dfm["date"] = pd.to_datetime(dfm["date"], utc=True)
                    dfm = dfm[dfm["date"] <= until]    # trim only NEW data, keep existing intact
                    combined = pd.concat([combined, dfm], ignore_index=True)
                    combined["date"] = pd.to_datetime(combined["date"], utc=True)
                    combined = (combined.drop_duplicates(subset=["date"])
                                .sort_values("date").reset_index(drop=True))
                    try:
                        safe_to_csv(combined, cache_file)
                        cached_months.add(mkey)
                        time.sleep(THROTTLE_SEC)
                    except KeyboardInterrupt:
                        cached_months.add(mkey)
                        print(f"\n⚠️  Ctrl+C — saved {mkey}, stopping.")
                        raise
                    break
                else:
                    print(f" empty")
                    print(f"     ⟳  {name} {mkey}: empty (attempt {attempt}/3)")
                    time.sleep(THROTTLE_SEC)
            except Exception as e:
                print(f"     ⚠️  {name} {mkey} attempt {attempt}/3: {str(e)[:50]}")
                time.sleep(THROTTLE_SEC)

        if not got_data:
            failed_months.append(mkey)
            consecutive_empty += 1

    if failed_months:
        print(f"     ❌ {name}: {len(failed_months)} months failed/empty")

    if fetched:
        manifest[name] = {"status": "ok", "rows": len(combined),
                           "first": str(combined["date"].min()),
                           "last":  str(combined["date"].max())}
        print(f"  ✅ {name:<12} {len(combined):>6} bars  "
              f"(fetched {fetched} mo, cached {skipped} mo)")
    elif len(existing):
        print(f"  💾 {name:<12} {len(existing):>6} bars  [cached, no change]")
        manifest[name] = {"status": "ok", "rows": len(existing)}
    else:
        print(f"  ❌ {name:<12} no data returned")
        manifest[name] = {"status": "no_data"}

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
        description="Fetch 30-min IBKR historical bars into market_data_cache/.",
    )
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

    global_until = parse_stamp(args.until) if args.until else pd.Timestamp.now(tz="UTC")
    global_since = parse_stamp(args.since) if args.since else None

    print("=" * 60)
    print("  IBKR HISTORICAL DATA FETCHER -- 30-min bars")
    print("=" * 60)
    print(f"  Until : {global_until.strftime('%Y-%m-%d %H:%M UTC')}")
    if global_since:
        print(f"  Since : {global_since.strftime('%Y-%m-%d')}  (--from, backfill mode)")
    else:
        print(f"  Since : per-instrument latest cached date")
    print(f"  Port  : {args.port}   Host: {args.host}   ClientID: {args.client_id}")
    print("=" * 60)

    manifest = ensure_cache_dir()

    print(f"\n🔌 Connecting to IBKR at {args.host}:{args.port} (client {args.client_id})...")
    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("   Is TWS/Gateway running with API enabled on this port?")
        return
    print("✅ Connected\n")

    if args.instruments:
        names = {n.upper() for n in args.instruments}
        todo = [x for x in INSTRUMENTS if x[0] in names]
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
    print(f"📥 Fetching {mode}")
    print(f"   ~{THROTTLE_SEC}s between requests to respect pacing limits\n")

    t0 = time.time()
    for name, contract, wts in todo:
        if global_since is not None:
            since = global_since
        else:
            latest = _csv_latest_date(name)
            if latest is not None:
                since = latest - pd.Timedelta(hours=1)
            else:
                since = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=2)

        if since >= global_until:
            print(f"  💾 {name:<12} already up to date ({since.date()})")
            continue

        try:
            fetch_one(ib, name, contract, wts, manifest, since, global_until)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            break

    ib.disconnect()
    elapsed = (time.time() - t0) / 60
    print(f"\n✅ Done in {elapsed:.1f} min")
    print(f"💾 Cache: {CACHE_DIR}/  -- reused automatically on next run")

    ok     = sum(1 for v in manifest.values() if v.get("status") == "ok")
    failed = sum(1 for v in manifest.values() if v.get("status") != "ok")
    print(f"\n   {ok} instruments cached, {failed} failed/skipped")


if __name__ == "__main__":
    main()
