"""
fetch_ibkr_history.py
---------------------
Downloads 30-min historical bars from Interactive Brokers, ONCE, and
caches them locally. Re-runs read the cache and only fetch missing data.

This breaks the yfinance 60-day wall: all your posts (Nov 2024 → now)
get real intraday labels, not just the last 57 days.

CACHING:
  market_data_cache/
    SPY_30min.csv ... (full history per instrument)
    _manifest.json  (tracks cached date ranges)
  Re-running = instant, offline. Only fetches gaps.

SETUP (one time):
  1. Open TWS or IB Gateway and log in
  2. In TWS: Edit → Global Config → API → Settings:
       ✅ Enable ActiveX and Socket Clients
       ✅ Socket port = 7497 (TWS paper) / 7496 (TWS live)
                       / 4002 (Gateway paper) / 4001 (Gateway live)
       ✅ Trust 127.0.0.1
  3. uv pip install ib_async pandas
  4. uv run python fetch_ibkr_history.py 202606170000 --from 202411010000

USAGE:
  # Full backfill from Nov 2016 to Nov 2024 (all instruments):
  uv run python fetch_ibkr_history.py 202411010000 --from 201611010000

  # Update cache through today, starting from Nov 2024:
  uv run python fetch_ibkr_history.py 202606170000 --from 202411010000

  # Only core 5 instruments:
  uv run python fetch_ibkr_history.py 202606170000 --from 202411010000 --core-only

Edit PORT below to match your TWS/Gateway mode.
"""

import os
import sys
import json
import time
import argparse
import datetime
import pandas as pd
from ib_async import IB, Stock, Future, Index, Forex, Crypto, ContFuture

NY = "America/New_York"

# ==========================================
# CONFIG — edit PORT if needed
# ==========================================
PORT      = 7497   # TWS desktop PAPER  (7496=TWS live · 4002/4001=Gateway paper/live)
HOST      = "127.0.0.1"
CLIENT_ID = 11

CACHE_DIR    = "market_data_cache"
BAR_SIZE     = "30 mins"
THROTTLE_SEC = 11          # safe: stays under 60 requests / 10 min
CHUNK        = "1 M"       # one month of 30-min bars per request

# ==========================================
# INSTRUMENT → IBKR CONTRACT MAPPING
# Core 5 first so they finish in the first ~20 min.
# ==========================================
def C_stock(sym):            return Stock(sym, "SMART", "USD")
def C_index(sym, exch="CBOE"): return Index(sym, exch, "USD")
def C_contfut(sym, exch):    return ContFuture(sym, exch, currency="USD")
def C_fx(pair):              return Forex(pair)

# (name, contract, whatToShow)
INSTRUMENTS = [
    # ---- CORE 5 ----
    ("SPY",   C_stock("SPY"),               "TRADES"),
    ("VIX",   C_index("VIX", "CBOE"),       "TRADES"),
    ("OIL",   C_contfut("CL", "NYMEX"),     "TRADES"),
    ("GOLD",  C_contfut("GC", "COMEX"),     "TRADES"),
    ("BTC",   Crypto("BTC", "PAXOS", "USD"),"AGGTRADES"),

    # ---- US equity sectors / breadth (react directly to policy posts) ----
    ("QQQ",   C_stock("QQQ"), "TRADES"),   # tech — tariff/China sensitive
    ("DIA",   C_stock("DIA"), "TRADES"),   # industrials-heavy Dow
    ("XLI",   C_stock("XLI"), "TRADES"),   # industrials — tariffs/trade
    ("XLF",   C_stock("XLF"), "TRADES"),   # financials — rates/dereg
    ("XLE",   C_stock("XLE"), "TRADES"),   # energy — oil/drilling posts

    # ---- commodities that US President posts actually move ----
    ("COPPER", C_contfut("HG", "COMEX"),   "TRADES"),  # tariffs, China demand
    ("NATGAS", C_contfut("NG", "NYMEX"),   "TRADES"),  # LNG/export policy

    # ---- major FX (tariff & trade-war transmission) ----
    ("EUR_USD", C_fx("EURUSD"), "MIDPOINT"),
    ("USD_JPY", C_fx("USDJPY"), "MIDPOINT"),
    ("GBP_USD", C_fx("GBPUSD"), "MIDPOINT"),
    ("USD_CNY", C_fx("USDCNH"), "MIDPOINT"),
    ("USD_CAD", C_fx("USDCAD"), "MIDPOINT"),
    ("USD_MXN", C_fx("USDMXN"), "MIDPOINT"),
    ("USD_CHF", C_fx("USDCHF"), "MIDPOINT"),
    ("AUD_USD", C_fx("AUDUSD"), "MIDPOINT"),

    # ---- bonds (Fed pressure, rate posts) ----
    ("US10Y", C_contfut("ZN", "CBOT"), "TRADES"),
    ("US2Y",  C_contfut("ZT", "CBOT"), "TRADES"),

    # ---- crypto ----
    ("ETH",   Crypto("ETH", "PAXOS", "USD"), "AGGTRADES"),
]

CORE_NAMES = {"SPY", "VIX", "OIL", "GOLD", "BTC"}

# Futures contract schedule per symbol: which months are actively traded.
# Used to auto-generate expiry strings for the full date range.
_FUTURES_MONTHS = {
    "CL": "monthly",       # crude oil: every month
    "NG": "monthly",       # nat gas: every month
    "HG": "monthly",       # copper: every month
    "GC": "even",          # gold: Feb Apr Jun Aug Oct Dec
    "SI": "bimonthly",     # silver: Mar May Jul Sep Dec
    "ZN": "quarterly",     # 10y note: Mar Jun Sep Dec
    "ZT": "quarterly",     # 2y note:  Mar Jun Sep Dec
}

_EVEN_MONTHS   = {2, 4, 6, 8, 10, 12}
_SILVER_MONTHS = {3, 5, 7, 9, 12}
_QUARTERLY     = {3, 6, 9, 12}


def _gen_futures_expiries(sym: str, since: pd.Timestamp, until: pd.Timestamp) -> list[str]:
    """
    Auto-generate YYYYMM contract strings for `sym` covering [since, until].
    Returns months whose active trading window overlaps the requested range.
    We include one extra quarter before `since` and after `until` as buffer.
    """
    schedule = _FUTURES_MONTHS.get(sym)
    if schedule is None:
        return []

    buf = pd.DateOffset(months=3)
    cur = (since - buf).replace(day=1)
    end = (until + buf).replace(day=1)

    months = []
    while cur <= end:
        m = cur.month
        include = (
            schedule == "monthly" or
            (schedule == "even" and m in _EVEN_MONTHS) or
            (schedule == "bimonthly" and m in _SILVER_MONTHS) or
            (schedule == "quarterly" and m in _QUARTERLY)
        )
        if include:
            months.append(cur.strftime("%Y%m"))
        cur = cur + pd.DateOffset(months=1)

    return months


# ============================================================
# CLI
# ============================================================
def parse_stamp(s: str, name: str = "time") -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.datetime.strptime(s.strip(), "%Y%m%d%H%M"), tz=NY)
    except ValueError:
        sys.exit(f"❌ {name} must be yyyymmddhhmm (NY local), got: {s!r}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Download 30-min IBKR bars into market_data_cache/ for a date range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "until", metavar="yyyymmddhhmm",
        help="End of the fetch window, NY local time (e.g. 202411010000).",
    )
    ap.add_argument(
        "--from", dest="since", metavar="yyyymmddhhmm", default=None,
        help="Start of the fetch window, NY local time.  "
             "Default: 30 days before --until.",
    )
    ap.add_argument(
        "--core-only", action="store_true",
        help="Fetch only the 5 core instruments (SPY VIX OIL GOLD BTC). "
             "Fast (~20 min). Flip off later for all 23.",
    )
    ap.add_argument(
        "--port", type=int, default=PORT,
        help=f"TWS/Gateway socket port (default {PORT}).  "
             "7497=TWS paper, 7496=TWS live, 4002=GW paper, 4001=GW live.",
    )
    ap.add_argument(
        "--host", default=HOST,
        help=f"TWS/Gateway host (default {HOST}).",
    )
    ap.add_argument(
        "--client-id", type=int, default=CLIENT_ID,
        help=f"IB client ID (default {CLIENT_ID}).",
    )
    return ap.parse_args()


# ============================================================
# cache helpers
# ============================================================
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


def month_starts(since: pd.Timestamp, until: pd.Timestamp) -> list[pd.Timestamp]:
    """Yield first-of-month dates from since to until (inclusive of until's month)."""
    cur = since.replace(day=1)
    end = until
    out = []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return out


# ============================================================
# main per-instrument fetcher
# ============================================================
def fetch_one(ib, name, contract, what_to_show, manifest,
              since: pd.Timestamp, until: pd.Timestamp):
    """Fetch bars for one instrument over [since, until], month by month, into cache."""
    cache_file = os.path.join(CACHE_DIR, f"{name}_30min.csv")

    since_utc = since.tz_convert("UTC")
    until_utc = until.tz_convert("UTC")

    # Load existing cache if present
    if os.path.exists(cache_file):
        existing = pd.read_csv(cache_file, parse_dates=['date'])
        existing['date'] = pd.to_datetime(existing['date'], utc=True)
        cached_months = set(existing['date'].dt.strftime('%Y-%m').unique())
        # Always re-fetch the latest cached month — may have been captured mid-month.
        if cached_months:
            latest = max(cached_months)
            cached_months.discard(latest)
            print(f"  ℹ️  {name}: will refresh latest cached month ({latest})")
    else:
        existing = pd.DataFrame()
        cached_months = set()

    # Qualify contract (resolves exact conId)
    try:
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            print(f"  ❌ {name:<12} could not qualify contract — skipping")
            manifest[name] = {"status": "failed_qualify"}
            return
    except Exception as e:
        print(f"  ❌ {name:<12} qualify error: {str(e)[:60]} — skipping")
        manifest[name] = {"status": "qualify_error"}
        return

    # ── CONTINUOUS FUTURES special path ──────────────────────────────────
    # ContFuture FORBIDS endDateTime (Error 10339) and only covers ~12 months.
    # Strategy:
    #   A) ContFuture with empty endDateTime — only if `until` is within the
    #      last 14 months (otherwise this contract covers none of our window).
    #   B) Dated expired contracts (includeExpired=True) — month-by-month back
    #      to `since`.  Dated futures DO allow endDateTime.
    is_contfut = type(contract).__name__ == "ContFuture"
    if is_contfut:
        new_frames = []
        c = qualified[0] if isinstance(qualified, list) else contract
        days_to_now = (pd.Timestamp.now(tz=NY) - until).days

        # ---- A) recent year via continuous (skip if until is far in the past) ----
        use_continuous = days_to_now < 400
        if use_continuous:
            have_recent = False
            if len(existing):
                latest_cached = existing['date'].max()
                if latest_cached >= (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=10)):
                    have_recent = True
                    print(f"     💾 {name}: recent year already cached, skipping continuous pull")

            if not have_recent:
                for dur in ["180 D", "120 D", "270 D", "365 D", "90 D"]:
                    try:
                        bars = ib.reqHistoricalData(
                            c, endDateTime="", durationStr=dur,
                            barSizeSetting=BAR_SIZE, whatToShow=what_to_show,
                            useRTH=0, formatDate=2, timeout=180,
                        )
                        if bars:
                            new_frames.append(pd.DataFrame([{
                                'date': b.date, 'open': b.open, 'high': b.high,
                                'low': b.low, 'close': b.close, 'volume': b.volume,
                            } for b in bars]))
                            print(f"     ✓ {name}: {len(bars)} recent bars ({dur})")
                            time.sleep(THROTTLE_SEC)
                            break
                        time.sleep(THROTTLE_SEC)
                    except Exception as e:
                        print(f"     ⚠️  {name} recent {dur}: {str(e)[:45]}")
                        time.sleep(THROTTLE_SEC)
        else:
            print(f"     ℹ️  {name}: --until is {days_to_now} days ago — "
                  f"skipping continuous path, using dated contracts only")

        # ---- B) dated expired contracts to cover [since, until] ----
        base_sym = contract.symbol
        exch     = contract.exchange
        expiries = _gen_futures_expiries(base_sym, since, until)
        print(f"     📅 {name}: {len(expiries)} contract months to check "
              f"({expiries[0] if expiries else '–'} → {expiries[-1] if expiries else '–'})")

        for ym in expiries:
            ym_month = f"{ym[:4]}-{ym[4:6]}"
            if ym_month in cached_months:
                continue
            try:
                fut = Future(symbol=base_sym, exchange=exch, currency="USD",
                             lastTradeDateOrContractMonth=ym, includeExpired=True)
                q = ib.qualifyContracts(fut)
                if not q:
                    print(f"     ⚠️  {name} {ym}: could not qualify expired contract")
                    continue
                # End at the expiry month-end, pull 90 days back (active window)
                end_dt = (
                    (pd.Timestamp(ym + "01") + pd.offsets.MonthEnd(0))
                    .strftime("%Y%m%d %H:%M:%S")
                )
                bars = ib.reqHistoricalData(
                    q[0], endDateTime=end_dt, durationStr="90 D",
                    barSizeSetting=BAR_SIZE, whatToShow=what_to_show,
                    useRTH=0, formatDate=2, timeout=90,
                )
                if bars:
                    new_frames.append(pd.DataFrame([{
                        'date': b.date, 'open': b.open, 'high': b.high,
                        'low': b.low, 'close': b.close, 'volume': b.volume,
                    } for b in bars]))
                    print(f"     ✓ {name}: {len(bars)} bars from expired {ym}")
                time.sleep(THROTTLE_SEC)
            except Exception as e:
                print(f"     ⚠️  {name} expired {ym}: {str(e)[:45]}")
                time.sleep(THROTTLE_SEC)

        if new_frames:
            combined = pd.concat(
                ([existing] if len(existing) else []) + new_frames, ignore_index=True
            )
            combined['date'] = pd.to_datetime(combined['date'], utc=True)
            combined = (combined
                .drop_duplicates(subset=['date'])
                .sort_values('date')
                .reset_index(drop=True))
            # Trim to requested window
            combined = combined[
                (combined['date'] >= since_utc) & (combined['date'] <= until_utc)
            ]
            combined.to_csv(cache_file, index=False)
            manifest[name] = {
                "status": "ok", "rows": len(combined),
                "first": str(combined['date'].min()),
                "last":  str(combined['date'].max()),
            }
            span = f"{combined['date'].min().date()} → {combined['date'].max().date()}"
            print(f"  ✅ {name:<12} {len(combined):>6} bars  ({span})")
        elif len(existing):
            span = (f"{existing['date'].min().date()} → "
                    f"{existing['date'].max().date()}")
            print(f"  💾 {name:<12} {len(existing):>6} bars (already complete, {span})")
            manifest[name] = {"status": "ok", "rows": len(existing)}
        else:
            print(f"  ❌ {name:<12} no data — futures subscription may be needed")
            manifest[name] = {"status": "no_data"}

        save_manifest(manifest)
        return
    # ── end continuous-futures path ───────────────────────────────────────

    # ── Stocks / FX / Crypto / Index path ────────────────────────────────
    months = month_starts(since, until)
    new_frames = []
    fetched, skipped, failed_months = 0, 0, []
    consecutive_empty = 0

    for m in months:
        mkey = m.strftime('%Y-%m')
        if mkey in cached_months:
            skipped += 1
            continue

        if consecutive_empty >= 2 and fetched == 0:
            print(f"     ⏭️  {name}: no data on first 2 months — "
                  f"likely unsubscribed, skipping rest")
            failed_months.append("(bailed early)")
            break

        # endDateTime = first day of NEXT month, pull "1 M" back.
        next_month = m + pd.offsets.MonthBegin(1)
        end_dt = (
            min(next_month, until + pd.Timedelta(days=1))
            .strftime('%Y%m%d %H:%M:%S')
        )

        got_data = False
        for attempt in range(1, 4):
            try:
                bars = ib.reqHistoricalData(
                    qualified[0] if isinstance(qualified, list) else contract,
                    endDateTime    = end_dt,
                    durationStr    = CHUNK,
                    barSizeSetting = BAR_SIZE,
                    whatToShow     = what_to_show,
                    useRTH         = 0,
                    formatDate     = 2,
                )
                if bars:
                    dfm = pd.DataFrame([{
                        'date': b.date, 'open': b.open, 'high': b.high,
                        'low':  b.low,  'close': b.close, 'volume': b.volume,
                    } for b in bars])
                    new_frames.append(dfm)
                    fetched += 1
                    got_data = True
                    consecutive_empty = 0
                    time.sleep(THROTTLE_SEC)
                    break
                else:
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

    if new_frames:
        combined = (pd.concat([existing] + new_frames, ignore_index=True)
                    if len(existing) else pd.concat(new_frames, ignore_index=True))
        combined['date'] = pd.to_datetime(combined['date'], utc=True)
        combined = (combined
            .drop_duplicates(subset=['date'])
            .sort_values('date')
            .reset_index(drop=True))
        combined = combined[
            (combined['date'] >= since_utc) & (combined['date'] <= until_utc)
        ]
        combined.to_csv(cache_file, index=False)
        manifest[name] = {
            "status": "ok", "rows": len(combined),
            "first": str(combined['date'].min()),
            "last":  str(combined['date'].max()),
        }
        print(f"  ✅ {name:<12} {len(combined):>6} bars  "
              f"(fetched {fetched} mo, cached {skipped} mo)")
    elif len(existing):
        print(f"  💾 {name:<12} fully cached ({len(existing)} bars) — no new fetch")
        manifest[name] = {"status": "ok", "rows": len(existing)}
    else:
        print(f"  ❌ {name:<12} no data returned (may need market-data subscription)")
        manifest[name] = {"status": "no_data"}

    save_manifest(manifest)


# ============================================================
# main
# ============================================================
def main():
    args = parse_args()

    until = parse_stamp(args.until, "until")
    if args.since:
        since = parse_stamp(args.since, "--from")
    else:
        since = until - pd.Timedelta(days=30)

    if since >= until:
        sys.exit("❌ --from must be before the positional until argument.")

    span_months = round((until - since).days / 30)
    span_years  = (until - since).days / 365.25

    print("=" * 64)
    print("  IBKR HISTORICAL DATA FETCHER — 30min bars, cached")
    print("=" * 64)
    print(f"  Since : {since:%Y-%m-%d %H:%M %Z}")
    print(f"  Until : {until:%Y-%m-%d %H:%M %Z}")
    print(f"  Span  : ~{span_years:.1f} years ({span_months} months)")
    print(f"  Port  : {args.port}   Host: {args.host}   ClientID: {args.client_id}")
    print("=" * 64)

    manifest = ensure_cache_dir()

    print(f"\n🔌 Connecting to IBKR at {args.host}:{args.port} "
          f"(client {args.client_id})...")
    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("   Is TWS/Gateway running with API enabled on this port?")
        return
    print("✅ Connected\n")

    todo = (
        [x for x in INSTRUMENTS if x[0] in CORE_NAMES]
        if args.core_only else INSTRUMENTS
    )
    mode = "5 CORE only" if args.core_only else f"all {len(INSTRUMENTS)}"
    print(f"📥 Fetching {mode} instruments  [{since:%Y-%m-%d} → {until:%Y-%m-%d}]")
    print(f"   ~{THROTTLE_SEC}s between requests to respect pacing limits\n")

    t0 = time.time()
    for name, contract, wts in todo:
        fetch_one(ib, name, contract, wts, manifest, since, until)

    ib.disconnect()
    elapsed = (time.time() - t0) / 60
    print(f"\n✅ Done in {elapsed:.1f} min")
    print(f"💾 Cache: {CACHE_DIR}/  — reused automatically on next run")

    ok     = sum(1 for v in manifest.values() if v.get("status") == "ok")
    failed = sum(1 for v in manifest.values() if v.get("status") != "ok")
    print(f"\n   {ok} instruments cached, {failed} failed/skipped")
    if failed:
        print("   Failed ones fall back to yfinance daily in build_final_training_set.py")


if __name__ == "__main__":
    main()
