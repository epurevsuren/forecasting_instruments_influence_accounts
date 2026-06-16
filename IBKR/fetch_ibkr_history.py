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
  4. uv run python fetch_ibkr_history.py

Edit PORT below to match your TWS/Gateway mode.
"""

import os
import json
import time
import datetime
import pandas as pd
from ib_async import IB, Stock, Future, Index, Forex, Crypto, ContFuture

# Expired dated-contract months to backfill Nov 2024 → mid 2025 for futures.
# Each contract trades actively for ~90 days before its expiry month.
# CL/NG: monthly. GC: even months. ZN/ZT: quarterly (Mar/Jun/Sep/Dec).
FUTURES_EXPIRIES = {
    "CL": ["202412", "202501", "202502", "202503", "202504", "202505"],
    "GC": ["202412", "202502", "202504", "202506"],
    "SI": ["202412", "202503", "202505"],
    "HG": ["202412", "202503", "202505"],
    "NG": ["202412", "202501", "202502", "202503", "202504", "202505"],
    "ZN": ["202412", "202503"],
    "ZT": ["202412", "202503"],
}

# ==========================================
# CONFIG — EDIT THESE TWO
# ==========================================
PORT = 7497   # TWS desktop PAPER  (7496=TWS live · 4002/4001=Gateway paper/live)
HOST = "127.0.0.1"
CLIENT_ID = 11

CACHE_DIR  = "market_data_cache"
START_DATE = "2024-11-01"

# Set True to fetch ONLY the 5 core instruments (~20 min).
# Flip to False later to fetch all 37 — cached core ones are skipped instantly.
CORE_ONLY = False
BAR_SIZE   = "30 mins"
THROTTLE_SEC = 11          # safe: stays under 60 requests / 10 min
CHUNK = "1 M"              # one month of 30-min bars per request

# ==========================================
# INSTRUMENT → IBKR CONTRACT MAPPING
# Core 5 first so they finish in the first ~20 min.
# ==========================================
def C_stock(sym):  return Stock(sym, "SMART", "USD")
def C_index(sym, exch="CBOE"): return Index(sym, exch, "USD")
def C_contfut(sym, exch): return ContFuture(sym, exch, currency="USD")
def C_fx(pair):    return Forex(pair)

# (name, contract, whatToShow)
INSTRUMENTS = [
    # ---- CORE 5 ----
    ("SPY",   C_stock("SPY"),            "TRADES"),
    ("VIX",   C_index("VIX", "CBOE"),    "TRADES"),
    ("OIL",   C_contfut("CL", "NYMEX"),  "TRADES"),
    ("GOLD",  C_contfut("GC", "COMEX"),  "TRADES"),
    ("BTC",   Crypto("BTC", "PAXOS", "USD"), "AGGTRADES"),

    # ---- US equity sectors / breadth (react directly to policy posts) ----
    ("QQQ",   C_stock("QQQ"), "TRADES"),   # tech — tariff/China sensitive
    ("DIA",   C_stock("DIA"), "TRADES"),   # industrials-heavy Dow
    ("XLI",   C_stock("XLI"), "TRADES"),   # industrials — tariffs/trade
    ("XLF",   C_stock("XLF"), "TRADES"),   # financials — rates/dereg
    ("XLE",   C_stock("XLE"), "TRADES"),   # energy — oil/drilling posts

    # ---- commodities that US President posts actually move ----
    ("COPPER", C_contfut("HG", "COMEX"),  "TRADES"),  # tariffs, China demand
    ("NATGAS", C_contfut("NG", "NYMEX"),  "TRADES"),  # LNG/export policy

    # ---- major FX (tariff & trade-war transmission) ----
    ("EUR_USD", C_fx("EURUSD"), "MIDPOINT"),   # EU tariffs
    ("USD_JPY", C_fx("USDJPY"), "MIDPOINT"),   # risk sentiment, BoJ
    ("GBP_USD", C_fx("GBPUSD"), "MIDPOINT"),   # UK trade
    ("USD_CNY", C_fx("USDCNH"), "MIDPOINT"),   # China trade war — key
    ("USD_CAD", C_fx("USDCAD"), "MIDPOINT"),   # Canada tariffs, oil
    ("USD_MXN", C_fx("USDMXN"), "MIDPOINT"),   # Mexico tariffs — very sensitive
    ("USD_CHF", C_fx("USDCHF"), "MIDPOINT"),   # safe-haven flows
    ("AUD_USD", C_fx("AUDUSD"), "MIDPOINT"),   # China-proxy / risk

    # ---- bonds (Fed pressure, rate posts) ----
    ("US10Y", C_contfut("ZN", "CBOT"), "TRADES"),
    ("US2Y",  C_contfut("ZT", "CBOT"), "TRADES"),

    # ---- crypto ----
    ("ETH",   Crypto("ETH", "PAXOS", "USD"), "AGGTRADES"),  # pro-crypto posts
]

# 23 instruments — all cached with full 2024-11→2026-05 history.
# CUT: SILVER (partial data only), USD_KRW + all Asian/European indices
#   (no market-data subscription on this account — they error out).
#   The labeler falls back to yfinance daily for anything not cached.


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


def month_starts(start_date, end_date):
    """Yield first-of-month dates from start to end (inclusive of end's month)."""
    cur = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date)
    out = []
    while cur <= end:
        out.append(cur)
        cur = (cur + pd.offsets.MonthBegin(1))
    return out


def fetch_one(ib, name, contract, what_to_show, manifest):
    """Fetch full history for one instrument, month by month, into cache."""
    cache_file = os.path.join(CACHE_DIR, f"{name}_30min.csv")

    # Load existing cache if present
    if os.path.exists(cache_file):
        existing = pd.read_csv(cache_file, parse_dates=['date'])
        cached_months = set(existing['date'].dt.strftime('%Y-%m').unique())
        # Always re-fetch the LATEST cached month — it may have been incomplete
        # (e.g. captured mid-month). Removing it forces a fresh pull.
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
        manifest[name] = {"status": f"qualify_error"}
        return

    today = pd.Timestamp.now()

    # ── CONTINUOUS FUTURES special path ──────────────────────────
    # ContFuture FORBIDS endDateTime entirely (Error 10339).
    # Strategy:
    #   A) ContFuture empty-endDateTime "365 D" → recent ~12 months
    #   B) DATED EXPIRED contracts (includeExpired=True) → older months
    #      back to Nov 2024. Dated Futures DO allow endDateTime.
    is_contfut = type(contract).__name__ == "ContFuture"
    if is_contfut:
        new_frames = []
        c = qualified[0] if isinstance(qualified, list) else contract

        # ---- A) recent year via continuous (skip if already well-cached) ----
        have_recent = False
        if len(existing):
            latest_cached = pd.to_datetime(existing['date']).max()
            if latest_cached >= (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=10)):
                have_recent = True
                print(f"     💾 {name}: recent year already cached, skipping continuous pull")

        if not have_recent:
            # Start smaller — thinly-traded futures (silver, copper) time out on
            # big pulls. Try ascending durations; first success wins.
            for dur in ["180 D", "120 D", "270 D", "365 D", "90 D"]:
                try:
                    bars = ib.reqHistoricalData(
                        c, endDateTime="", durationStr=dur, barSizeSetting=BAR_SIZE,
                        whatToShow=what_to_show, useRTH=0, formatDate=2, timeout=180,
                    )
                    if bars:
                        new_frames.append(pd.DataFrame([{
                            'date': b.date,'open':b.open,'high':b.high,
                            'low':b.low,'close':b.close,'volume':b.volume} for b in bars]))
                        print(f"     ✓ {name}: {len(bars)} recent bars ({dur})")
                        time.sleep(THROTTLE_SEC)
                        break
                    time.sleep(THROTTLE_SEC)
                except Exception as e:
                    print(f"     ⚠️  {name} recent {dur}: {str(e)[:45]}")
                    time.sleep(THROTTLE_SEC)

        # ---- B) older months via dated expired contracts ----
        # Quarterly contract months that cover Nov 2024 → mid 2025.
        # Each quarterly contract trades for ~3 months before expiry, so
        # fetching these expiries with includeExpired backfills the gap.
        base_sym  = contract.symbol
        exch      = contract.exchange
        expiries  = FUTURES_EXPIRIES.get(base_sym, [])
        for ym in expiries:
            # Skip if we already have data spanning this contract's active window
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
                # end at expiry month-end, pull 90 days back (its active window)
                end_dt = (pd.Timestamp(ym + "01") + pd.offsets.MonthEnd(0)).strftime('%Y%m%d %H:%M:%S')
                bars = ib.reqHistoricalData(
                    q[0], endDateTime=end_dt, durationStr="90 D", barSizeSetting=BAR_SIZE,
                    whatToShow=what_to_show, useRTH=0, formatDate=2, timeout=90,
                )
                if bars:
                    new_frames.append(pd.DataFrame([{
                        'date': b.date,'open':b.open,'high':b.high,
                        'low':b.low,'close':b.close,'volume':b.volume} for b in bars]))
                    print(f"     ✓ {name}: {len(bars)} bars from expired {ym}")
                time.sleep(THROTTLE_SEC)
            except Exception as e:
                print(f"     ⚠️  {name} expired {ym}: {str(e)[:45]}")
                time.sleep(THROTTLE_SEC)

        if new_frames:
            combined = pd.concat(([existing] if len(existing) else []) + new_frames,
                                  ignore_index=True)
            combined['date'] = pd.to_datetime(combined['date'], utc=True)
            combined = combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            combined = combined[combined['date'] >= pd.Timestamp(START_DATE, tz='UTC')]
            combined.to_csv(cache_file, index=False)
            manifest[name] = {"status":"ok","rows":len(combined),
                              "first":str(combined['date'].min()),
                              "last":str(combined['date'].max())}
            span = f"{combined['date'].min().date()} → {combined['date'].max().date()}"
            print(f"  ✅ {name:<12} {len(combined):>6} bars  ({span})")
        elif len(existing):
            # No NEW data but cache already complete — keep it, not a failure
            span = f"{existing['date'].min().date()} → {existing['date'].max().date()}"
            print(f"  💾 {name:<12} {len(existing):>6} bars (already complete, {span})")
            manifest[name] = {"status":"ok","rows":len(existing)}
        else:
            print(f"  ❌ {name:<12} no data — futures subscription may be needed")
            manifest[name] = {"status":"no_data"}
        save_manifest(manifest)
        return
    # ── end continuous-futures path ──────────────────────────────

    months = month_starts(START_DATE, today)
    new_frames = []
    fetched, skipped, failed_months = 0, 0, []
    consecutive_empty = 0   # for early bail-out on unsubscribed instruments

    for m in months:
        mkey = m.strftime('%Y-%m')
        if mkey in cached_months:
            skipped += 1
            continue

        # Early bail-out: if the first 2 attempted months both return nothing,
        # this instrument almost certainly lacks a data subscription.
        # Stop wasting 10 min retrying all 18 months × 3 attempts.
        if consecutive_empty >= 2 and fetched == 0:
            print(f"     ⏭️  {name}: no data on first months — likely unsubscribed, skipping rest")
            failed_months.append("(bailed early)")
            break

        # endDateTime = first day of NEXT month, pull "1 M" back.
        next_month = m + pd.offsets.MonthBegin(1)
        if next_month > today:
            end_dt = today.strftime('%Y%m%d %H:%M:%S')
        else:
            end_dt = next_month.strftime('%Y%m%d %H:%M:%S')

        # Retry each month up to 3 times — handles transient session/pacing errors
        got_data = False
        for attempt in range(1, 4):
            try:
                bars = ib.reqHistoricalData(
                    qualified[0] if isinstance(qualified, list) else contract,
                    endDateTime   = end_dt,
                    durationStr   = CHUNK,
                    barSizeSetting= BAR_SIZE,
                    whatToShow    = what_to_show,
                    useRTH        = 0,        # include extended hours
                    formatDate    = 2,       # UTC tz-aware
                )
                if bars:
                    dfm = pd.DataFrame([{
                        'date':  b.date, 'open': b.open, 'high': b.high,
                        'low':   b.low,  'close': b.close, 'volume': b.volume,
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
        print(f"     ❌ {name}: {len(failed_months)} months failed")

    # Merge new + existing, dedup, save
    if new_frames:
        combined = pd.concat([existing] + new_frames, ignore_index=True) if len(existing) else pd.concat(new_frames, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'], utc=True)
        combined = combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        combined.to_csv(cache_file, index=False)
        manifest[name] = {
            "status": "ok",
            "rows": len(combined),
            "first": str(combined['date'].min()),
            "last":  str(combined['date'].max()),
        }
        print(f"  ✅ {name:<12} {len(combined):>6} bars  (fetched {fetched} mo, cached {skipped} mo)")
    elif len(existing):
        print(f"  💾 {name:<12} fully cached ({len(existing)} bars) — no new fetch")
        manifest[name] = {"status": "ok", "rows": len(existing)}
    else:
        print(f"  ❌ {name:<12} no data returned (may need market-data subscription)")
        manifest[name] = {"status": "no_data"}

    save_manifest(manifest)   # save after each instrument (resumable)


def main():
    print("=" * 60)
    print("  IBKR HISTORICAL DATA FETCHER — 30min bars, cached")
    print("=" * 60)

    manifest = ensure_cache_dir()

    print(f"\n🔌 Connecting to IBKR at {HOST}:{PORT} (client {CLIENT_ID})...")
    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("   Is TWS/Gateway running with API enabled on this port?")
        return
    print("✅ Connected\n")

    CORE_NAMES = {"SPY", "VIX", "OIL", "GOLD", "BTC"}
    todo = [x for x in INSTRUMENTS if x[0] in CORE_NAMES] if CORE_ONLY else INSTRUMENTS

    mode = "5 CORE only" if CORE_ONLY else f"all {len(INSTRUMENTS)}"
    print(f"📥 Fetching {mode} instruments...")
    print(f"   ~{THROTTLE_SEC}s between requests to respect pacing limits\n")

    start = time.time()
    for name, contract, wts in todo:
        # Skip if already fully cached and marked ok
        if manifest.get(name, {}).get("status") == "ok" and \
           os.path.exists(os.path.join(CACHE_DIR, f"{name}_30min.csv")):
            # still call fetch_one — it appends only NEW months incrementally
            pass
        fetch_one(ib, name, contract, wts, manifest)

    ib.disconnect()
    elapsed = (time.time() - start) / 60
    print(f"\n✅ Done in {elapsed:.1f} min")
    print(f"💾 Cache: {CACHE_DIR}/  — reused automatically on next run")

    ok    = sum(1 for v in manifest.values() if v.get("status") == "ok")
    failed= sum(1 for v in manifest.values() if v.get("status") != "ok")
    print(f"\n   {ok} instruments cached, {failed} failed/skipped")
    if failed:
        print("   Failed ones fall back to yfinance daily in build_final_training_set.py")


if __name__ == "__main__":
    main()