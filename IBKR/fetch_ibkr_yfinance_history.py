"""
fetch_ibkr_yfinance_history.py
------------------------------
Downloads 30-min historical bars from Interactive Brokers, ONCE, and
caches them locally. Re-runs read the cache and only fetch missing data.

GAP FILL:
  IBKR futures (ContFuture) reach back ~2–5 years for 30-min bars.
  For any period that IBKR can't cover (too old), this script automatically
  falls back to yfinance daily bars and prepends them to the same cache CSV.
  This gives you full 8-year coverage: 30-min resolution where IBKR can
  provide it, daily resolution for the historical gap.

CACHING:
  market_data_cache/
    SPY_30min.csv ... (full history per instrument — may mix 30-min + 1d bars)
    _manifest.json  (tracks cached date ranges)
  Re-running = instant, offline. Only fetches gaps.

SETUP (one time):
  1. Open TWS or IB Gateway and log in
  2. In TWS: Edit → Global Config → API → Settings:
       ✅ Enable ActiveX and Socket Clients
       ✅ Socket port = 7497 (TWS paper) / 7496 (TWS live)
                       / 4002 (Gateway paper) / 4001 (Gateway live)
       ✅ Trust 127.0.0.1
  3. uv pip install ib_async pandas yfinance
  4. uv run python fetch_ibkr_yfinance_history.py 202606170000 --from 202411010000

USAGE:
  # Full backfill from Nov 2016 to Nov 2024 (all instruments):
  uv run python fetch_ibkr_yfinance_history.py 202411010000 --from 201611010000

  # Update cache through today:
  uv run python fetch_ibkr_yfinance_history.py 202606170000 --from 202411010000

  # Only core 5 instruments:
  uv run python fetch_ibkr_yfinance_history.py 202606170000 --from 202411010000 --core-only

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
def C_stock(sym):              return Stock(sym, "SMART", "USD")
def C_index(sym, exch="CBOE"): return Index(sym, exch, "USD")
def C_contfut(sym, exch):      return ContFuture(sym, exch, currency="USD")
def C_fx(pair):                return Forex(pair)

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

# ==========================================
# yfinance ticker map for gap fill
# Used when IBKR can't reach far enough back (futures pre-2022, crypto pre-2020).
# ==========================================
_YF_TICKERS = {
    "SPY":     "SPY",
    "VIX":     "^VIX",
    "OIL":     "CL=F",
    "GOLD":    "GC=F",
    "BTC":     "BTC-USD",
    "QQQ":     "QQQ",
    "DIA":     "DIA",
    "XLI":     "XLI",
    "XLF":     "XLF",
    "XLE":     "XLE",
    "COPPER":  "HG=F",
    "NATGAS":  "NG=F",
    "EUR_USD": "EURUSD=X",
    "USD_JPY": "JPY=X",
    "GBP_USD": "GBPUSD=X",
    "USD_CNY": "CNY=X",
    "USD_CAD": "CAD=X",
    "USD_MXN": "MXN=X",
    "USD_CHF": "CHF=X",
    "AUD_USD": "AUDUSD=X",
    "US10Y":   "ZN=F",
    "US2Y":    "ZT=F",
    "ETH":     "ETH-USD",
}

# Futures contract schedule per symbol: which months are actively traded.
_FUTURES_MONTHS = {
    "CL": "monthly",
    "NG": "monthly",
    "HG": "monthly",
    "GC": "even",          # Feb Apr Jun Aug Oct Dec
    "SI": "bimonthly",     # Mar May Jul Sep Dec
    "ZN": "quarterly",     # Mar Jun Sep Dec
    "ZT": "quarterly",
}

_EVEN_MONTHS   = {2, 4, 6, 8, 10, 12}
_SILVER_MONTHS = {3, 5, 7, 9, 12}
_QUARTERLY     = {3, 6, 9, 12}


def _gen_futures_expiries(sym: str, since: pd.Timestamp, until: pd.Timestamp) -> list[str]:
    """
    Auto-generate YYYYMM contract strings for `sym` covering [since, until].
    Includes one extra quarter before `since` and after `until` as buffer.
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
            (schedule == "even"      and m in _EVEN_MONTHS)   or
            (schedule == "bimonthly" and m in _SILVER_MONTHS) or
            (schedule == "quarterly" and m in _QUARTERLY)
        )
        if include:
            months.append(cur.strftime("%Y%m"))
        cur = cur + pd.DateOffset(months=1)

    return months


def _yfinance_gap_fill(name: str, since: pd.Timestamp, until: pd.Timestamp,
                       have: pd.DataFrame) -> pd.DataFrame:
    """
    Download daily bars from yfinance for any gap not covered by `have`.

    If `have` is non-empty:  fills [since → have['date'].min() - 1 day]
    If `have` is empty:      fills [since → until]

    Returns a DataFrame with columns [date, open, high, low, close, volume],
    or an empty DataFrame if there is no gap or yfinance has no data.
    """
    try:
        import yfinance as yf
    except ImportError:
        print(f"     ⚠️  yfinance not installed — skipping gap fill for {name}")
        return pd.DataFrame()

    ticker = _YF_TICKERS.get(name)
    if ticker is None:
        return pd.DataFrame()

    since_utc = since.tz_convert("UTC")
    until_utc = until.tz_convert("UTC")

    if len(have):
        have_dates = pd.to_datetime(have["date"], utc=True)
        gap_end_utc = have_dates.min() - pd.Timedelta(days=1)
    else:
        gap_end_utc = until_utc

    # No meaningful gap (< 2 days) → nothing to fill
    if gap_end_utc < since_utc + pd.Timedelta(days=2):
        return pd.DataFrame()

    yf_start = str(since_utc.date())
    yf_end   = str((gap_end_utc + pd.Timedelta(days=1)).date())

    print(f"     📈 {name}: yfinance daily gap fill "
          f"[{since_utc.date()} → {gap_end_utc.date()}]...")

    try:
        raw = yf.download(
            ticker,
            start=yf_start,
            end=yf_end,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            print(f"     ⚠️  {name}: yfinance returned no data")
            return pd.DataFrame()

        # Flatten MultiIndex columns (yfinance returns them for a single ticker too)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [str(c[0]).lower() for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]

        raw = raw.reset_index()

        # Normalise the date/datetime column name
        date_col = next((c for c in raw.columns if c in ("date", "datetime")), None)
        if date_col is None:
            print(f"     ⚠️  {name}: yfinance result has no date column — {raw.columns.tolist()}")
            return pd.DataFrame()
        raw = raw.rename(columns={date_col: "date"})
        raw["date"] = pd.to_datetime(raw["date"], utc=True)

        # Pick OHLCV columns (auto_adjust replaces close with adjusted close)
        col_map = {}
        for want in ("open", "high", "low", "close", "volume"):
            # exact match first, then 'adj close' → close
            if want in raw.columns:
                col_map[want] = want
            elif want == "close" and "adj close" in raw.columns:
                col_map["adj close"] = "close"

        df = raw[["date"] + list(col_map.keys())].rename(columns=col_map)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = float("nan")
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.dropna(subset=["close"])
        df = df[(df["date"] >= since_utc) & (df["date"] <= gap_end_utc + pd.Timedelta(days=1))]

        if df.empty:
            print(f"     ⚠️  {name}: yfinance gap data empty after filtering")
            return pd.DataFrame()

        print(f"     ✓ {name}: {len(df)} daily bars from yfinance "
              f"[{df['date'].min().date()} → {df['date'].max().date()}]")
        return df.reset_index(drop=True)

    except Exception as e:
        print(f"     ⚠️  {name}: yfinance error: {str(e)[:70]}")
        return pd.DataFrame()


def _merge_with_gap_fill(name: str, since: pd.Timestamp, until: pd.Timestamp,
                         ibkr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge `ibkr_df` with yfinance daily gap fill and return the combined,
    deduplicated, sorted DataFrame (not yet trimmed to [since, until]).
    If no gap fill is possible/needed, returns `ibkr_df` unchanged.
    """
    yf_fill = _yfinance_gap_fill(name, since, until, ibkr_df)
    if yf_fill.empty:
        return ibkr_df

    parts = ([ibkr_df] if len(ibkr_df) else []) + [yf_fill]
    merged = pd.concat(parts, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], utc=True)
    merged = (merged
              .drop_duplicates(subset=["date"])
              .sort_values("date")
              .reset_index(drop=True))
    return merged


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
        description="Download 30-min IBKR bars (+ yfinance daily gap fill) into market_data_cache/.",
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
        help="Fetch only the 5 core instruments (SPY VIX OIL GOLD BTC).",
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
        existing = pd.read_csv(cache_file, parse_dates=["date"])
        existing["date"] = pd.to_datetime(existing["date"], utc=True)
        cached_months = set(existing["date"].dt.strftime("%Y-%m").unique())
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
    # ContFuture FORBIDS endDateTime (Error 10339). Strategy:
    #   A) ContFuture with empty endDateTime, progressively longer durations.
    #      IBKR supports up to "5 Y" for 30-min bars. Data trimmed to [since, until].
    #   B) Dated expired contracts — only for the last ~2 years (IBKR retention window).
    #      Bail out after 5 consecutive qualify failures.
    #   C) yfinance daily gap fill — for any period still not covered after A+B.
    is_contfut = type(contract).__name__ == "ContFuture"
    if is_contfut:
        new_frames = []
        c = qualified[0] if isinstance(qualified, list) else contract

        # ---- A) continuous contract ----
        cache_covers = False
        if len(existing):
            ec_min = existing["date"].min()
            ec_max = existing["date"].max()
            if (ec_min <= since_utc + pd.Timedelta(days=5) and
                    ec_max >= until_utc - pd.Timedelta(days=5)):
                cache_covers = True
                print(f"     💾 {name}: cache already covers [{since.date()} → {until.date()}],"
                      f" skipping continuous pull")

        if not cache_covers:
            CONTFUT_DURATIONS = ["5 Y", "3 Y", "2 Y", "365 D", "270 D", "180 D"]
            for dur in CONTFUT_DURATIONS:
                try:
                    bars = ib.reqHistoricalData(
                        c, endDateTime="", durationStr=dur,
                        barSizeSetting=BAR_SIZE, whatToShow=what_to_show,
                        useRTH=0, formatDate=2, timeout=300,
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

        # ---- B) dated expired contracts — only within IBKR's 2-year retention window ----
        base_sym = contract.symbol
        exch     = contract.exchange
        expiries = _gen_futures_expiries(base_sym, since, until)

        frames_so_far = ([existing] if len(existing) else []) + new_frames
        if frames_so_far:
            tmp = pd.concat(frames_so_far, ignore_index=True)
            tmp["date"] = pd.to_datetime(tmp["date"], utc=True)
            earliest_have = tmp["date"].min()
        else:
            earliest_have = until_utc

        # Hard cutoff: IBKR only retains expired contract definitions for ~2 years.
        ibkr_cutoff = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=2)).strftime("%Y%m")

        gap_months, n_too_old = [], 0
        for ym in expiries:
            ym_ts = pd.Timestamp(ym + "01", tz="UTC")
            if ym_ts < earliest_have - pd.Timedelta(days=45):
                if ym >= ibkr_cutoff:
                    gap_months.append(ym)
                else:
                    n_too_old += 1

        if n_too_old:
            print(f"     ⏭️  {name}: {n_too_old} gap months before "
                  f"{ibkr_cutoff[:4]}-{ibkr_cutoff[4:6]} skipped "
                  f"(IBKR only retains ~2 years of expired definitions)")

        if gap_months:
            print(f"     📅 {name}: {len(gap_months)} gap months to try via dated contracts "
                  f"({gap_months[0]} → {gap_months[-1]})")
            consec_qualify_fails = 0
            for ym in gap_months:
                ym_month = f"{ym[:4]}-{ym[4:6]}"
                if ym_month in cached_months:
                    continue
                if consec_qualify_fails >= 5:
                    print(f"     ⏭️  {name}: 5 consecutive qualify failures — stopping.")
                    break
                try:
                    fut = Future(symbol=base_sym, exchange=exch, currency="USD",
                                 lastTradeDateOrContractMonth=ym, includeExpired=True)
                    q = ib.qualifyContracts(fut)
                    # qualifyContracts returns [None] (not []) on Error 200
                    if not q or q[0] is None:
                        consec_qualify_fails += 1
                        continue
                    consec_qualify_fails = 0
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
                            "date": b.date, "open": b.open, "high": b.high,
                            "low": b.low, "close": b.close, "volume": b.volume,
                        } for b in bars]))
                        print(f"     ✓ {name}: {len(bars)} bars from expired {ym}")
                    time.sleep(THROTTLE_SEC)
                except Exception as e:
                    consec_qualify_fails += 1
                    print(f"     ⚠️  {name} expired {ym}: {str(e)[:45]}")
                    time.sleep(THROTTLE_SEC)
        else:
            if not n_too_old:
                print(f"     ✓ {name}: ContFuture already covers window — skipping dated contracts")

        # ---- Combine IBKR data ----
        if new_frames:
            combined = pd.concat(
                ([existing] if len(existing) else []) + new_frames, ignore_index=True
            )
            combined["date"] = pd.to_datetime(combined["date"], utc=True)
            combined = (combined
                        .drop_duplicates(subset=["date"])
                        .sort_values("date")
                        .reset_index(drop=True))
        elif len(existing):
            combined = existing.copy()
            combined["date"] = pd.to_datetime(combined["date"], utc=True)
        else:
            combined = pd.DataFrame()

        # ---- C) yfinance daily gap fill ----
        combined = _merge_with_gap_fill(name, since, until, combined)

        # Trim to requested window and save
        if len(combined):
            combined = combined[
                (combined["date"] >= since_utc) & (combined["date"] <= until_utc)
            ].reset_index(drop=True)

        if len(combined):
            combined.to_csv(cache_file, index=False)
            manifest[name] = {
                "status": "ok", "rows": len(combined),
                "first": str(combined["date"].min()),
                "last":  str(combined["date"].max()),
            }
            span = f"{combined['date'].min().date()} → {combined['date'].max().date()}"
            src  = "fetched" if new_frames else "cached"
            print(f"  ✅ {name:<12} {len(combined):>6} bars  ({span})  [{src}]")
        else:
            print(f"  ❌ {name:<12} no data — futures subscription or yfinance ticker may be needed")
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
        mkey = m.strftime("%Y-%m")
        if mkey in cached_months:
            skipped += 1
            continue

        if consecutive_empty >= 2 and fetched == 0:
            print(f"     ⏭️  {name}: no data on first 2 months — "
                  f"likely unsubscribed or pre-launch, skipping rest")
            failed_months.append("(bailed early)")
            break

        next_month = m + pd.offsets.MonthBegin(1)
        end_dt = (
            min(next_month, until + pd.Timedelta(days=1))
            .strftime("%Y%m%d %H:%M:%S")
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
                        "date": b.date, "open": b.open, "high": b.high,
                        "low":  b.low,  "close": b.close, "volume": b.volume,
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

    # Build combined from IBKR data
    if new_frames:
        combined = (pd.concat([existing] + new_frames, ignore_index=True)
                    if len(existing) else pd.concat(new_frames, ignore_index=True))
        combined["date"] = pd.to_datetime(combined["date"], utc=True)
        combined = (combined
                    .drop_duplicates(subset=["date"])
                    .sort_values("date")
                    .reset_index(drop=True))
    elif len(existing):
        combined = existing.copy()
        combined["date"] = pd.to_datetime(combined["date"], utc=True)
    else:
        combined = pd.DataFrame()

    # yfinance daily gap fill — covers BTC/ETH pre-PAXOS, any early bail-out months
    combined = _merge_with_gap_fill(name, since, until, combined)

    # Trim and save
    if len(combined):
        combined = combined[
            (combined["date"] >= since_utc) & (combined["date"] <= until_utc)
        ].reset_index(drop=True)

    if len(combined):
        combined.to_csv(cache_file, index=False)
        manifest[name] = {
            "status": "ok", "rows": len(combined),
            "first": str(combined["date"].min()),
            "last":  str(combined["date"].max()),
        }
        print(f"  ✅ {name:<12} {len(combined):>6} bars  "
              f"(fetched {fetched} mo, cached {skipped} mo)")
    else:
        print(f"  ❌ {name:<12} no data returned (IBKR subscription + yfinance both failed)")
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
    print("  IBKR + yfinance HISTORICAL DATA FETCHER — 30min/daily bars")
    print("=" * 64)
    print(f"  Since : {since:%Y-%m-%d %H:%M %Z}")
    print(f"  Until : {until:%Y-%m-%d %H:%M %Z}")
    print(f"  Span  : ~{span_years:.1f} years ({span_months} months)")
    print(f"  Port  : {args.port}   Host: {args.host}   ClientID: {args.client_id}")
    print(f"  Gap fill: yfinance daily bars where IBKR can't reach")
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
    print(f"   ~{THROTTLE_SEC}s between IBKR requests to respect pacing limits\n")

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
        print("   Check above for ❌ lines — these had no IBKR data and no yfinance fallback")


if __name__ == "__main__":
    main()
