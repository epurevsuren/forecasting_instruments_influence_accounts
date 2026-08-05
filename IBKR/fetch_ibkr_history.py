"""
fetch_ibkr_history.py
---------------------
Downloads historical bars from Interactive Brokers and caches them
locally as CSV files. Re-runs read the cache and only fetch what is missing.

Bar sizes : 1m, 5m, 15m, 30m  (each gets its own CSV)

DEFAULT BEHAVIOUR (no arguments):
  Fetches 1m, 15m AND 30m bars. For each instrument and each of those sizes,
  reads the latest date already in that size's CSV and fetches forward to now.
  New instruments with no CSV start from now - 2 years. (5m exists but is only
  fetched when asked for explicitly via --bar-size.)
  Crypto (BTC, ETH) is EXCLUDED by default — it is fetched from Binance
  (fetch_binance_history.py). Fetch it from IBKR only via --instruments BTC.

WITH ARGUMENTS:
  python fetch_ibkr_history.py --bar-size 15m --from 20241101 --until 20260101
  Backfills the window, deciding coverage by DATE: a month is fetched if it has
  any missing business day inside the window, so interior HOLES (e.g. a run that
  stopped mid-June while July already exists) are filled — not skipped. Existing
  bars are kept for dates already present. --bar-size accepts several sizes
  (e.g. --bar-size 1m 30m).

REPAIRING DISTORTED DATA:
  python fetch_ibkr_history.py --refetch --from 20260601 --instruments SPY VIX --bar-size 1m
  --refetch CLEARS existing bars in [--from, --until] first, then re-fetches them
  fresh, so the new bars REPLACE the old (distorted) ones.

CSV naming  : {name}_{N}min.csv  e.g. SPY_30min.csv, SPY_1min.csv

Edit the CONFIG block below to match your TWS/Gateway setup.
"""

import argparse
import os
import json
import time
import datetime
import pandas as pd
import bars as barsdb   # DuckDB engine: read / dedup-write cache CSVs. Aliased because
                        # `bars` is a local var (the fetched bar list) inside fetch_one.
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

# Bar sizes fetched when --bar-size is not given (each in its own CSV).
# 5m is intentionally excluded — available on demand via --bar-size 5m.
DEFAULT_BAR_SIZES = ["1m", "15m", "30m"]

# Set True to fetch ONLY the 5 core instruments (~20 min).
CORE_ONLY = False

# ==========================================
# INSTRUMENT -> IBKR CONTRACT MAPPING
# ==========================================
def C_stock(sym):              return Stock(sym, "SMART", "USD")
def C_index(sym, exch="CBOE"): return Index(sym, exch, "USD")
def C_contfut(sym, exch):      return ContFuture(sym, exch, currency="USD")
def C_fx(pair):                return Forex(pair)

# Instruments loaded DYNAMICALLY from ../DP/instruments.json (master registry).
# Each entry's "ibkr" object: kind (stock|index|contfut|fx|crypto), symbol,
# exchange, what (whatToShow), core (True = member of the --core fetch set).
# Add or remove an instrument THERE — no code edits needed.
import json as _json
_INSTRUMENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DP", "instruments.json")


def _build_contract(spec):
    kind = spec["kind"]
    if kind == "stock":   return C_stock(spec["symbol"])
    if kind == "index":   return C_index(spec["symbol"], spec.get("exchange", "CBOE"))
    if kind == "contfut": return C_contfut(spec["symbol"], spec["exchange"])
    if kind == "fx":      return C_fx(spec["symbol"])
    if kind == "crypto":  return Crypto(spec["symbol"], spec.get("exchange", "PAXOS"), "USD")
    raise ValueError(f"Unknown ibkr contract kind: {kind!r}")


with open(_INSTRUMENTS_FILE, encoding="utf-8") as _f:
    _REGISTRY = _json.load(_f)["instruments"]

INSTRUMENTS = [(name, _build_contract(v["ibkr"]), v["ibkr"]["what"])
               for name, v in _REGISTRY.items() if "ibkr" in v]

CORE_NAMES = {name for name, v in _REGISTRY.items()
              if v.get("ibkr", {}).get("core")}

# Crypto is fetched from Binance (fetch_binance_history.py), NOT IBKR, so it is
# EXCLUDED from the default and --core-only runs. Still fetchable on demand via
# an explicit --instruments BTC / ETH.
CRYPTO_NAMES = {name for name, v in _REGISTRY.items()
                if v.get("ibkr", {}).get("kind") == "crypto"}

# Bare futures symbols (investing.com style) -> IBKR exchange. Ad-hoc
# --instruments tokens not in instruments.json resolve through this first —
# futures are the main ad-hoc use case.
FUTURES_EXCHANGES = {
    # NYMEX energy / platinum group
    "CL": "NYMEX", "NG": "NYMEX", "HO": "NYMEX", "RB": "NYMEX",
    "PA": "NYMEX", "PL": "NYMEX",
    # COMEX metals
    "GC": "COMEX", "SI": "COMEX", "HG": "COMEX",
    # CBOT rates / grains / mini Dow
    "ZN": "CBOT", "ZT": "CBOT", "ZF": "CBOT", "ZB": "CBOT", "UB": "CBOT",
    "ZC": "CBOT", "ZS": "CBOT", "ZW": "CBOT", "ZL": "CBOT", "ZM": "CBOT",
    "YM": "CBOT",
    # CME equity index / FX / livestock
    "ES": "CME", "NQ": "CME", "RTY": "CME", "LE": "CME", "HE": "CME", "GF": "CME",
    "6E": "CME", "6J": "CME", "6B": "CME", "6A": "CME", "6C": "CME",
    "6M": "CME", "6N": "CME",
    # CBOE futures
    "VX": "CFE",
}
_FUT_QUALIFY_EXCHANGES = ("NYMEX", "COMEX", "CBOT", "CME", "CFE")


def resolve_symbol(ib, sym):
    """
    Resolve a bare ad-hoc symbol (investing.com style: SI, ZC, ES, AAPL...)
    to (name, contract, whatToShow). Futures first — the main ad-hoc case:
      1. FUTURES_EXCHANGES map            (SI -> ContFuture SI @ COMEX)
      2. live ContFuture qualification against the major exchanges
      3. fallback: Stock @ SMART
    The CSV lands in the same market_data_cache/ as registry instruments, so
    the bars are usable anywhere — inside or outside this project.
    """
    s = sym.upper()
    if s in FUTURES_EXCHANGES:
        return (s, C_contfut(s, FUTURES_EXCHANGES[s]), "TRADES")
    for exch in _FUT_QUALIFY_EXCHANGES:
        try:
            c = C_contfut(s, exch)
            if ib.qualifyContracts(c):
                print(f"  🔎 {s}: resolved as future on {exch}")
                return (s, c, "TRADES")
        except Exception:
            pass
    print(f"  🔎 {s}: not a known future — fetching as stock (SMART)")
    return (s, C_stock(s), "TRADES")


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
    """Return the latest date in the cache CSV for (name, suffix), or None.
    Uses DuckDB (bars.latest_date) — reads only max(date), no full load."""
    try:
        return barsdb.latest_date(os.path.join(CACHE_DIR, f"{name}_{suffix}.csv"))
    except Exception:
        return None


# ==========================================
# Per-instrument fetcher
# ==========================================
def _ibkr_duration(span) -> str:
    """IBKR durationStr sized to `span` (a Timedelta, + a 60s buffer), so a
    default resume fetches ONLY [last cached bar -> now] instead of a fixed week/
    month. <= 1 day -> request in seconds; otherwise in days."""
    total = int(span.total_seconds())
    secs = max(60, total + 60)                  # small buffer so the last bar is covered
    if secs <= 86400:                           # <= 1 day -> request in seconds (minimal)
        return f"{secs} S"
    days = (total + 86399) // 86400             # ceil to whole days
    return f"{days} D"


def _month_days_covered(existing_days, m, since, until) -> bool:
    """True if EVERY business day of month `m` that falls inside [since, until]
    already has >=1 bar in the cache. Used to decide, at DATE granularity, whether
    a month can be skipped — so an interior hole (e.g. June 19-30 missing while
    July exists) is detected and re-fetched instead of being skipped as 'cached'.
    Business days (Mon-Fri) are a trading-calendar proxy; a missing holiday just
    triggers a harmless re-fetch of that month."""
    lo = max(m.normalize(), since.floor("D"))
    m_end = (m + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
    hi = min(m_end.normalize(), until.floor("D"))
    if hi < lo:
        return True
    for d in pd.bdate_range(lo, hi):
        if d not in existing_days:
            return False
    return True


def fetch_one(ib, name, contract, what_to_show, manifest,
              since, until, suffix, bar_size_str, chunk, throttle, contfut_durations,
              default_mode=False, refetch=False):
    """
    Fetch bars for one instrument over [since, until].
    `since` and `until` are UTC Timestamps.

    Coverage is decided by DATE, not by month:
      * default mode  -> since = last cached bar; the run resumes from there and
        always (re)fetches through the current window, filling the partial tail.
      * --from backfill -> a month is skipped only if EVERY business day it spans
        inside [since, until] is already cached (so interior holes are re-fetched).
      * --refetch -> existing bars in [since, until] are cleared first, so the
        freshly-fetched bars REPLACE them (use to repair distorted data).
    Dedup on write keeps existing bars for overlapping dates (refetch pre-clears
    the window so its fresh bars win).
    """
    cache_file   = os.path.join(CACHE_DIR, f"{name}_{suffix}.csv")
    manifest_key = f"{name}_{suffix}"

    # ── Load existing cache ───────────────────────────────────────────────────
    if os.path.exists(cache_file):
        existing = barsdb.read(cache_file)          # DuckDB read (tz-aware UTC date)
        if existing is None or len(existing) == 0 or "date" not in existing.columns:
            existing = pd.DataFrame()
            cached_months = set()
        else:
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

            # Default mode RESUMES from the last cached bar: the request for the
            # current window is sized to exactly [last bar -> now] (see
            # _ibkr_duration), and no month is skipped merely because it holds
            # some bars — coverage is decided per business day, so interior holes
            # are re-fetched. A historical --from backfill sets default_mode=False.
            resume_msg = ", resuming from last cached bar" if default_mode else ""

            print(f"  ℹ️  {name} ({suffix}): cache {cache_earliest} → {cache_latest} "
                  f"({len(existing):,} bars){gap_msg}{resume_msg}")
    else:
        existing      = pd.DataFrame()
        cached_months = set()

    # ── --refetch: clear existing bars in [since, until] so fresh ones replace ──
    if refetch and len(existing):
        in_win = (existing["date"] >= since) & (existing["date"] <= until)
        n_drop = int(in_win.sum())
        if n_drop:
            existing = existing[~in_win].reset_index(drop=True)
            cached_months = set(existing["date"].dt.strftime("%Y-%m").unique()) if len(existing) else set()
            print(f"     ♻️  --refetch: cleared {n_drop:,} {name} {suffix} bars in "
                  f"[{since.date()} .. {until.date()}] — will re-fetch fresh")

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

        def _flush(bar_list, before=None, after=None):
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
            if after is not None:                    # --refetch: keep only the requested window
                df = df[df["date"] >= after]
            if df.empty:
                return
            base = [combined] if len(combined) else []
            combined = pd.concat(base + [df], ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"], utc=True)
            combined = (combined.drop_duplicates(subset=["date"], keep="last")
                        .sort_values("date").reset_index(drop=True))
            try:
                barsdb.write_csv(cache_file, combined, list(combined.columns))  # DuckDB dedup-write
            except KeyboardInterrupt:
                print(f"\n⚠️  Ctrl+C -- saved {len(combined):,} bars, stopping.")
                raise
            got_new = True

        # Path A: fetch recent data via ContFuture (endDateTime="" = now).
        # Skip only if the cache already reaches ~now.
        #
        # BUG FIXED 2026-08-05: this window was 10 DAYS, so on a daily run the
        # futures (OIL, GOLD, COPPER, NATGAS, US10Y, US2Y) could NEVER update —
        # a cache 2 days old counted as "fresh" and the fetch was skipped, while
        # the stock/FX path updated normally. The futures would have to rot for
        # 10+ days before the fetcher touched them. Cost of the fix is one
        # ContFuture request per instrument per run (~11s each).
        FRESH_DAYS = float(os.environ.get("IBKR_FRESH_DAYS", "1"))
        _cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=FRESH_DAYS)
        cache_is_fresh = (len(existing) > 0 and existing["date"].max() >= _cut)
        if cache_is_fresh or refetch:
            if refetch:
                print(f"     ♻️  {name}: --refetch → refilling [{since.date()} .. {until.date()}] "
                      f"from dated contracts (skipping the recent ContFuture fetch)")
            else:
                _age = (pd.Timestamp.now(tz="UTC") - existing["date"].max()).total_seconds() / 86400
                print(f"     💾 {name}: cached to {existing['date'].max().date()} "
                      f"({_age:.1f}d old, within {FRESH_DAYS:g}d) — skipping recent fetch")
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

        base_sym = contract.symbol
        exch     = contract.exchange

        if refetch:
            # --refetch cleared [since, until]; refill it directly from the dated
            # contracts that traded during that window. Widen the delivery-month
            # search ~one quarter each side so a contract ROLL inside the window is
            # covered (e.g. ZN rolls Mar->Jun); every fetched bar is then clipped to
            # [since, until] and dedup keeps the LATEST-expiry (post-roll) contract
            # on any overlap, matching the continuous-future stitch. The cached-month
            # skip is bypassed since those months are being replaced on purpose.
            expiries = _expired_months(base_sym,
                                       since - pd.Timedelta(days=95),
                                       until + pd.Timedelta(days=95))
            skip_cached_months = False
            if expiries:
                print(f"     🔁 {name}: refetching {len(expiries)} dated contract(s) "
                      f"covering [{since.date()} .. {until.date()}] "
                      f"({expiries[0]} -> {expiries[-1]})")
        else:
            gap_end  = cf_earliest - pd.Timedelta(days=1)
            expiries = _expired_months(base_sym, since, gap_end)
            skip_cached_months = True
            if expiries:
                print(f"     🔙 {name}: backfilling {len(expiries)} expired contracts "
                      f"({expiries[0]} -> {expiries[-1]})")

        # Smaller bar sizes need shorter backfill windows to stay under IBKR bar limits.
        backfill_dur = "30 D" if suffix in ("1min", "5min") else "90 D"

        for ym in expiries:
            ym_month = f"{ym[:4]}-{ym[4:6]}"
            if skip_cached_months and ym_month in cached_months:
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
                    if refetch:
                        # clip to the requested window; dedup keep='last' + ascending
                        # expiry order → the post-roll contract wins any overlap.
                        _flush(bars, after=since)
                    else:
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

    # Days already present, for DATE-level (not month-level) skip decisions.
    existing_days = set(existing["date"].dt.floor("D")) if len(existing) else set()
    until_mkey    = until.strftime("%Y-%m")

    # 1m bars use weekly sub-chunking: "1 W" chunk is too small for a full month
    # in one request, so we slice the month into weekly windows and combine.
    use_weekly_sub = chunk.endswith(" W")

    for m in months:
        mkey = m.strftime("%Y-%m")

        # Skip a month ONLY when it is already covered on every business day it
        # spans in-window (date granularity — an interior hole is NOT skipped).
        # Never skip on --refetch; in default mode never skip the current window's
        # month, so today's partial tail is always topped up.
        if not refetch:
            force = default_mode and mkey == until_mkey
            if not force and _month_days_covered(existing_days, m, since, until):
                skipped += 1
                continue

        if consecutive_empty >= 2 and fetched == 0:
            print(f"     ⏭️  {name}: no data on first 2 months -- likely unsubscribed, skipping rest")
            failed_months.append("(bailed early)")
            break

        next_month = m + pd.offsets.MonthBegin(1)
        limit_ts   = min(next_month, until + pd.Timedelta(days=1))

        # (start, end) sub-windows covering [max(m, since), limit_ts]. The first
        # month starts at `since` (not month start), and each request's duration
        # is SIZED TO ITS WINDOW (not a fixed week/month) — so a default resume
        # pulls only [last cached bar -> now]. 1m is sliced weekly (IBKR bar cap).
        win_start = max(m, since)
        if use_weekly_sub:
            sub_windows = []
            cur_w = win_start
            while cur_w < limit_ts:
                end_w = min(cur_w + pd.Timedelta(weeks=1), limit_ts)
                sub_windows.append((cur_w, end_w))
                cur_w = end_w
        else:
            sub_windows = [(win_start, limit_ts)]

        print(f"     → {name} ({suffix}) {mkey} ...", end="", flush=True)
        month_bars = []

        for start_w, end_w in sub_windows:
            if end_w <= start_w:
                continue
            end_dt  = end_w.strftime("%Y%m%d %H:%M:%S")
            dur_str = _ibkr_duration(end_w - start_w)
            for attempt in range(1, 4):
                try:
                    bars = ib.reqHistoricalData(
                        qualified[0] if isinstance(qualified, list) else contract,
                        endDateTime=end_dt, durationStr=dur_str,
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
                barsdb.write_csv(cache_file, combined, list(combined.columns))  # DuckDB dedup-write
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
        n_new = len(combined) - len(existing)
        manifest[manifest_key] = {"status": "ok", "rows": len(combined),
                                  "first": str(combined["date"].min()),
                                  "last":  str(combined["date"].max())}
        print(f"  ✅ {name:<12} {len(combined):>6} bars  "
              f"(+{n_new:,} new, {skipped} mo already covered)")
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
    ap.add_argument("--bar-size", dest="bar_sizes", nargs="+", default=None,
                    choices=list(BAR_CONFIG.keys()), metavar="SIZE",
                    help="Bar size(s) to fetch, space-separated. Default: 1m 15m 30m "
                         "(each stored in its own CSV). E.g. --bar-size 5m  or  --bar-size 1m 30m.")
    ap.add_argument("--from", dest="since", metavar="YYYYMMDD[hhmm]", default=None,
                    help="Start of fetch window. Default: per-instrument latest cached date.")
    ap.add_argument("--until", metavar="YYYYMMDD[hhmm]", default=None,
                    help="End of fetch window. Default: now.")
    ap.add_argument("--instruments", nargs="+", metavar="SYMBOL", default=None,
                    help="Fetch only these. Registry names (instruments.json) work "
                         "as-is (OIL GOLD SPY). ANY other symbol works bare, "
                         "investing.com style — futures resolve first (SI -> COMEX "
                         "silver, ZC -> CBOT corn, ES -> CME e-mini), unknown "
                         "symbols fall back to stock. Bars land in the same cache.")
    ap.add_argument("--core-only", action="store_true",
                    help="Fetch only the core instruments (SPY VIX OIL GOLD; crypto excluded — "
                         "it comes from Binance).")
    ap.add_argument("--refetch", action="store_true",
                    help="Clear existing bars in the [--from, --until] window and re-fetch them "
                         "fresh (repairs distorted/corrupted data). Pair with --from and "
                         "--instruments, e.g. --refetch --from 20260601 --instruments SPY VIX --bar-size 1m.")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--client-id", type=int, default=CLIENT_ID)
    args = ap.parse_args()

    bar_sizes = args.bar_sizes if args.bar_sizes else DEFAULT_BAR_SIZES
    sizes_str = " ".join(bar_sizes)

    global_until = parse_stamp(args.until) if args.until else pd.Timestamp.now(tz="UTC")
    global_since = parse_stamp(args.since) if args.since else None

    print("=" * 60)
    print(f"  IBKR HISTORICAL DATA FETCHER -- {sizes_str} bars")
    print("=" * 60)
    print(f"  Bar sizes: {sizes_str}  ->  one CSV each per instrument")
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
        # Explicit selection — fetch EXACTLY what's named (crypto included if
        # asked). Registry names resolve from instruments.json; any OTHER bare
        # symbol resolves investing.com style via resolve_symbol() (futures first).
        registry = {x[0]: x for x in INSTRUMENTS}
        todo = []
        for token in args.instruments:
            name = token.upper()
            if name in registry:
                todo.append(registry[name])
            else:
                todo.append(resolve_symbol(ib, name))
        mode = ", ".join(x[0] for x in todo)
    elif args.core_only or CORE_ONLY:
        # core set minus crypto (crypto comes from Binance)
        todo = [x for x in INSTRUMENTS if x[0] in CORE_NAMES and x[0] not in CRYPTO_NAMES]
        mode = "CORE only"
    else:
        # default: everything EXCEPT crypto (crypto comes from Binance)
        todo = [x for x in INSTRUMENTS if x[0] not in CRYPTO_NAMES]
        mode = f"all {len(todo)} (non-crypto)"

    # Transparency: on any non-explicit run, note the crypto we deliberately skip.
    if not args.instruments and CRYPTO_NAMES:
        _c = sorted(CRYPTO_NAMES)
        print(f"  ⏭️  skipping crypto ({', '.join(_c)}) — fetched via Binance "
              f"(fetch_binance_history.py); use --instruments {_c[0]} to force IBKR.")
    print(f"📥 Fetching {mode}  ({sizes_str} bars)\n")

    t0 = time.time()
    stopped = False
    # Outer loop over bar sizes so a single IBKR connection fetches all of them.
    for bar_size in bar_sizes:
        bar_size_str, suffix, chunk, throttle, contfut_durations = BAR_CONFIG[bar_size]
        print("─" * 60)
        print(f"  ▶  {bar_size} bars  ->  {suffix}.csv   (~{throttle}s between requests)")
        print("─" * 60)

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
                          suffix, bar_size_str, chunk, throttle, contfut_durations,
                          default_mode=(global_since is None), refetch=args.refetch)
            except KeyboardInterrupt:
                print("\n🛑 Stopped by user.")
                stopped = True
                break
        if stopped:
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
