"""
bars.py — DuckDB engine for the market_data_cache bar CSVs
==========================================================
The bars stay on disk as market_data_cache/<NAME>_<bar>.csv (portable, git-
friendly, unchanged for every other tool). This module just uses DuckDB as the
ENGINE for the heavy operations — reading, resampling, dedup/merge and writing —
which is far faster and lower-RAM than pandas read_csv + concat + drop_duplicates
+ resample on multi-million-row files (BTC 1-min ~7.6M rows).

Verified against the pandas paths on real data:
  * time_bucket == pandas resample(label='left', closed='left')  (OHLC 0 delta)
  * sentinel-aware VWAP/barCount == recompute_30min logic         (avg Δ<=1e-6)
  * write_csv round-trip is BYTE-IDENTICAL to the existing CSVs

Typical use (drop-in for the slow pandas bits):
    import bars
    df   = bars.read(bars.path("GOLD", "1min"))          # fast CSV -> DataFrame
    r30  = bars.resample(bars.path("GOLD", "1min"), 30,   # 1m -> 30m OHLC (+sentinels)
                         cols=["date","open","high","low","close","volume","average","barCount"])
    r30v = bars.resample(bars.path("GOLD", "15min"), 30,  # real 30m VWAP+barCount from 15m
                         cols=[...,"average","barCount"], vwap_col="average")
    out  = bars.union_fill(existing, incoming, cols, prefer_existing=True)
    bars.write_csv(bars.path("GOLD", "30min"), out, cols)
"""
import os
import duckdb
import pandas as pd

_HERE     = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "market_data_cache")

BAR_MIN = {"1min": 1, "5min": 5, "15min": 15, "30min": 30}
SENT_VOL, SENT_AVG, SENT_BC = -1.0, -1.0, -1


def connect():
    """A DuckDB connection pinned to UTC (so TIMESTAMPTZ math stays in UTC)."""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    return con


def path(name: str, bar: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}_{bar}.csv")


def _src(con, source) -> str:
    """SQL table expression for `source`: a CSV path (read_csv_auto) or a pandas
    DataFrame (registered as a temp view `_bars_src`)."""
    if isinstance(source, str):
        return f"(SELECT * FROM read_csv_auto('{source}', null_padding=true))"
    con.register("_bars_src", source)
    return "(SELECT * FROM _bars_src)"


def read(source, con=None) -> "pd.DataFrame | None":
    """CSV path (or DataFrame) -> pandas DataFrame with tz-aware UTC `date`."""
    if isinstance(source, str) and not os.path.exists(source):
        return None
    own = con is None
    con = con or connect()
    try:
        if isinstance(source, str):
            df = con.execute(f"SELECT * FROM read_csv_auto('{source}', null_padding=true)").df()
        else:
            df = source.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df
    finally:
        if own:
            con.close()


def resample(source, out_min: int, cols: list, vwap_col: str = None, con=None) -> pd.DataFrame:
    """Resample finer bars in `source` to `out_min`-minute bars using DuckDB's
    time_bucket (== IBKR label='left'/closed='left'): open=first, high=max,
    low=min, close=last. Empty buckets are naturally omitted.

    If `vwap_col` is given AND present in the source, the target vwap column gets
    a sentinel-aware volume-weighted VWAP and `barCount` gets a sentinel-aware
    sum (rebuilds a real 30-min VWAP/count from 15-min). Otherwise any
    volume/average/wap/barCount columns in `cols` are filled with the -1
    sentinels (FX / quote-only fills). Returns a DataFrame with exactly `cols`.
    """
    own = con is None
    con = con or connect()
    try:
        src = _src(con, source)
        sel = [
            f"time_bucket(INTERVAL '{out_min} minutes', date::TIMESTAMPTZ) AS date",
            "arg_min(open,  date::TIMESTAMPTZ) AS open",
            "max(high) AS high",
            "min(low)  AS low",
            "arg_max(close, date::TIMESTAMPTZ) AS close",
        ]
        if vwap_col and vwap_col in cols:
            sel.append(
                f'round(CASE WHEN count(*) FILTER ("{vwap_col}">=0)=0 THEN -1.0 '
                f'WHEN sum(volume) FILTER (volume>0 AND "{vwap_col}">=0) > 0 '
                f'THEN sum("{vwap_col}"*volume) FILTER (volume>0 AND "{vwap_col}">=0) '
                f'/ sum(volume) FILTER (volume>0 AND "{vwap_col}">=0) '
                f'ELSE avg("{vwap_col}") FILTER ("{vwap_col}">=0) END, 6) AS "{vwap_col}"'
            )
            if "barCount" in cols:
                sel.append('CAST(CASE WHEN count(*) FILTER ("barCount">=0)=0 THEN -1 '
                           'ELSE sum("barCount") FILTER ("barCount">=0) END AS BIGINT) AS "barCount"')
        q = f"SELECT {', '.join(sel)} FROM {src} GROUP BY 1 ORDER BY 1"
        out = con.execute(q).df()
        out["date"] = pd.to_datetime(out["date"], utc=True)
        for c in cols:                          # fill columns not produced above
            if c in out.columns:
                continue
            out[c] = SENT_BC if c == "barCount" else (SENT_VOL if c == "volume" else SENT_AVG)
        if "barCount" in cols:
            out["barCount"] = out["barCount"].fillna(SENT_BC).astype("int64")
        return out[cols]
    finally:
        try:
            con.unregister("_bars_src")
        except Exception:
            pass
        if own:
            con.close()


def union_fill(existing, incoming, cols, prefer_existing=True, con=None) -> pd.DataFrame:
    """Union `existing` + `incoming` (DataFrames or None) on `date`, keeping the
    preferred side on overlapping timestamps. Returns a DataFrame with `cols`.
    Uses DuckDB anti-join for speed on large frames."""
    def _norm(df):
        if df is None or len(df) == 0:
            return None
        df = df.reindex(columns=cols).copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df
    a, b = _norm(existing), _norm(incoming)
    if a is None and b is None:
        return pd.DataFrame(columns=cols)
    if a is None:
        return b.sort_values("date").reset_index(drop=True)
    if b is None:
        return a.sort_values("date").reset_index(drop=True)

    keep, add = (a, b) if prefer_existing else (b, a)   # `keep` wins overlaps
    own = con is None
    con = con or connect()
    try:
        con.register("_keep", keep)
        con.register("_add", add)
        out = con.execute(
            "SELECT * FROM _keep UNION ALL "
            "SELECT * FROM _add WHERE date NOT IN (SELECT date FROM _keep) "
            "ORDER BY date"
        ).df()
        out["date"] = pd.to_datetime(out["date"], utc=True)
        if "barCount" in cols:
            out["barCount"] = out["barCount"].fillna(SENT_BC).astype("int64")
        return out[cols]
    finally:
        for v in ("_keep", "_add"):
            try:
                con.unregister(v)
            except Exception:
                pass
        if own:
            con.close()


def latest_date(source, con=None):
    """Fast max(date) of a cache CSV (or DataFrame) as a tz-aware UTC Timestamp,
    or None. Reads only the date column via DuckDB — no full load."""
    if isinstance(source, str) and not os.path.exists(source):
        return None
    own = con is None
    con = con or connect()
    try:
        src = _src(con, source)
        r = con.execute(f"SELECT max(date::TIMESTAMPTZ) FROM {src}").fetchone()[0]
        return pd.Timestamp(r) if r is not None else None
    finally:
        try:
            con.unregister("_bars_src")
        except Exception:
            pass
        if own:
            con.close()


def write_csv(dst: str, source, cols: list, con=None) -> None:
    """Write `cols` from `source` (DataFrame or CSV path) to `dst`, sorted by
    date and DEDUPED by date — keeping the LAST occurrence in input order (the
    freshest fetched bar) so no duplicate timestamp can ever be written — in the
    EXACT cache format (tz-aware UTC date -> 'YYYY-MM-DD HH:MM:SS+00:00'; byte-
    compatible). Atomic: writes a temp file then renames it into place."""
    own = con is None
    con = con or connect()
    tmp = dst + ".tmp"
    try:
        src = _src(con, source)
        rest = ", ".join(f'"{c}"' for c in cols if c != "date")
        con.execute(
            f"COPY (WITH _u AS (SELECT *, row_number() OVER () AS _rid FROM {src}), "
            f"_d AS (SELECT *, row_number() OVER (PARTITION BY date::TIMESTAMPTZ ORDER BY _rid DESC) AS _rn FROM _u) "
            f"SELECT strftime(date::TIMESTAMPTZ, '%Y-%m-%d %H:%M:%S') || '+00:00' AS date, {rest} "
            f"FROM _d WHERE _rn = 1 ORDER BY date::TIMESTAMPTZ) "
            f"TO '{tmp}' (HEADER, DELIMITER ',')"
        )
        os.replace(tmp, dst)
    finally:
        try:
            con.unregister("_bars_src")
        except Exception:
            pass
        if own:
            con.close()
