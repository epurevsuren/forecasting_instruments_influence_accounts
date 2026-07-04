"""
simulate_tp_sl.py  —  LAYER 2: TP / STOP-LOSS TRADE SIMULATOR (1-min bars)
===========================================================================
The second layer of the two-layer strategy:

  Layer 1 (existing)  30-min bars -> signals / features / labels / backtest
  Layer 2 (this)      1-min bars  -> realistic TP/stop-loss trade simulation

Consumes a backtest_simulator.py results CSV (backtest_results/backtest_*.csv)
and, for every (post, instrument) cell the model flagged TRADE, simulates the
actual trade against the cached 1-min bars in ../IBKR/market_data_cache/:

  ENTRY   the 1-min bar containing the post timestamp; entry price = wap of
          that bar (IBKR calls the column `average`, Binance `wap` — both
          accepted). If wap <= 0 (FX MIDPOINT bars: volume/wap/barCount = -1
          is expected IBKR behaviour, not a bug) fall back to the bar close.
  FILTER  skip the trade when 0 <= barCount < --min-barcount (thin bar ->
          unreliable wap, wide spread). barCount == -1 (FX) skips the filter.
  LEVELS  TP  = entry * (1 ± |pred| * --tp-mult / 100)
          SL  = entry * (1 ∓ |pred| * --sl-mult / 100)
          i.e. sized from the model's own predicted move (2:1 RR by default).
          --tp-pct / --sl-pct override with fixed percentages.
  SCAN    walk forward bar by bar (starting at the bar AFTER entry):
              long : low <= SL -> stop | high >= TP -> take-profit
              short: high >= SL -> stop | low <= TP -> take-profit
          If BOTH trigger inside one bar, assume the STOP hit first
          (conservative). Timeout after --max-hold-min minutes -> exit at the
          last bar in the window.
  EXIT    exit price = wap of the exit bar (fallback close, same FX rule).
  EXTRAS  entry_vol_ratio = entry-bar volume / mean(volume of previous 30
          bars) — the volume-spike secondary signal (-1 when FX/no volume).

Every simulated trade is APPENDED to the output CSV as soon as its instrument
finishes (incremental save — a crash never loses completed instruments), each
step prints progress, and 1-min files are read via DuckDB filtered to just the
date span needed (one instrument in RAM at a time).

USAGE
-----
  uv run python simulate_tp_sl.py                          # latest backtest csv
  uv run python simulate_tp_sl.py --csv backtest_results/backtest_x_to_y.csv
  uv run python simulate_tp_sl.py --tp-pct 0.4 --sl-pct 0.2 --max-hold-min 120
"""
import os
import re
import sys
import glob
import json
import argparse
import datetime
import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_HERE     = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.normpath(os.path.join(_HERE, "..", "IBKR", "market_data_cache"))
RESULTS_DIR = os.path.join(_HERE, "backtest_results")
OUT_DIR     = os.path.join(_HERE, "tp_sl_results")

# DEFAULTS = Peter's winning config (2026-07-02 grid search, 82% win, Σ+160%):
# TP targets the model's 1-hour predicted move; the SL is catastrophe
# insurance only — floored OUTSIDE 1-min wick noise so it can't be swept.
# Small predictions are skipped entirely (reward can't clear the noise).
TP_MULT_DEFAULT      = 1.0   # TP distance = |pred| * this
SL_MULT_DEFAULT      = 2.0   # SL distance = |pred| * this (wide: insurance, not exit)
MIN_PRED_DEFAULT     = 0.3   # skip trades with |pred| below this %
SL_NOISE_MULT_DEFAULT = 5.0  # SL floor = this x median 1-min bar range
MAX_HOLD_DEFAULT  = 60     # minutes — matches the 1h reaction window
MIN_BARCOUNT      = 10     # skip entry bars thinner than this (when barCount >= 0)
ENTRY_GAP_MAX_MIN = 5      # post in a market gap: enter at next bar if within this many min
VOL_LOOKBACK      = 30     # bars for the entry volume-spike ratio


# ============================================================ input loading ----
def latest_backtest_csv():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "backtest_*.csv")))
    if not files:
        sys.exit(f"❌ No backtest_*.csv in {RESULTS_DIR} — run backtest_simulator.py first.")
    return files[-1]


def load_trades(csv_path):
    """Return (posts_df, instruments) — one row per post, instruments detected
    from the <INST>_decision columns."""
    print(f"📂 Loading backtest results: {csv_path}")
    df = pd.read_csv(csv_path)
    instruments = [c[:-9] for c in df.columns if c.endswith("_decision")]
    df["post_ts"] = pd.to_datetime(df["date"], utc=True)  # NY iso -> UTC
    print(f"  {len(df)} posts | {len(instruments)} instruments "
          f"({df['post_ts'].min():%Y-%m-%d} -> {df['post_ts'].max():%Y-%m-%d})")
    n_trades = int(sum((df[f"{i}_decision"] == "TRADE").sum() for i in instruments))
    print(f"  TRADE-flagged (post, instrument) cells: {n_trades}")
    return df, instruments


def load_1min(name, t_min, t_max):
    """DuckDB-filtered load of {name}_1min.csv for [t_min, t_max] (UTC).
    Keeps ALL original columns; adds a unified 'wap' (from `average` when the
    file is IBKR-named). Returns a DataFrame indexed by UTC timestamp, or None."""
    import duckdb
    path = os.path.join(CACHE_DIR, f"{name}_1min.csv")
    if not os.path.exists(path):
        return None
    try:
        q = (f"SELECT * FROM read_csv_auto('{path}') "
             f"WHERE date >= '{t_min:%Y-%m-%d %H:%M}' AND date <= '{t_max:%Y-%m-%d %H:%M}'")
        bars = duckdb.query(q).df()
    except Exception as e:
        print(f"  ⚠️  {name}: 1-min read failed ({str(e)[:60]})")
        return None
    if bars.empty:
        return None
    if "wap" not in bars.columns:
        bars["wap"] = bars["average"] if "average" in bars.columns else np.nan
    bars["date"] = pd.to_datetime(bars["date"], utc=True)
    bars = bars.sort_values("date").set_index("date")
    return bars


# ============================================================ trade engine ----
def price_of(bar):
    """Entry/exit price: wap when valid, else close (FX MIDPOINT: wap == -1)."""
    w = bar.get("wap", np.nan)
    if pd.notna(w) and w > 0:
        return float(w)
    return float(bar["close"])


def simulate_one(bars, post_ts, direction, tp_dist_pct, sl_dist_pct,
                 max_hold_min, min_barcount):
    """Simulate one trade. Returns dict or (None, reason)."""
    idx = bars.index
    # entry bar = bar containing post_ts, else next bar within ENTRY_GAP_MAX_MIN
    pos = idx.searchsorted(post_ts, side="right") - 1
    if pos >= 0 and (post_ts - idx[pos]) < pd.Timedelta(minutes=1):
        entry_pos = pos
    else:
        nxt = pos + 1
        if nxt >= len(idx) or (idx[nxt] - post_ts) > pd.Timedelta(minutes=ENTRY_GAP_MAX_MIN):
            return None, "no_bar_at_post"
        entry_pos = nxt
    if entry_pos >= len(idx) - 1:
        return None, "at_cache_end"

    ebar = bars.iloc[entry_pos]
    bc = float(ebar.get("barCount", -1)) if pd.notna(ebar.get("barCount", np.nan)) else -1
    vol = float(ebar.get("volume", -1)) if pd.notna(ebar.get("volume", np.nan)) else -1
    # Thin-bar filter only applies to instruments that actually TRADE:
    #   FX MIDPOINT bars   -> volume = -1 (expected IBKR sentinel)
    #   VIX (calc. index)  -> volume = 0, barCount = index updates/min (0-4)
    # Neither says anything about liquidity, so the filter is skipped there.
    if vol > 0 and 0 <= bc < min_barcount:
        return None, "thin_bar"

    entry = price_of(ebar)
    if not np.isfinite(entry) or entry <= 0:
        return None, "bad_entry_price"

    # volume-spike secondary signal
    vol_ratio = -1.0
    if vol > 0 and entry_pos >= 1:
        prev = bars["volume"].iloc[max(0, entry_pos - VOL_LOOKBACK):entry_pos]
        prev = prev[prev > 0]
        if len(prev):
            vol_ratio = round(vol / prev.mean(), 3)

    if direction > 0:   # long
        tp_price = entry * (1 + tp_dist_pct / 100)
        sl_price = entry * (1 - sl_dist_pct / 100)
    else:               # short
        tp_price = entry * (1 - tp_dist_pct / 100)
        sl_price = entry * (1 + sl_dist_pct / 100)

    deadline = idx[entry_pos] + pd.Timedelta(minutes=max_hold_min)
    window = bars.iloc[entry_pos + 1:]
    window = window[window.index <= deadline]
    if window.empty:
        return None, "no_forward_bars"

    outcome, exit_bar, bars_held = "TIMEOUT", window.iloc[-1], len(window)
    for k in range(len(window)):
        b = window.iloc[k]
        hi, lo = float(b["high"]), float(b["low"])
        if direction > 0:
            hit_sl, hit_tp = lo <= sl_price, hi >= tp_price
        else:
            hit_sl, hit_tp = hi >= sl_price, lo <= tp_price
        if hit_sl:                       # stop first when both (conservative)
            outcome, exit_bar, bars_held = "STOP", b, k + 1
            break
        if hit_tp:
            outcome, exit_bar, bars_held = "TP", b, k + 1
            break

    exit_price = price_of(exit_bar)
    pnl_pct = (exit_price - entry) / entry * 100 * (1 if direction > 0 else -1)

    return {
        "entry_time": idx[entry_pos].isoformat(),
        "entry_price": round(entry, 6),
        "exit_time": exit_bar.name.isoformat(),
        "exit_price": round(exit_price, 6),
        "direction": "LONG" if direction > 0 else "SHORT",
        "outcome": outcome,
        "bars_held": bars_held,
        "tp_price": round(tp_price, 6),
        "sl_price": round(sl_price, 6),
        "entry_barCount": int(bc),
        "entry_vol_ratio": vol_ratio,
        "pnl_pct": round(pnl_pct, 4),
    }, None


# ================================================================== main ----
def main():
    ap = argparse.ArgumentParser(
        description="Layer-2 TP/stop-loss trade simulator on cached 1-min bars.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--csv", default=None, help="backtest results CSV (default: latest in backtest_results/)")
    ap.add_argument("--tp-mult", type=float, default=TP_MULT_DEFAULT,
                    help=f"TP distance = |pred| x this (default {TP_MULT_DEFAULT})")
    ap.add_argument("--sl-mult", type=float, default=SL_MULT_DEFAULT,
                    help=f"SL distance = |pred| x this (default {SL_MULT_DEFAULT} -> 2:1 RR)")
    ap.add_argument("--tp-pct", type=float, default=None, help="fixed TP distance in %% (overrides --tp-mult)")
    ap.add_argument("--sl-pct", type=float, default=None, help="fixed SL distance in %% (overrides --sl-mult)")
    ap.add_argument("--max-hold-min", type=int, default=MAX_HOLD_DEFAULT,
                    help=f"timeout exit after this many minutes (default {MAX_HOLD_DEFAULT})")
    ap.add_argument("--min-barcount", type=int, default=MIN_BARCOUNT,
                    help=f"skip entries on bars with 0 <= barCount < this (default {MIN_BARCOUNT})")
    ap.add_argument("--min-pred", type=float, default=MIN_PRED_DEFAULT,
                    help=f"skip trades with |pred| below this %% (small predicted reward "
                         f"can't clear 1-min noise; default {MIN_PRED_DEFAULT}, 0 = off)")
    ap.add_argument("--sl-noise-mult", type=float, default=SL_NOISE_MULT_DEFAULT,
                    help=f"floor the SL distance at this x the instrument's 1-min noise "
                         f"(median (high-low)/close of its bars) so stops can't be swept "
                         f"by wick noise. Default {SL_NOISE_MULT_DEFAULT}, 0 = off.")
    ap.add_argument("--out", default=None, help="output trades CSV (default tp_sl_results/trades_<stamp>.csv)")
    args = ap.parse_args()

    csv_path = args.csv or latest_backtest_csv()
    posts, instruments = load_trades(csv_path)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(OUT_DIR, f"trades_{stamp}.csv")

    print("=" * 78)
    print("  LAYER-2 TP/STOP-LOSS SIMULATION  (1-min bars, entry/exit at wap)")
    if args.tp_pct is not None:
        print(f"  TP = {args.tp_pct}% fixed   SL = {args.sl_pct or args.tp_pct/2}% fixed")
    else:
        print(f"  TP = |pred| x {args.tp_mult}   SL = |pred| x {args.sl_mult}   (per-trade sizing)")
    print(f"  Max hold: {args.max_hold_min} min | min barCount: {args.min_barcount} "
          f"| stop-first on same-bar conflict")
    if args.min_pred > 0 or args.sl_noise_mult > 0:
        print(f"  Filters: min |pred| = {args.min_pred}%  |  "
              f"SL floor = {args.sl_noise_mult} x 1-min noise")
    print(f"  Output (incremental append): {out_path}")
    print("=" * 78)

    t_min = posts["post_ts"].min() - pd.Timedelta(hours=1)
    t_max = posts["post_ts"].max() + pd.Timedelta(minutes=args.max_hold_min + 90)

    summary = {}
    skip_totals = {}
    wrote_header = False

    for n_i, inst in enumerate(instruments, 1):
        dcol, pcol = f"{inst}_decision", f"{inst}_pred"
        todo = posts[posts[dcol] == "TRADE"]
        if todo.empty:
            print(f"\n[{n_i}/{len(instruments)}] ⏭️  {inst}: no TRADE-flagged posts")
            continue

        print(f"\n[{n_i}/{len(instruments)}] 📥 {inst}: loading 1-min bars "
              f"({len(todo)} trade candidates)...", flush=True)
        bars = load_1min(inst, t_min, t_max)
        if bars is None:
            print(f"  ❌ {inst}: no 1-min cache file / no bars in span — skipped")
            skip_totals["no_1min_data"] = skip_totals.get("no_1min_data", 0) + len(todo)
            continue
        print(f"  💾 {len(bars)} bars ({bars.index.min():%Y-%m-%d} -> {bars.index.max():%Y-%m-%d})")

        # 1-min noise level: median intrabar range in % — the size of a typical
        # wick. SL must sit outside a few of these or it WILL get swept.
        # NB: quiet ETFs (DIA/XLI/XLF/...) have high==low on >50% of 1-min bars,
        # which makes the plain median 0 — measure over NONZERO ranges only.
        _rng = ((bars["high"] - bars["low"]) / bars["close"]) * 100
        _rng = _rng[_rng > 0]
        noise_pct = float(_rng.median()) if len(_rng) else 0.0
        if args.sl_noise_mult > 0:
            print(f"  📏 1-min noise: {noise_pct:.4f}%  -> SL floor "
                  f"{args.sl_noise_mult * noise_pct:.4f}%")

        rows, skips = [], {}
        # ONE OPEN POSITION PER INSTRUMENT: chained posts minutes apart share
        # one market reaction — a second entry while the first trade is still
        # open would ride the SAME move twice. Skip until the position exits.
        open_until = None
        for _, p in todo.sort_values("post_ts").iterrows():
            pred = float(p[pcol])
            if pred == 0 or not np.isfinite(pred):
                skips["zero_pred"] = skips.get("zero_pred", 0) + 1
                continue
            if abs(pred) < args.min_pred:
                skips["small_pred"] = skips.get("small_pred", 0) + 1
                continue
            if open_until is not None and p["post_ts"] <= open_until:
                skips["position_open"] = skips.get("position_open", 0) + 1
                continue
            tp_d = args.tp_pct if args.tp_pct is not None else abs(pred) * args.tp_mult
            sl_d = args.sl_pct if args.sl_pct is not None else abs(pred) * args.sl_mult
            if args.sl_noise_mult > 0:
                sl_d = max(sl_d, args.sl_noise_mult * noise_pct)
            trade, reason = simulate_one(bars, p["post_ts"], np.sign(pred),
                                         tp_d, sl_d, args.max_hold_min, args.min_barcount)
            if trade is None:
                skips[reason] = skips.get(reason, 0) + 1
                continue
            trade.update({"instrument": inst, "post_id": p["id"], "platform": p["platform"],
                          "account": p["account"], "post_time": p["date"],
                          "pred_pct": round(pred, 4),
                          "text": str(p["text"])})
            rows.append(trade)
            open_until = pd.Timestamp(trade["exit_time"])   # position stays open until exit

        del bars  # free RAM before the next instrument

        for k, v in skips.items():
            skip_totals[k] = skip_totals.get(k, 0) + v

        if rows:
            rdf = pd.DataFrame(rows)
            lead = ["instrument", "post_id", "platform", "account", "post_time", "pred_pct",
                    "direction", "entry_time", "entry_price", "exit_time", "exit_price",
                    "outcome", "bars_held", "tp_price", "sl_price",
                    "entry_barCount", "entry_vol_ratio", "pnl_pct", "text"]
            rdf = rdf[lead]
            rdf.to_csv(out_path, mode="a", header=not wrote_header, index=False,
                       encoding="utf-8", lineterminator="\n")
            wrote_header = True

            n = len(rdf)
            tp_n = (rdf["outcome"] == "TP").sum()
            sl_n = (rdf["outcome"] == "STOP").sum()
            to_n = (rdf["outcome"] == "TIMEOUT").sum()
            wins = (rdf["pnl_pct"] > 0).sum()
            summary[inst] = {"n": int(n), "tp": int(tp_n), "stop": int(sl_n),
                             "timeout": int(to_n), "win_rate": round(wins / n, 3),
                             "total_pnl_pct": round(float(rdf["pnl_pct"].sum()), 4),
                             "avg_pnl_pct": round(float(rdf["pnl_pct"].mean()), 4)}
            skip_str = f"  (skipped: {skips})" if skips else ""
            print(f"  ✅ {inst}: {n} trades  TP {tp_n} | STOP {sl_n} | TIMEOUT {to_n}  "
                  f"ΣP&L {rdf['pnl_pct'].sum():+.3f}%  -> appended{skip_str}", flush=True)
        else:
            print(f"  ⚠️  {inst}: 0 executable trades (skipped: {skips})")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print(f"  TP/SL SIMULATION SUMMARY  (exit at wap; stop-first; "
          f"{args.max_hold_min}min timeout)")
    print("=" * 78)
    if not summary:
        print("  ⚠️  No trades executed at all. Skip reasons:", skip_totals)
        return
    print(f"  {'instrument':<10} {'n':>5} {'TP':>5} {'STOP':>5} {'T/O':>5} "
          f"{'win%':>7} {'avg P&L':>9} {'Σ P&L':>9}")
    tot_n = tot_pnl = 0
    for inst, s in sorted(summary.items(), key=lambda kv: -kv[1]["total_pnl_pct"]):
        flag = "✅" if s["total_pnl_pct"] > 0 else "🔴"
        print(f"  {flag} {inst:<10} {s['n']:>4} {s['tp']:>5} {s['stop']:>5} {s['timeout']:>5} "
              f"{s['win_rate']:>6.1%} {s['avg_pnl_pct']:>+8.4f}% {s['total_pnl_pct']:>+8.3f}%")
        tot_n += s["n"]
        tot_pnl += s["total_pnl_pct"]
    print("-" * 78)
    print(f"  OVERALL   {tot_n} trades   Σ P&L = {tot_pnl:+.3f}%  "
          f"(sum of per-trade %% moves, 1 unit per trade)")
    if skip_totals:
        print(f"  Skipped candidates: {skip_totals}")

    meta = {"backtest_csv": os.path.basename(csv_path),
            "params": {"tp_mult": args.tp_mult, "sl_mult": args.sl_mult,
                       "tp_pct": args.tp_pct, "sl_pct": args.sl_pct,
                       "max_hold_min": args.max_hold_min, "min_barcount": args.min_barcount},
            "skips": skip_totals, "instruments": summary,
            "overall": {"n": tot_n, "total_pnl_pct": round(tot_pnl, 4)}}
    jpath = out_path.replace(".csv", "_summary.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  💾 Trades CSV : {out_path}")
    print(f"  💾 Summary    : {jpath}")
    print("✅ Done")


if __name__ == "__main__":
    main()
