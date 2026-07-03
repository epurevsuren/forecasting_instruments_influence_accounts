"""
sweep_tp_sl.py  —  parameter sweep over the Layer-2 TP/SL simulator
====================================================================
Loads each instrument's 1-min bars ONCE and evaluates ALL configs in the same
pass (bar loading dominates runtime, so N configs cost barely more than 1).

Configs swept (name, tp_mult, sl_mult, tp_pct, sl_pct, max_hold_min):
  baseline      TP=|pred|x1.0  SL=|pred|x0.5   60min   (current default)
  wide_stop     TP=|pred|x1.0  SL=|pred|x1.0   60min   (1:1 RR)
  wider_stop    TP=|pred|x1.0  SL=|pred|x2.0   60min   (risk 2 to make 1 -> high win%)
  atr_floorish  TP=|pred|x1.5  SL=|pred|x1.5   60min
  horizon       no TP/SL, exit at the 60min mark       (pure direction bet ->
                win% should approach the backtest direction accuracy)
  horizon120    no TP/SL, exit at 120min

USAGE
-----
  uv run python sweep_tp_sl.py                     # latest backtest csv, all posts
  uv run python sweep_tp_sl.py --csv X --rows 0:77 --out-prefix /tmp/sweep_c0
      (--rows lets a driver shard the posts and merge partial JSONs later)

Each config's per-trade results are saved to <out-prefix>_<config>.csv and a
combined summary printed + saved to <out-prefix>_summary.json.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import simulate_tp_sl as S

NO_STOP = 1e9   # "no TP/SL" sentinel distance (percent) — nothing ever triggers

CONFIGS = [
    # name          tp_mult sl_mult tp_pct  sl_pct  hold
    ("baseline",     1.0,    0.5,   None,   None,    60),
    ("wide_stop",    1.0,    1.0,   None,   None,    60),
    ("wider_stop",   1.0,    2.0,   None,   None,    60),
    ("tp15_sl15",    1.5,    1.5,   None,   None,    60),
    ("horizon",      None,   None,  NO_STOP, NO_STOP, 60),
    ("horizon120",   None,   None,  NO_STOP, NO_STOP, 120),
]
MAX_HOLD_SPAN = max(c[5] for c in CONFIGS)


def main():
    ap = argparse.ArgumentParser(description="Sweep TP/SL configs in one bar-loading pass.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--rows", default=None, metavar="A:B",
                    help="slice of post rows to process (for sharded runs)")
    ap.add_argument("--out-prefix", default=os.path.join(S.OUT_DIR, "sweep"))
    ap.add_argument("--min-barcount", type=int, default=S.MIN_BARCOUNT)
    args = ap.parse_args()

    csv_path = args.csv or S.latest_backtest_csv()
    posts, instruments = S.load_trades(csv_path)
    if args.rows:
        a, b = (int(x) for x in args.rows.split(":"))
        posts = posts.iloc[a:b]
        print(f"  Row slice {a}:{b} -> {len(posts)} posts")

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    t_min = posts["post_ts"].min() - pd.Timedelta(hours=1)
    t_max = posts["post_ts"].max() + pd.Timedelta(minutes=MAX_HOLD_SPAN + 90)

    results = {name: [] for name, *_ in CONFIGS}

    for n_i, inst in enumerate(instruments, 1):
        dcol, pcol = f"{inst}_decision", f"{inst}_pred"
        todo = posts[posts[dcol] == "TRADE"]
        if todo.empty:
            continue
        print(f"[{n_i}/{len(instruments)}] 📥 {inst} ({len(todo)} candidates)...", flush=True)
        bars = S.load_1min(inst, t_min, t_max)
        if bars is None:
            print(f"  ❌ no 1-min data")
            continue

        n_ok = 0
        for _, p in todo.iterrows():
            pred = float(p[pcol])
            if pred == 0 or not np.isfinite(pred):
                continue
            for name, tpm, slm, tpp, slp, hold in CONFIGS:
                tp_d = tpp if tpp is not None else abs(pred) * tpm
                sl_d = slp if slp is not None else abs(pred) * slm
                trade, _reason = S.simulate_one(bars, p["post_ts"], np.sign(pred),
                                                tp_d, sl_d, hold, args.min_barcount)
                if trade is None:
                    continue
                trade.update({"instrument": inst, "post_id": p["id"],
                              "pred_pct": round(pred, 4)})
                results[name].append(trade)
                n_ok += 1
        del bars
        print(f"  ✅ {inst}: {n_ok} simulated cells across {len(CONFIGS)} configs", flush=True)

    # ------------------------------------------------------------- outputs ----
    summary = {}
    print("\n" + "=" * 78)
    print(f"  SWEEP SUMMARY  ({os.path.basename(csv_path)}"
          f"{', rows ' + args.rows if args.rows else ''})")
    print("=" * 78)
    print(f"  {'config':<12} {'n':>5} {'TP':>5} {'STOP':>6} {'T/O':>5} "
          f"{'win%':>7} {'avg P&L':>9} {'Σ P&L':>9}")
    for name, *_ in CONFIGS:
        rows = results[name]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(f"{args.out_prefix}_{name}.csv", index=False,
                  encoding="utf-8", lineterminator="\n")
        n = len(df)
        s = {"n": int(n),
             "tp": int((df.outcome == "TP").sum()),
             "stop": int((df.outcome == "STOP").sum()),
             "timeout": int((df.outcome == "TIMEOUT").sum()),
             "win_rate": round(float((df.pnl_pct > 0).mean()), 4),
             "avg_pnl_pct": round(float(df.pnl_pct.mean()), 4),
             "total_pnl_pct": round(float(df.pnl_pct.sum()), 4)}
        summary[name] = s
        print(f"  {name:<12} {s['n']:>5} {s['tp']:>5} {s['stop']:>6} {s['timeout']:>5} "
              f"{s['win_rate']:>6.1%} {s['avg_pnl_pct']:>+8.4f}% {s['total_pnl_pct']:>+8.3f}%")

    with open(f"{args.out_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"backtest_csv": os.path.basename(csv_path), "rows": args.rows,
                   "configs": {c[0]: {"tp_mult": c[1], "sl_mult": c[2], "tp_pct": c[3],
                                      "sl_pct": c[4], "max_hold_min": c[5]} for c in CONFIGS},
                   "summary": summary}, f, indent=2)
    print(f"\n  💾 {args.out_prefix}_<config>.csv + _summary.json")
    print("✅ Done")


if __name__ == "__main__":
    main()
