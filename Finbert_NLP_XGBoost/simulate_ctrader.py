"""
simulate_ctrader.py  —  LAYER 3: cTRADER/PEPPERSTONE ACCOUNT SIMULATION
=========================================================================
Replays the Layer-2 trade events (tp_sl_results/trades_*.csv — entry/exit
price+time per tweet-trade) through a realistic cTrader demo account:

  ACCOUNT     balance $1000, account leverage 1:30 (Pepperstone demo),
              equity = balance + floating PnL, margin level = equity/margin.
  SYMBOLS     cTrader contract specs per instrument: contract size, pip size,
              per-class leverage cap (ESMA/Pepperstone: FX majors 30:1,
              minors/gold 20:1, commodities/indices 10-20:1, crypto 2:1,
              US share CFDs 5:1). Instruments with no CFD (US10Y/US2Y) skip.
  ORDERS      market orders, volume in lots (min 0.01, step 0.01), fills at
              the Layer-2 entry/exit prices (1-min wap).
  SIZING      risk-based on CURRENT equity: lots = (equity × RISK_PCT) /
              (SL distance × contract size), capped by free-margin budget.
              Every trade compounds with the account.
  STOP-OUT    Pepperstone cTrader smart stop-out = 50% margin level. This
              simulator EVADES it BY CONSTRUCTION: hard caps
                total open risk   <= MAX_OPEN_RISK_PCT of equity (all SLs hit)
                total used margin <= MAX_MARGIN_PCT of equity
              give a worst-case margin level of
                (1 - MAX_OPEN_RISK) / MAX_MARGIN  ~ 235%  >>  50%.
              Concurrent trades within the hour queue against these budgets —
              when the budget is full, new signals are skipped, not sized down
              into stop-out territory.

No API keys needed — pure simulation. Concepts map 1:1 to cTrader OpenAPI
(symbols, lots, margin, market orders) for the production bridge later.

USAGE
-----
  uv run python simulate_ctrader.py                        # latest Layer-2 CSV
  uv run python simulate_ctrader.py --balance 1000 --risk-pct 2
  uv run python simulate_ctrader.py --csv tp_sl_results/trades_X.csv
"""
import os
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

_HERE   = os.path.dirname(os.path.abspath(__file__))
TP_DIR  = os.path.join(_HERE, "tp_sl_results")
OUT_DIR = os.path.join(_HERE, "ctrader_results")

# ---------------------------------------------------------------- account ----
BALANCE_DEFAULT     = 1000.0
ACCOUNT_LEVERAGE    = 30      # Pepperstone demo 1:30
STOP_OUT_LEVEL      = 50.0    # cTrader smart stop-out (% margin level)
MIN_LOTS, LOT_STEP  = 0.01, 0.01

# DEFAULT SIZING = MARGIN MODE (this is CFD trading — the leverage IS the
# product). Each trade takes a chunk of equity as margin at full class
# leverage; the ONLY brake is the actual stop-out equation:
#     worst-case equity (ALL open SLs hit)  >=  used margin x (stop-out+buffer)
# solved per-entry for the max safe lots. The Layer-2 SLs (max(2x|pred|,
# 5x noise)) bound each trade's loss, so stop-out is dodged by the SL firing
# long before margin level can reach 50%.
MARGIN_PER_TRADE_PCT = 25.0   # % of equity committed as margin per trade
MAX_TOTAL_MARGIN_PCT = 80.0   # % of equity as margin across all open trades
STOPOUT_BUFFER_PP    = 20.0   # worst-case margin level >= stop-out + this (pp)
RISK_PCT             = 2.0    # only used with --sizing risk (conservative mode)

# ------------------------------------------------- cTrader contract specs ----
# name -> (cTrader symbol, contract size per lot, pip size, class leverage)
# Class leverage caps the ACCOUNT leverage (effective = min of the two).
SPECS = {
    "EUR_USD": ("EURUSD", 100_000, 0.0001, 30),
    "GBP_USD": ("GBPUSD", 100_000, 0.0001, 30),
    "AUD_USD": ("AUDUSD", 100_000, 0.0001, 30),
    "USD_JPY": ("USDJPY", 100_000, 0.01,   30),
    "USD_CAD": ("USDCAD", 100_000, 0.0001, 30),
    "USD_CHF": ("USDCHF", 100_000, 0.0001, 30),
    "USD_CNY": ("USDCNH", 100_000, 0.0001, 20),
    "USD_MXN": ("USDMXN", 100_000, 0.0001, 20),
    "GOLD":    ("XAUUSD", 100,     0.01,   20),
    "OIL":     ("XTIUSD", 100,     0.01,   10),
    "NATGAS":  ("XNGUSD", 1_000,   0.001,  10),
    "COPPER":  ("COPPER", 1_000,   0.001,  10),
    "SPY":     ("SPY.US", 100,     0.01,    5),   # US share CFD
    "QQQ":     ("QQQ.US", 100,     0.01,    5),
    "DIA":     ("DIA.US", 100,     0.01,    5),
    "XLI":     ("XLI.US", 100,     0.01,    5),
    "XLF":     ("XLF.US", 100,     0.01,    5),
    "XLE":     ("XLE.US", 100,     0.01,    5),
    "VIX":     ("VIX",    100,     0.01,   10),
    "BTC":     ("BTCUSD", 1,       1.0,     2),
    "ETH":     ("ETHUSD", 1,       0.01,    2),
    # US10Y / US2Y: no CFD on Pepperstone cTrader -> skipped
}


def quote_ccy_is_usd(name):
    """PnL below assumes USD quote ccy. USD_JPY etc. are quoted in JPY/CHF/...:
    convert PnL by dividing by the exit price (good approximation)."""
    return name.split("_")[-1] not in ("JPY", "CAD", "CHF", "CNY", "MXN")


def latest_trades_csv():
    files = sorted(glob.glob(os.path.join(TP_DIR, "trades_*.csv")))
    files = [f for f in files if "_summary" not in f]
    if not files:
        sys.exit(f"❌ No trades_*.csv in {TP_DIR} — run simulate_tp_sl.py first.")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(description="cTrader/Pepperstone account simulation over Layer-2 trades.")
    ap.add_argument("--csv", default=None, help="Layer-2 trades CSV (default: latest in tp_sl_results/)")
    ap.add_argument("--balance", type=float, default=BALANCE_DEFAULT)
    ap.add_argument("--leverage", type=int, default=ACCOUNT_LEVERAGE)
    ap.add_argument("--risk-pct", type=float, default=RISK_PCT)
    ap.add_argument("--max-open-risk", type=float, default=MAX_OPEN_RISK_PCT)
    ap.add_argument("--max-margin", type=float, default=MAX_MARGIN_PCT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    csv_path = args.csv or latest_trades_csv()
    df = pd.read_csv(csv_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(OUT_DIR, f"ctrader_{stamp}.csv")

    worst_ml_bound = (1 - args.max_open_risk / 100) / (args.max_margin / 100) * 100
    print("=" * 78)
    print("  LAYER-3 cTRADER ACCOUNT SIMULATION  (Pepperstone demo mechanics)")
    print(f"  Balance ${args.balance:.0f} | leverage 1:{args.leverage} | "
          f"stop-out {STOP_OUT_LEVEL:.0f}% margin level")
    print(f"  Sizing: risk {args.risk_pct}%/trade | caps: open risk "
          f"<={args.max_open_risk}%, margin <={args.max_margin}% of equity")
    print(f"  Worst-case margin level bound (ALL SLs hit at once): "
          f"{worst_ml_bound:.0f}%  (stop-out impossible by construction)")
    print(f"  Trades in: {csv_path} ({len(df)} events)")
    print("=" * 78)

    balance = args.balance
    open_pos = []          # [{exit_time, margin, risk, ...}]
    ledger, skips = [], {}
    peak, max_dd = balance, 0.0
    min_ml_seen = float("inf")

    def close_expired(now):
        nonlocal balance, peak, max_dd
        for p in [p for p in open_pos if p["exit_time"] <= now]:
            balance += p["pnl_usd"]
            open_pos.remove(p)
            peak = max(peak, balance)
            max_dd = max(max_dd, (peak - balance) / peak * 100)
            ledger.append(p["row"])

    for _, t in df.iterrows():
        inst = t["instrument"]
        close_expired(t["entry_time"])

        if inst not in SPECS:
            skips["no_cfd_symbol"] = skips.get("no_cfd_symbol", 0) + 1
            continue
        if any(p["instrument"] == inst for p in open_pos):
            skips["position_open"] = skips.get("position_open", 0) + 1
            continue

        symbol, contract, pip, class_lev = SPECS[inst]
        lev = min(args.leverage, class_lev)
        entry, exitp = float(t["entry_price"]), float(t["exit_price"])
        direction = 1 if t["direction"] == "LONG" else -1
        sl_dist = abs(entry - float(t["sl_price"]))
        if sl_dist <= 0:
            skips["bad_sl"] = skips.get("bad_sl", 0) + 1
            continue

        equity      = balance                        # realized (conservative)
        used_margin = sum(p["margin"] for p in open_pos)
        open_risk   = sum(p["risk"] for p in open_pos)

        # convert 1 unit of quote-ccy PnL to USD (approx: divide by price)
        usd_conv = 1.0 if quote_ccy_is_usd(inst) else 1.0 / exitp

        # risk-based lots, then margin-budget cap, then floor to lot step
        risk_usd  = equity * args.risk_pct / 100
        lots_risk = risk_usd / (sl_dist * contract * usd_conv)
        margin_per_lot = contract * entry * usd_conv / lev
        margin_budget  = equity * args.max_margin / 100 - used_margin
        risk_budget    = equity * args.max_open_risk / 100 - open_risk
        if margin_budget <= 0 or risk_budget <= 0:
            skips["budget_full"] = skips.get("budget_full", 0) + 1
            continue
        lots_margin = margin_budget / margin_per_lot
        lots_riskbudget = risk_budget / (sl_dist * contract * usd_conv)
        lots = np.floor(min(lots_risk, lots_margin, lots_riskbudget) / LOT_STEP) * LOT_STEP
        if lots < MIN_LOTS:
            skips["lot_too_small"] = skips.get("lot_too_small", 0) + 1
            continue

        margin = lots * margin_per_lot
        risk   = lots * sl_dist * contract * usd_conv
        pnl    = direction * (exitp - entry) * lots * contract * usd_conv
        pips   = direction * (exitp - entry) / pip

        ml_now = (equity / (used_margin + margin) * 100) if (used_margin + margin) > 0 else float("inf")
        min_ml_seen = min(min_ml_seen, ml_now)

        row = {
            "entry_time": t["entry_time"].isoformat(), "exit_time": t["exit_time"].isoformat(),
            "instrument": inst, "symbol": symbol, "direction": t["direction"],
            "lots": round(lots, 2), "entry": entry, "exit": exitp,
            "pips": round(pips, 1), "sl_dist_pips": round(sl_dist / pip, 1),
            "margin_usd": round(margin, 2), "risk_usd": round(risk, 2),
            "pnl_usd": round(pnl, 2), "outcome": t["outcome"],
            "balance_after": None,  # filled at close
            "post_id": t.get("post_id", ""), "pred_pct": t.get("pred_pct", ""),
        }
        open_pos.append({"instrument": inst, "exit_time": t["exit_time"],
                         "margin": margin, "risk": risk, "pnl_usd": pnl, "row": row})

    close_expired(pd.Timestamp.max.tz_localize("UTC"))
    for r in ledger:                     # fill running balances in close order
        pass
    bal = args.balance
    for r in sorted(ledger, key=lambda r: r["exit_time"]):
        bal += r["pnl_usd"]
        r["balance_after"] = round(bal, 2)

    led = pd.DataFrame(sorted(ledger, key=lambda r: r["entry_time"]))
    led.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")

    n = len(led)
    wins = int((led["pnl_usd"] > 0).sum()) if n else 0
    print(f"\n{'='*78}\n  ACCOUNT SUMMARY\n{'='*78}")
    print(f"  Executed trades : {n}  (win {wins}/{n} = {wins/max(n,1):.1%})")
    print(f"  Start balance   : ${args.balance:,.2f}")
    print(f"  Final balance   : ${balance:,.2f}   ({(balance/args.balance-1)*100:+.1f}%)")
    print(f"  Max drawdown    : {max_dd:.1f}%")
    print(f"  Min margin level seen at entry: "
          f"{'n/a' if min_ml_seen == float('inf') else f'{min_ml_seen:.0f}%'}"
          f"  (stop-out at {STOP_OUT_LEVEL:.0f}%)")
    if n:
        by = led.groupby("instrument")["pnl_usd"].agg(["count", "sum"]).sort_values("sum", ascending=False)
        print(f"\n  {'instrument':<10} {'n':>4} {'PnL $':>10}")
        for inst, r in by.iterrows():
            print(f"  {inst:<10} {int(r['count']):>4} {r['sum']:>+10.2f}")
    print(f"\n  Skipped: {skips}")
    meta = {"source_csv": os.path.basename(csv_path),
            "params": {"balance": args.balance, "leverage": args.leverage,
                       "risk_pct": args.risk_pct, "max_open_risk": args.max_open_risk,
                       "max_margin": args.max_margin, "stop_out": STOP_OUT_LEVEL},
            "final_balance": round(balance, 2), "return_pct": round((balance/args.balance-1)*100, 2),
            "max_drawdown_pct": round(max_dd, 2), "trades": n, "wins": wins,
            "min_margin_level": None if min_ml_seen == float("inf") else round(min_ml_seen, 1),
            "skips": skips}
    with open(out_path.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  💾 Ledger : {out_path}")
    print(f"  💾 Summary: {out_path.replace('.csv', '_summary.json')}")
    print("✅ Done")


if __name__ == "__main__":
    main()
