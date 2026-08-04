"""
simulate_ctrader.py  —  LAYER 3: cTRADER/PEPPERSTONE ACCOUNT SIMULATION
=========================================================================
Replays the Layer-2 trade events (tp_sl_results/trades_*.csv — entry/exit
price+time per tweet-trade) through a realistic cTrader account:

  ACCOUNT     balance, account leverage 1:30 (Pepperstone), equity, margin
              level = equity/margin; smart stop-out at 50%.
  SYMBOLS     real specs from DP/instruments.json "ctrader" blocks (synced
              from the Open API by cTrader/fetch_ctrader_symbols.py): lot
              size, pip position, min/max/step lots, class leverage, swaps.
              LIVE max-lot caps by default (VIX 1000 live vs 100 demo) —
              --demo-caps reverts. Instruments without a spec are skipped.
  ORDERS      market orders in lots; TP fills AT the TP price (limit), SL at
              the stop price. Wire numbers recorded per trade via the SAME
              helpers as the production bridge (cTrader/ctrader_bridge.py):
              wire_volume (cents of units), wire_relative_tp/sl (1/100_000).
  SIZING      MARGIN mode (default, CFD style): --margin-per-trade % of
              CURRENT equity as margin at full class leverage — compounds.
              RISK mode (--sizing risk): classic %-risk sizing.
              BURST ALLOCATION: same-minute signals ranked by MARGIN-RELATIVE
              expected return = |pred| x class leverage (FX small %% x 30:1
              beats crypto big %% x 2:1 — raw |pred| ordering starved FX).
  STOP-OUT    exact per-entry solve: worst-case equity (ALL open SLs hit)
              >= used margin x (stop-out + buffer). --no-sl / sentinel-SL
              trades skip the solve (margin caps only).

USAGE
-----
  uv run python simulate_ctrader.py                        # latest Layer-2 CSV (mtime)
  uv run python simulate_ctrader.py --balance 50000
  uv run python simulate_ctrader.py --csv tp_sl_results/trades_X.csv --no-sl
  uv run python simulate_ctrader.py --demo-caps            # demo max-lot limits
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
ACCOUNT_LEVERAGE    = 30      # Pepperstone 1:30
STOP_OUT_LEVEL      = 50.0    # cTrader smart stop-out (% margin level)
MIN_LOTS, LOT_STEP  = 0.01, 0.01

# DEFAULT SIZING = MARGIN MODE (this is CFD trading — the leverage IS the
# product). Each trade takes a chunk of equity as margin at full class
# leverage; the ONLY brake is the actual stop-out equation:
#     worst-case equity (ALL open SLs hit)  >=  used margin x (stop-out+buffer)
# solved per-entry for the max safe lots.
MARGIN_PER_TRADE_PCT = 25.0   # % of equity committed as margin per trade
MAX_TOTAL_MARGIN_PCT = 80.0   # % of equity as margin across all open trades
STOPOUT_BUFFER_PP    = 20.0   # worst-case margin level >= stop-out + this (pp)
RISK_PCT             = 2.0    # only used with --sizing risk (conservative mode)

# ---------------------------------------------------------------------------
# DRAWDOWN FSM  (Algothon 2023 winner, CookieAlgorists — their one idea that
# belongs at Layer 3 rather than in the predictor)
# ---------------------------------------------------------------------------
# Their observation: every model in the stack is STATELESS. It has no memory of
# how it has been doing lately, so a regime it can no longer read produces a
# run of losses and nothing in the system notices. Their fix was a finite state
# machine over recent performance — count consecutive losing days, and on a
# streak switch to a defensive state.
#
# They left the hard part open on stage: once you are IN the drawdown state,
# what do you do — liquidate, pause, or attenuate? We attenuate. Liquidating
# realises the loss and forfeits the recovery; pausing means a strategy that
# stops exactly when the streak was about to end. Cutting size keeps you in the
# game at reduced exposure, which is also what a real desk does.
#
#   NORMAL   --(DD_TRIGGER consecutive losses)-->  DEFENSIVE   (size x DD_SIZE)
#   DEFENSIVE --(DD_RECOVER consecutive wins)-->   NORMAL
#
# States are counted over TRADES, not days — our trades are 60-minute event
# windows and several can close in one session.
DD_ENABLED  = True
DD_TRIGGER  = 4      # consecutive losing trades that flip us defensive
DD_RECOVER  = 2      # consecutive winners that restore full size
DD_SIZE     = 0.40   # size multiplier while defensive

# ------------------------------------------------- cTrader contract specs ----
# Loaded DYNAMICALLY from DP/instruments.json "ctrader" blocks — values synced
# from the broker's own Open API (fetch_ctrader_symbols.py). Correct any
# symbol in the JSON — no code changes needed. No "ctrader" key = skipped.
_INSTRUMENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DP", "instruments.json")
with open(_INSTRUMENTS_FILE, encoding="utf-8") as _f:
    SPECS = {name: v["ctrader"]
             for name, v in json.load(_f)["instruments"].items() if "ctrader" in v}
# lot step: whole shares step 1, everything else 0.01 unless the API said so
for _s in SPECS.values():
    _s.setdefault("lot_step", 1.0 if _s["min_lots"] >= 1 else 0.01)
    _s["pip"] = 10.0 ** (-_s["pip_position"])

# WIRE-TRUTH CONVERSIONS — imported from the PRODUCTION bridge so simulation
# and live order placement share ONE implementation. Every simulated trade
# records the exact ProtoOANewOrderReq numbers — the ledger is a replayable
# order stream.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "cTrader"))
from ctrader_bridge import lots_to_volume, pips_to_relative  # noqa: E402


def quote_ccy_is_usd(name):
    """PnL below assumes USD quote ccy. USD_JPY etc. are quoted in JPY/CHF/...:
    convert PnL by dividing by the exit price (good approximation)."""
    return name.split("_")[-1] not in ("JPY", "CAD", "CHF", "CNY", "MXN")


def latest_trades_csv():
    files = [f for f in glob.glob(os.path.join(TP_DIR, "trades_*.csv"))
             if "_summary" not in f]
    if not files:
        sys.exit(f"❌ No trades_*.csv in {TP_DIR} — run simulate_tp_sl.py first.")
    # MOST RECENTLY WRITTEN, not alphabetical — 'trades_tponly_*' used to sort
    # after 'trades_<date>_*' and silently hijack default runs.
    return max(files, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description="cTrader/Pepperstone account simulation over Layer-2 trades.")
    ap.add_argument("--csv", default=None, help="Layer-2 trades CSV (default: newest in tp_sl_results/)")
    ap.add_argument("--balance", type=float, default=BALANCE_DEFAULT)
    ap.add_argument("--leverage", type=int, default=ACCOUNT_LEVERAGE)
    ap.add_argument("--sizing", choices=["margin", "risk"], default="margin",
                    help="margin (default): commit --margin-per-trade %% of equity as "
                         "margin at full class leverage — CFD style. "
                         "risk: conservative %%-risk sizing (--risk-pct).")
    ap.add_argument("--margin-per-trade", type=float, default=MARGIN_PER_TRADE_PCT,
                    help=f"margin mode: %% of equity committed per trade (default {MARGIN_PER_TRADE_PCT})")
    ap.add_argument("--max-margin-total", type=float, default=MAX_TOTAL_MARGIN_PCT,
                    help=f"max %% of equity as margin across open trades (default {MAX_TOTAL_MARGIN_PCT})")
    ap.add_argument("--stopout-buffer", type=float, default=STOPOUT_BUFFER_PP,
                    help=f"worst-case margin level must stay >= stop-out + this many "
                         f"percentage points (default {STOPOUT_BUFFER_PP})")
    ap.add_argument("--risk-pct", type=float, default=RISK_PCT,
                    help=f"risk mode: %% of equity risked per trade (default {RISK_PCT})")
    ap.add_argument("--demo-caps", action="store_true",
                    help="use DEMO max-lot caps. Default: LIVE caps — instruments with "
                         "a 'max_lots_live' in the registry (VIX: 1000 vs demo 100) use "
                         "the live limit, since production trades the live account.")
    ap.add_argument("--no-sl", action="store_true",
                    help="TP-ONLY mode: positions carry no stop-loss (exit = TP limit or "
                         "60-min timeout). Risk is UNDEFINED per trade, so the stop-out "
                         "solve is skipped — margin caps are the only protection.")
    ap.add_argument("--no-drawdown-guard", action="store_true",
                    help="Disable the drawdown FSM (default: ON). It cuts size "
                         f"to x{DD_SIZE} after {DD_TRIGGER} consecutive losing "
                         f"trades and restores after {DD_RECOVER} wins.")
    ap.add_argument("--dd-trigger", type=int, default=DD_TRIGGER,
                    help=f"consecutive losses that flip to DEFENSIVE (default {DD_TRIGGER})")
    ap.add_argument("--dd-recover", type=int, default=DD_RECOVER,
                    help=f"consecutive wins that restore full size (default {DD_RECOVER})")
    ap.add_argument("--dd-size", type=float, default=DD_SIZE,
                    help=f"size multiplier while DEFENSIVE (default {DD_SIZE})")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    csv_path = args.csv or latest_trades_csv()
    df = pd.read_csv(csv_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    # BURST ALLOCATION: one post fires many instruments in the same minute —
    # allocate margin by MARGIN-RELATIVE EXPECTED RETURN: |pred| x class
    # leverage (the % return on committed margin if the prediction lands).
    # Raw |pred| alone starves FX: EURUSD 0.4% x 30:1 = 12% on margin beats
    # ETH 1.0% x 2:1 = 2%, yet sorted by raw pred ETH ate the budget first.
    _lev = {n: s.get("leverage", 1) for n, s in SPECS.items()}
    df["_conv"] = (pd.to_numeric(df.get("pred_pct"), errors="coerce").abs().fillna(0.0)
                   * df["instrument"].map(_lev).fillna(0.0))
    df = df.sort_values(["entry_time", "_conv"],
                        ascending=[True, False]).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(OUT_DIR, f"ctrader_{stamp}.csv")

    print("=" * 78)
    print("  LAYER-3 cTRADER ACCOUNT SIMULATION  (Pepperstone mechanics)")
    print(f"  Balance ${args.balance:.0f} | leverage 1:{args.leverage} | "
          f"stop-out {STOP_OUT_LEVEL:.0f}% margin level | "
          f"{'DEMO' if args.demo_caps else 'LIVE'} max-lot caps")
    if args.sizing == "margin":
        print(f"  Sizing: MARGIN mode — {args.margin_per_trade}% of equity as margin per "
              f"trade at full class leverage (CFD style)")
    else:
        print(f"  Sizing: RISK mode — {args.risk_pct}% of equity risked per trade")
    dd_enabled = DD_ENABLED and not args.no_drawdown_guard
    dd_trigger, dd_recover, dd_size = args.dd_trigger, args.dd_recover, args.dd_size
    print(f"  Drawdown FSM: " + (f"ON — {dd_trigger} losses -> size x{dd_size}, "
          f"{dd_recover} wins -> full" if dd_enabled else "OFF (--no-drawdown-guard)"))
    print(f"  Guards: total margin <= {args.max_margin_total}% of equity; per-entry "
          f"stop-out solve keeps worst-case margin level >= "
          f"{STOP_OUT_LEVEL + args.stopout_buffer:.0f}% even if ALL open SLs hit")
    print(f"  Trades in: {csv_path} ({len(df)} events)")
    print("=" * 78)

    balance = args.balance
    open_pos = []          # [{instrument, exit_time, margin, risk, pnl_usd, row}]
    ledger, skips = [], {}
    peak, max_dd = balance, 0.0
    min_ml_seen = float("inf")
    min_worst_ml = [float("inf")]   # worst-case projected margin level (all SLs hit)

    # ---- drawdown FSM state (see the DD_* block at the top) ---------------
    fsm = {"state": "NORMAL", "losses": 0, "wins": 0,
           "entered": 0, "trades_defensive": 0}

    def close_expired(now):
        nonlocal balance, peak, max_dd
        for p in [p for p in open_pos if p["exit_time"] <= now]:
            balance += p["pnl_usd"]
            open_pos.remove(p)
            peak = max(peak, balance)
            max_dd = max(max_dd, (peak - balance) / peak * 100)
            ledger.append(p["row"])
            # ---- FSM transitions, driven by REALISED outcomes only --------
            if not dd_enabled:
                continue
            if p["pnl_usd"] < 0:
                fsm["losses"] += 1
                fsm["wins"] = 0
                if fsm["state"] == "NORMAL" and fsm["losses"] >= dd_trigger:
                    fsm["state"] = "DEFENSIVE"
                    fsm["entered"] += 1
                    print(f"    🛡️  DRAWDOWN: {fsm['losses']} losses in a row "
                          f"at {now:%Y-%m-%d %H:%M} — size x{dd_size:.2f} "
                          f"until {dd_recover} wins")
            else:
                fsm["wins"] += 1
                fsm["losses"] = 0
                if fsm["state"] == "DEFENSIVE" and fsm["wins"] >= dd_recover:
                    fsm["state"] = "NORMAL"
                    print(f"    ✅ RECOVERED: {fsm['wins']} wins in a row "
                          f"at {now:%Y-%m-%d %H:%M} — full size restored")

    for _, t in df.iterrows():
        inst = t["instrument"]
        close_expired(t["entry_time"])

        if inst not in SPECS:
            skips["no_cfd_symbol"] = skips.get("no_cfd_symbol", 0) + 1
            continue
        if any(p["instrument"] == inst for p in open_pos):
            skips["position_open"] = skips.get("position_open", 0) + 1
            continue

        spec = SPECS[inst]
        symbol, contract, pip = spec["symbol"], spec["lot_size"], spec["pip"]
        min_lot, lot_step = spec["min_lots"], spec["lot_step"]
        # LIVE caps by default (production trades the live account): demo API
        # reports tighter limits (VIX 100 vs live 1000).
        max_lot = (spec["max_lots"] if args.demo_caps
                   else spec.get("max_lots_live", spec["max_lots"]))
        lev = min(args.leverage, spec["leverage"])
        entry, exitp = float(t["entry_price"]), float(t["exit_price"])
        direction = 1 if t["direction"] == "LONG" else -1
        sl_dist = abs(entry - float(t["sl_price"]))
        # SENTINEL-SL AUTO-DETECT: a TP-only Layer-2 CSV carries stops parked
        # ~999999% away. Treating them as real risk would size every trade to
        # zero — detect (>50% of entry) and handle the trade as no-SL.
        trade_no_sl = args.no_sl or (entry > 0 and sl_dist / entry > 0.5)
        if trade_no_sl and not args.no_sl:
            if skips.get("_sentinel_sl_notice") is None:
                print("  ℹ️  CSV contains sentinel (no-SL) stops — treating those trades "
                      "as TP-only. Use --no-sl to silence this notice.")
            skips["_sentinel_sl_notice"] = skips.get("_sentinel_sl_notice", 0) + 1
        if not trade_no_sl and sl_dist <= 0:
            skips["bad_sl"] = skips.get("bad_sl", 0) + 1
            continue

        equity      = balance                        # realized (conservative)
        used_margin = sum(p["margin"] for p in open_pos)
        open_risk   = sum(p["risk"] for p in open_pos)

        # convert 1 unit of quote-ccy PnL to USD (approx: divide by price)
        usd_conv = 1.0 if quote_ccy_is_usd(inst) else 1.0 / exitp

        margin_per_lot = contract * entry * usd_conv / lev
        risk_per_lot   = 0.0 if trade_no_sl else sl_dist * contract * usd_conv

        # ---- target lots per sizing mode ----
        # DRAWDOWN FSM: while DEFENSIVE, commit less. Applied to the TARGET so
        # every downstream guard (margin budget, stop-out solve, lot caps) still
        # binds normally — this can only ever make a position smaller.
        _dd_mult = dd_size if (dd_enabled and fsm["state"] == "DEFENSIVE") else 1.0
        if _dd_mult < 1.0:
            fsm["trades_defensive"] += 1
        if args.sizing == "margin":
            lots_target = (equity * args.margin_per_trade * _dd_mult / 100) / margin_per_lot
        else:
            lots_target = (equity * args.risk_pct * _dd_mult / 100) / max(risk_per_lot, 1e-9)

        # ---- total-margin budget ----
        margin_budget = equity * args.max_margin_total / 100 - used_margin
        if margin_budget <= 0:
            skips["budget_full"] = skips.get("budget_full", 0) + 1
            continue
        lots_budget = margin_budget / margin_per_lot

        # ---- THE stop-out guard (exact, solved per entry) ----
        # Require: worst-case equity if ALL open SLs hit  >=  k x used margin
        #   equity - open_risk - l*risk_per_lot >= k*(used_margin + l*margin_per_lot)
        if trade_no_sl:
            lots_guard = float("inf")     # no SL -> no defined risk to solve for
        else:
            k = (STOP_OUT_LEVEL + args.stopout_buffer) / 100.0
            denom = risk_per_lot + k * margin_per_lot
            lots_guard = (equity - open_risk - k * used_margin) / denom if denom > 0 else 0.0
            if lots_guard <= 0:
                skips["stopout_guard"] = skips.get("stopout_guard", 0) + 1
                continue

        lots = np.floor(min(lots_target, lots_budget, lots_guard, max_lot)
                        / lot_step) * lot_step
        lots = round(lots, 2)
        if lots < min_lot:
            skips["lot_too_small"] = skips.get("lot_too_small", 0) + 1
            continue

        margin = lots * margin_per_lot
        risk   = lots * risk_per_lot
        # cTrader hard rule: estimated margin can never exceed the balance
        if used_margin + margin > equity:
            skips["margin_exceeds_balance"] = skips.get("margin_exceeds_balance", 0) + 1
            continue
        pnl  = direction * (exitp - entry) * lots * contract * usd_conv
        pips = direction * (exitp - entry) / pip

        ml_now = (equity / (used_margin + margin) * 100) if (used_margin + margin) > 0 else float("inf")
        min_ml_seen = min(min_ml_seen, ml_now)
        ml_worst = ((equity - open_risk - risk) / (used_margin + margin) * 100
                    if (used_margin + margin) > 0 else float("inf"))
        min_worst_ml[0] = min(min_worst_ml[0], ml_worst)

        # cTrader New-Market-Order style numbers for THIS position
        tp_price  = float(t.get("tp_price", exitp))
        sl_price  = float(t.get("sl_price", entry))
        tp_dist   = abs(tp_price - entry)
        pip_value = lots * contract * pip * usd_conv          # $ per pip, this qty
        est_tp    = lots * contract * tp_dist * usd_conv      # $ at take-profit
        est_sl    = risk                                      # $ at stop-loss

        # exact wire numbers (SAME helpers as the production bridge)
        wire_volume, _wl = lots_to_volume(lots, spec)
        wire_tp = pips_to_relative(tp_dist / pip, spec)
        wire_sl = None if trade_no_sl else pips_to_relative(sl_dist / pip, spec)

        row = {
            "entry_time": t["entry_time"].isoformat(), "exit_time": t["exit_time"].isoformat(),
            "instrument": inst, "symbol": symbol, "direction": t["direction"],
            "lots": round(lots, 2), "entry": entry, "exit": exitp,
            "tp_price": tp_price, "sl_price": None if trade_no_sl else sl_price,
            "tp_pips": round(tp_dist / pip, 1),
            "sl_pips": None if trade_no_sl else round(sl_dist / pip, 1),
            "tp_pct": round(tp_dist / entry * 100, 3),
            "sl_pct": None if trade_no_sl else round(sl_dist / entry * 100, 3),
            "pip_value_usd": round(pip_value, 2),
            "est_profit_tp_usd": round(est_tp, 2), "est_loss_sl_usd": round(-est_sl, 2),
            "margin_usd": round(margin, 2),
            "margin_pct_equity": round(margin / equity * 100, 1),
            "pips": round(pips, 1), "risk_usd": round(risk, 2),
            "pnl_usd": round(pnl, 2), "outcome": t["outcome"],
            "balance_after": None,  # filled at close
            "wire_volume": wire_volume, "wire_relative_tp": wire_tp,
            "wire_relative_sl": wire_sl,
            "post_id": t.get("post_id", ""), "pred_pct": t.get("pred_pct", ""),
        }
        open_pos.append({"instrument": inst, "exit_time": t["exit_time"],
                         "margin": margin, "risk": risk, "pnl_usd": pnl, "row": row})

        # backtest-style per-trade line (cTrader order-ticket numbers)
        _lots_fmt = f"{lots:>8.0f}" if lot_step >= 1 else f"{lots:>8.2f}"
        print(f"  {t['entry_time']:%Y-%m-%d %H:%M}  {symbol:<8} {t['direction']:<5} "
              f"{_lots_fmt} lots @ {entry:<10.5g} pipval ${pip_value:.2f}")
        _sl_str = ("SL --         (no stop-loss)" if trade_no_sl else
                   f"SL {sl_price:<10.5g} ({sl_dist/pip:>6.0f} pips, "
                   f"{sl_dist/entry*100:>5.2f}%) est -${est_sl:,.2f}")
        print(f"      TP {tp_price:<10.5g} ({tp_dist/pip:>6.0f} pips, {tp_dist/entry*100:>5.2f}%) "
              f"est +${est_tp:,.2f}  |  {_sl_str}")
        print(f"      wire: volume {wire_volume}  relativeTP {wire_tp}"
              + (f"  relativeSL {wire_sl}" if wire_sl is not None else "  (no SL)"))
        print(f"      margin ${margin:,.2f} ({margin/equity*100:.1f}% eq)  ->  "
              f"{t['outcome']:<7} {pnl:+,.2f} USD")

    close_expired(pd.Timestamp.max.tz_localize("UTC"))
    bal = args.balance
    for r in sorted(ledger, key=lambda r: r["exit_time"]):
        bal += r["pnl_usd"]
        r["balance_after"] = round(bal, 2)

    led = pd.DataFrame(sorted(ledger, key=lambda r: r["entry_time"]))
    led.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")

    n = len(led)
    wins = int((led["pnl_usd"] > 0).sum()) if n else 0
    print(f"\n{'='*78}\n  ACCOUNT SUMMARY\n{'='*78}")
    print(f"  Executed trades : {n}  (win {wins}/{max(n,1)} = {wins/max(n,1):.1%})")
    print(f"  Start balance   : ${args.balance:,.2f}")
    print(f"  Final balance   : ${balance:,.2f}   ({(balance/args.balance-1)*100:+.1f}%)")
    print(f"  Max drawdown    : {max_dd:.1f}%")
    if dd_enabled:
        _n = len(ledger) or 1
        print(f"  Drawdown FSM    : entered DEFENSIVE {fsm['entered']}x; "
              f"{fsm['trades_defensive']} of {_n} trades sized down "
              f"({fsm['trades_defensive'] / _n:.0%})"
              + ("   [never triggered]" if not fsm["entered"] else ""))
        print(f"                    compare with --no-drawdown-guard to see "
              f"whether it actually helped")
    print(f"  Min margin level at entry     : "
          f"{'n/a' if min_ml_seen == float('inf') else f'{min_ml_seen:.0f}%'}"
          f"  (stop-out at {STOP_OUT_LEVEL:.0f}%)")
    print(f"  Min WORST-CASE margin level   : "
          f"{'n/a' if min_worst_ml[0] == float('inf') else f'{min_worst_ml[0]:.0f}%'}"
          f"  (all open SLs hit at once — guard floor "
          f"{STOP_OUT_LEVEL + args.stopout_buffer:.0f}%)")
    if n:
        by = led.groupby("instrument")["pnl_usd"].agg(["count", "sum"]).sort_values("sum", ascending=False)
        print(f"\n  {'instrument':<10} {'n':>4} {'PnL $':>10}")
        for inst, r in by.iterrows():
            print(f"  {inst:<10} {int(r['count']):>4} {r['sum']:>+10.2f}")
    print(f"\n  Skipped: {skips}")
    meta = {"source_csv": os.path.basename(csv_path),
            "params": {"balance": args.balance, "leverage": args.leverage,
                       "sizing": args.sizing, "margin_per_trade": args.margin_per_trade,
                       "max_margin_total": args.max_margin_total,
                       "stopout_buffer": args.stopout_buffer,
                       "risk_pct": args.risk_pct, "stop_out": STOP_OUT_LEVEL,
                       "no_sl": args.no_sl, "demo_caps": args.demo_caps},
            "final_balance": round(balance, 2), "return_pct": round((balance/args.balance-1)*100, 2),
            "max_drawdown_pct": round(max_dd, 2), "trades": n, "wins": wins,
            "min_margin_level": None if min_ml_seen == float("inf") else round(min_ml_seen, 1),
            "min_worst_case_margin_level": None if min_worst_ml[0] == float("inf") else round(min_worst_ml[0], 1),
            "skips": skips}
    with open(out_path.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  💾 Ledger : {out_path}")
    print(f"  💾 Summary: {out_path.replace('.csv', '_summary.json')}")
    print("✅ Done")


if __name__ == "__main__":
    main()
