"""
ctrader_bridge.py  —  ORDER BRIDGE on the OFFICIAL ctrader-open-api LIBRARY
============================================================================
Production bridge v1 (Spotware's own Python package, protobuf over TCP):
places a MARKET order with server-side TP (and optional SL) using the exact
cTrader wire conversions — the two places where a wrong factor means a 100x
mis-sized order:

  LOTS -> VOLUME     ProtoOANewOrderReq.volume is in CENTS OF UNITS
                     (0.01 units):   volume = lots x lot_size_units x 100
                     e.g. 100 lots VIX (lot_size 1)  -> volume 10_000
                          0.10 lots EURUSD (100_000) -> volume 1_000_000

  PIPS -> RELATIVE   relativeTakeProfit / relativeStopLoss are in 1/100_000
                     of a price unit: relative = pips x 10^(5 - pip_position)
                     e.g. VIX (pip_position 1): 15 pips -> 150_000 (=1.50)
                          EURUSD (pip_position 4): 63.3 pips -> 633 (=0.00633)

Specs (lot_size, pip_position, min/max/step lots) come from
DP/instruments.json — kept truthful by fetch_ctrader_symbols.py.

SAFETY: --dry-run is the DEFAULT — prints the exact protobuf numbers without
connecting. Sending requires an explicit --send. Demo endpoint unless --live.

SETUP
-----
    uv add ctrader-open-api        # Spotware official (Twisted + protobuf)

USAGE
-----
    uv run python ctrader_bridge.py VIX sell --lots 100 --tp-pips 15
    uv run python ctrader_bridge.py VIX sell --lots 100 --tp-pips 15 --sl-pips 7 --send
    uv run python ctrader_bridge.py EUR_USD buy --lots 0.1 --tp-pips 63.3 --send
    uv run python ctrader_bridge.py --positions          # list open positions
"""
import os
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
INSTRUMENTS_FILE = os.path.join(_ROOT, "DP", "instruments.json")

sys.path.insert(0, _HERE)
from fetch_ctrader_symbols import load_creds   # same OpenAPI.txt parser


# ======================================================= conversion helpers ----
def load_spec(instrument: str) -> dict:
    reg = json.load(open(INSTRUMENTS_FILE, encoding="utf-8"))["instruments"]
    key = instrument.upper()
    if key not in reg or "ctrader" not in reg[key]:
        tradeable = [k for k, v in reg.items() if "ctrader" in v]
        sys.exit(f"❌ '{instrument}' has no ctrader spec. Tradeable: {', '.join(tradeable)}")
    return reg[key]["ctrader"]


def lots_to_volume(lots: float, spec: dict) -> int:
    """lots -> ProtoOA volume (cents of units), clamped to min/step/max."""
    step = spec.get("lot_step", 0.01)
    lots = max(spec["min_lots"], min(spec["max_lots"], round(lots / step) * step))
    return int(round(lots * spec["lot_size"] * 100)), lots


def pips_to_relative(pips: float, spec: dict) -> int:
    """pips -> relative TP/SL (1/100_000 price units)."""
    return int(round(pips * 10 ** (5 - spec["pip_position"])))


def describe_order(instrument, side, lots, tp_pips, sl_pips):
    spec = load_spec(instrument)
    volume, lots_clamped = lots_to_volume(lots, spec)
    rel_tp = pips_to_relative(tp_pips, spec) if tp_pips else None
    rel_sl = pips_to_relative(sl_pips, spec) if sl_pips else None
    pip = 10 ** (-spec["pip_position"])
    print(f"📋 ORDER  {spec['symbol']}  {side.upper()}")
    print(f"   lots      : {lots_clamped:g}  (requested {lots:g}; min {spec['min_lots']}, "
          f"step {spec.get('lot_step', 0.01)}, max {spec['max_lots']})")
    print(f"   volume    : {volume}  (cents of units; {volume/100:g} units)")
    if rel_tp is not None:
        print(f"   TP        : {tp_pips:g} pips = {tp_pips*pip:g} price units "
              f"-> relativeTakeProfit {rel_tp}")
    if rel_sl is not None:
        print(f"   SL        : {sl_pips:g} pips = {sl_pips*pip:g} price units "
              f"-> relativeStopLoss {rel_sl}")
    else:
        print(f"   SL        : none (TP-only)")
    return spec, volume, rel_tp, rel_sl


# ================================================================ live send ----
def send_order(args, spec, volume, rel_tp, rel_sl):
    try:
        from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAApplicationAuthReq, ProtoOAAccountAuthReq,
            ProtoOAGetAccountListByAccessTokenReq, ProtoOANewOrderReq,
            ProtoOAReconcileReq, ProtoOAErrorRes, ProtoOAExecutionEvent,
            ProtoOASymbolsListReq)
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
            ProtoOAOrderType, ProtoOATradeSide)
        from twisted.internet import reactor
    except ImportError:
        sys.exit("❌ ctrader-open-api not installed:  uv add ctrader-open-api")

    creds = load_creds(args.creds)
    missing = [k for k in ("client_id", "client_secret", "access_token") if not creds.get(k)]
    if missing:
        sys.exit(f"❌ Missing credentials: {', '.join(missing)} (see OpenAPI.txt)")

    host = EndPoints.PROTOBUF_LIVE_HOST if args.live else EndPoints.PROTOBUF_DEMO_HOST
    client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
    state = {"acc": None, "symbol_id": None}

    def fail(err):
        print(f"❌ {err}")
        if reactor.running:
            reactor.stop()

    def on_connected(_):
        print(f"🔌 Connected to {host}")
        req = ProtoOAApplicationAuthReq()
        req.clientId = creds["client_id"]
        req.clientSecret = creds["client_secret"]
        d = client.send(req)
        d.addCallbacks(after_app_auth, fail)

    def after_app_auth(_):
        print("✅ Application authenticated")
        req = ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = creds["access_token"]
        client.send(req).addCallbacks(after_accounts, fail)

    def after_accounts(msg):
        res = Protobuf.extract(msg)
        accounts = list(res.ctidTraderAccount)
        if not accounts:
            return fail("access token has no accounts")
        acc = accounts[0]
        state["acc"] = acc.ctidTraderAccountId
        print(f"✅ Account {acc.traderLogin} ({'LIVE' if acc.isLive else 'DEMO'})")
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = state["acc"]
        req.accessToken = creds["access_token"]
        client.send(req).addCallbacks(after_acc_auth, fail)

    def after_acc_auth(_):
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = state["acc"]
        client.send(req).addCallbacks(after_symbols, fail)

    def after_symbols(msg):
        res = Protobuf.extract(msg)
        wanted = spec["symbol"].upper() if spec else None
        for s in res.symbol:
            if s.symbolName.upper() == wanted:
                state["symbol_id"] = s.symbolId
                break
        if args.positions:
            req = ProtoOAReconcileReq()
            req.ctidTraderAccountId = state["acc"]
            client.send(req).addCallbacks(show_positions, fail)
            return
        if state["symbol_id"] is None:
            return fail(f"symbol {wanted} not found at broker")
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = state["acc"]
        req.symbolId = state["symbol_id"]
        req.orderType = ProtoOAOrderType.MARKET
        req.tradeSide = (ProtoOATradeSide.BUY if args.side == "buy"
                         else ProtoOATradeSide.SELL)
        req.volume = volume
        if rel_tp is not None:
            req.relativeTakeProfit = rel_tp
        if rel_sl is not None:
            req.relativeStopLoss = rel_sl
        req.comment = "tweet-pipeline"
        print("📤 Sending market order...")
        client.send(req).addCallbacks(lambda m: None, fail)

    def show_positions(msg):
        res = Protobuf.extract(msg)
        print(f"📊 Open positions: {len(res.position)}")
        for p in res.position:
            td = p.tradeData
            print(f"   #{p.positionId} symbolId {td.symbolId} "
                  f"{'BUY' if td.tradeSide == 1 else 'SELL'} vol {td.volume} "
                  f"entry {p.price} swap {p.swap} P&L(gross) {p.moneyDigits}")
        reactor.stop()

    def on_message(_, msg):
        ev = Protobuf.extract(msg)
        if ev.DESCRIPTOR.name == "ProtoOAExecutionEvent":
            pos = getattr(ev, "position", None)
            order = getattr(ev, "order", None)
            print(f"⚡ ExecutionEvent: {ev.executionType}"
                  + (f"  position #{pos.positionId} @ {pos.price}" if pos and pos.positionId else "")
                  + (f"  order #{order.orderId}" if order and order.orderId else ""))
            if ev.executionType in (2, 3):        # ORDER_FILLED / ORDER_REPLACED
                print("✅ Order filled — position live with server-side TP/SL")
                reactor.stop()
        elif ev.DESCRIPTOR.name == "ProtoOAErrorRes":
            fail(f"{ev.errorCode}: {ev.description}")

    client.setConnectedCallback(on_connected)
    client.setDisconnectedCallback(lambda _, reason: print(f"🔌 Disconnected: {reason}"))
    client.setMessageReceivedCallback(on_message)
    client.startService()
    reactor.run()


# ==================================================================== main ----
def main():
    ap = argparse.ArgumentParser(description="cTrader order bridge (official ctrader-open-api).")
    ap.add_argument("instrument", nargs="?", help="registry name, e.g. VIX / EUR_USD / GOLD")
    ap.add_argument("side", nargs="?", choices=["buy", "sell"])
    ap.add_argument("--lots", type=float, help="volume in lots (converted per symbol spec)")
    ap.add_argument("--tp-pips", type=float, default=None, help="take-profit distance in pips")
    ap.add_argument("--sl-pips", type=float, default=None, help="stop-loss distance in pips (optional)")
    ap.add_argument("--send", action="store_true",
                    help="actually SEND the order (default is dry-run: print wire numbers only)")
    ap.add_argument("--positions", action="store_true", help="list open positions and exit")
    ap.add_argument("--live", action="store_true", help="live endpoint (default demo)")
    ap.add_argument("--creds", default=os.path.join(_ROOT, "OpenAPI.txt"))
    args = ap.parse_args()

    if args.positions:
        send_order(args, None, None, None, None)
        return

    if not (args.instrument and args.side and args.lots):
        sys.exit("usage: ctrader_bridge.py INSTRUMENT buy|sell --lots N [--tp-pips N] "
                 "[--sl-pips N] [--send]   or   --positions")

    spec, volume, rel_tp, rel_sl = describe_order(
        args.instrument, args.side, args.lots, args.tp_pips, args.sl_pips)

    if not args.send:
        print("\n🔎 DRY RUN — nothing sent. Add --send to place the order on "
              + ("LIVE" if args.live else "DEMO") + ".")
        return
    send_order(args, spec, volume, rel_tp, rel_sl)


if __name__ == "__main__":
    main()
