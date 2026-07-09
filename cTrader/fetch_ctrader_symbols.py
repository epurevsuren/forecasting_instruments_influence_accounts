"""
fetch_ctrader_symbols.py  —  PULL REAL SYMBOL SPECS FROM cTRADER OPEN API
==========================================================================
Authenticates against the Spotware Open API (JSON over WebSocket) with your
Pepperstone demo credentials and updates the "ctrader" blocks in
DP/instruments.json with the REAL values — no more screenshot-guessing:

    lot_size      ProtoOASymbol.lotSize    (units per 1.00 lot)
    pip_position  ProtoOASymbol.pipPosition
    digits        price digits
    min_lots / max_lots / lot_step   (from min/max/stepVolume)
    swap_long / swap_short           (pips per day)

Also prints your trader account (balance, leverage) so the simulator's
--balance/--leverage match reality.

CREDENTIALS (OpenAPI.txt in the project root — NOT committed anywhere):
    Access token      <token>          <- you already have these two
    Refresh token     <token>
    Client id         <app client id>  <- ADD from Open API portal ->
    Client secret     <app secret>        your application's page
Labels are matched case-insensitively; the value may follow on the same
line (after ':') or on the next line. Env vars CTRADER_CLIENT_ID /
CTRADER_CLIENT_SECRET / CTRADER_ACCESS_TOKEN override the file.

USAGE
-----
    uv add websockets
    uv run python fetch_ctrader_symbols.py               # demo, updates registry
    uv run python fetch_ctrader_symbols.py --dry-run     # show diff only
    uv run python fetch_ctrader_symbols.py --live        # live environment
    uv run python fetch_ctrader_symbols.py --expected-margin VIX 100
                                    # ask the SERVER for the est. margin of
                                    # 100 lots VIX — cross-check the simulator

OTHER USEFUL CALLS THIS UNLOCKS LATER (same connection):
    ProtoOAGetTrendbarsReq   — M1 bars from Pepperstone's OWN feed (the
                               actually-tradeable prices; can fill VIX/US10Y
                               gaps in the IBKR cache)
    ProtoOASubscribeSpotsReq — live bid/ask -> real spread stats per symbol
    ProtoOANewOrderReq       — the production bridge (market order + TP/SL)
"""
import os
import sys
import json
import ssl
import asyncio
import argparse
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
INSTRUMENTS_FILE = os.path.join(_ROOT, "DP", "instruments.json")
BACKUP_DIR       = os.path.join(_ROOT, "DP", "config_backups")

HOST_DEMO = "wss://demo.ctraderapi.com:5036"
HOST_LIVE = "wss://live.ctraderapi.com:5036"

# ProtoOA payload types (Open API 2.0)
PT = dict(APP_AUTH_REQ=2100, APP_AUTH_RES=2101, ACC_AUTH_REQ=2102, ACC_AUTH_RES=2103,
          ERROR_RES=2142, SYMBOLS_LIST_REQ=2114, SYMBOLS_LIST_RES=2115,
          SYMBOL_BY_ID_REQ=2116, SYMBOL_BY_ID_RES=2117,
          TRADER_REQ=2121, TRADER_RES=2122,
          ACCOUNTS_BY_TOKEN_REQ=2149, ACCOUNTS_BY_TOKEN_RES=2150,
          EXPECTED_MARGIN_REQ=2139, EXPECTED_MARGIN_RES=2140,
          HEARTBEAT=51)


def load_creds(path):
    """Parse OpenAPI.txt: 'Label[: ] value' or label line followed by value line."""
    keys = {"access token": "access_token", "refresh token": "refresh_token",
            "client id": "client_id", "client secret": "client_secret",
            "account id": "account_id"}
    out = {}
    if os.path.exists(path):
        lines = [l.strip() for l in open(path, encoding="utf-8").read().splitlines()]
        for i, line in enumerate(lines):
            low = line.lower()
            for label, key in keys.items():
                if low.startswith(label):
                    rest = line[len(label):].strip(" :\t")
                    if rest:
                        out[key] = rest
                    else:                       # value on the next non-empty line
                        for nxt in lines[i + 1:]:
                            if nxt:
                                out[key] = nxt
                                break
    out.setdefault("client_id",     os.environ.get("CTRADER_CLIENT_ID", ""))
    out.setdefault("client_secret", os.environ.get("CTRADER_CLIENT_SECRET", ""))
    out.setdefault("access_token",  os.environ.get("CTRADER_ACCESS_TOKEN", ""))
    if os.environ.get("CTRADER_CLIENT_ID"):
        out["client_id"] = os.environ["CTRADER_CLIENT_ID"]
    if os.environ.get("CTRADER_CLIENT_SECRET"):
        out["client_secret"] = os.environ["CTRADER_CLIENT_SECRET"]
    return out


class OpenApi:
    def __init__(self, host):
        self.host, self.ws, self._id = host, None, 0

    async def connect(self):
        import websockets
        ctx = ssl.create_default_context()
        self.ws = await websockets.connect(self.host, ssl=ctx, max_size=2**24)

    async def call(self, ptype, payload, expect):
        self._id += 1
        await self.ws.send(json.dumps(
            {"clientMsgId": str(self._id), "payloadType": ptype, "payload": payload}))
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=30))
            mt = msg.get("payloadType")
            if mt == PT["HEARTBEAT"]:
                continue
            if mt == PT["ERROR_RES"] or "errorCode" in (msg.get("payload") or {}):
                p = msg.get("payload", {})
                sys.exit(f"❌ API error: {p.get('errorCode')} — {p.get('description')}")
            if mt == expect:
                return msg.get("payload", {})
            # unrelated push (spots etc.) — keep waiting


async def main_async(args):
    creds = load_creds(args.creds)
    missing = [k for k in ("client_id", "client_secret", "access_token") if not creds.get(k)]
    if missing:
        sys.exit(f"❌ Missing credentials: {', '.join(missing)}.\n"
                 f"   Add 'Client id' / 'Client secret' lines to {args.creds}\n"
                 f"   (from https://openapi.ctrader.com -> your application).")

    api = OpenApi(HOST_LIVE if args.live else HOST_DEMO)
    print(f"🔌 Connecting to {api.host} ...")
    await api.connect()

    await api.call(PT["APP_AUTH_REQ"],
                   {"clientId": creds["client_id"], "clientSecret": creds["client_secret"]},
                   PT["APP_AUTH_RES"])
    print("✅ Application authenticated")

    accs = await api.call(PT["ACCOUNTS_BY_TOKEN_REQ"],
                          {"accessToken": creds["access_token"]},
                          PT["ACCOUNTS_BY_TOKEN_RES"])
    accounts = accs.get("ctidTraderAccount", [])
    if not accounts:
        sys.exit("❌ Access token has no trading accounts attached.")
    acc = accounts[0]
    if creds.get("account_id"):
        acc = next((a for a in accounts
                    if str(a["ctidTraderAccountId"]) == str(creds["account_id"])), acc)
    acc_id = acc["ctidTraderAccountId"]
    print(f"✅ Account {acc.get('traderLogin', '?')} "
          f"({'LIVE' if acc.get('isLive') else 'DEMO'}, id {acc_id})")

    await api.call(PT["ACC_AUTH_REQ"],
                   {"ctidTraderAccountId": acc_id, "accessToken": creds["access_token"]},
                   PT["ACC_AUTH_RES"])

    trader = (await api.call(PT["TRADER_REQ"], {"ctidTraderAccountId": acc_id},
                             PT["TRADER_RES"])).get("trader", {})
    bal = trader.get("balance", 0) / 100.0
    lev = trader.get("leverageInCents", 0) / 100.0
    print(f"💰 Balance: ${bal:,.2f}   account leverage: 1:{lev:.0f}")
    print(f"   -> simulator flags: --balance {bal:.0f} --leverage {lev:.0f}")

    # ---- symbols ----
    listing = await api.call(PT["SYMBOLS_LIST_REQ"], {"ctidTraderAccountId": acc_id},
                             PT["SYMBOLS_LIST_RES"])
    by_name = {s["symbolName"].upper(): s["symbolId"] for s in listing.get("symbol", [])}
    print(f"📚 Broker offers {len(by_name)} symbols")

    if args.search:
        for term in args.search:
            hits = sorted(n for n in by_name if term.upper() in n)
            print(f"\n🔎 '{term}': {len(hits)} match(es)")
            for h in hits[:40]:
                print(f"   {h}")
        await api.ws.close()
        return

    reg = json.load(open(INSTRUMENTS_FILE, encoding="utf-8"))
    wanted = {}   # symbolId -> instrument name
    for name, inst in reg["instruments"].items():
        ct = inst.get("ctrader")
        if not ct:
            continue
        sid = by_name.get(ct["symbol"].upper())
        if sid is None:
            print(f"  ⚠️  {name}: '{ct['symbol']}' not offered by this broker — "
                  f"check the exact name in cTrader and fix instruments.json")
            continue
        wanted[sid] = name

    details = await api.call(PT["SYMBOL_BY_ID_REQ"],
                             {"ctidTraderAccountId": acc_id, "symbolId": list(wanted)},
                             PT["SYMBOL_BY_ID_RES"])

    print(f"\n{'instrument':<10} {'field':<14} {'registry':>14} {'broker':>14}")
    changes = 0
    for s in details.get("symbol", []):
        name = wanted.get(s["symbolId"])
        ct = reg["instruments"][name]["ctrader"]
        lot_units = s["lotSize"] / 100.0            # volumes are in 0.01 units
        real = {
            "lot_size":     lot_units,
            "pip_position": s["pipPosition"],
            "digits":       s.get("digits"),
            "min_lots":     s["minVolume"] / s["lotSize"],
            "max_lots":     s["maxVolume"] / s["lotSize"],
            "lot_step":     s["stepVolume"] / s["lotSize"],
            "swap_long":    s.get("swapLong"),
            "swap_short":   s.get("swapShort"),
        }
        for k, v in real.items():
            if v is None:
                continue
            old = ct.get(k)
            if old != v:
                print(f"{name:<10} {k:<14} {str(old):>14} {str(v):>14}")
                changes += 1
            ct[k] = v
        ct["_verified"] = f"OpenAPI {datetime.date.today()}"

    if args.expected_margin:
        sym, lots = args.expected_margin
        sid = next((i for i, n in wanted.items() if n == sym.upper()), None)
        if sid:
            ct = reg["instruments"][sym.upper()]["ctrader"]
            vol = int(float(lots) * ct["lot_size"] * 100)
            em = await api.call(PT["EXPECTED_MARGIN_REQ"],
                                {"ctidTraderAccountId": acc_id, "symbolId": sid,
                                 "volume": [vol]}, PT["EXPECTED_MARGIN_RES"])
            for m in em.get("margin", []):
                print(f"\n🧮 Server expected margin for {lots} lots {sym}: "
                      f"buy ${m.get('buyMargin', 0)/100:,.2f} / "
                      f"sell ${m.get('sellMargin', 0)/100:,.2f}  "
                      f"(cross-check the simulator's margin_usd)")

    if changes == 0:
        print("\n✅ Registry already matches the broker exactly.")
    elif args.dry_run:
        print(f"\n🔎 DRY RUN — {changes} field(s) differ; run without --dry-run to apply.")
    else:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(BACKUP_DIR, f"instruments.{ts}.json"), "w",
                  encoding="utf-8") as f:
            f.write(open(INSTRUMENTS_FILE, encoding="utf-8").read())
        tmp = INSTRUMENTS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        json.load(open(tmp, encoding="utf-8"))
        os.replace(tmp, INSTRUMENTS_FILE)
        print(f"\n💾 Applied {changes} correction(s) to {INSTRUMENTS_FILE} "
              f"(backup in {BACKUP_DIR}/)")
    await api.ws.close()
    print("✅ Done")


def main():
    ap = argparse.ArgumentParser(description="Sync instruments.json ctrader specs from the Open API.")
    ap.add_argument("--creds", default=os.path.join(_ROOT, "OpenAPI.txt"),
                    help="credentials file (default <project>/OpenAPI.txt)")
    ap.add_argument("--live", action="store_true", help="live environment (default demo)")
    ap.add_argument("--dry-run", action="store_true", help="show differences, change nothing")
    ap.add_argument("--expected-margin", nargs=2, metavar=("SYMBOL", "LOTS"), default=None,
                    help="also ask the server for expected margin, e.g. --expected-margin VIX 100")
    ap.add_argument("--search", nargs="+", metavar="TERM", default=None,
                    help="list broker symbol names containing TERM(s) and exit — for "
                         "finding the exact names of OIL/NATGAS/XLF etc. "
                         "(e.g. --search OIL GAS XLF CRUDE XTI)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
