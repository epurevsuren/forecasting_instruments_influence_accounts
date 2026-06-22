"""
verify_fx_bid_ask.py
====================
Pulls the SAME recent AUD_USD bars four ways (MIDPOINT, BID_ASK, BID, ASK)
and prints them aligned, so you can see IBKR's BID_ASK field overloading
with your own eyes:

    BID_ASK  ->  open = time-avg bid, high = max ask, low = min bid, close = time-avg ask

That makes (high - low) on a BID_ASK bar the max spread excursion in the bar.

Run on the machine where TWS/Gateway is up (port 7497 paper / 7496 live).
    python verify_fx_bid_ask.py
"""
from ib_insync import IB, Forex

HOST, PORT, CID = "127.0.0.1", 7497, 21   # use a clientId not already in use


def last_bar(ib, contract, what):
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr="600 S",
        barSizeSetting="1 min", whatToShow=what, useRTH=False,
        formatDate=2,
    )
    return bars[-1] if bars else None


def main():
    ib = IB()
    ib.connect(HOST, PORT, clientId=CID)
    print(f"[ok] connected {HOST}:{PORT}")
    fx = Forex("AUDUSD")
    ib.qualifyContracts(fx)

    rows = {}
    for what in ("MIDPOINT", "BID_ASK", "BID", "ASK"):
        b = last_bar(ib, fx, what)
        rows[what] = b
        print(f"\n--- {what} ---")
        if b is None:
            print("  (no data)")
            continue
        print(f"  date    {b.date}")
        print(f"  open    {b.open}")
        print(f"  high    {b.high}")
        print(f"  low     {b.low}")
        print(f"  close   {b.close}")
        print(f"  volume  {b.volume}")
        print(f"  average {b.average}")
        print(f"  barCount {b.barCount}")

    ba = rows.get("BID_ASK")
    bid = rows.get("BID")
    ask = rows.get("ASK")
    if ba and bid and ask:
        print("\n=== interpretation check (BID_ASK overloading) ===")
        print(f"  BID_ASK.open  (time-avg bid) {ba.open:.5f}   vs BID.close {bid.close:.5f}")
        print(f"  BID_ASK.high  (max ask)      {ba.high:.5f}   vs ASK.high  {ask.high:.5f}")
        print(f"  BID_ASK.low   (min bid)      {ba.low:.5f}   vs BID.low   {bid.low:.5f}")
        print(f"  BID_ASK.close (time-avg ask) {ba.close:.5f}   vs ASK.close {ask.close:.5f}")
        print(f"  => bar spread excursion (high-low) = {ba.high - ba.low:.5f}")

    ib.disconnect()


if __name__ == "__main__":
    main()
