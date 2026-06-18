@echo off
cd /d %~dp0

echo ============================================================
echo  Backfill BTC + ETH from Binance (2017-08 to now)
echo  No API key required -- Binance public endpoint
echo  Bar sizes: 15m, 5m, 3m, 1m (30m already cached)
echo ============================================================

echo.
echo [1/4] 15-minute bars...
python fetch_binance_history.py --bar-size 15m --from 20170801

echo.
echo [2/4] 5-minute bars...
python fetch_binance_history.py --bar-size 5m --from 20170801

echo.
echo [3/4] 3-minute bars...
python fetch_binance_history.py --bar-size 3m --from 20170801

echo.
echo [4/4] 1-minute bars...
python fetch_binance_history.py --bar-size 1m --from 20170801

echo.
echo ============================================================
echo  All bar sizes complete (15m, 5m, 3m, 1m).
echo  CSVs written to market_data_cache/
echo ============================================================
pause
