@echo off
cd /d %~dp0

echo ============================================================
echo  Backfill COPPER NATGAS US10Y US2Y USD_CNY USD_MXN
echo ============================================================

echo.
echo COPPER NATGAS US10Y US2Y USD_CNY USD_MXN 15-minute bars...
python fetch_ibkr_history.py --bar-size 15m --instruments COPPER NATGAS US10Y US2Y --from 20241101

echo.
echo US2Y 15-minute bars...
python fetch_ibkr_history.py --bar-size 15m --instruments USD_CNY --from 20121031

echo.
echo US2Y 15-minute bars...
python fetch_ibkr_history.py --bar-size 15m --instruments USD_MXN --from 20120101
echo.
echo ============================================================
echo  CSVs written to market_data_cache/
echo ============================================================
pause
