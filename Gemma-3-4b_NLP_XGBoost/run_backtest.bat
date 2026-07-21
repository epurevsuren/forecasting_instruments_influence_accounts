@echo off
REM ===========================================================================
REM run_backtest.bat - wrapper for backtest_simulator.py with a properly
REM UTF-8-encoded log file.
REM
REM WHY: PowerShell's Tee-Object defaults to UTF-16LE ("Unicode") output,
REM regardless of `chcp 65001` / PYTHONUTF8=1 — those only affect the console
REM and Python's own stdout encoding, not Tee-Object's file encoding. The
REM result is a UTF-16LE backtest.log full of NUL bytes between characters
REM (and a garbled BOM/emoji) when opened as UTF-8/ASCII. Windows PowerShell
REM 5.1's Tee-Object also doesn't support -Encoding (that's PS7+), so instead
REM we pipe through ForEach-Object and write each line to the console AND to
REM the log file via Add-Content -Encoding utf8.
REM
REM USAGE
REM   run_backtest.bat 202606140900 --from 202601010000
REM   run_backtest.bat 202606140900 --from 202606130900 --fine-tune
REM ===========================================================================
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1

set TS=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i

if not exist logs mkdir logs

REM %* passes through all args (e.g. "202606140900 --from 202411010000 --fine-tune")
powershell -NoProfile -Command "uv run python backtest_simulator.py %* 2>>'logs\backtest_%TS%.err' | ForEach-Object { $_; Add-Content -Path 'logs\backtest_%TS%.log' -Value $_ -Encoding utf8 }"
REM If you don't use uv, replace "uv run python" with "python" above.
