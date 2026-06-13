@echo off
REM ===========================================================================
REM run_daily_truths.bat - wrapper for Windows Task Scheduler.
REM
REM Schedule this once per day at the LOCAL time that equals 22:00 UTC (the gap
REM between the New York close and the Tokyo open). Examples:
REM   UTC+8      (Beijing)      -> 06:00 next day
REM   UTC+10/+11 (Sydney)       -> 08:00 / 09:00 next day
REM   UTC+0      (London)       -> 22:00
REM   UTC-5      (New York,EST) -> 17:00
REM
REM One-time setup (PowerShell, run once):
REM   setx TRUTHSOCIAL_USERNAME "epurevsuren"
REM   setx TRUTHSOCIAL_PASSWORD "7Qy#J9d3Jmw7!Tx"
REM
REM Register the daily task (adjust HH:MM to your local 22:00-UTC equivalent):
REM   schtasks /Create /TN "TrumpTruthsDaily" /TR "D:\Coding\forecasting_instruments_influence_accounts\DP\run_daily_truths.bat" /SC DAILY /ST 06:00
REM ===========================================================================
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1

REM Per-run timestamp (yyyyMMdd_HHmmss), used so each run gets its OWN log files
REM instead of overwriting/appending into one ever-growing log.
set TS=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i

if not exist logs mkdir logs

REM Each step's STDOUT is piped through PowerShell's Tee-Object so you see it
REM live in the console (when run interactively) AND it's saved to its own
REM timestamped log file under logs\. STDERR is redirected straight to a
REM separate .err file via cmd (not through the pipe) so PowerShell doesn't
REM wrap normal stderr chatter (login-token notices, progress bars) as red
REM "NativeCommandError" blocks.
powershell -NoProfile -Command "uv run python daily_truths_retriever.py 2>>'logs\daily_truths_retriever_%TS%.err' | Tee-Object -FilePath 'logs\daily_truths_retriever_%TS%.log'"
REM Score the freshly retrieved posts: incremental mode (default) only scores the
REM NEW ids in trump_truths.csv and APPENDS them to the trump_truths_scored
REM table in ..\database.db (DuckDB).
REM Runs even if the retriever found nothing new (it then just reports 0 to score).
powershell -NoProfile -Command "uv run python signal_scorer.py 2>>'logs\signal_scorer_%TS%.err' | Tee-Object -FilePath 'logs\signal_scorer_%TS%.log'"
REM Label the freshly scored posts: incremental mode (default) labels only the
REM posts not yet in the trump_truths_labeled table and APPENDS them to the
REM truth_training_set_FINAL (+ HIGH_SIGNAL + labeled) tables in database.db.
REM Posts whose market reaction data isn't available yet (after-hours / weekend)
REM are deferred to the next run instead of being mislabeled 0.
powershell -NoProfile -Command "uv run python build_final_training_set.py 2>>'logs\build_final_training_set_%TS%.err' | Tee-Object -FilePath 'logs\build_final_training_set_%TS%.log'"
REM If you don't use uv, replace "uv run python" with "python" in the Tee-Object lines above.
