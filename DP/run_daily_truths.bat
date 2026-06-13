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
set PYTHONUTF8=1
uv run python daily_truths_retriever.py >> daily_truths_retriever.log 2>&1
REM Score the freshly retrieved posts: incremental mode (default) only scores the
REM NEW ids in trump_truths.csv and APPENDS them to trump_truths_scored.csv.
REM Runs even if the retriever found nothing new (it then just reports 0 to score).
uv run python signal_scorer.py >> signal_scorer.log 2>&1
REM Label the freshly scored posts: incremental mode (default) labels only the
REM posts not yet in trump_truths_labeled.csv and APPENDS them to
REM truth_training_set_FINAL.csv (+ HIGH_SIGNAL + labeled). Posts whose market
REM reaction data isn't available yet (after-hours / weekend) are deferred to
REM the next run instead of being mislabeled 0.
uv run python build_final_training_set.py >> build_final_training_set.log 2>&1
REM If you don't use uv, replace "uv run python" with "python" in the lines above.
