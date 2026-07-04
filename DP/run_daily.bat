@echo off
REM ===========================================================================
REM run_daily.bat — unified daily pipeline (replaces run_daily_truths.bat)
REM
REM Pipeline order:
REM   1. daily_truths_retriever.py   — fetch new Trump Truth Social posts
REM   2. x_tweets_retriever.py       — fetch new geopolitical tweets (last 7 days)
REM   3. filter_english_tweets.py    — rebuild x_tweets_en.csv from x_tweets.csv
REM   4. sync_unified_feed.py        — merge both sources → unified_feed (DuckDB)
REM   5. signal_scorer.py            — NLP score new posts → posts_scored (DuckDB)
REM   6. build_final_training_set.py — label with market impact → training tables
REM
REM SCHEDULE: once per day at your local equivalent of 22:00 UTC
REM   (gap between NY close 16:00 EST and Tokyo open 09:00 JST next day)
REM   UTC-5 (New York EST)  → 17:00
REM   UTC+0 (London)        → 22:00
REM   UTC+8 (Beijing)       → 06:00 next day
REM
REM ONE-TIME SETUP (run once in PowerShell, then restart the shell):
REM   setx TRUTHSOCIAL_USERNAME "your_username"
REM   setx TRUTHSOCIAL_PASSWORD "your_password"
REM
REM X/TWITTER COOKIES: the tweet retriever uses DP\.x_cookies.json.
REM   Cookies typically expire in ~1 year. When tweets stop loading (0 results
REM   across all accounts), re-export cookies from x.com via the Cookie-Editor
REM   browser extension and overwrite DP\.x_cookies.json.
REM
REM REGISTER SCHEDULED TASK (adjust HH:MM to your local 22:00-UTC equivalent):
REM   schtasks /Create /TN "DailyPipeline" ^
REM     /TR "D:\Coding\forecasting_instruments_influence_accounts\DP\run_daily.bat" ^
REM     /SC DAILY /ST 06:00
REM ===========================================================================

cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1

REM Per-run timestamp for log files (no two runs share a log)
set TS=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i

REM Lookback window for tweet retrieval: 7 days prevents gaps on weekend/holiday
REM runs. Duplicate tweet IDs are skipped automatically.
set SINCE=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd -Date (Get-Date).AddDays(-7)"') do set SINCE=%%i

if not exist logs mkdir logs

echo [%TS%] Starting daily pipeline...

REM ---------------------------------------------------------------------------
REM Step 1: Trump Truth Social posts
REM ---------------------------------------------------------------------------
powershell -NoProfile -Command ^
  "uv run python daily_truths_retriever.py ^
   2>>'logs\truths_retriever_%TS%.err' ^
   | Tee-Object -FilePath 'logs\truths_retriever_%TS%.log'"

REM ---------------------------------------------------------------------------
REM Step 2: Geopolitical tweets (incremental — skips already-saved IDs)
REM ---------------------------------------------------------------------------
powershell -NoProfile -Command ^
  "uv run python x_tweets_retriever.py --since %SINCE% ^
   2>>'logs\x_tweets_retriever_%TS%.err' ^
   | Tee-Object -FilePath 'logs\x_tweets_retriever_%TS%.log'"

REM ---------------------------------------------------------------------------
REM Step 3: Rebuild English-only tweet CSV
REM ---------------------------------------------------------------------------
powershell -NoProfile -Command ^
  "uv run python filter_english_tweets.py ^
   2>>'logs\filter_english_tweets_%TS%.err' ^
   | Tee-Object -FilePath 'logs\filter_english_tweets_%TS%.log'"

REM ---------------------------------------------------------------------------
REM Step 4: Sync both sources into unified_feed (DuckDB)
REM ---------------------------------------------------------------------------
powershell -NoProfile -Command ^
  "uv run python sync_unified_feed.py ^
   2>>'logs\sync_unified_feed_%TS%.err' ^
   | Tee-Object -FilePath 'logs\sync_unified_feed_%TS%.log'"

REM ---------------------------------------------------------------------------
REM Step 4.5: LLM config curator (OPTIONAL — runs only if GEMINI_API_KEY set).
REM   Proposes strictly-additive deltas to scorer_config.json /
REM   influence_accounts.json / events.json from the last days' posts.
REM   Validated + backed up (config_backups\) + changelogged. Runs BEFORE the
REM   scorer so today's posts are scored with today's config. New flags reach
REM   posts_scored via DuckDB schema evolution (old posts default 0) — no
REM   full re-score, no retrain needed until the drift notice says so.
REM ---------------------------------------------------------------------------
if defined GEMINI_API_KEY (
  powershell -NoProfile -Command ^
    "uv run python update_configs_llm.py ^
     2>>'logs\config_curator_%TS%.err' ^
     | Tee-Object -FilePath 'logs\config_curator_%TS%.log'"
) else (
  echo [skip] Step 4.5 config curator: GEMINI_API_KEY not set
)

REM ---------------------------------------------------------------------------
REM Step 5: Score new posts (incremental: only new (id,source) pairs)
REM ---------------------------------------------------------------------------
powershell -NoProfile -Command ^
  "uv run python signal_scorer.py ^
   2>>'logs\signal_scorer_%TS%.err' ^
   | Tee-Object -FilePath 'logs\signal_scorer_%TS%.log'"

REM ---------------------------------------------------------------------------
REM Step 6: Label new posts with market impact (deferred if data not yet ready)
REM ---------------------------------------------------------------------------
powershell -NoProfile -Command ^
  "uv run python build_final_training_set.py ^
   2>>'logs\build_final_training_set_%TS%.err' ^
   | Tee-Object -FilePath 'logs\build_final_training_set_%TS%.log'"

echo [%TS%] Pipeline complete. Logs in DP\logs\
