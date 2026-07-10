# 0. Fresh start — delete ONLY the DuckDB file (keep all CSVs!)
Set-Location D:\Coding\forecasting_instruments_influence_accounts
Remove-Item database.db

# 1. Top up TruthSocial posts (auto-resumes from last captured post)
#    Needs TRUTHSOCIAL_USERNAME / TRUTHSOCIAL_PASSWORD env vars set
cd DP
uv run python daily_truths_retriever.py

# 2. Top up X/geo tweets since your last run (~mid-June; adjust date)
#    If it returns 0 for ALL accounts, cookies expired -> re-export DP\.x_cookies.json
uv run python x_tweets_retriever.py

# 3. Rebuild English-only tweet file
uv run python filter_english_tweets.py

# 4. Rebuild unified_feed table (full, since DB is fresh)
uv run python sync_unified_feed.py --full

# 5. NLP-score everything -> posts_scored
uv run python signal_scorer.py --full

# 6. Label with market impact -> training tables
#    (uses IBKR 30-min cache + auto-extends with yfinance past cache end)
uv run python build_final_training_set.py --full

# 7+8. Train models (re-embeds all posts once — the slow step)
cd ..\Finbert_NLP_XGBoost
uv run python train_finbert_nlp_xgb.py

# 9. Direction backtest over the full window (until = now, NY time)
.\run_backtest.bat 202607090000 --from 201611010000

# 10. Layer-2 TP/SL simulation (auto-picks the latest backtest CSV)
uv run python simulate_tp_sl.py