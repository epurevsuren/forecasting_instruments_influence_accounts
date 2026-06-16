"""
migrate_to_duckdb.py
---------------------
ONE-TIME migration: load the existing CSV/JSON artifacts into database.db
(DuckDB, project root), then move the old files aside.

Migrates:
  DP/trump_truths_scored.csv          -> trump_truths_scored        table
  DP/truth_training_set_TEST.csv      -> truth_training_set_TEST    table
  DP/truth_training_set_FINAL.csv     -> truth_training_set_FINAL   table   (if present)
  DP/truth_training_set_HIGH_SIGNAL.csv -> truth_training_set_HIGH_SIGNAL table (if present)
  DP/trump_truths_labeled.csv         -> trump_truths_labeled       table   (if present)
  Finbert_NLP_XGBoost/finbert_nlp_xgb_models/eval_report.json -> eval_report table

NOT migrated:
  DP/trump_truths.csv                  -- stays as the raw input file (unchanged)
  IBKR/market_data_cache/*.csv         -- stays as files (unchanged)
  Finbert_NLP_XGBoost/finbert_embeddings_v2.npy
      -- has no per-row id, so it can't be mapped into the new id-keyed
         finbert_embeddings_v2 table. Left untouched here; the next run of
         train_finbert_nlp_xgb.py will simply re-embed (cache miss) and
         write a fresh id-keyed finbert_embeddings_v2 table automatically.
  Finbert_NLP_XGBoost/finbert_nlp_xgb_models/*_Impact.json, config.json
      -- binary model artifacts, stay as files (per scope decision).

After a successful load, source files are renamed to "<name>.migrated_bak"
next to where they were (nothing is deleted).

Run once from DP/:  python migrate_to_duckdb.py
"""
import os
import json
import pandas as pd
import db

_HERE = os.path.dirname(os.path.abspath(__file__))
_FNX  = os.path.join(_HERE, "..", "Finbert_NLP_XGBoost")


def _migrate_csv(path, table, dtype=None):
    if not os.path.exists(path):
        print(f"  ⏭️  {path} not found — skipping")
        return
    print(f"  📂 Loading {path} -> {table}")
    df = pd.read_csv(path, dtype=dtype) if dtype else pd.read_csv(path)
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)
    db.write_table(table, df)
    print(f"     ✅ {len(df)} rows written")
    bak = path + ".migrated_bak"
    os.rename(path, bak)
    print(f"     📦 moved source -> {bak}")


def _migrate_eval_report():
    path = os.path.join(_FNX, "finbert_nlp_xgb_models", "eval_report.json")
    if not os.path.exists(path):
        print(f"  ⏭️  {path} not found — skipping")
        return
    print(f"  📂 Loading {path} -> eval_report")
    with open(path, encoding='utf-8') as f:
        report = json.load(f)
    eval_df = pd.DataFrame([{"instrument": k, **v} for k, v in report.items()])
    db.write_table("eval_report", eval_df)
    print(f"     ✅ {len(eval_df)} rows written")
    bak = path + ".migrated_bak"
    os.rename(path, bak)
    print(f"     📦 moved source -> {bak}")


def main():
    print("=" * 60)
    print("  ONE-TIME MIGRATION: CSV/JSON -> database.db (DuckDB)")
    print(f"  DB: {db.DB_PATH}")
    print("=" * 60)

    _migrate_csv(os.path.join(_HERE, "trump_truths_scored.csv"),
                  "trump_truths_scored", dtype={'id': str})
    _migrate_csv(os.path.join(_HERE, "truth_training_set_TEST.csv"),
                  "truth_training_set_TEST")
    _migrate_csv(os.path.join(_HERE, "truth_training_set_FINAL.csv"),
                  "truth_training_set_FINAL")
    _migrate_csv(os.path.join(_HERE, "truth_training_set_HIGH_SIGNAL.csv"),
                  "truth_training_set_HIGH_SIGNAL")
    _migrate_csv(os.path.join(_HERE, "trump_truths_labeled.csv"),
                  "trump_truths_labeled", dtype={'id': str})
    _migrate_eval_report()

    print("\n✅ Migration done. Tables now in database.db:")
    with db.get_connection() as con:
        for (name,) in con.execute(
                "SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall():
            n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            print(f"   - {name} ({n} rows)")


if __name__ == '__main__':
    main()
