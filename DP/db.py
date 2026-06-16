"""
db.py
-----
Shared DuckDB helper for database.db at the project root
(D:\\Coding\\forecasting_instruments_influence_accounts\\database.db).

Replaces the old intermediate CSV/NPY/JSON artifacts (trump_truths_scored.csv,
truth_training_set_TEST/FINAL.csv, finbert_embeddings_v2.npy, eval_report.json)
with tables in a single local DuckDB file. trump_truths.csv and
IBKR/market_data_cache/*.csv stay as plain files.

This is the canonical copy (DP/db.py). Finbert_NLP_XGBoost/db.py is a thin
shim that imports this file, same pattern as signal_scorer.py.

Usage:
    import db
    df = db.read_table("trump_truths_scored")     # -> DataFrame or None if missing
    db.write_table("trump_truths_scored", df)      # overwrite/create table
    db.append_table("trump_truths_scored", new_df) # append rows (creates if missing)
    if db.table_exists("eval_report"): ...
"""

import os
import duckdb
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", "database.db")


def get_connection():
    """New connection to database.db. Caller should close() it, or use a
    `with` block (duckdb connections support context-manager close)."""
    return duckdb.connect(DB_PATH)


def table_exists(table, con=None):
    own = con is None
    con = con or get_connection()
    try:
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchall()
        return len(rows) > 0
    finally:
        if own:
            con.close()


def read_table(table, con=None):
    """Return the full table as a DataFrame, or None if it doesn't exist."""
    own = con is None
    con = con or get_connection()
    try:
        if not table_exists(table, con):
            return None
        return con.execute(f"SELECT * FROM {table}").fetchdf()
    finally:
        if own:
            con.close()


def write_table(table, df, con=None):
    """Create or overwrite `table` with the contents of `df`."""
    own = con is None
    con = con or get_connection()
    try:
        con.register("_tmp_write_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_write_df")
        con.unregister("_tmp_write_df")
    finally:
        if own:
            con.close()


<<<<<<< HEAD
def query(sql, con=None):
    """Run arbitrary SQL and return a DataFrame (or None on error)."""
    own = con is None
    con = con or get_connection()
    try:
        return con.execute(sql).fetchdf()
    except Exception:
        return None
    finally:
        if own:
            con.close()


def rename_table(old_name, new_name, con=None):
    """Rename a table; no-op if old_name doesn't exist or new_name already does."""
    own = con is None
    con = con or get_connection()
    try:
        if table_exists(old_name, con) and not table_exists(new_name, con):
            con.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")
            return True
        return False
    finally:
        if own:
            con.close()


=======
>>>>>>> origin/main
def append_table(table, df, con=None):
    """Append rows of `df` to `table`, creating it (with df's schema) if it
    doesn't exist yet. Column order/types follow `df`."""
    own = con is None
    con = con or get_connection()
    try:
        con.register("_tmp_append_df", df)
        if table_exists(table, con):
            cols = con.execute(f"SELECT * FROM {table} LIMIT 0").fetchdf().columns.tolist()
            df2 = df.reindex(columns=cols)
            con.unregister("_tmp_append_df")
            con.register("_tmp_append_df", df2)
            con.execute(f"INSERT INTO {table} SELECT * FROM _tmp_append_df")
        else:
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_append_df")
        con.unregister("_tmp_append_df")
    finally:
        if own:
            con.close()
