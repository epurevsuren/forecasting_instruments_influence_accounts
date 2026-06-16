"""
db.py
-----
Shared DuckDB helper for database.db at the project root
(D:\\Coding\\forecasting_instruments_influence_accounts\\database.db).

Replaces the old intermediate CSV/NPY/JSON artifacts (posts_scored.csv,
truth_training_set_TEST/FINAL.csv, finbert_embeddings_v2.npy, eval_report.json)
with tables in a single local DuckDB file. truth_social.csv (raw TruthSocial
posts) and IBKR/market_data_cache/*.csv stay as plain files — they are the
raw inputs that feed into the DuckDB pipeline.

This is the canonical copy (DP/db.py). Finbert_NLP_XGBoost/db.py is a thin
shim that imports this file, same pattern as signal_scorer.py.

Usage:
    import db
    df = db.read_table("posts_scored")     # -> DataFrame or None if missing
    db.write_table("posts_scored", df)      # overwrite/create table
    db.append_table("posts_scored", new_df) # append rows (creates if missing)
    if db.table_exists("eval_report"): ...
"""

import os
import duckdb
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", "database.db")

# Tables that require a PRIMARY KEY on (platform, id).
# write_table builds a proper CREATE TABLE ... PRIMARY KEY DDL so the key
# appears in duckdb_constraints().  append_table uses ON CONFLICT DO NOTHING.
_PK_TABLES: dict[str, tuple[str, ...]] = {
    "unified_feed":             ("platform", "id"),
    "posts_scored":             ("platform", "id"),
    "posts_labeled":            ("platform", "id"),
    "training_set_FINAL":       ("platform", "id"),
    "training_set_HIGH_SIGNAL": ("platform", "id"),
}


def _create_with_pk(table: str, df: pd.DataFrame, con) -> None:
    """
    Drop-and-recreate `table` with a proper PRIMARY KEY constraint.
    Schema is inferred from `df` via DuckDB's DESCRIBE; data is bulk-inserted.
    The temporary view _tmp_pk_df must already be registered by the caller.
    """
    pk_cols = _PK_TABLES[table]

    # Infer column types from the DataFrame via DuckDB
    col_rows = con.execute("DESCRIBE SELECT * FROM _tmp_pk_df").fetchall()
    # col_rows: (column_name, column_type, null, key, default, extra)
    col_defs = ", ".join(f'"{r[0]}" {r[1]}' for r in col_rows)
    pk_def = f"PRIMARY KEY ({', '.join(pk_cols)})"

    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} ({col_defs}, {pk_def})")
    con.execute(f"INSERT INTO {table} SELECT * FROM _tmp_pk_df")


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
    """Create or overwrite `table` with the contents of `df`.
    Tables in _PK_TABLES get a real PRIMARY KEY constraint."""
    own = con is None
    con = con or get_connection()
    try:
        con.register("_tmp_pk_df", df)
        if table in _PK_TABLES:
            _create_with_pk(table, df, con)
        else:
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_pk_df")
        con.unregister("_tmp_pk_df")
    finally:
        if own:
            con.close()


def query(sql, con=None):
    """Run arbitrary SQL and return a DataF