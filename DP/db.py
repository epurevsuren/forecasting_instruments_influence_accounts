"""
db.py
-----
Shared DuckDB helper for database.db at the project root
(D:\\Coding\\forecasting_instruments_influence_accounts\\database.db).

Replaces the old intermediate CSV/NPY/JSON artifacts (posts_scored.csv,
truth_training_set_TEST/FINAL.csv, embedding .npy caches, eval_report.json)
with tables in a single local DuckDB file. truth_social.csv (raw TruthSocial
posts) and IBKR/market_data_cache/*.csv stay as plain files — they are the
raw inputs that feed into the DuckDB pipeline.

This is the canonical copy (DP/db.py). Gemma-3-4b_NLP_XGBoost/db.py is a thin
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


def _dtype_to_duck(dtype) -> str:
    """Map a pandas dtype to a DuckDB column type for ALTER TABLE ADD COLUMN."""
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    return "VARCHAR"


_DUCK_DEFAULTS = {"BOOLEAN": "FALSE", "BIGINT": "0", "DOUBLE": "0"}


def _evolve_schema(table, df, con):
    """
    SCHEMA EVOLUTION — the key to daily scorer_config.json changes without
    full re-scores: for every column in `df` that the table doesn't have yet,
    run  ALTER TABLE ADD COLUMN ... DEFAULT 0/false.  In DuckDB this is a
    METADATA-ONLY operation: all existing rows instantly read the default
    (old posts get flag=0/false) with no table rewrite and no re-scoring.
    """
    existing = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    added = []
    for c in df.columns:
        if c not in existing:
            duck = _dtype_to_duck(df[c].dtype)
            ddl = f'ALTER TABLE {table} ADD COLUMN "{c}" {duck}'
            dflt = _DUCK_DEFAULTS.get(duck)
            if dflt is not None:
                ddl += f" DEFAULT {dflt}"
            con.execute(ddl)
            added.append(c)
    if added:
        print(f"  🧬 {table}: schema evolved +{len(added)} column(s) "
              f"{added[:6]}{'...' if len(added) > 6 else ''} "
              f"(historical rows read default 0/false — no rewrite)")


def append_table(table, df, con=None):
    """Append rows of `df` to `table`, creating it (with df's schema) if it
    doesn't exist yet.  For PK tables, duplicates are silently skipped
    (ON CONFLICT DO NOTHING).  NEW columns in `df` evolve the table schema
    (ALTER ... DEFAULT 0) instead of being dropped — see _evolve_schema."""
    own = con is None
    con = con or get_connection()
    try:
        con.register("_tmp_pk_df", df)
        if table_exists(table, con):
            _evolve_schema(table, df, con)
            cols = con.execute(f"SELECT * FROM {table} LIMIT 0").fetchdf().columns.tolist()
            df2 = df.reindex(columns=cols)
            con.unregister("_tmp_pk_df")
            con.register("_tmp_pk_df", df2)
            if table in _PK_TABLES:
                con.execute(
                    f"INSERT OR IGNORE INTO {table} SELECT * FROM _tmp_pk_df"
                )
            else:
                con.execute(f"INSERT INTO {table} SELECT * FROM _tmp_pk_df")
        else:
            # Table doesn't exist yet — create it with PK if applicable
            if table in _PK_TABLES:
                _create_with_pk(table, df, con)
            else:
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_pk_df")
        con.unregister("_tmp_pk_df")
    finally:
        if own:
            con.close()
