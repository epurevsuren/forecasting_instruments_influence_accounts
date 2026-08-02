"""
db.py (shim) - Gemma-3-4b_NLP_XGBoost
-----------------------------------
The canonical db.py lives in DP/. This shim loads it under a distinct
internal module name (avoiding a name collision between this file and
DP/db.py, both called "db") and re-exports its DuckDB helpers so scripts
in this folder can `import db` and reach the same database.db at the
project root.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_DP_DB_PATH = os.path.join(_HERE, "..", "DP", "db.py")

_spec = importlib.util.spec_from_file_location("_dp_db", _DP_DB_PATH)
_dp_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp_db)

DB_PATH        = _dp_db.DB_PATH
get_connection = _dp_db.get_connection
table_exists   = _dp_db.table_exists
read_table     = _dp_db.read_table
write_table    = _dp_db.write_table
append_table   = _dp_db.append_table
query          = _dp_db.query   # needed by predict's post-memory lookup
