r"""
sort_truths.py  -  one-time utility: sort truth_social.csv by date ASCENDING.

Rewrites the file in place (after a .bak backup), keeping the exact same columns,
the original date strings and the UTF-8 BOM encoding - only the ROW ORDER changes.
Stable sort, so same-timestamp rows keep their relative order; rows whose date
can't be parsed are moved to the end.

Run:  python sort_truths.py
      python sort_truths.py path\to\other.csv
"""
import os
import sys
import shutil
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_HERE, "truth_social.csv")


def sort_csv(path=DEFAULT_CSV):
    if not os.path.exists(path):
        sys.exit(f"Not found: {path}")

    # utf-8-sig strips a leading BOM from the header if present (and is harmless
    # if absent), so the 'id' column name stays clean.
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if "date" not in df.columns:
        sys.exit("No 'date' column to sort by.")

    n = len(df)
    key = pd.to_datetime(df["date"], utc=True, format="mixed", errors="coerce")
    bad = int(key.isna().sum())

    # Stable sort by the parsed datetime; unparseable dates go last.
    order = key.sort_values(kind="mergesort", na_position="last").index
    df_sorted = df.loc[order].reset_index(drop=True)

    backup = path + ".bak"
    shutil.copy2(path, backup)   # safety copy before overwriting
    df_sorted.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")

    valid = key.dropna()
    print(f"Sorted {n} rows by date ASC -> {path}")
    print(f"  backup saved: {backup}")
    if len(valid):
        print(f"  range: {valid.min()}  ->  {valid.max()}")
    if bad:
        print(f"  note: {bad} row(s) had an unparseable date (moved to the end)")


if __name__ == "__main__":
    sort_csv(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)
