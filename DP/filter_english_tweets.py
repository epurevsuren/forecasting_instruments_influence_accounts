"""
filter_english_tweets.py
========================
Filters x_tweets.csv to English-only tweets and writes x_tweets_en.csv.
Run any time after x_tweets_retriever.py to refresh the filtered copy.

Usage:
  python filter_english_tweets.py
"""
import os
import pandas as pd

_HERE    = os.path.dirname(os.path.abspath(__file__))
SRC      = os.path.join(_HERE, "x_tweets.csv")
DST      = os.path.join(_HERE, "x_tweets_en.csv")

df = pd.read_csv(SRC)

total   = len(df)
en_df   = df[df["language"] == "en"].copy()
en_rows = len(en_df)

en_df.to_csv(DST, index=False, encoding="utf-8-sig", lineterminator="\n")

print(f"[src]  {SRC}")
print(f"       {total} rows total")
print(f"[dst]  {DST}")
print(f"       {en_rows} English rows ({en_rows/total*100:.1f}%)")
print()
print("Per-account English ratio:")
for acc, grp in df.groupby("account"):
    n     = len(grp)
    en    = (grp["language"] == "en").sum()
    print(f"  {acc:<22} {en:>4}/{n:<4}  ({en/n*100:.0f}%)")
