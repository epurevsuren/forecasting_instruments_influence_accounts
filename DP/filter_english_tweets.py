"""
filter_english_tweets.py
========================
Filters x_tweets.csv to English-only tweets and writes x_tweets_en.csv.
Run any time after x_tweets_retriever.py to refresh the filtered copy.

It ALSO canonicalizes archived account handles -> their original account (via
account_aliases.py, driven by influence_accounts.json): both the `account`
column and any @mentions in the text. Archived office handles (@WhiteHouse45,
@WhiteHouse46, @POTUS45, @POTUS46Archive, @VP45/@VP46archive, @DeptofDefense ...)
are per-administration snapshots of the SAME office, so collapsing them keeps the
backtest from fragmenting one account into several.

Usage:
  python filter_english_tweets.py
"""
import os
import pandas as pd

from account_aliases import load_alias_map, canonical_account, canonical_mentions

_HERE    = os.path.dirname(os.path.abspath(__file__))
SRC      = os.path.join(_HERE, "x_tweets.csv")
DST      = os.path.join(_HERE, "x_tweets_en.csv")

df = pd.read_csv(SRC)

total   = len(df)
en_df   = df[df["language"] == "en"].copy()
en_rows = len(en_df)

# --- canonicalize archived handles -> original account (JSON-driven) ----------
amap = load_alias_map()
if amap and en_rows:
    before = en_df["account"].astype(str).str.lstrip("@").str.lower()
    en_df["account"] = en_df["account"].map(lambda a: canonical_account(a, amap))
    if "text" in en_df.columns:
        en_df["text"] = en_df["text"].map(lambda t: canonical_mentions(t, amap))
    n_remapped = int((before != en_df["account"].astype(str).str.lower()).sum())
    print(f"[alias] {len(amap)} archived->canonical mapping(s); "
          f"remapped {n_remapped} row account(s) + @mentions")
    print("        " + ", ".join(f"{k}->{v}" for k, v in sorted(amap.items())))

en_df.to_csv(DST, index=False, encoding="utf-8-sig", lineterminator="\n")

print(f"[src]  {SRC}")
print(f"       {total} rows total")
print(f"[dst]  {DST}")
print(f"       {en_rows} English rows ({en_rows/total*100:.1f}%)")
print()
print("Per-account English ratio (after canonicalization):")
# group on the canonicalized handle so archived variants aggregate together
_grp_acc = df["account"].map(lambda a: canonical_account(a, amap)) if amap else df["account"]
for acc, grp in df.assign(_acc=_grp_acc).groupby("_acc"):
    n  = len(grp)
    en = (grp["language"] == "en").sum()
    print(f"  {str(acc):<22} {en:>4}/{n:<4}  ({en/n*100:.0f}%)")
