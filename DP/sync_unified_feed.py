"""
sync_unified_feed.py
====================
Merges TruthSocial posts (trump_truths.csv) and English geopolitical tweets
(x_tweets_en.csv) into the unified_feed DuckDB table, sorted by
America/New_York time.

Each row is augmented with:
  source          'truthsocial' | 'x_twitter'
  platform        'truthsocial' | 'x_twitter'  (factual, permanent per account)
  account         'realDonaldTrump' | twitter handle
  account_name    display name
  account_rank    0=primary influence account | entity rank (1-N) | None for institutions
  mention_count   times the primary account mentioned this entity | None for institutions
  entity_weight   1.0 (rank-0 primary) | mention_count/MAX_MENTION | institution weight
  event_weight    1.0 (rank-0 primary) | EventManager.get_account_multiplier(handle)

Rank-0 primary account gets entity_weight=1.0, event_weight=1.0 (highest trust).
Geo X/Twitter accounts are discounted by SOURCE_DISCOUNT in signal_scorer.py.

Dedup is keyed on (id, source): re-running is always safe.

CLI:
  python sync_unified_feed.py           # incremental — only new (id, source) pairs
  python sync_unified_feed.py --full    # full rebuild of unified_feed
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import db
from event_manager import EventManager

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ------------------------------------------------------------------ paths ----
ENTITIES_FILE = os.path.join(_HERE, "geopolitical_entities.json")
TRUTHS_CSV    = os.path.join(_HERE, "trump_truths.csv")
TWEETS_CSV    = os.path.join(_HERE, "x_tweets_en.csv")
FEED_TABLE    = "unified_feed"
TZ            = "America/New_York"

# Maximum entity mention_count in our tracked list (netanyahu = 28).
# Used to normalise entity_weight to [0, 1].
MAX_MENTION_COUNT = 28

# Institution entity_weight by handle (they have no mention_count in entities JSON).
# WhiteHouse ranks above DeptofWar because it IS the executive branch.
INSTITUTION_WEIGHTS: dict[str, float] = {
    "WhiteHouse":      0.80,
    "DeptofWar":       0.75,
    "IDF":             0.70,
    "StateDept":       0.65,
    "IsraelMFA":       0.65,
    "mfa_russia":      0.65,
    "ChineseEmbinUS":  0.60,
    "SEPeaceMissions": 0.60,
    "GovernmentRF":    0.55,
    "KremlinRussia_E": 0.55,
}

UNIFIED_FEED_COLS = [
    "id", "source", "platform",
    "account", "account_name", "account_rank", "mention_count",
    "entity_weight", "event_weight",
    "date",          # America/New_York, ISO string
    "text", "language", "url",
    "favorites", "retweets", "replies",
]


# ---------------------------------------------------------- entity lookup ----

def _build_entity_map() -> dict:
    """
    Returns {handle_lower: {handle, name, rank, mention_count,
                            entity_weight, event_weight}}.
    """
    with open(ENTITIES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    em = EventManager()
    out: dict = {}

    # Named geo leaders
    for ent in data.get("entities", []):
        raw = ent.get("twitter_handle")
        if not raw:
            continue
        h = raw.lstrip("@")
        mc = int(ent.get("mention_count", 0))
        ew = round(mc / MAX_MENTION_COUNT, 4)
        out[h.lower()] = {
            "handle":        h,
            "name":          ent.get("name", h),
            "rank":          int(ent.get("rank", 99)),
            "mention_count": mc,
            "entity_weight": ew,
            "event_weight":  em.get_account_multiplier(h),
        }

    # Institutions (no mention_count)
    for inst in data.get("institutions", {}).get("entries", []):
        raw = inst.get("twitter_handle")
        if not raw:
            continue
        h = raw.lstrip("@")
        ew = INSTITUTION_WEIGHTS.get(h, 0.50)
        out[h.lower()] = {
            "handle":        h,
            "name":          inst.get("name", h),
            "rank":          None,
            "mention_count": None,
            "entity_weight": ew,
            "event_weight":  em.get_account_multiplier(h),
        }

    return out


# ------------------------------------------------ classify / clean helpers ----

def _classify(text: str) -> str:
    t = str(text).strip()
    if not t:
        return "noise"
    bare = re.sub(r"https?://\S+", "", t).strip()
    if len(bare) < 5 and re.search(r"https?://", t):
        return "url_only"
    return "original"


def _clean_text(text: str) -> str:
    t = str(text).replace("\r\n", " ").replace("\r", " ")
    t = re.sub(r"([^\s])https://", r"\1 https://", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _to_ny(series: pd.Series) -> pd.Series:
    """Convert a date series (any tz) to America/New_York ISO strings."""
    dt = pd.to_datetime(series, format="mixed", utc=True)
    return dt.dt.tz_convert(TZ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------- build sources ----

def _build_truths() -> pd.DataFrame:
    """Load trump_truths.csv → unified schema rows."""
    if not os.path.exists(TRUTHS_CSV):
        print(f"  ⚠️  {TRUTHS_CSV} not found — skipping Trump truths")
        return pd.DataFrame(columns=UNIFIED_FEED_COLS)

    df = pd.read_csv(TRUTHS_CSV, dtype={"id": str})
    # Filter noise (URL-only posts — keep originals and RT/quote)
    df["_type"] = df["text"].apply(_classify)
    df = df[df["_type"] != "url_only"].copy()
    df = df.drop(columns=["_type"])

    rows = pd.DataFrame(index=df.index)
    rows["id"]           = df["id"].astype(str)
    rows["source"]       = "truthsocial"
    rows["platform"]     = "truthsocial"
    rows["account"]      = "realDonaldTrump"
    rows["account_name"] = "Donald J. Trump"
    rows["account_rank"] = 0
    rows["mention_count"] = None
    rows["entity_weight"] = 1.0
    rows["event_weight"]  = 1.0
    rows["date"]          = _to_ny(df["date"])
    rows["text"]          = df["text"].apply(_clean_text)
    rows["language"]      = "en"
    # Truth Social post URL (construct from id if no url column)
    if "url" in df.columns:
        rows["url"] = df["url"].fillna(
            "https://truthsocial.com/@realDonaldTrump/posts/" + df["id"].astype(str)
        )
    else:
        rows["url"] = "https://truthsocial.com/@realDonaldTrump/posts/" + df["id"].astype(str)
    rows["favorites"] = df.get("favorites", pd.Series(0, index=df.index)).fillna(0).astype(int)
    rows["retweets"]  = df.get("retweets",  pd.Series(0, index=df.index)).fillna(0).astype(int)
    rows["replies"]   = df.get("replies",   pd.Series(0, index=df.index)).fillna(0).astype(int)
    return rows[UNIFIED_FEED_COLS]


def _build_tweets(entity_map: dict) -> pd.DataFrame:
    """Load x_tweets_en.csv → unified schema rows."""
    if not os.path.exists(TWEETS_CSV):
        print(f"  ⚠️  {TWEETS_CSV} not found — run filter_english_tweets.py first")
        return pd.DataFrame(columns=UNIFIED_FEED_COLS)

    df = pd.read_csv(TWEETS_CSV, dtype={"id": str})

    # Filter URL-only tweets
    df["_type"] = df["text"].apply(_classify)
    df = df[df["_type"] != "url_only"].copy()
    df = df.drop(columns=["_type"])

    # Unknown handles get low-weight defaults
    def _lookup(handle: str) -> dict:
        info = entity_map.get(str(handle).lower(), {})
        return {
            "account_rank":   info.get("rank", None),
            "mention_count":  info.get("mention_count", None),
            "entity_weight":  info.get("entity_weight", 0.30),
            "event_weight":   info.get("event_weight",  1.00),
        }

    handles = df["account"].fillna("").astype(str)
    lookup_df = pd.DataFrame([_lookup(h) for h in handles], index=df.index)

    rows = pd.DataFrame(index=df.index)
    rows["id"]           = df["id"].astype(str)
    rows["source"]       = "x_twitter"
    rows["platform"]     = "x_twitter"
    rows["account"]      = handles
    rows["account_name"] = df.get("account_name", handles).fillna(handles)
    rows["account_rank"]  = lookup_df["account_rank"]
    rows["mention_count"] = lookup_df["mention_count"]
    rows["entity_weight"] = lookup_df["entity_weight"]
    rows["event_weight"]  = lookup_df["event_weight"]
    rows["date"]          = _to_ny(df["date"])
    rows["text"]          = df["text"].apply(_clean_text)
    rows["language"]      = df.get("language", pd.Series("en", index=df.index)).fillna("en")
    rows["url"]           = df.get("url", pd.Series(None, index=df.index))
    rows["favorites"]     = df.get("favorites", pd.Series(0, index=df.index)).fillna(0).astype(int)
    rows["retweets"]      = df.get("retweets",  pd.Series(0, index=df.index)).fillna(0).astype(int)
    rows["replies"]       = df.get("replies",   pd.Series(0, index=df.index)).fillna(0).astype(int)
    return rows[UNIFIED_FEED_COLS]


# -------------------------------------------------------------- sync main ----

def sync(full: bool = False) -> int:
    print("=" * 64)
    print("  SYNC UNIFIED FEED")
    print(f"  {'FULL REBUILD' if full else 'INCREMENTAL'}: "
          f"trump_truths.csv + x_tweets_en.csv → {FEED_TABLE}")
    print("=" * 64)

    entity_map = _build_entity_map()
    print(f"\n📋 Entity map loaded: {len(entity_map)} handles")

    print("\n📂 Building TruthSocial post rows...")
    truths = _build_truths()
    print(f"  {len(truths)} rows (post URL-only filter)")

    print("📂 Building X/Twitter geo-account rows...")
    tweets = _build_tweets(entity_map)
    print(f"  {len(tweets)} rows (post URL-only filter)")

    # Merge and sort by date
    combined = pd.concat([truths, tweets], ignore_index=True)
    combined["_dt"] = pd.to_datetime(combined["date"], format="mixed", utc=True)
    combined = combined.sort_values("_dt").drop(columns=["_dt"]).reset_index(drop=True)

    if full:
        db.write_table(FEED_TABLE, combined)
        print(f"\n💾 Full rebuild: {FEED_TABLE} = {len(combined)} rows")
        _print_summary(combined)
        return len(combined)

    # Incremental: find (id, source) pairs not yet in the table
    existing = db.read_table(FEED_TABLE)
    if existing is None:
        db.write_table(FEED_TABLE, combined)
        print(f"\n💾 First run: {FEED_TABLE} = {len(combined)} rows")
        _print_summary(combined)
        return len(combined)

    existing_keys = set(zip(existing["id"].astype(str), existing["source"]))
    new = combined[
        ~combined.apply(lambda r: (r["id"], r["source"]), axis=1).isin(existing_keys)
    ].copy()

    print(f"\n  Already in {FEED_TABLE}: {len(existing_keys)} | new: {len(new)}")
    if new.empty:
        print("✅ unified_feed is up to date — nothing to add.")
        return 0

    db.append_table(FEED_TABLE, new)
    total = len(existing) + len(new)
    print(f"\n💾 Appended {len(new)} rows → {FEED_TABLE} ({total} total)")
    _print_summary(new, label="new rows")
    return len(new)


def _print_summary(df: pd.DataFrame, label: str = "rows") -> None:
    if df.empty:
        return
    n_primary = int((df["platform"] == "truthsocial").sum()) if "platform" in df.columns else 0
    n_twitter = len(df) - n_primary
    print(f"\n  {label}: {len(df)} total  |  TruthSocial: {n_primary}  |  X/Twitter: {n_twitter}")
    if n_twitter:
        tw = df[df["platform"] == "x_twitter"] if "platform" in df.columns else df
        by_acc = tw.groupby("account").size().sort_values(ascending=False)
        print(f"  X/Twitter accounts: {', '.join(f'{a}({n})' for a, n in by_acc.head(10).items())}")
    print()


# ----------------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sync unified_feed DuckDB table.")
    ap.add_argument("--full", action="store_true",
                    help="Full rebuild — overwrite unified_feed entirely.")
    args = ap.parse_args()
    sync(full=args.full)
