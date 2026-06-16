"""
daily_truths_retriever.py
=========================
Daily, incremental retriever for Truth Social posts. Accounts to retrieve are
driven by the `primary_accounts` list in DP/geopolitical_entities.json — no
account is hardcoded here, so adding a future candidate (e.g. for the 2028
election) only requires updating that JSON file.

What this script does
---------------------
For each primary account listed in geopolitical_entities.json:
  * loads the ids already in truth_social.csv for that account,
  * pulls only recent statuses (from the last captured post, with a small
    overlap window to catch any posts missed on a previous run),
  * skips replies / re-truths / empty posts and anything already stored,
  * appends only the NEW rows.

Output schema (truth_social.csv)
---------------------------------
    id, account, date, text, url, favorites, retweets, replies

The `account` column is the TruthSocial handle (e.g. "realDonaldTrump").
Every subsequent pipeline step (sync_unified_feed.py → signal_scorer.py →
build_final_training_set.py → train / predict) reads this column to assign
the correct entity_weight, account_rank, and is_primary flag.

WHEN TO RUN  (the NY <-> Asia gap)
-----------------------------------
Target: 22:00 UTC, once per day.

The New York equity session closes at ~20:00 UTC (EDT) / 21:00 UTC (EST)
and the Tokyo session opens at 00:00 UTC (09:00 JST). The window that is a
gap in both seasons is 21:00-00:00 UTC — the low-liquidity FX lull between
New York and Asia. 22:00 UTC sits inside it:
  * ~1-2 h AFTER the NY close → a full, market-moving NY session of posts is
    already captured, and
  * ~2 h BEFORE Tokyo opens   → the dataset is refreshed before the Asian
    session can trade on it.

CREDENTIALS
-----------
Set these environment variables before running (NOT hardcoded here):

    TRUTHSOCIAL_USERNAME
    TRUTHSOCIAL_PASSWORD

Run daily (auto):   python daily_truths_retriever.py
Manual, one day:    python daily_truths_retriever.py 20260607
Backfill a range:   python daily_truths_retriever.py --since 20260501
Dry run:            python daily_truths_retriever.py 20260607 --dry-run

Every mode de-duplicates against the ids already in truth_social.csv, so a
post is never written twice — re-running the same day just reports 0 new.
"""
import os
import re
import sys
import json
import time
import random
import argparse
import warnings
from datetime import datetime, timedelta, timezone

UTC = timezone.utc  # py3.11 exposes datetime.UTC; this keeps the script on 3.10 too

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ------------------------------------------------------------------ paths ----
_HERE           = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE     = os.path.join(_HERE, "truth_social.csv")
ENTITIES_FILE   = os.path.join(_HERE, "geopolitical_entities.json")

# ----------------------------------------------------------------- config ----
LOOKBACK_DAYS    = 3      # re-scan window before the last post (dedup handles repeats)
CHECKPOINT_EVERY = 100    # append + sleep every N new rows (be gentle on the API)
COLUMNS          = ["id", "account", "date", "text", "url",
                    "favorites", "retweets", "replies"]


# --------------------------------------------------------- account loading ----
def load_primary_accounts() -> list[dict]:
    """
    Return the `primary_accounts` list from geopolitical_entities.json.

    Each entry is expected to have at minimum:
        { "account": "realDonaldTrump", "name": "Donald J. Trump", ... }

    Adding a new TruthSocial account for a future candidate only requires
    appending an entry to geopolitical_entities.json — no code change needed.
    """
    try:
        with open(ENTITIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"❌ {ENTITIES_FILE} not found — cannot determine accounts to retrieve.")

    accounts = data.get("primary_accounts", [])
    if not accounts:
        sys.exit("❌ 'primary_accounts' list is empty in geopolitical_entities.json.")

    # Only TruthSocial accounts are relevant for this retriever
    ts_accounts = [a for a in accounts if a.get("platform") == "truthsocial"]
    if not ts_accounts:
        sys.exit("❌ No primary_accounts with platform='truthsocial' found.")
    return ts_accounts


# ----------------------------------------------------------------- helpers ----
def normalize_text(html: str) -> str:
    """HTML status content → clean single-line text (same rules as the notebook)."""
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text()
    text = text.replace("\r\n", " ").replace("\r", " ")
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"https://\s+", "https://", text)
    text = re.sub(r"http://\s+",  "http://",  text)
    text = re.sub(r"([^\s])https://", r"\1 https://", text)
    text = re.sub(r"([^\s])http://",  r"\1 http://",  text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", " ", text)
    text = re.sub(r"\n{3,}", " ", text)
    return text.strip()


def load_existing(account: str) -> tuple[set, "pd.Timestamp | None"]:
    """
    Return (set_of_ids_for_account, latest_date_utc_or_None) from the CSV.

    Deduplicates per account so that multiple primary accounts can share a
    single truth_social.csv without cross-account ID collisions.
    """
    if not os.path.exists(OUTPUT_FILE):
        return set(), None
    df = pd.read_csv(OUTPUT_FILE, dtype={"id": str})

    # Support legacy files without an 'account' column (single-account era)
    if "account" in df.columns:
        df = df[df["account"] == account]

    ids = set(df["id"].astype(str)) if "id" in df.columns else set()
    last_date = None
    if "date" in df.columns and len(df):
        parsed = pd.to_datetime(df["date"], utc=True, format="mixed", errors="coerce")
        if parsed.notna().any():
            last_date = parsed.max()
    return ids, last_date


def save_rows(rows: list[dict]) -> None:
    """Append rows to truth_social.csv. BOM only when first creating the file."""
    if not rows:
        return
    exists = os.path.exists(OUTPUT_FILE)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not exists,
        index=False,
        encoding="utf-8-sig" if not exists else "utf-8",
        lineterminator="\n",
    )


def get_api():
    """Authenticate with TruthSocial using environment credentials."""
    from truthbrush.api import Api
    user = os.environ.get("TRUTHSOCIAL_USERNAME")
    pw   = os.environ.get("TRUTHSOCIAL_PASSWORD")
    if not user or not pw:
        sys.exit(
            "Set TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD environment "
            "variables before running (credentials are intentionally not "
            "stored in this file)."
        )
    return Api(username=user, password=pw)


def is_keepable(post: dict) -> bool:
    """Original posts only: skip replies and re-truths."""
    return post.get("in_reply_to_id") is None and post.get("reblog") is None


def row_from_post(post: dict, text: str, account: str) -> dict:
    """Build a CSV row dict from a raw TruthSocial API post."""
    return {
        "id":        str(post.get("id")),
        "account":   account,
        "date":      post.get("created_at"),
        "text":      text,
        "url":       post.get("url"),
        "favorites": post.get("favourites_count"),
        "retweets":  post.get("reblogs_count"),
        "replies":   post.get("replies_count"),
    }


def post_dt(post: dict) -> "datetime | None":
    """Parse a status' created_at into a tz-aware UTC datetime (or None)."""
    raw = post.get("created_at")
    if not raw:
        return None
    try:
        return pd.to_datetime(raw, utc=True).to_pydatetime()
    except Exception:
        return None


# -------------------------------------------------------------------- main ----
def retrieve(account: str, created_after=None, created_before=None,
             lookback_days: int = LOOKBACK_DAYS, dry_run: bool = False) -> int:
    """
    Fetch TruthSocial posts for `account` in [created_after, created_before)
    and append only the ones not already in truth_social.csv.

    With no window given, defaults to daily incremental mode: start from the
    last stored post for this account minus the overlap, up to now.

    Returns the number of new posts added.
    """
    existing_ids, last_date = load_existing(account)
    now = datetime.now(UTC)

    if created_before is None:
        created_before = now
    if created_after is None:
        if last_date is not None:
            created_after = (last_date - timedelta(days=lookback_days)).to_pydatetime()
        else:
            created_after = now - timedelta(days=lookback_days)

    print(f"\n[account] @{account}")
    print(f"[csv]     {OUTPUT_FILE}")
    print(f"          existing rows for this account: {len(existing_ids)}"
          + (f", latest: {last_date}" if last_date is not None else " (new)"))
    print(f"[fetch]   from {created_after.isoformat()} to {created_before.isoformat()}"
          + ("   [DRY RUN]" if dry_run else ""))

    api = get_api()
    # truthbrush.pull_statuses only supports the lower bound (created_after);
    # the upper bound (created_before) is enforced client-side below.
    results = api.pull_statuses(account, created_after=created_after)

    rows, added, scanned, dup = [], 0, 0, 0
    for post in results:
        scanned += 1
        cdt = post_dt(post)
        if cdt is not None and cdt >= created_before:
            continue                                    # newer than the window
        if not is_keepable(post):
            continue
        pid = str(post.get("id"))
        if pid in existing_ids:
            dup += 1
            continue                                    # already stored
        text = normalize_text(post.get("content", ""))
        if not text:
            continue

        existing_ids.add(pid)
        rows.append(row_from_post(post, text, account))
        added += 1

        if not dry_run and len(rows) >= CHECKPOINT_EVERY:
            save_rows(rows)
            rows = []
            time.sleep(random.uniform(1.5, 3.5))

    if not dry_run:
        save_rows(rows)

    print(f"[done]    scanned {scanned}, skipped {dup} duplicates, "
          f"{'would add' if dry_run else 'added'} {added} new posts.")
    return added


def parse_date(s: str) -> datetime:
    """Accept YYYYMMDD or YYYY-MM-DD → tz-aware UTC midnight."""
    return datetime.strptime(s.strip().replace("-", ""), "%Y%m%d").replace(tzinfo=UTC)


def main():
    ap = argparse.ArgumentParser(
        description="Daily / manual Truth Social retriever for all primary accounts.")
    ap.add_argument("date", nargs="?",
                    help="YYYYMMDD: manual mode — fetch just that one UTC day.")
    ap.add_argument("--since",
                    help="YYYYMMDD or YYYY-MM-DD: backfill from this date through now.")
    ap.add_argument("--account",
                    help="Restrict retrieval to a single TruthSocial handle "
                         "(e.g. realDonaldTrump). Default: all primary accounts.")
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS,
                    help=f"Daily-mode overlap before the last stored post "
                         f"(default {LOOKBACK_DAYS}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch and report new posts without writing to the CSV.")
    args = ap.parse_args()

    created_after = created_before = None
    if args.date:
        day = parse_date(args.date)
        created_after, created_before = day, day + timedelta(days=1)
    elif args.since:
        created_after = parse_date(args.since)

    # Load account list from geopolitical_entities.json so no account
    # is ever hardcoded here. Adding a 2028 candidate = edit the JSON only.
    all_accounts = load_primary_accounts()
    if args.account:
        all_accounts = [a for a in all_accounts
                        if a["account"].lower() == args.account.lower()]
        if not all_accounts:
            sys.exit(f"❌ Account '{args.account}' not found in primary_accounts.")

    total_added = 0
    for acct in all_accounts:
        handle = acct["account"]
        total_added += retrieve(
            account=handle,
            created_after=created_after,
            created_before=created_before,
            lookback_days=args.lookback_days,
            dry_run=args.dry_run,
        )

    if len(all_accounts) > 1:
        print(f"\n✅ Total new posts across all accounts: {total_added}")


if __name__ == "__main__":
    main()
