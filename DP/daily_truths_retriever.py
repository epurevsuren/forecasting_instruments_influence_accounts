"""
daily_truths_retriever.py
=========================
Daily, incremental retriever for @realDonaldTrump Truth Social posts. Built from
TruthSocial.ipynb, but turned into a repeatable job that ENRICHES the existing
DP/trump_truths.csv instead of re-scraping from scratch:

  * loads the ids already in trump_truths.csv,
  * pulls only recent statuses (from the last captured post, with a small overlap),
  * skips replies / re-truths / empty posts and anything already stored,
  * appends only the NEW rows, keeping the exact same schema:
        id, date, text, url, favorites, retweets, replies

WHEN TO RUN  (the NY <-> Asia gap)
----------------------------------
Target: 22:00 UTC, once per day.

In UTC the New York equity session closes at 20:00 (EDT) / 21:00 (EST) and the
Tokyo session opens at 00:00 (09:00 JST, Japan has no DST). The window that is a
gap in BOTH seasons is 21:00-00:00 UTC, which is also the low-liquidity FX lull
between the New York and Asian sessions. 22:00 UTC sits inside it:

  * ~1-2 h AFTER the NY close  -> a full, market-moving NY session of posts is
    already captured, and
  * ~2 h BEFORE Tokyo opens    -> the dataset is refreshed before the Asian
    session can trade on it.

Running in this gap (rather than mid-session) also means the scrape never
competes with the moment you actually want the scored signal to be ready.

CREDENTIALS
-----------
Set these environment variables before running (NOT hardcoded here on purpose,
so the password never lands in the repo):

    TRUTHSOCIAL_USERNAME
    TRUTHSOCIAL_PASSWORD

Run daily (auto):   python daily_truths_retriever.py
Manual, one day:    python daily_truths_retriever.py 20260607
Backfill a range:   python daily_truths_retriever.py --since 20260501
Dry run:            python daily_truths_retriever.py 20260607 --dry-run

Every mode de-duplicates against the ids already in trump_truths.csv, so a post
is never written twice — re-running the same day just reports 0 new.
"""
import os
import re
import sys
import time
import random
import argparse
import warnings
from datetime import datetime, timedelta, timezone

UTC = timezone.utc  # py3.11 exposes datetime.UTC; this keeps the script on 3.10 too

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ------------------------------------------------------------------ config ----
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(_HERE, "trump_truths.csv")   # resolve next to this file

USERNAME          = "realDonaldTrump"
LOOKBACK_DAYS     = 3      # re-scan window before the last post (dedup handles repeats)
CHECKPOINT_EVERY  = 100    # append + sleep every N new rows (be gentle on the API)
COLUMNS           = ["id", "date", "text", "url", "favorites", "retweets", "replies"]


# ----------------------------------------------------------------- helpers ----
def normalize_text(html):
    """HTML status content -> clean single-line text (same rules as the notebook)."""
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text()
    text = text.replace("\r\n", " ").replace("\r", " ")
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"https://\s+", "https://", text)
    text = re.sub(r"http://\s+", "http://", text)
    text = re.sub(r"([^\s])https://", r"\1 https://", text)
    text = re.sub(r"([^\s])http://", r"\1 http://", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", " ", text)
    text = re.sub(r"\n{3,}", " ", text)
    return text.strip()


def load_existing():
    """Return (set_of_ids, latest_date_utc_or_None) from the current CSV."""
    if not os.path.exists(OUTPUT_FILE):
        return set(), None
    df = pd.read_csv(OUTPUT_FILE)
    ids = set(df["id"].astype(str)) if "id" in df.columns else set()
    last_date = None
    if "date" in df.columns and len(df):
        parsed = pd.to_datetime(df["date"], utc=True, format="mixed", errors="coerce")
        if parsed.notna().any():
            last_date = parsed.max()
    return ids, last_date


def save_rows(rows):
    """Append rows to the CSV. BOM only when first creating the file."""
    if not rows:
        return
    exists = os.path.exists(OUTPUT_FILE)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not exists,
        index=False,
        encoding="utf-8-sig" if not exists else "utf-8",  # avoid mid-file BOM on append
        lineterminator="\n",
    )


def get_api():
    from truthbrush.api import Api
    user = os.environ.get("TRUTHSOCIAL_USERNAME")
    pw   = os.environ.get("TRUTHSOCIAL_PASSWORD")
    if not user or not pw:
        sys.exit("Set TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD environment "
                 "variables before running (credentials are intentionally not "
                 "stored in this file).")
    return Api(username=user, password=pw)


def is_keepable(post):
    """Original post only: no replies, no re-truths."""
    return post.get("in_reply_to_id") is None and post.get("reblog") is None


def row_from_post(post, text):
    return {
        "id":        str(post.get("id")),
        "date":      post.get("created_at"),
        "text":      text,
        "url":       post.get("url"),
        "favorites": post.get("favourites_count"),
        "retweets":  post.get("reblogs_count"),
        "replies":   post.get("replies_count"),
    }


def post_dt(post):
    """Parse a status' created_at into a tz-aware UTC datetime (or None)."""
    raw = post.get("created_at")
    if not raw:
        return None
    try:
        return pd.to_datetime(raw, utc=True).to_pydatetime()
    except Exception:
        return None


# -------------------------------------------------------------------- main ----
def retrieve(created_after=None, created_before=None,
             lookback_days=LOOKBACK_DAYS, dry_run=False):
    """
    Fetch statuses in [created_after, created_before) and append only the ones
    not already in the CSV. With no window given, defaults to daily incremental
    (from the last stored post minus the overlap, up to now).
    """
    existing_ids, last_date = load_existing()
    now = datetime.now(UTC)

    if created_before is None:
        created_before = now
    if created_after is None:                      # daily incremental default
        if last_date is not None:
            created_after = (last_date - timedelta(days=lookback_days)).to_pydatetime()
        else:
            created_after = now - timedelta(days=lookback_days)

    print(f"[csv]   {OUTPUT_FILE}")
    print(f"        existing rows: {len(existing_ids)}"
          + (f", latest: {last_date}" if last_date is not None else " (new file)"))
    print(f"[fetch] {USERNAME} from {created_after.isoformat()} to {created_before.isoformat()}"
          + ("   [DRY RUN]" if dry_run else ""))

    api = get_api()
    # truthbrush.pull_statuses only supports the lower bound (created_after);
    # the upper bound (created_before) is enforced client-side below.
    results = api.pull_statuses(USERNAME, created_after=created_after)

    rows, added, scanned, dup = [], 0, 0, 0
    for post in results:
        scanned += 1
        cdt = post_dt(post)
        if cdt is not None and cdt >= created_before:   # newer than the window
            continue
        if not is_keepable(post):
            continue
        pid = str(post.get("id"))
        if pid in existing_ids:                    # already stored -> never duplicate
            dup += 1
            continue
        text = normalize_text(post.get("content", ""))
        if not text:
            continue

        existing_ids.add(pid)
        rows.append(row_from_post(post, text))
        added += 1

        if not dry_run and len(rows) >= CHECKPOINT_EVERY:
            save_rows(rows)
            rows = []
            time.sleep(random.uniform(1.5, 3.5))

    if not dry_run:
        save_rows(rows)

    print(f"[done]  scanned {scanned} statuses, skipped {dup} already in file, "
          f"{'would add' if dry_run else 'added'} {added} new truths.")
    return added


def parse_date(s):
    """Accept YYYYMMDD or YYYY-MM-DD -> tz-aware UTC midnight."""
    return datetime.strptime(s.strip().replace("-", ""), "%Y%m%d").replace(tzinfo=UTC)


def main():
    ap = argparse.ArgumentParser(description="Daily / manual Truth Social retriever.")
    ap.add_argument("date", nargs="?",
                    help="YYYYMMDD: manual mode — fetch just that one UTC day "
                         "(de-duplicated against the file).")
    ap.add_argument("--since",
                    help="YYYYMMDD or YYYY-MM-DD: backfill from this date through now.")
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS,
                    help=f"Daily-mode overlap before the last stored post (default {LOOKBACK_DAYS}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch and report new posts without writing to the CSV.")
    args = ap.parse_args()

    created_after = created_before = None
    if args.date:                                  # manual single-day mode
        day = parse_date(args.date)
        created_after, created_before = day, day + timedelta(days=1)
    elif args.since:                               # range backfill
        created_after = parse_date(args.since)

    retrieve(created_after=created_after, created_before=created_before,
             lookback_days=args.lookback_days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
