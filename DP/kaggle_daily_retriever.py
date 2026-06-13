"""
kaggle_daily_retriever.py
==========================
Kaggle-notebook version of daily_truths_retriever.py, for scheduled execution
on Kaggle to keep the "Donald Trump Truths Dataset"
(epurevsuren/donald-trump-truths-dataset) up to date automatically.

WHY A SEPARATE FILE
--------------------
The local DP/daily_truths_retriever.py reads/writes a local trump_truths.csv
next to itself. On Kaggle there is no local file to begin with — the CSV
lives in the dataset and must be:
  1. loaded as a notebook INPUT (read-only, mounted under /kaggle/input/...),
  2. appended to in memory, and
  3. written to /kaggle/working/trump_truths.csv as the notebook OUTPUT,
     which (with "auto-update dataset on save" enabled) becomes the new
     dataset version.

Everything else — normalization, dedup-by-id, is_keepable filter, schema —
is identical to the local script, copied here so this file is self-contained
and can be pasted directly into a Kaggle notebook cell.

TIMEZONE
--------
trump_truths.csv dates are UTC (ISO 8601 with 'Z', e.g. 2026-06-08T17:12:57.168Z).
Kaggle kernels run in UTC and this script is UTC-only throughout
(datetime.now(UTC)), so no timezone conversion is needed regardless of
Kaggle's host location.

SCHEDULE
--------
In the Notebook Editor -> Settings -> "Schedule a notebook to run":
  - Frequency: Daily
  - Time: 22:00 UTC   (same target window as the local run_daily_truths.bat —
    see DP/daily_truths_retriever.py for why 22:00 UTC was chosen)

SETUP CHECKLIST (one-time, in the Kaggle notebook UI)
------------------------------------------------------
1. Internet: Settings -> Internet -> ON (required to call the Truth Social API).
2. Secrets: Add-ons -> Secrets -> add
       TRUTHSOCIAL_USERNAME
       TRUTHSOCIAL_PASSWORD
3. Input: Add Data -> attach the existing dataset
       epurevsuren/donald-trump-truths-dataset
   (mounted read-only under /kaggle/input/donald-trump-truths-dataset/)
4. Output -> Create Dataset (first run only), then enable
   "Always update this dataset when this notebook is run" so each scheduled
   run publishes a new version automatically.
5. requirements: pip install truthbrush beautifulsoup4 pandas (first cell).

USAGE (as a notebook cell)
---------------------------
    !pip install -q truthbrush beautifulsoup4
    %run kaggle_daily_retriever.py
"""
import os
import re
import time
import random
import warnings
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ------------------------------------------------------------------ config ----
DATASET_SLUG  = "epurevsuren/donald-trump-truths-dataset"
INPUT_CSV     = "/kaggle/input/donald-trump-truths-dataset/trump_truths.csv"
OUTPUT_CSV    = "/kaggle/working/trump_truths.csv"

USERNAME          = "realDonaldTrump"
LOOKBACK_DAYS     = 3      # re-scan window before the last post (dedup handles repeats)
CHECKPOINT_EVERY  = 100    # batch size before a short sleep (be gentle on the API)
COLUMNS           = ["id", "date", "text", "url", "favorites", "retweets", "replies"]


# ----------------------------------------------------------------- helpers ----
def normalize_text(html):
    """HTML status content -> clean single-line text (same rules as the local script)."""
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text()
    text = text.replace("\r\n", " ").replace("\r", " ")
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
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
    """Return (df, set_of_ids, latest_date_utc_or_None) from the input dataset CSV."""
    if not os.path.exists(INPUT_CSV):
        return pd.DataFrame(columns=COLUMNS), set(), None
    df = pd.read_csv(INPUT_CSV)
    ids = set(df["id"].astype(str)) if "id" in df.columns else set()
    last_date = None
    if "date" in df.columns and len(df):
        parsed = pd.to_datetime(df["date"], utc=True, format="mixed", errors="coerce")
        if parsed.notna().any():
            last_date = parsed.max()
    return df, ids, last_date


def get_api():
    from truthbrush.api import Api
    # On Kaggle, secrets are exposed as env vars only after UserSecretsClient
    # attaches them — try that first, fall back to plain env vars.
    user = os.environ.get("TRUTHSOCIAL_USERNAME")
    pw   = os.environ.get("TRUTHSOCIAL_PASSWORD")
    if not user or not pw:
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            user = user or secrets.get_secret("TRUTHSOCIAL_USERNAME")
            pw   = pw or secrets.get_secret("TRUTHSOCIAL_PASSWORD")
        except Exception:
            pass
    if not user or not pw:
        raise SystemExit(
            "Add TRUTHSOCIAL_USERNAME / TRUTHSOCIAL_PASSWORD as Kaggle notebook "
            "Secrets (Add-ons -> Secrets)."
        )
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
def retrieve():
    """Daily incremental fetch: from (last stored post - LOOKBACK_DAYS) to now,
    de-duplicated against the input dataset's ids. Writes the merged CSV to
    /kaggle/working/trump_truths.csv (the notebook output)."""
    existing_df, existing_ids, last_date = load_existing()
    now = datetime.now(UTC)

    created_before = now
    if last_date is not None:
        created_after = (last_date - timedelta(days=LOOKBACK_DAYS)).to_pydatetime()
    else:
        created_after = now - timedelta(days=LOOKBACK_DAYS)

    print(f"[input] {INPUT_CSV}")
    print(f"        existing rows: {len(existing_ids)}"
          + (f", latest: {last_date}" if last_date is not None else " (no existing file found)"))
    print(f"[fetch] {USERNAME} from {created_after.isoformat()} to {created_before.isoformat()}")

    api = get_api()
    results = api.pull_statuses(USERNAME, created_after=created_after)

    new_rows, added, scanned, dup = [], 0, 0, 0
    for post in results:
        scanned += 1
        cdt = post_dt(post)
        if cdt is not None and cdt >= created_before:   # newer than the window
            continue
        if not is_keepable(post):
            continue
        pid = str(post.get("id"))
        if pid in existing_ids:                         # already stored -> never duplicate
            dup += 1
            continue
        text = normalize_text(post.get("content", ""))
        if not text:
            continue

        existing_ids.add(pid)
        new_rows.append(row_from_post(post, text))
        added += 1

        if len(new_rows) % CHECKPOINT_EVERY == 0:
            time.sleep(random.uniform(1.5, 3.5))

    print(f"[done]  scanned {scanned} statuses, skipped {dup} already in dataset, "
          f"added {added} new truths.")

    if new_rows:
        merged = pd.concat([existing_df, pd.DataFrame(new_rows, columns=COLUMNS)],
                            ignore_index=True)
    else:
        merged = existing_df

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False, lineterminator="\n", encoding="utf-8")
    print(f"[saved] {OUTPUT_CSV} ({len(merged)} rows total, {added} new)")
    return added


if __name__ == "__main__":
    retrieve()
