"""
tmp_extend_x_tweets.py   (TEMPORARY / one-off)
==============================================
Extend DP/x_tweets.csv with two historical archives, mapped into the
x_tweets schema:

  trump_tweets.csv  ->  @realDonaldTrump   window 2016-11-01 .. 2021-01-20
  JoeBiden.csv      ->  @JoeBiden          window 2020-11-01 .. 2025-01-20

Target schema (from x_tweets.csv):
  id, date, text, url, favorites, retweets, replies, account, account_name, language

Dates are normalised to UTC ISO-8601 (e.g. 2018-03-04T12:34:56+00:00) to match
x_tweets. Rows are deduped by id (existing x_tweets rows win) and sorted newest
-first to match the base file's ordering.

By default this is NON-DESTRUCTIVE: it writes x_tweets_extended.csv and leaves
x_tweets.csv untouched. Set OVERWRITE = True to replace x_tweets.csv in place
(a .bak backup is made first).
"""
import os
import shutil
import pandas as pd

HERE       = os.path.dirname(os.path.abspath(__file__))
X_PATH     = os.path.join(HERE, "x_tweets.csv")
TRUMP_PATH = os.path.join(HERE, "trump_tweets.csv")
BIDEN_PATH = os.path.join(HERE, "JoeBiden.csv")

OVERWRITE  = False   # False -> write x_tweets_extended.csv ; True -> replace x_tweets.csv (+ .bak)
OUT_PATH   = X_PATH if OVERWRITE else os.path.join(HERE, "x_tweets_extended.csv")

# Target column order (must match x_tweets.csv exactly).
COLUMNS = ["id", "date", "text", "url", "favorites", "retweets", "replies",
           "account", "account_name", "language"]

TRUMP_WINDOW = ("2016-11-01", "2021-01-20")   # inclusive of both end days
BIDEN_WINDOW = ("2020-11-01", "2025-01-20")

# The archives include retweets / deleted tweets. Default = keep everything
# (don't silently drop data). Flip to True to exclude them.
DROP_TRUMP_RETWEETS = False
DROP_TRUMP_DELETED  = False

LANG = "en"   # Trump/Biden tweet in English; set to "" to leave language blank.


def _window_mask(dt_utc, start, end):
    """Boolean mask for start 00:00 .. end 23:59:59 (both days inclusive), UTC."""
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)   # end day inclusive
    return (dt_utc >= lo) & (dt_utc < hi)


def load_trump():
    print(f"[trump] reading {os.path.basename(TRUMP_PATH)} ...", flush=True)
    t = pd.read_csv(TRUMP_PATH, dtype={"id": str})
    # trump dates are naive UTC (e.g. '2018-03-04 12:34:56'); localise to UTC.
    dt = pd.to_datetime(t["date"], utc=True, errors="coerce")
    mask = _window_mask(dt, *TRUMP_WINDOW)
    if DROP_TRUMP_RETWEETS:
        mask &= t["isRetweet"].astype(str).str.lower().ne("t")
    if DROP_TRUMP_DELETED:
        mask &= t["isDeleted"].astype(str).str.lower().ne("t")
    t, dt = t[mask].copy(), dt[mask]
    out = pd.DataFrame({
        "id":           t["id"].astype(str),
        "date":         dt.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "text":         t["text"],
        "url":          "https://twitter.com/realDonaldTrump/status/" + t["id"].astype(str),
        "favorites":    t["favorites"],
        "retweets":     t["retweets"],
        "replies":      pd.NA,                 # not present in this archive
        "account":      "realDonaldTrump",
        "account_name": "Donald J. Trump",
        "language":     LANG,
    })
    print(f"[trump] {len(out):,} tweets in {TRUMP_WINDOW[0]} .. {TRUMP_WINDOW[1]}")
    return out


def load_biden():
    print(f"[biden] reading {os.path.basename(BIDEN_PATH)} ...", flush=True)
    b = pd.read_csv(BIDEN_PATH, dtype={"id": str})
    # biden 'date' is already tz-aware (e.g. '2022-12-17 22:35:00+00:00').
    dt = pd.to_datetime(b["date"], utc=True, errors="coerce")
    mask = _window_mask(dt, *BIDEN_WINDOW)
    b, dt = b[mask].copy(), dt[mask]
    out = pd.DataFrame({
        "id":           b["id"].astype(str),
        "date":         dt.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "text":         b["content"],
        "url":          b["url"],
        "favorites":    b["likeCount"],
        "retweets":     b["retweetCount"],
        "replies":      b["replyCount"],
        "account":      "JoeBiden",
        "account_name": "Joe Biden",
        "language":     LANG,
    })
    print(f"[biden] {len(out):,} tweets in {BIDEN_WINDOW[0]} .. {BIDEN_WINDOW[1]}")
    return out


def main():
    print(f"[base]  reading {os.path.basename(X_PATH)} ...", flush=True)
    x = pd.read_csv(X_PATH, dtype={"id": str}, encoding="utf-8-sig")
    missing = [c for c in COLUMNS if c not in x.columns]
    if missing:
        raise SystemExit(f"x_tweets.csv is missing expected columns: {missing}")
    x = x[COLUMNS]
    print(f"[base]  {len(x):,} existing rows")

    add = pd.concat([load_trump(), load_biden()], ignore_index=True)[COLUMNS]

    before = len(x)
    combined = pd.concat([x, add], ignore_index=True)
    # dedup by id, keep first occurrence -> existing x_tweets rows win over archives
    combined = combined.drop_duplicates(subset="id", keep="first")

    # normalise numeric columns to clean nullable ints (no trailing .0, blank for NA)
    for c in ("favorites", "retweets", "replies"):
        combined[c] = pd.to_numeric(combined[c], errors="coerce").astype("Int64")

    # sort newest-first to match x_tweets ordering
    order = pd.to_datetime(combined["date"], utc=True, errors="coerce")
    combined = (combined.assign(_o=order)
                        .sort_values("_o", ascending=False)
                        .drop(columns="_o")
                        .reset_index(drop=True))

    added = len(combined) - before
    print(f"[merge] +{added:,} new rows  ->  {len(combined):,} total")
    print(f"[merge] date span: {combined['date'].min()}  ..  {combined['date'].max()}")
    print("[merge] per-account added:")
    print(add['account'].value_counts().to_string())

    if OVERWRITE and os.path.exists(X_PATH):
        bak = X_PATH + ".bak"
        shutil.copy2(X_PATH, bak)
        print(f"[save]  backed up original -> {os.path.basename(bak)}")

    combined.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[done]  wrote {os.path.basename(OUT_PATH)}  ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
