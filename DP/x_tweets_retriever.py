"""
x_tweets_retriever.py
======================
Retrieves tweets for the geopolitical accounts in DP/influence_accounts.json
using Playwright (local headless Chromium with your real X cookies).

SETUP
-----
1. uv add playwright
2. uv run playwright install chromium
3. Export cookies from x.com via Cookie-Editor browser extension
   -> Save JSON array to DP/.x_cookies.json

USAGE
-----
  python x_tweets_retriever.py                               # latest 40 tweets per account
  python x_tweets_retriever.py --since 20241101             # backfill from Nov 1 2024
  python x_tweets_retriever.py --since 20250101 --until 20250601
  python x_tweets_retriever.py --handles ZelenskyyUa,IDF --since 20260601
  python x_tweets_retriever.py --handles ZelenskyyUa --dry-run
  python x_tweets_retriever.py --handles JoeBiden --since 20221217 --refetch  # re-scrape covered months

ROBUSTNESS / RESUME
-------------------
X throttles scraping to ~600-700 tweets per session, then silently serves empty
pages. To handle that: each monthly chunk is deduped and written to disk
IMMEDIATELY (an interruption never loses progress), every chunk logs its yield,
and rate-limit/empty streaks print a warning instead of stalling quietly. Months
already present in x_tweets.csv are SKIPPED on re-run, so simply re-running the
SAME command resumes forward into un-fetched months until the range is complete.
Use --refetch to force re-scraping covered months (e.g. to fill a partial month).

COOKIE ROTATION: on an X rate-limit the run pauses 1 minute and switches to the
next cookie set in DP/.x_cookies/ (.x_cookies.json, .x_cookies1.json ..
.x_cookies6.json), retrying the SAME month with a fresh logged-in session. It
cycles the pool; once every set has been throttled on one month it takes a longer
cooldown -- so ONE invocation runs unattended far past one account's ~600-700 cap.

OUTPUT SCHEMA (mirrors truth_social.csv + two extra columns)
-------------------------------------------------------------
  id, date, text, url, favorites, retweets, replies, account, account_name
"""
import os
import glob
import re
import sys
import json
import time
import asyncio
import argparse
import random
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import jmespath
import pandas as pd
from langdetect import detect as _langdetect, DetectorFactory
DetectorFactory.seed = 0

UTC = timezone.utc

# ------------------------------------------------------------------ config ----
_HERE         = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE   = os.path.join(_HERE, "x_tweets.csv")
ENTITIES_FILE = os.path.join(_HERE, "influence_accounts.json")
COOKIES_FILE  = os.path.join(_HERE, ".x_cookies.json")
COOKIES_DIR   = os.path.join(_HERE, ".x_cookies")   # folder of rotating cookie sets (.x_cookies*.json)

COLUMNS = ["id", "date", "text", "url", "favorites", "retweets", "replies",
           "account", "account_name", "language"]

MAX_PER_ACCOUNT_DEFAULT = 40
SLEEP_BETWEEN_ACCOUNTS  = (4.0, 8.0)
SLEEP_BETWEEN_CHUNKS    = (2.0, 4.0)

# Robustness / resume (X throttles scraping ~600-700 tweets per session, then
# silently returns empty pages). Each monthly chunk is saved to disk IMMEDIATELY,
# so an interruption never loses progress; re-running the same command resumes.
MIN_MONTH_COVERAGE  = 1     # skip a month that already has >= this many rows (resume). --refetch overrides
GAP_DAYS_DEFAULT    = 3     # with an explicit --since, refetch a month whose day coverage has a hole longer than this
EMPTY_STREAK_WARN   = 3     # consecutive empty months -> warn about likely X throttling
RATE_LIMIT_BACKOFF  = 30     # pause after an X rate-limit signal (30 seconds), then rotate to the next cookie set
LONG_COOLDOWN       = 247    # longer pause once EVERY cookie set has been throttled on the same month

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/125.0.0.0 Safari/537.36")


# ----------------------------------------------------------------- helpers ----
def normalize_text(text):
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\r", " ")
    text = re.sub(r"([^\s])https://", r"\1 https://", text)
    text = re.sub(r"([^\s])http://",  r"\1 http://",  text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_date(s):
    return datetime.strptime(s.strip().replace("-", ""), "%Y%m%d").replace(tzinfo=UTC)


def monthly_chunks(since, until):
    """Yield (chunk_since, chunk_until) pairs covering [since, until) by calendar month."""
    cur = since.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur < until:
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1)
        else:
            nxt = cur.replace(month=cur.month + 1)
        yield max(cur, since), min(nxt, until)
        cur = nxt


def _is_expired(entry) -> bool:
    """
    True when the account's influence window has ENDED (active_to or
    expiration_date strictly in the past) — resigned, dead, out of office,
    or moved platforms. ONLY expiry excludes an account: a not-yet-started
    active_from (e.g. Kremlin might reactivate, an election candidate not
    yet sworn in) still gets fetched — who knows when they start mattering,
    and the active-window damping in sync_unified_feed handles the weighting.
    """
    for key in ("active_to", "expiration_date"):
        v = entry.get(key)
        if v is None or str(v).strip().upper() in ("", "N/A", "NONE", "NULL"):
            continue
        try:
            if pd.Timestamp(v).date() < datetime.now(UTC).date():
                return True
        except (ValueError, TypeError):
            continue
    return False


def load_tracked_accounts(handles_filter=None):
    with open(ENTITIES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # (entries, apply_expiry_filter). Archives = out-of-office / former accounts
    # (e.g. @JoeBiden, @POTUS46Archive, former PMs). They are NEVER fetched on a
    # default/scheduled run -- only when the user explicitly passes --handles --
    # and their expiry is ignored so their historical tweets can be backfilled.
    accounts, seen, skipped, non_x = [], set(), [], []
    sections = [(data.get("entities", []), True),
                (data.get("institutions", {}).get("entries", []), True)]
    if handles_filter:
        sections.append((data.get("archives", {}).get("entries", []), False))
    for entries, apply_expiry in sections:
        for entry in entries:
            handle = entry.get("account")
            if not handle:
                continue
            handle = handle.lstrip("@")
            # X/Twitter retriever ONLY — skip accounts whose platform is Truth
            # Social (e.g. @JDVance1, whose X handle is @JDVance and which does not
            # exist on X). Those are fetched by daily_truths_retriever.py instead.
            if str(entry.get("platform", "")).strip().lower() == "truthsocial":
                if handle.lower() not in seen:
                    non_x.append(handle)
                    seen.add(handle.lower())
                continue
            if handle.lower() in seen:
                continue
            seen.add(handle.lower())
            if apply_expiry and _is_expired(entry):
                skipped.append(handle)
                continue
            accounts.append({"handle": handle, "name": entry.get("name", handle)})
    if non_x:
        print(f"[skip] {len(non_x)} Truth Social account(s) not fetched by the X retriever "
              f"(handled by daily_truths_retriever.py): {', '.join(non_x)}")
    if skipped:
        print(f"[skip] {len(skipped)} expired account(s) not fetched: {', '.join(skipped)}")

    if handles_filter:
        wanted = {h.lstrip("@").lower() for h in handles_filter}
        accounts = [a for a in accounts if a["handle"].lower() in wanted]

    return accounts


def load_existing_coverage():
    """Return (set_of_ids, {handle_lower: {'YYYY-MM': count}}).

    The per-handle monthly counts drive RESUME: a monthly chunk already covered
    in x_tweets.csv is skipped on re-run, so successive runs march forward into
    un-fetched months instead of re-hitting X's per-session cap on the same
    early tweets."""
    if not os.path.exists(OUTPUT_FILE):
        return set(), {}
    df = pd.read_csv(OUTPUT_FILE, dtype={"id": str})
    ids = set(df["id"].astype(str)) if "id" in df.columns else set()
    cov: dict = {}
    if "account" in df.columns and "date" in df.columns:
        dt = pd.to_datetime(df["date"], utc=True, errors="coerce")
        months = dt.dt.strftime("%Y-%m")
        for h, mth in zip(df["account"].astype(str).str.lower(), months):
            if not isinstance(mth, str):
                continue
            cov.setdefault(h, {})
            cov[h][mth] = cov[h].get(mth, 0) + 1
        # DAY-LEVEL presence, so an INTRA-MONTH GAP is visible. Month counts
        # alone cannot see one: netanyahu had 44 rows in 2026-07 and the month
        # was skipped as "covered" while 07-07..07-15 was entirely missing.
        days = dt.dt.strftime("%Y-%m-%d")
        for h, dy in zip(df["account"].astype(str).str.lower(), days):
            if isinstance(dy, str):
                _COV_DAYS.setdefault(h, set()).add(dy)
    return ids, cov


# {handle_lower: {'YYYY-MM-DD', ...}} — filled by load_existing_coverage()
_COV_DAYS: dict = {}


def month_has_gap(handle, mkey, lo, hi, max_gap_days):
    """True when the account's rows inside [lo, hi) leave a hole longer than
    `max_gap_days`, i.e. the month is only PARTIALLY covered.

    Bounded by the account's own first/last day in that month so a dormant
    account (no tweets at all after the 3rd) is not re-fetched forever — only
    holes BETWEEN known activity count, plus a hole running up to the end of
    the requested window when the account was clearly still posting.
    """
    have = sorted(d for d in _COV_DAYS.get(str(handle).lower(), set())
                  if lo.strftime("%Y-%m-%d") <= d < hi.strftime("%Y-%m-%d"))
    if not have:
        return True, 999, lo              # nothing at all in range -> fetch
    days = [datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=UTC) for d in have]
    edges = [lo] + days + [min(hi, datetime.now(UTC))]
    worst, at = 0, None
    for a, b in zip(edges, edges[1:]):
        gap = (b - a).days
        if gap > worst:
            worst, at = gap, a
    if worst > max_gap_days:
        return True, worst, at
    return False, worst, at


def save_rows(rows):
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


def save_new(raw_rows, handle, name, existing_ids, dry_run):
    """Dedup raw_rows against existing_ids and write to disk IMMEDIATELY.
    Returns the number of new rows written. Mutates existing_ids."""
    rows = []
    for r in raw_rows:
        tid = str(r.get("id", ""))
        if not tid or tid in existing_ids:
            continue
        existing_ids.add(tid)
        r = dict(r)
        r["account"], r["account_name"] = handle, name
        rows.append({c: r.get(c) for c in COLUMNS})
    if rows and not dry_run:
        save_rows(rows)
    return len(rows)


# ---------------------------------------------------------------- X parsing ----
# X MOVES THESE FIELDS. Each entry is tried in order and the first non-null
# wins, so a schema change degrades one field instead of silently returning
# zero tweets. `id` is the killer: if it resolves to None the tweet is DROPPED,
# which is exactly how a rename turns into "0 tweets updated" with no error.
#   - screen_name moved legacy.* -> core.* (X user-object restructure, 2024-25)
#   - long posts carry text in note_tweet.*, not legacy.full_text
_TWEET_JMES = jmespath.compile("""{
    id:              legacy.id_str,
    created_at:      legacy.created_at,
    text:            legacy.full_text,
    favorite_count:  legacy.favorite_count,
    retweet_count:   legacy.retweet_count,
    reply_count:     legacy.reply_count,
    screen_name:     core.user_results.result.legacy.screen_name
}""")

# Fallbacks applied when the primary path above yields None for a field.
_TWEET_FALLBACKS = {
    "id": ["rest_id", "legacy.id_str", "tweet.rest_id"],
    "created_at": ["legacy.created_at", "tweet.legacy.created_at"],
    "text": ["note_tweet.note_tweet_results.result.text",
             "legacy.full_text", "tweet.legacy.full_text"],
    "favorite_count": ["legacy.favorite_count"],
    "retweet_count": ["legacy.retweet_count"],
    "reply_count": ["legacy.reply_count"],
    "screen_name": ["core.user_results.result.core.screen_name",
                    "core.user_results.result.legacy.screen_name"],
}
_FALLBACK_C = {k: [jmespath.compile(p) for p in v]
               for k, v in _TWEET_FALLBACKS.items()}


def _parse_tweet(result):
    """Primary JMES + per-field fallbacks. Returns dict or None."""
    parsed = _TWEET_JMES.search(result) or {}
    for field, exprs in _FALLBACK_C.items():
        if parsed.get(field) in (None, ""):
            for e in exprs:
                v = e.search(result)
                if v not in (None, ""):
                    parsed[field] = v
                    break
    return parsed if parsed.get("id") else None

_CREATED_AT_FMT = "%a %b %d %H:%M:%S %z %Y"


def _parse_created_at(s):
    try:
        return datetime.strptime(s, _CREATED_AT_FMT)
    except Exception:
        return None


DEBUG = os.environ.get("X_DEBUG", "0") == "1"

# Operation names that carry a tweet timeline. X renames these periodically —
# the old filter was hard-coded to UserTweets/SearchTimeline only, so a rename
# meant every response was ignored and the run reported "0 tweets updated"
# with no error at all.
_OP_MATCH = ("UserTweets", "SearchTimeline", "UserTweetsAndReplies",
             "UserMedia", "TweetDetail", "HomeTimeline", "ListLatestTweets")


def _is_timeline_response(url):
    if "/graphql/" not in url and "/i/api/" not in url:
        return False
    return any(op.lower() in url.lower() for op in _OP_MATCH)


def _unwrap_tweet_result(result):
    """Handle TweetWithVisibilityResults wrapper (extra nesting X added ~2023)."""
    if result.get("__typename") == "TweetWithVisibilityResults":
        result = result.get("tweet", {})
    return result


def _find_instructions(body):
    """Locate timeline instructions wherever X has moved them this month.

    The two hard-coded paths below have both been valid at different times;
    `timeline_v2` in particular was dropped for plain `timeline` in newer
    responses. When both miss we RECURSIVELY search for any 'instructions'
    list, so a path rename degrades to a slower search instead of silently
    returning zero tweets.
    """
    d = body.get("data", {})
    paths = [
        ("search_by_raw_query", "search_timeline", "timeline", "instructions"),
        ("user", "result", "timeline_v2", "timeline", "instructions"),
        ("user", "result", "timeline", "timeline", "instructions"),
    ]
    for p in paths:
        cur = d
        for k in p:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        if isinstance(cur, list) and cur:
            return cur, ".".join(p)

    # last resort: walk the tree for the first 'instructions' list
    stack = [(d, "data")]
    while stack:
        node, where = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "instructions" and isinstance(v, list) and v:
                    return v, f"{where}.{k} (RECURSIVE)"
                if isinstance(v, (dict, list)):
                    stack.append((v, f"{where}.{k}"))
        elif isinstance(node, list):
            for v in node[:20]:
                if isinstance(v, (dict, list)):
                    stack.append((v, where))
    return [], None


# Set by --debug; collects why extraction produced nothing.
_DIAG = {"responses": 0, "graphql": 0, "matched": 0, "json_fail": 0,
         "no_instructions": 0, "entries": 0, "no_id": 0, "ops": {},
         "dumped": False, "last_err": ""}


def _extract_from_graphql(body):
    tweets = []
    instructions, _path = _find_instructions(body)
    if not instructions:
        _DIAG["no_instructions"] += 1
    for instr in instructions:
        # Handle both entries[] and items[] (module-level timeline items)
        entries = instr.get("entries", [])
        for entry in entries:
            content = entry.get("content", {})
            # Standard single-tweet entry
            item = content.get("itemContent", {})
            items = [item] if item else []
            # Conversation module: content.items[].item.itemContent
            for mod_item in content.get("items", []):
                sub = mod_item.get("item", {}).get("itemContent", {})
                if sub:
                    items.append(sub)
            for item in items:
                if item.get("__typename") != "TimelineTweet":
                    continue
                _DIAG["entries"] += 1
                result = _unwrap_tweet_result(
                    item.get("tweet_results", {}).get("result", {})
                )
                parsed = _parse_tweet(result)
                if parsed:
                    tweets.append(parsed)
                else:
                    # a tweet WAS present but no id could be resolved — the
                    # single most likely cause of "0 tweets updated"
                    _DIAG["no_id"] += 1
                    if DEBUG and not _DIAG["dumped"]:
                        _dump_sample(result, "tweet_result_no_id")
    return tweets


def _dump_sample(obj, tag):
    """Write one raw payload so a schema change can be read, not guessed."""
    try:
        p = os.path.join(_HERE, f"_x_debug_{tag}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        _DIAG["dumped"] = True
        print(f"    🧪 dumped raw payload -> {os.path.basename(p)}")
    except Exception as e:                                    # noqa: BLE001
        print(f"    ⚠️  dump failed: {e}")


def print_diag():
    d = _DIAG
    print("\n" + "=" * 70)
    print("  X RETRIEVER DIAGNOSTICS")
    print("=" * 70)
    print(f"  responses seen           : {d['responses']}")
    print(f"  graphql responses        : {d['graphql']}")
    print(f"  matched the URL filter   : {d['matched']}")
    print(f"  response.json() failed   : {d['json_fail']}"
          + (f"   last: {d['last_err'][:60]}" if d['last_err'] else ""))
    print(f"  no timeline instructions : {d['no_instructions']}")
    print(f"  TimelineTweet entries    : {d['entries']}")
    print(f"  entries with NO id       : {d['no_id']}   <- schema change if > 0")
    if d["ops"]:
        print("  graphql operations seen  :")
        for k, v in sorted(d["ops"].items(), key=lambda x: -x[1])[:12]:
            print(f"      {v:>4}x  {k}")
    print("-" * 70)
    if d["graphql"] and not d["matched"]:
        print("  ➜ GraphQL traffic exists but NOTHING matched the URL filter.")
        print("    X renamed the operation — add the name(s) above to _OP_MATCH.")
    elif d["matched"] and not d["entries"]:
        print("  ➜ Responses matched but held no TimelineTweet entries:")
        print("    either the timeline path moved (see 'no instructions') or")
        print("    the session is logged out / rate-limited.")
    elif d["entries"] and d["no_id"] == d["entries"]:
        print("  ➜ Tweets ARE arriving but no id resolves — X moved the field.")
        print("    Read _x_debug_tweet_result_no_id.json and add the new path")
        print("    to _TWEET_FALLBACKS['id'].")
    elif not d["responses"]:
        print("  ➜ No responses captured at all. The page.on('response') hook")
        print("    never fired — check the Playwright version/API.")
    print("=" * 70)


# ----------------------------------------------------------------- cookies ----
def _cookie_files():
    """Rotation pool: sorted .x_cookies*.json inside .x_cookies/, else the legacy
    single DP/.x_cookies.json. Rotating between logged-in X sessions lets the run
    continue past one account's ~600-700-tweet throttle without re-running."""
    files = []
    if os.path.isdir(COOKIES_DIR):
        files = sorted(glob.glob(os.path.join(COOKIES_DIR, ".x_cookies*.json")))
    if not files and os.path.exists(COOKIES_FILE):
        files = [COOKIES_FILE]
    return files


def _playwright_cookies(path=COOKIES_FILE):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    result = []
    for c in (raw if isinstance(raw, list) else []):
        if "name" not in c or "value" not in c:
            continue
        ss = c.get("sameSite", "")
        same_site = {"no_restriction": "None", "lax": "Lax", "strict": "Strict"}.get(ss, "None")
        result.append({
            "name":     c["name"],
            "value":    c["value"],
            "domain":   c.get("domain", ".x.com"),
            "path":     c.get("path", "/"),
            "httpOnly": c.get("httpOnly", False),
            "secure":   c.get("secure", True),
            "sameSite": same_site,
        })
    return result


def _detect_lang(text):
    try:
        if not isinstance(text, str) or len(text.strip()) < 10:
            return "und"
        return _langdetect(text)
    except Exception:
        return "und"


def _tweet_to_row(t, handle):
    cdt = _parse_created_at(t.get("created_at", ""))
    tid = str(t.get("id", ""))
    screen_name = t.get("screen_name") or handle
    text = normalize_text(t.get("text", ""))
    return {
        "id":        tid,
        "date":      cdt.isoformat() if cdt else t.get("created_at", ""),
        "text":      text,
        "url":       "https://x.com/" + screen_name + "/status/" + tid,
        "favorites": t.get("favorite_count"),
        "retweets":  t.get("retweet_count"),
        "replies":   t.get("reply_count"),
        "language":  _detect_lang(text),
        "_cdt":      cdt,
    }


# ------------------------------------------------------------ Playwright fetch ----
async def _scroll_until_stable(page, collected, max_scrolls=5, wait_secs=1.5,
                               target=None):
    """Scroll until no new tweets appear, `target` is reached, or max_scrolls.

    `target` matters: fetch_latest used a hard max_scrolls=2, so a request for
    --max-per-account 200 still stopped at ~57 tweets. The scroll count now
    scales with what was actually asked for, and stops the moment we have it.
    """
    for _ in range(max_scrolls):
        if target and len(collected) >= target:
            break                      # already have what was asked for
        prev = len(collected)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(wait_secs)
        if len(collected) == prev:
            break  # no new tweets loaded — stop early


async def fetch_search_page(page, handle, chunk_since, chunk_until):
    """Fetch one monthly search chunk. Returns (rows, rate_limited).
    rate_limited is True if X returned an HTTP 429 or a GraphQL error payload
    (the tell-tale of throttling that otherwise stops the scroll silently)."""
    collected = []
    rate_limited = {"hit": False}

    async def on_response(response):
        try:
            if response.status == 429:
                rate_limited["hit"] = True
                return
        except Exception:
            pass
        _DIAG["responses"] += 1
        _u = response.url
        if "/graphql/" in _u or "/i/api/" in _u:
            _DIAG["graphql"] += 1
            _op = _u.split("/")[-1].split("?")[0]
            _DIAG["ops"][_op] = _DIAG["ops"].get(_op, 0) + 1
        if not _is_timeline_response(_u):
            return
        _DIAG["matched"] += 1
        try:
            body = await response.json()
            if isinstance(body, dict) and body.get("errors"):
                rate_limited["hit"] = True
            if DEBUG and not _DIAG["dumped"] and isinstance(body, dict):
                _dump_sample(body, "graphql_body")
            collected.extend(_extract_from_graphql(body))
        except Exception as e:                                # noqa: BLE001
            _DIAG["json_fail"] += 1
            _DIAG["last_err"] = f"{type(e).__name__}: {e}"
            if DEBUG:
                print(f"    ⚠️  response.json() failed on {_u[:70]}: {e}")

    page.on("response", on_response)
    try:
        q = ("from:" + handle
             + " since:" + chunk_since.strftime("%Y-%m-%d")
             + " until:" + chunk_until.strftime("%Y-%m-%d"))
        url = "https://x.com/search?q=" + quote(q) + "&f=live&src=typed_query"
        await page.goto(url, wait_until="load", timeout=30000)
        try:
            await page.wait_for_selector("[data-testid='tweet']", timeout=12000)
        except Exception:
            pass
        await _scroll_until_stable(page, collected)
    finally:
        page.remove_listener("response", on_response)

    rows = []
    for t in collected:
        r = _tweet_to_row(t, handle)
        cdt = r.pop("_cdt")
        if cdt and (cdt < chunk_since or cdt >= chunk_until):
            continue
        rows.append(r)
    return rows, rate_limited["hit"]


async def fetch_latest(page, handle, max_tweets):
    """Fetch latest tweets from a profile page (no date filter).

    Returns (rows, rate_limited) — SAME contract as fetch_search_page. It used
    to return rows only, which is why the default daily run had no throttle
    handling: X starts serving empty HTTP-200 pages at ~600-700 tweets per
    session and the loop simply printed '+0 new' for every remaining account.
    """
    collected = []
    rate_limited = {"hit": False}

    async def on_response(response):
        _DIAG["responses"] += 1
        _u = response.url
        try:
            if response.status == 429:
                rate_limited["hit"] = True
        except Exception:
            pass
        if "/graphql/" in _u or "/i/api/" in _u:
            _DIAG["graphql"] += 1
            _op = _u.split("/")[-1].split("?")[0]
            _DIAG["ops"][_op] = _DIAG["ops"].get(_op, 0) + 1
        if not _is_timeline_response(_u):
            return
        _DIAG["matched"] += 1
        try:
            body = await response.json()
            if isinstance(body, dict) and body.get("errors"):
                rate_limited["hit"] = True
            if DEBUG and not _DIAG["dumped"] and isinstance(body, dict):
                _dump_sample(body, "graphql_body")
            collected.extend(_extract_from_graphql(body))
        except Exception as e:                                # noqa: BLE001
            _DIAG["json_fail"] += 1
            _DIAG["last_err"] = f"{type(e).__name__}: {e}"
            if DEBUG:
                print(f"    ⚠️  response.json() failed on {_u[:70]}: {e}")

    page.on("response", on_response)
    try:
        await page.goto("https://x.com/" + handle, wait_until="load", timeout=30000)
        try:
            await page.wait_for_selector("[data-testid='tweet']", timeout=12000)
        except Exception:
            pass
        # Scroll budget scales with the request. X loads ~20 tweets per scroll,
        # so aim for max_tweets/15 plus headroom; the loop still exits early
        # the moment a scroll yields nothing new (dormant/short profiles).
        _budget = max(2, min(40, int(max_tweets / 15) + 2))
        await _scroll_until_stable(page, collected, max_scrolls=_budget,
                                   target=max_tweets)
    finally:
        page.remove_listener("response", on_response)

    rows = []
    for t in collected:
        r = _tweet_to_row(t, handle)
        r.pop("_cdt")
        rows.append(r)
    # SILENT THROTTLE: X answers 200 with an empty timeline once the session
    # cap (~600-700 tweets) is reached. No 429, no errors payload — an empty
    # result on a live account IS the signal.
    if not rows:
        rate_limited["hit"] = True
    return rows[:max_tweets], rate_limited["hit"]


# -------------------------------------------------------------------- main ----
async def fetch_profile_scroll(page, handle, since=None, max_tweets=500):
    """
    Scroll a profile page backwards in time, collecting tweets until we reach
    `since` or run out. Used as fallback when search gives 0 results (e.g.
    accounts whose language is not indexed well in X search).
    """
    collected = []

    async def on_response(response):
        url = response.url
        if not any(k in url for k in ("UserTweets", "UserTweetsAndReplies")):
            return
        try:
            body = await response.json()
            before = len(collected)
            collected.extend(_extract_from_graphql(body))
            after = len(collected)
            endpoint = url.split("?")[0].split("/")[-1]
            if after > before:
                print(f"      [profile] {handle}: +{after - before} from {endpoint}", flush=True)

        except Exception as e:
            print(f"      [profile-err] {handle}: {e}", flush=True)

    page.on("response", on_response)
    try:
        await page.goto("https://x.com/" + handle, wait_until="load", timeout=30000)
        try:
            await page.wait_for_selector("[data-testid='tweet']", timeout=12000)
        except Exception:
            pass

        for _ in range(40):  # up to 40 scrolls (~800-1000 tweets)
            prev = len(collected)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2.0)
            if len(collected) == prev:
                break  # no more tweets loading
            if since and collected:
                dates = [_parse_created_at(t.get("created_at", "")) for t in collected]
                dates = [d for d in dates if d]
                if dates and min(dates) < since:
                    break  # scrolled past the start date
    finally:
        page.remove_listener("response", on_response)

    rows = []
    for t in collected:
        r = _tweet_to_row(t, handle)
        cdt = r.pop("_cdt")
        if since and cdt and cdt < since:
            continue
        rows.append(r)
    return rows[:max_tweets]


async def retrieve(handles_filter=None, since=None, until=None,
                   max_per_account=MAX_PER_ACCOUNT_DEFAULT, dry_run=False,
                   refetch=False, gap_days=GAP_DAYS_DEFAULT):
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        sys.exit("Run: uv add playwright && uv run playwright install chromium")

    accounts = load_tracked_accounts(handles_filter)
    if not accounts:
        print("[done]  no matching tracked accounts found.")
        return 0

    existing_ids, coverage = load_existing_coverage()
    now = datetime.now(UTC)
    if until is None:
        until = now

    if since:
        chunks = list(monthly_chunks(since, until))
        date_info = ("  [" + str(since.date()) + " -> " + str(until.date())
                     + "]  (" + str(len(chunks)) + " monthly chunks)")
    else:
        chunks = []
        date_info = ""

    print("[csv]   " + OUTPUT_FILE)
    print("        existing rows: " + str(len(existing_ids)))
    print("[fetch] " + str(len(accounts)) + " account(s)" + date_info
          + ("   [DRY RUN]" if dry_run else ""))

    cookie_files = _cookie_files()
    cookie_idx = 0
    if cookie_files:
        print("[cookies] " + str(len(cookie_files)) + " set(s) for rotation: "
              + ", ".join(os.path.basename(c) for c in cookie_files))
    else:
        print("[warn]  no cookies found -- scraping without authentication")

    total_added, total_scanned = 0, 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox",
                  "--disable-blink-features=AutomationControlled",
                  "--disable-dev-shm-usage"],
        )
        # No locale= here: omitting it lets X return tweets in all languages
        context = await browser.new_context(
            user_agent=_UA,
            viewport={"width": 1280, "height": 900},
        )

        async def _use_cookies(idx):
            """Swap the browser context to cookie set `idx` (wraps around the pool)."""
            if not cookie_files:
                return None
            path = cookie_files[idx % len(cookie_files)]
            cks = _playwright_cookies(path)
            try:
                await context.clear_cookies()
            except Exception:
                pass
            if cks:
                await context.add_cookies(cks)
            return os.path.basename(path)

        await _use_cookies(cookie_idx)

        page = await context.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )

        for i, acc in enumerate(accounts):
            handle, name = acc["handle"], acc["name"]
            hkey = handle.lower()
            acc_added = 0

            if chunks:
                empty_streak = 0
                rotations = 0
                ci = 0
                while ci < len(chunks):
                    cs, ce = chunks[ci]
                    mkey = cs.strftime("%Y-%m")
                    have = coverage.get(hkey, {}).get(mkey, 0)
                    # RESUME: skip months already covered (unless --refetch), so a
                    # re-run marches forward instead of re-hitting X's session cap.
                    #
                    # GAP-AWARE (2026-08-03). A month count cannot see a HOLE:
                    # netanyahu had 44 rows in 2026-07 so the month was skipped
                    # as "covered" while 07-07..07-15 was entirely missing, and
                    # no amount of re-running would ever fill it. When the user
                    # passes --since EXPLICITLY they are asking for a specific
                    # range, so we check DAY coverage and fetch the month if it
                    # has a hole longer than --gap-days. Default runs (no
                    # --since) keep the cheap month-level skip.
                    if not refetch and have >= MIN_MONTH_COVERAGE:
                        _gap, _worst, _at = (
                            month_has_gap(handle, mkey, cs, ce, gap_days)
                            if since is not None else (False, 0, None))
                        if not _gap:
                            print(f"  [{handle}] {mkey}: skip (already have {have})")
                            ci += 1
                            continue
                        print(f"  [{handle}] {mkey}: have {have} but a "
                              f"{_worst}-day GAP from "
                              f"{_at:%Y-%m-%d} — refetching this month")
                    try:
                        chunk_rows, rate_limited = await fetch_search_page(page, handle, cs, ce)
                    except Exception as e:
                        print(f"  [{handle}] {mkey}: [warn] {str(e)[:247]}")
                        chunk_rows, rate_limited = [], False
                    # SAVE THIS CHUNK NOW -- interruption never loses progress
                    added = save_new(chunk_rows, handle, name, existing_ids, dry_run)
                    acc_added += added
                    total_scanned += len(chunk_rows)
                    if not rate_limited:
                        coverage.setdefault(hkey, {})[mkey] = have + added
                    print(f"  [{handle}] {mkey}: +{added} new "
                          f"({len(chunk_rows)} scanned, account total {acc_added})"
                          + ("  [DRY]" if dry_run else "  -> saved"))
                    # On throttle: pause 1 min, ROTATE to the next cookie set, and
                    # retry the SAME month. After the whole pool has been throttled on
                    # one month, take a single longer cooldown to let X's limits reset.
                    if rate_limited:
                        rotations += 1
                        cookie_idx += 1
                        nm = await _use_cookies(cookie_idx)
                        pause = (LONG_COOLDOWN if cookie_files
                                 and rotations % len(cookie_files) == 0
                                 else RATE_LIMIT_BACKOFF)
                        print(f"  [warn] X rate-limit at {mkey}; progress saved. "
                              f"pausing {pause}s"
                              + (f" and switching cookies -> {nm}" if nm else "")
                              + ", retrying same month.")
                        await asyncio.sleep(pause)
                        continue
                    rotations = 0
                    if len(chunk_rows) == 0:
                        empty_streak += 1
                        if empty_streak == EMPTY_STREAK_WARN:
                            print(f"  [warn] {empty_streak} empty months in a row for {handle} "
                                  f"-- likely X throttling (or no posts). Progress is saved; "
                                  f"re-run the SAME command to resume where it stopped.")
                    else:
                        empty_streak = 0
                    ci += 1
                    if ci < len(chunks):
                        await asyncio.sleep(random.uniform(*SLEEP_BETWEEN_CHUNKS))
                # Fallback profile-scroll only when the account has nothing at all
                if acc_added == 0 and not coverage.get(hkey):
                    try:
                        fb = await fetch_profile_scroll(page, handle, since=since)
                        acc_added += save_new(fb, handle, name, existing_ids, dry_run)
                        if acc_added:
                            print(f"  [{handle}] fallback profile-scroll: +{acc_added} new")
                    except Exception as e:
                        print(f"  [warn fallback] {handle}: {str(e)[:247]}")
            else:
                # DEFAULT DAILY RUN. This branch had NO throttle handling: once
                # X hit the ~600-700/session cap it served empty 200s and every
                # remaining account printed "+0 new" while the cookie pool sat
                # unused. Now it rotates through the pool like the --since path.
                raw_rows = []
                for _attempt in range(max(len(cookie_files), 1)):
                    try:
                        raw_rows, _rl = await fetch_latest(page, handle, max_per_account)
                    except Exception as e:
                        print("  [warn] " + handle + ": " + str(e))
                        raw_rows, _rl = [], False
                    if not _rl:
                        break
                    if not cookie_files:
                        print(f"  [throttled] {handle}: X returned nothing and "
                              f"there is no cookie pool to rotate to — stopping "
                              f"this account.")
                        break
                    cookie_idx += 1
                    nm = await _use_cookies(cookie_idx)
                    print(f"  [throttled] {handle}: empty/limited — pausing "
                          f"{RATE_LIMIT_BACKOFF}s and rotating to {nm} "
                          f"(try {_attempt + 2}/{len(cookie_files)})")
                    await asyncio.sleep(RATE_LIMIT_BACKOFF)
                acc_added = save_new(raw_rows, handle, name, existing_ids, dry_run)
                total_scanned += len(raw_rows)

            total_added += acc_added
            print(f"  [{handle.ljust(18)}] DONE: +{acc_added} new this run"
                  + ("  [DRY]" if dry_run else ""))
            if i < len(accounts) - 1:
                await asyncio.sleep(random.uniform(*SLEEP_BETWEEN_ACCOUNTS))

        await browser.close()

    print("[done]  scanned " + str(total_scanned) + " across "
          + str(len(accounts)) + " account(s); "
          + ("would add " if dry_run else "added ") + str(total_added)
          + " new this run (covered months skipped; re-run to fetch more).")
    return total_added


def main():
    ap = argparse.ArgumentParser(
        description="Retrieve tweets for tracked geopolitical X accounts via Playwright.")
    ap.add_argument("--handles",
                    help="Comma-separated handles (with or without @). "
                         "Default: all accounts in influence_accounts.json.")
    ap.add_argument("--since",
                    help="YYYYMMDD or YYYY-MM-DD: fetch tweets on or after this date.")
    ap.add_argument("--until",
                    help="YYYYMMDD or YYYY-MM-DD: fetch tweets up to this date. Default: today.")
    ap.add_argument("--max-per-account", type=int, default=MAX_PER_ACCOUNT_DEFAULT,
                    help="Max tweets per account when no --since is given (default 40).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch and report without writing to the CSV.")
    ap.add_argument("--refetch", action="store_true",
                    help="Re-scrape months already present in x_tweets.csv "
                         "(default: skip covered months so re-runs resume forward).")
    ap.add_argument("--gap-days", type=int, default=GAP_DAYS_DEFAULT,
                    help=f"With --since: refetch a month whose DAY coverage has "
                         f"a hole longer than this many days, even if the month "
                         f"already has rows (default {GAP_DAYS_DEFAULT}). This is "
                         f"what fills a mid-month gap; month counts alone cannot "
                         f"see one.")
    ap.add_argument("--debug", action="store_true",
                    help="Diagnose a '0 tweets' run: count every response, log "
                         "the GraphQL operation names X actually returned, dump "
                         "the first payload to _x_debug_*.json, and print why "
                         "extraction produced nothing.")
    args = ap.parse_args()
    if args.debug:
        globals()["DEBUG"] = True
        os.environ["X_DEBUG"] = "1"

    handles_filter = [h.strip() for h in args.handles.split(",")] if args.handles else None
    since = parse_date(args.since) if args.since else None
    until = parse_date(args.until) if args.until else None

    asyncio.run(retrieve(
        handles_filter=handles_filter,
        since=since,
        until=until,
        max_per_account=args.max_per_account,
        dry_run=args.dry_run,
        refetch=args.refetch,
        gap_days=args.gap_days,
    ))
    # Always print the accounting when nothing came back — a silent "0 tweets
    # updated" is what made this impossible to diagnose in the first place.
    if args.debug or _DIAG["entries"] == 0:
        print_diag()


if __name__ == "__main__":
    main()