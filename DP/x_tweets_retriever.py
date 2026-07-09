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
    accounts, seen, skipped = [], set(), []
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
            if handle.lower() in seen:
                continue
            seen.add(handle.lower())
            if apply_expiry and _is_expired(entry):
                skipped.append(handle)
                continue
            accounts.append({"handle": handle, "name": entry.get("name", handle)})
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
    return ids, cov


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
_TWEET_JMES = jmespath.compile("""{
    id:              legacy.id_str,
    created_at:      legacy.created_at,
    text:            legacy.full_text,
    favorite_count:  legacy.favorite_count,
    retweet_count:   legacy.retweet_count,
    reply_count:     legacy.reply_count,
    screen_name:     core.user_results.result.legacy.screen_name
}""")

_CREATED_AT_FMT = "%a %b %d %H:%M:%S %z %Y"


def _parse_created_at(s):
    try:
        return datetime.strptime(s, _CREATED_AT_FMT)
    except Exception:
        return None


def _unwrap_tweet_result(result):
    """Handle TweetWithVisibilityResults wrapper (extra nesting X added ~2023)."""
    if result.get("__typename") == "TweetWithVisibilityResults":
        result = result.get("tweet", {})
    return result


def _extract_from_graphql(body):
    tweets = []
    instructions = (
        body.get("data", {})
            .get("search_by_raw_query", {})
            .get("search_timeline", {})
            .get("timeline", {})
            .get("instructions", [])
    )
    if not instructions:
        instructions = (
            body.get("data", {})
                .get("user", {})
                .get("result", {})
                .get("timeline_v2", {})
                .get("timeline", {})
                .get("instructions", [])
        )
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
                result = _unwrap_tweet_result(
                    item.get("tweet_results", {}).get("result", {})
                )
                parsed = _TWEET_JMES.search(result)
                if parsed and parsed.get("id"):
                    tweets.append(parsed)
    return tweets


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
async def _scroll_until_stable(page, collected, max_scrolls=5, wait_secs=1.5):
    """Scroll page until no new tweets appear or max_scrolls reached."""
    for _ in range(max_scrolls):
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
        if "UserTweets" not in response.url and "SearchTimeline" not in response.url:
            return
        try:
            body = await response.json()
            if isinstance(body, dict) and body.get("errors"):
                rate_limited["hit"] = True
            collected.extend(_extract_from_graphql(body))
        except Exception:
            pass

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
    """Fetch latest tweets from a profile page (no date filter)."""
    collected = []

    async def on_response(response):
        if "UserTweets" not in response.url and "SearchTimeline" not in response.url:
            return
        try:
            body = await response.json()
            collected.extend(_extract_from_graphql(body))
        except Exception:
            pass

    page.on("response", on_response)
    try:
        await page.goto("https://x.com/" + handle, wait_until="load", timeout=30000)
        try:
            await page.wait_for_selector("[data-testid='tweet']", timeout=12000)
        except Exception:
            pass
        await _scroll_until_stable(page, collected, max_scrolls=2)
    finally:
        page.remove_listener("response", on_response)

    rows = []
    for t in collected:
        r = _tweet_to_row(t, handle)
        r.pop("_cdt")
        rows.append(r)
    return rows[:max_tweets]


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
                   refetch=False):
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
                    if not refetch and have >= MIN_MONTH_COVERAGE:
                        print(f"  [{handle}] {mkey}: skip (already have {have})")
                        ci += 1
                        continue
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
                try:
                    raw_rows = await fetch_latest(page, handle, max_per_account)
                except Exception as e:
                    print("  [warn] " + handle + ": " + str(e))
                    continue
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
    args = ap.parse_args()

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
    ))


if __name__ == "__main__":
    main()