"""
x_tweets_retriever.py
======================
Retrieves tweets for the geopolitical accounts in DP/geopolitical_entities.json
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

OUTPUT SCHEMA (mirrors trump_truths.csv + two extra columns)
-------------------------------------------------------------
  id, date, text, url, favorites, retweets, replies, account, account_name
"""
import os
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
ENTITIES_FILE = os.path.join(_HERE, "geopolitical_entities.json")
COOKIES_FILE  = os.path.join(_HERE, ".x_cookies.json")

COLUMNS = ["id", "date", "text", "url", "favorites", "retweets", "replies",
           "account", "account_name", "language"]

MAX_PER_ACCOUNT_DEFAULT = 40
SLEEP_BETWEEN_ACCOUNTS  = (4.0, 8.0)
SLEEP_BETWEEN_CHUNKS    = (2.0, 4.0)

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


def load_tracked_accounts(handles_filter=None):
    with open(ENTITIES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    accounts, seen = [], set()
    sections = [data.get("entities", []),
                data.get("institutions", {}).get("entries", [])]
    for entries in sections:
        for entry in entries:
            handle = entry.get("twitter_handle")
            if not handle:
                continue
            handle = handle.lstrip("@")
            if handle.lower() in seen:
                continue
            seen.add(handle.lower())
            accounts.append({"handle": handle, "name": entry.get("name", handle)})

    if handles_filter:
        wanted = {h.lstrip("@").lower() for h in handles_filter}
        accounts = [a for a in accounts if a["handle"].lower() in wanted]

    return accounts


def load_existing():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    return set(df["id"].astype(str)) if "id" in df.columns else set()


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
def _playwright_cookies():
    if not os.path.exists(COOKIES_FILE):
        return []
    with open(COOKIES_FILE, encoding="utf-8") as f:
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
    """Fetch one monthly search chunk; returns list of raw tweet dicts."""
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
    return rows


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
                   max_per_account=MAX_PER_ACCOUNT_DEFAULT, dry_run=False):
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        sys.exit("Run: uv add playwright && uv run playwright install chromium")

    accounts = load_tracked_accounts(handles_filter)
    if not accounts:
        print("[done]  no matching tracked accounts found.")
        return 0

    existing_ids = load_existing()
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

    pw_cookies = _playwright_cookies()
    if not pw_cookies:
        print("[warn]  no cookies found -- scraping without authentication")

    total_added, total_scanned, total_dup = 0, 0, 0

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
        if pw_cookies:
            await context.add_cookies(pw_cookies)

        page = await context.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )

        for i, acc in enumerate(accounts):
            handle, name = acc["handle"], acc["name"]

            if chunks:
                all_rows = []
                for ci, (cs, ce) in enumerate(chunks):
                    try:
                        chunk_rows = await fetch_search_page(page, handle, cs, ce)
                    except Exception as e:
                        print("    [warn] chunk " + cs.strftime("%Y-%m") + ": " + str(e))
                        chunk_rows = []
                    all_rows.extend(chunk_rows)
                    if ci < len(chunks) - 1:
                        await asyncio.sleep(random.uniform(*SLEEP_BETWEEN_CHUNKS))
                # Fallback: if search gave nothing, try scrolling the profile page
                if not all_rows:
                    try:
                        all_rows = await fetch_profile_scroll(page, handle, since=since)
                        if all_rows:
                            print("    [fallback profile-scroll] " + handle + ": " + str(len(all_rows)) + " tweets")
                    except Exception as e:
                        print("    [warn fallback] " + handle + ": " + str(e))
                raw_rows = all_rows
            else:
                try:
                    raw_rows = await fetch_latest(page, handle, max_per_account)
                except Exception as e:
                    print("  [warn] " + handle + ": " + str(e))
                    continue

            rows, added, dup = [], 0, 0
            seen_in_batch = set()
            for r in raw_rows:
                tid = str(r["id"])
                if tid in existing_ids or tid in seen_in_batch:
                    dup += 1
                    continue
                existing_ids.add(tid)
                seen_in_batch.add(tid)
                r["account"]      = handle
                r["account_name"] = name
                rows.append({c: r.get(c) for c in COLUMNS})
                added += 1

            if not dry_run:
                save_rows(rows)

            scanned = len(raw_rows)
            total_added   += added
            total_scanned += scanned
            total_dup     += dup
            print("  [" + handle.ljust(18) + "] scanned " + str(scanned).rjust(4)
                  + ", skipped " + str(dup).rjust(4) + " dup, "
                  + ("would add " if dry_run else "added ") + str(added).rjust(4) + " new")

            if i < len(accounts) - 1:
                await asyncio.sleep(random.uniform(*SLEEP_BETWEEN_ACCOUNTS))

        await browser.close()

    print("[done]  scanned " + str(total_scanned) + " tweets across "
          + str(len(accounts)) + " account(s), skipped " + str(total_dup)
          + " already in file, "
          + ("would add " if dry_run else "added ") + str(total_added) + " new.")
    return total_added


def main():
    ap = argparse.ArgumentParser(
        description="Retrieve tweets for tracked geopolitical X accounts via Playwright.")
    ap.add_argument("--handles",
                    help="Comma-separated handles (with or without @). "
                         "Default: all accounts in geopolitical_entities.json.")
    ap.add_argument("--since",
                    help="YYYYMMDD or YYYY-MM-DD: fetch tweets on or after this date.")
    ap.add_argument("--until",
                    help="YYYYMMDD or YYYY-MM-DD: fetch tweets up to this date. Default: today.")
    ap.add_argument("--max-per-account", type=int, default=MAX_PER_ACCOUNT_DEFAULT,
                    help="Max tweets per account when no --since is given (default 40).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch and report without writing to the CSV.")
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
    ))


if __name__ == "__main__":
    main()
