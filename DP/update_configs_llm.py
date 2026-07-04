"""
update_configs_llm.py  —  DAILY LLM CONFIG CURATOR
====================================================
Reads the last N days of posts from the unified_feed table and asks an LLM
(Gemini by default — integrates with Google Search grounding; Claude optional)
to propose STRICTLY ADDITIVE deltas to the three dynamic config files:

    scorer_config.json       new flag keywords / new flags
    influence_accounts.json  new influence accounts (leaders, institutions,
                             BILLIONAIRES/magnates) with active_from/active_to
                             windows, mention-count increments
    events.json              new events, status transitions on existing events

WHY DELTAS, NOT REWRITES
------------------------
The LLM never sees or produces a whole config file — it can only emit a small
JSON of allowed operations, each validated against hard rules before being
applied. A hallucinating model can therefore at worst add a few bad keywords
(visible in the changelog, trivially revertible from backups) — it can NEVER
truncate, restructure, or delete existing config. This is what makes daily
unattended runs safe.

HARD RULES (validator, non-negotiable)
--------------------------------------
  * additive only: nothing is ever deleted or renamed
  * caps/day: ≤3 new flags, ≤10 keywords per flag, ≤5 new accounts, ≤5 new
    events, ≤10 event updates
  * flags match ^flag_[a-z0-9_]+$; keywords are lowercase 2..40 chars
  * event status must be one of event_manager.STATUS_WEIGHTS; updates may only
    touch status / notes / end_date of EXISTING ids
  * new accounts need name + twitter_handle + country (alpha-2) + category
    (leader|institution|billionaire|candidate) + active_from (ISO date;
    active_to null = open) — the active window is what makes back-simulation
    work (Biden era, Trump term 1, Musk...)
  * every applied change: timestamped backup in DP/config_backups/ +
    JSONL changelog entry; files written atomically (tmp + os.replace)

DOWNSTREAM (why this is cheap daily)
------------------------------------
New flags flow into posts_scored via DuckDB schema evolution (old posts
default 0/false — no re-score). Models keep their frozen feature list until
the next retrain (predict prints a drift notice). New accounts start being
retrieved/weighted on the next daily run. Nothing needs a full rebuild.

USAGE
-----
  uv run python update_configs_llm.py                 # last 2 days, Gemini, APPLY
  uv run python update_configs_llm.py --dry-run       # show proposal only
  uv run python update_configs_llm.py --days 7 --provider claude

  Env: GEMINI_API_KEY (default provider) or ANTHROPIC_API_KEY (--provider claude)
       GEMINI_MODEL / CLAUDE_MODEL override the model names.

DAILY PIPELINE ORDER: retrieve → sync_unified_feed → THIS → signal_scorer
(so today's posts are scored with today's config).
"""
import os
import re
import sys
import json
import argparse
import datetime

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import db
from event_manager import STATUS_WEIGHTS

SCORER_FILE   = os.path.join(_HERE, "scorer_config.json")
ACCOUNTS_FILE = os.path.join(_HERE, "influence_accounts.json")
EVENTS_FILE   = os.path.join(_HERE, "events.json")
BACKUP_DIR    = os.path.join(_HERE, "config_backups")
CHANGELOG     = os.path.join(BACKUP_DIR, "changelog.jsonl")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")

# ---- hard caps per daily run --------------------------------------------------
# Deliberately TIGHT (Peter: "no rush edits"): a config that changes slowly is
# a config the model can trust. Most days the correct output is NO changes;
# the caps make even a chatty LLM unable to churn the files.
MAX_NEW_FLAGS      = 1
MAX_KW_PER_FLAG    = 5
MAX_NEW_ACCOUNTS   = 2
MAX_NEW_EVENTS     = 2
MAX_EVENT_UPDATES  = 5
MAX_POSTS_TO_LLM   = 150

_FLAG_RE    = re.compile(r"^flag_[a-z0-9_]{2,40}$")
_ID_RE      = re.compile(r"^[a-z0-9_]{3,50}$")
_HANDLE_RE  = re.compile(r"^@?[A-Za-z0-9_]{2,30}$")
_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
_DATE_RE    = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CATEGORIES  = {"leader", "institution", "billionaire", "candidate"}
VALID_STATUS = set(STATUS_WEIGHTS.keys())


# ============================================================ file helpers ----
def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    json.load(open(tmp, encoding="utf-8"))          # re-parse guard before replace
    os.replace(tmp, path)


def _backup(path, ts):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    base = os.path.basename(path).replace(".json", "")
    dst = os.path.join(BACKUP_DIR, f"{base}.{ts}.json")
    with open(path, encoding="utf-8") as src, open(dst, "w", encoding="utf-8") as out:
        out.write(src.read())
    return dst


def _log_change(entry):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    with open(CHANGELOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ========================================================== input assembly ----
def load_recent_posts(days):
    feed = db.read_table("unified_feed")
    if feed is None or feed.empty:
        sys.exit("❌ unified_feed is empty — run sync_unified_feed.py first.")
    feed["date"] = pd.to_datetime(feed["date"], format="mixed", utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    recent = feed[feed["date"] >= cutoff].sort_values("date")
    if len(recent) > MAX_POSTS_TO_LLM:
        recent = recent.tail(MAX_POSTS_TO_LLM)
    return recent


def current_state_summary():
    sc = _load(SCORER_FILE)
    acc = _load(ACCOUNTS_FILE)
    ev = _load(EVENTS_FILE)
    flags = {k: len(v) for k, v in sc["policy_flags"].items()}
    handles = sorted({(e.get("account") or e.get("twitter_handle") or "").lstrip("@")
                      for e in (acc.get("primary_accounts", []) + acc.get("entities", [])
                                + acc.get("institutions", {}).get("entries", []))} - {""})
    events = [{"id": e["id"], "status": e["status"], "name": e["name"]}
              for e in ev.get("events", [])]
    return sc, acc, ev, flags, handles, events


def build_prompt(posts, flags, handles, events):
    digest = "\n".join(
        f"- [{r['date']:%Y-%m-%d %H:%M}] @{r['account']}: {str(r['text'])[:280]}"
        for _, r in posts.iterrows())
    return f"""You are the daily config curator for a market-impact NLP pipeline that
scores posts by world leaders, institutions and billionaires and predicts
1-hour market moves. Based ONLY on the posts below, propose MINIMAL additive
updates.

STRICT CONSERVATISM — THE DEFAULT ANSWER IS ALL-EMPTY LISTS:
Return empty lists unless the evidence is strong and repeated. Never propose
a change because it is merely plausible or "nice to have". Concretely:
* add_flag_keywords / new_flags: only when >=3 SEPARATE posts use
  market-moving vocabulary that NO existing flag catches. A synonym or
  variant spelling of an existing keyword is NOT a new keyword.
* new_accounts: allowed — but only for a genuinely INFLUENTIAL account:
  repeatedly referenced or quoted across these posts AND able to move
  markets by itself (head of state, central bank chief, major billionaire).
  One mention is not influence.
* new_events: only developments a trader would reprice TODAY. Commentary on
  an existing event is update_events (status change), not a new event.
* If unsure, leave it out. On most days the correct proposal is empty —
  that is a SUCCESS, not a failure.

CURRENT POLICY FLAGS (name: keyword count): {json.dumps(flags)}
CURRENT TRACKED HANDLES: {json.dumps(handles)}
CURRENT EVENTS (id/status): {json.dumps(events)}

POSTS (last window):
{digest}

Propose ONLY:
1. add_flag_keywords — new lowercase keywords for EXISTING flags, only if the
   posts show market-relevant vocabulary the flag misses.
2. new_flags — a genuinely NEW market-impact theme visible in the posts
   (e.g. a new conflict, commodity shock, regulation wave). Rare.
3. new_accounts — a NEW account (world leader, institution, BILLIONAIRE/
   magnate, election candidate) that the posts show moving markets. Include
   its influence window: active_from = when its market influence began
   (ISO date), active_to = null if ongoing.
4. increment_mentions — handle: +count when tracked entities were mentioned
   by the rank-0 account in these posts.
5. new_events / update_events — market-relevant world events beginning,
   escalating, pausing or ending. update_events may only change status
   ({sorted(VALID_STATUS)}), notes, end_date of existing ids.
6. close_account_window — when a tracked leader LEAVES power (election loss,
   resignation, death, term end, moved platforms): set their active_to date.
   This never deletes the account or its history — it only ends the window.

ELECTIONS & LEADERSHIP TRANSITIONS (important — elections move markets):
When posts show an upcoming election, a leadership handover, or an old
leader endorsing a successor (in ANY country): (a) propose a new_event for
the election/transition itself (e.g. us_election_2028) — election runs are
tradeable; (b) propose new_accounts for serious successors/candidates
(category 'candidate', active_from = when their influence starts, e.g.
campaign launch or inauguration); (c) once the outgoing leader actually
leaves office, propose close_account_window for them. Successor examples to
watch for: JD Vance or Trump family members on TruthSocial, new heads of
state on X after any national election.

Respond with JSON exactly matching the schema. Do NOT propose removals or
renames — they are forbidden and will be rejected."""


# ============================================================= LLM backends ----
def propose_gemini(prompt):
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
    from typing import Optional

    class NewAccount(BaseModel):
        name: str
        twitter_handle: str
        country: str = Field(description="ISO alpha-2, e.g. US")
        category: str = Field(description="leader|institution|billionaire|candidate")
        active_from: str = Field(description="ISO date the influence window opens")
        active_to: Optional[str] = Field(default=None, description="ISO date or null if ongoing")
        reason: str

    class NewEvent(BaseModel):
        id: str = Field(description="snake_case id")
        name: str
        category: str
        start_date: str
        status: str
        priority: str
        description: str
        affected_accounts: list[str]
        tags: list[str]

    class EventUpdate(BaseModel):
        id: str
        status: Optional[str] = None
        notes: Optional[str] = None
        end_date: Optional[str] = None

    class FlagKeywords(BaseModel):
        flag: str
        keywords: list[str]

    class MentionInc(BaseModel):
        handle: str
        increment: int

    class WindowClose(BaseModel):
        handle: str
        active_to: str = Field(description="ISO date the influence window ends")
        reason: str

    class ConfigDeltas(BaseModel):
        add_flag_keywords: list[FlagKeywords]
        new_flags: list[FlagKeywords]
        new_accounts: list[NewAccount]
        increment_mentions: list[MentionInc]
        new_events: list[NewEvent]
        update_events: list[EventUpdate]
        close_account_window: list[WindowClose]

    client = genai.Client()
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ConfigDeltas))
    return json.loads(resp.text)


def propose_claude(prompt):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": prompt + "\n\nRespond with ONLY the JSON object, keys: add_flag_keywords (list of {flag, keywords}), new_flags (same), new_accounts, increment_mentions (list of {handle, increment}), new_events, update_events, close_account_window (list of {handle, active_to, reason})."}])
    text = msg.content[0].text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("Claude response contained no JSON object")
    return json.loads(m.group(0))


# ============================================================== validation ----
def validate(deltas, sc, acc, ev):
    """Enforce the hard rules. Returns (clean_deltas, rejects[])."""
    rejects, clean = [], {"add_flag_keywords": {}, "new_flags": {},
                          "new_accounts": [], "increment_mentions": {},
                          "new_events": [], "update_events": [],
                          "close_account_window": []}
    existing_flags = set(sc["policy_flags"].keys())
    existing_ids   = {e["id"] for e in ev.get("events", [])}
    existing_handles = {(e.get("account") or e.get("twitter_handle") or "").lstrip("@").lower()
                        for e in (acc.get("primary_accounts", []) + acc.get("entities", [])
                                  + acc.get("institutions", {}).get("entries", []))}

    def norm_kws(kws):
        out = []
        for k in kws[:MAX_KW_PER_FLAG]:
            k = str(k).strip().lower()
            if 2 <= len(k) <= 40 and k not in out:
                out.append(k)
        return out

    for item in (deltas.get("add_flag_keywords") or []):
        f, kws = item.get("flag", ""), norm_kws(item.get("keywords") or [])
        if f in existing_flags and kws:
            fresh = [k for k in kws if k not in sc["policy_flags"][f]]
            if fresh:
                clean["add_flag_keywords"][f] = fresh
        else:
            rejects.append(f"add_flag_keywords: unknown flag or empty kws: {f}")

    for item in (deltas.get("new_flags") or [])[:MAX_NEW_FLAGS]:
        f, kws = item.get("flag", ""), norm_kws(item.get("keywords") or [])
        if _FLAG_RE.match(f) and f not in existing_flags and len(kws) >= 2:
            clean["new_flags"][f] = kws
        else:
            rejects.append(f"new_flags: invalid/duplicate: {f}")

    for a in (deltas.get("new_accounts") or [])[:MAX_NEW_ACCOUNTS]:
        h = str(a.get("twitter_handle", "")).lstrip("@")
        ok = (a.get("name") and _HANDLE_RE.match(h or "-")
              and h.lower() not in existing_handles
              and _COUNTRY_RE.match(str(a.get("country", "")))
              and str(a.get("category", "")) in CATEGORIES
              and _DATE_RE.match(str(a.get("active_from", ""))))
        at = a.get("active_to")
        if ok and (at is None or _DATE_RE.match(str(at))):
            clean["new_accounts"].append(a)
        else:
            rejects.append(f"new_accounts: invalid/duplicate: {a.get('name')}")

    for m in (deltas.get("increment_mentions") or []):
        h, inc = str(m.get("handle", "")).lstrip("@"), int(m.get("increment", 0))
        if h.lower() in existing_handles and 1 <= inc <= 20:
            clean["increment_mentions"][h] = inc
        else:
            rejects.append(f"increment_mentions: unknown handle/bad count: {h}")

    for e in (deltas.get("new_events") or [])[:MAX_NEW_EVENTS]:
        ok = (_ID_RE.match(str(e.get("id", ""))) and e["id"] not in existing_ids
              and e.get("name") and _DATE_RE.match(str(e.get("start_date", "")))
              and str(e.get("status")) in VALID_STATUS
              and str(e.get("priority")) in ("high", "medium", "low"))
        if ok:
            clean["new_events"].append(e)
        else:
            rejects.append(f"new_events: invalid/duplicate: {e.get('id')}")

    for u in (deltas.get("update_events") or [])[:MAX_EVENT_UPDATES]:
        ok = (u.get("id") in existing_ids
              and (u.get("status") is None or u["status"] in VALID_STATUS)
              and (u.get("end_date") is None or _DATE_RE.match(str(u["end_date"]))))
        if ok and (u.get("status") or u.get("notes") or u.get("end_date")):
            clean["update_events"].append(u)
        else:
            rejects.append(f"update_events: invalid/unknown id: {u.get('id')}")

    # close_account_window: the ONE allowed mutation — end an existing
    # account's influence window (election loss / resignation / death /
    # platform move). Never deletes; never reopens; rank-0 primary accounts
    # are protected (a presidency handover is a human decision).
    rank0 = {str(a.get("account", "")).lower()
             for a in acc.get("primary_accounts", []) if a.get("rank", 99) == 0}
    for w in (deltas.get("close_account_window") or [])[:MAX_NEW_ACCOUNTS]:
        h = str(w.get("handle", "")).lstrip("@")
        if h.lower() in rank0:
            rejects.append(f"close_account_window: {h} is a rank-0 primary — manual only")
        elif (h.lower() in existing_handles and _DATE_RE.match(str(w.get("active_to", "")))
              and w.get("reason")):
            clean["close_account_window"].append({"handle": h,
                                                  "active_to": w["active_to"],
                                                  "reason": str(w["reason"])})
        else:
            rejects.append(f"close_account_window: unknown handle/bad date: {h}")

    return clean, rejects


# ================================================================== apply ----
def apply_deltas(clean, sc, acc, ev, ts):
    changed = []

    if clean["add_flag_keywords"] or clean["new_flags"]:
        _backup(SCORER_FILE, ts)
        for f, kws in clean["add_flag_keywords"].items():
            sc["policy_flags"][f].extend(kws)
            changed.append(f"scorer: +{len(kws)} kw -> {f}")
        for f, kws in clean["new_flags"].items():
            sc["policy_flags"][f] = kws
            changed.append(f"scorer: NEW flag {f} ({len(kws)} kw)")
        try:            # version bump: "1.4" -> "1.5"
            maj, minor = str(sc.get("version", "1.0")).split(".")[:2]
            sc["version"] = f"{maj}.{int(minor) + 1}"
        except Exception:
            pass
        _atomic_write(SCORER_FILE, sc)

    if clean["new_accounts"] or clean["increment_mentions"] or clean["close_account_window"]:
        _backup(ACCOUNTS_FILE, ts)
        max_rank = max([int(e.get("rank", 0)) for e in acc.get("entities", [])] or [0])
        for a in clean["new_accounts"]:
            max_rank += 1
            acc.setdefault("entities", []).append({
                "rank": max_rank, "name": a["name"],
                "country": a["country"], "role": a.get("category", "leader"),
                "category": a["category"], "mention_count": 1,
                "platform": "x_twitter",
                "twitter_handle": "@" + str(a["twitter_handle"]).lstrip("@"),
                "active_from": a["active_from"], "active_to": a.get("active_to"),
                "expiration_date": "N/A",
                "note": f"[LLM curator {ts}] {a.get('reason', '')}"})
            changed.append(f"accounts: NEW {a['category']} {a['name']} "
                           f"(window {a['active_from']} -> {a.get('active_to') or 'open'})")
        for h, inc in clean["increment_mentions"].items():
            for e in acc.get("entities", []):
                if str(e.get("twitter_handle", "")).lstrip("@").lower() == h.lower():
                    e["mention_count"] = int(e.get("mention_count", 0)) + inc
                    changed.append(f"accounts: {h} mention_count +{inc}")
        for w in clean["close_account_window"]:
            hl = w["handle"].lower()
            for section in (acc.get("entities", []),
                            acc.get("institutions", {}).get("entries", [])):
                for e in section:
                    eh = str(e.get("twitter_handle") or e.get("account") or "").lstrip("@").lower()
                    if eh == hl:
                        e["active_to"] = w["active_to"]
                        e["note"] = (str(e.get("note", "")) +
                                     f" [window closed {w['active_to']} by LLM curator {ts}: {w['reason']}]").strip()
                        changed.append(f"accounts: {w['handle']} window closed at "
                                       f"{w['active_to']} ({w['reason'][:60]})")
        _atomic_write(ACCOUNTS_FILE, acc)

    if clean["new_events"] or clean["update_events"]:
        _backup(EVENTS_FILE, ts)
        for e in clean["new_events"]:
            e.setdefault("end_date", "N/A")
            e.setdefault("affected_assets", [])
            e.setdefault("notes", f"[LLM curator {ts}]")
            ev["events"].append(e)
            changed.append(f"events: NEW {e['id']} ({e['status']})")
        for u in clean["update_events"]:
            for e in ev["events"]:
                if e["id"] == u["id"]:
                    old = e["status"]
                    if u.get("status"):
                        e["status"] = u["status"]
                    if u.get("end_date"):
                        e["end_date"] = u["end_date"]
                    if u.get("notes"):
                        e["notes"] = f"{u['notes']} [LLM curator {ts}, was {old}]"
                    changed.append(f"events: {u['id']} {old} -> {e['status']}")
        _atomic_write(EVENTS_FILE, ev)

    return changed


# ==================================================================== main ----
def main():
    ap = argparse.ArgumentParser(description="Daily LLM curator for the three dynamic config JSONs.")
    ap.add_argument("--days", type=int, default=2, help="posts lookback window (default 2)")
    ap.add_argument("--provider", choices=["gemini", "claude"], default="gemini",
                    help="LLM backend (default gemini — GEMINI_API_KEY)")
    ap.add_argument("--dry-run", action="store_true", help="show validated proposal, change nothing")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 66)
    print(f"  DAILY CONFIG CURATOR  —  {args.provider} | last {args.days} day(s)"
          f"{' | DRY RUN' if args.dry_run else ''}")
    print("=" * 66)

    posts = load_recent_posts(args.days)
    print(f"📂 {len(posts)} recent posts from unified_feed")
    if posts.empty:
        print("✅ No recent posts — nothing to curate.")
        return

    sc, acc, ev, flags, handles, events = current_state_summary()
    prompt = build_prompt(posts, flags, handles, events)

    print(f"🤖 Asking {args.provider} for config deltas...")
    deltas = propose_gemini(prompt) if args.provider == "gemini" else propose_claude(prompt)

    clean, rejects = validate(deltas, sc, acc, ev)
    n_ops = (len(clean['add_flag_keywords']) + len(clean['new_flags'])
             + len(clean['new_accounts']) + len(clean['increment_mentions'])
             + len(clean['new_events']) + len(clean['update_events'])
             + len(clean['close_account_window']))
    print(f"\n🔎 Proposal: {n_ops} valid op(s), {len(rejects)} rejected")
    for r in rejects:
        print(f"  🚫 {r}")
    print(json.dumps(clean, indent=2, ensure_ascii=False))

    if args.dry_run or n_ops == 0:
        print("\n✅ Dry run / nothing to apply." if args.dry_run else "\n✅ Configs already current.")
        return

    changed = apply_deltas(clean, sc, acc, ev, ts)
    _log_change({"ts": ts, "provider": args.provider, "days": args.days,
                 "applied": changed, "rejected": rejects})
    print(f"\n💾 Applied {len(changed)} change(s); backups in {BACKUP_DIR}/")
    for c in changed:
        print(f"  ✅ {c}")
    print(f"📒 Changelog: {CHANGELOG}")


if __name__ == "__main__":
    main()
