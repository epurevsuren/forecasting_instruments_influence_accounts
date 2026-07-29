"""
gemini_impact_annotator.py — OPTIONAL Gemini annotation layer for DP
=====================================================================
Independent, second-opinion annotation of posts, written to DuckDB table
`gemini_annotations` and consumed ONLY when build_final_training_set.py is
run with --use-gemini. Nothing in the daily pipeline changes unless you
ask for it: no flag (or no table) => byte-identical behaviour to today.

Evolves DP/tweet_truth_impact_analyzer.py (same genai client, pydantic
response schema and resume-by-id logic) and fixes what would not scale:
DuckDB-native instead of CSV, real batching instead of sleep(12.5) per row
(that is ~10 days for 70k), cost estimate + hard cap before any spend.

-----------------------------------------------------------------------
THE LEAKAGE FIREWALL — read before adding any new field
-----------------------------------------------------------------------
A GROUNDED call (Google Search enabled) can see what happened AFTER the
post. Anything it returns is therefore usable ONLY on the LABEL side —
curation, quality flags, sample weighting. It can NEVER be a model
feature: at live prediction time there is no future to search, and in a
backtest it is textbook lookahead leakage (this is precisely how public
"90% accuracy" tweet projects get their numbers).

So the two modes write DIFFERENT column prefixes and must never be mixed:

  --mode grounded   search ON   -> gl_*  LABEL-SIDE ONLY. Never a feature.
  --mode text       search OFF  -> gf_*  Feature-safe (text only, no future).

One grounded call contaminates ALL of its own outputs, which is why the
feature-safe fields need their own ungrounded call rather than riding
along on the grounded one.

-----------------------------------------------------------------------
COST (verified 2026-07-28) — grounding is billed PER QUERY, not by token
-----------------------------------------------------------------------
  Gemini 3.x family: 5,000 free grounded queries/month, then ~$14/1,000.
  HIGH_SIGNAL is ~2,800 posts => fits inside the free monthly allowance.
  NEVER ground all 70k posts: that is $980-$2,450.
  Text mode on Flash-Lite + batch is ~$5 for the full 70k.
This script prints an estimate and refuses to exceed --cost-cap.

USAGE
-----
  # 1. pilot first — 500 grounded HIGH_SIGNAL posts, see the sample output
  python gemini_impact_annotator.py --mode grounded --limit 500

  # 2. full HIGH_SIGNAL set (still inside the free grounded allowance)
  python gemini_impact_annotator.py --mode grounded

  # 3. optional: feature-safe text pass (can cover every post)
  python gemini_impact_annotator.py --mode text --table training_set_FINAL

  # then, and only then:
  python build_final_training_set.py --full --use-gemini

Env: GEMINI_API_KEY (required), GEMINI_MODEL / GEMINI_TEXT_MODEL override.
"""
import os
import re
import sys
import json
import time
import argparse
import datetime

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import db  # DuckDB helper -> ../database.db

try:
    from dotenv import load_dotenv
    # search DP/.env, then the project root, then cwd — whichever exists
    for _envp in (os.path.join(_HERE, ".env"),
                  os.path.join(_HERE, "..", ".env"),
                  ".env"):
        if os.path.exists(_envp):
            load_dotenv(_envp, override=False)
            break
    else:
        load_dotenv()
except ImportError:
    pass

# The google-genai SDK accepts EITHER name; mirror that so a key set as
# GOOGLE_API_KEY (what tweet_truth_impact_analyzer.py relies on via its bare
# genai.Client()) works here too.
if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

# STORAGE: a plain CSV in DP/, NOT database.db — the DB is single-writer and
# the trainer/backtest hold it for hours, so a long annotation run must not
# depend on it. Written append-only with a flush after EVERY post (same
# pattern as tweet_truth_impact_analyzer.py), so Ctrl-C or a crash never
# costs more than the post in flight. DuckDB still reads it happily
# (read_csv_auto) exactly like the market_data_cache bars.
ANNOT_CSV = os.path.join(_HERE, "gemini_annotations.csv")

# Gemini 3.x is required for grounding + structured output in ONE call; on
# 2.5 the API rejects response_schema together with the google_search tool.
# Tried in order — first one the API accepts wins (same fallback-chain
# pattern the Gemma loader uses).
# Probed in order (2026-07-28 on Peter's key: gemini-3-flash = 404,
# 3.x previews = 429 no quota, 2.5-flash = 400 grounding+JSON incompatible
# -> handled by the no-schema fallback in _call()). 2.5-flash is therefore
# the realistic workhorse; the 3.x names stay first in case quota appears.
GROUNDED_MODELS = [os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                   "gemini-3-flash-preview", "gemini-2.5-flash-lite",
                   "gemini-2.0-flash"]
TEXT_MODELS     = [os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite"),
                   "gemini-2.5-flash"]

GROUNDED_USD_PER_1K = 14.0     # after the free monthly allowance
FREE_GROUNDED_MONTH = 5000
TOKENS_USD_PER_CALL = 0.0004   # ~700 in + 150 out on a Flash-class model

# Instruments from THE MASTER REGISTRY (never hardcode — project rule)
with open(os.path.join(_HERE, "instruments.json"), encoding="utf-8") as _f:
    INSTRUMENTS = list(json.load(_f)["instruments"].keys())


# ===================================================================
# RESPONSE SCHEMAS
# ===================================================================
def _schemas():
    from pydantic import BaseModel, Field

    class GroundedVerdict(BaseModel):
        """LABEL-SIDE ONLY (search-grounded — knows the future)."""
        market_moving: bool = Field(
            description="Did THIS post itself measurably move markets within "
                        "about an hour? False for routine politics/endorsements.")
        instruments_moved: list[str] = Field(
            description="Which of the provided instrument names reportedly moved "
                        "because of this post. Empty if none.")
        confounder_present: bool = Field(
            description="Was there a DIFFERENT major market event (FOMC/CPI/NFP/"
                        "earnings/other geopolitics) within 60 minutes AFTER this "
                        "post that would dominate price action?")
        confounder_desc: str = Field(
            description="One short phrase naming the competing event, or empty.")
        confidence: float = Field(
            description="0.0-1.0 confidence in this verdict given what you found.")
        evidence: str = Field(
            description="One sentence citing what the search actually showed.")

    class TextFeatures(BaseModel):
        """FEATURE-SAFE (no search — depends on nothing after the post)."""
        event_type: str = Field(
            description="One of: tariff_trade, military_conflict, peace_deescalation, "
                        "monetary_policy, fiscal_policy, crypto_policy, energy_policy, "
                        "public_health, personnel, endorsement, commentary, other")
        is_new_information: bool = Field(
            description="True only if this ANNOUNCES something new. False for "
                        "restatements, opinions, reactions to known news.")
        specificity: int = Field(
            description="0=vague, 1=named entity, 2=named entity+action, "
                        "3=entity+action+number/date")
        actionability: int = Field(
            description="0=rhetoric, 1=intention, 2=decision announced, 3=in force now")
        direction: int = Field(
            description="Expected risk direction: -1 risk-off, 0 neutral, +1 risk-on")
        magnitude_class: int = Field(
            description="0=none, 1=small, 2=moderate, 3=large expected market reaction")

    return GroundedVerdict, TextFeatures


def _load_done(mode):
    """pids already annotated SUCCESSFULLY for this mode.
    Rows that errored are deliberately NOT written, so a rerun retries them —
    that is what makes the whole run resumable without ever losing a post."""
    if not os.path.exists(ANNOT_CSV):
        return set()
    try:
        prev = pd.read_csv(ANNOT_CSV, usecols=lambda c: c in ("pid", "mode"))
        return set(prev[prev["mode"] == mode]["pid"].astype(str))
    except Exception as e:                                     # noqa: BLE001
        print(f"  ⚠️  could not parse {ANNOT_CSV} ({e}) — starting fresh")
        return set()


def _append_row(rec, fieldnames):
    """Append ONE annotated post and flush immediately (checkpoint = 1 post)."""
    import csv
    new = not os.path.exists(ANNOT_CSV) or os.path.getsize(ANNOT_CSV) == 0
    with open(ANNOT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(rec)
        f.flush()
        os.fsync(f.fileno())


def _print_saved(i, n, row, rec, pfx):
    """Echo EXACTLY what just went into the CSV — full post text and every
    annotated field, nothing truncated. This is the audit trail: what you
    read in the console is literally the row on disk."""
    import textwrap
    w = 100
    when = str(row.get("date", ""))[:16]
    who = str(row.get("account", "") or "?")
    print("\n" + "─" * w)
    print(f"[{i}/{n}] 💾 SAVED → {os.path.basename(ANNOT_CSV)}   "
          f"pid={row['pid']}  @{who}  {when}")
    body = re.sub(r"\s+", " ", str(row.get("text", ""))).strip()
    for j, line in enumerate(textwrap.wrap(body, w - 12) or [""]):
        print(f"  {'POST   :' if j == 0 else '        '} {line}")
    for k, v in rec.items():
        label = k[len(pfx):] if k.startswith(pfx) else k
        if isinstance(v, str) and len(v) > w - 14:
            wrapped = textwrap.wrap(v, w - 14)
            print(f"  {label:<10}: {wrapped[0]}")
            for extra in wrapped[1:]:
                print(f"  {'':<10}  {extra}")
        else:
            print(f"  {label:<10}: {v}")


def _prompt(mode, row):
    when = str(row.get("date", ""))[:16]
    who = str(row.get("account", "") or "unknown")
    txt = str(row.get("text", ""))[:1500]
    if mode == "grounded":
        return (
            f"A social-media post by @{who} at {when} (New York time).\n\n"
            f'POST: "{txt}"\n\n'
            f"Using web search, establish what actually happened in the markets in "
            f"the hour after this post.\n"
            f"1. Did THIS post move markets? Routine politics, endorsements and "
            f"commentary almost never do — say false unless you find evidence.\n"
            f"2. Was there a DIFFERENT major scheduled or breaking event within 60 "
            f"minutes after it (FOMC, CPI, NFP, jobs, earnings, other geopolitics) "
            f"that would dominate the price action? This matters more than "
            f"question 1 — we use it to discard contaminated training windows.\n"
            f"3. Which of these instruments reportedly moved: {INSTRUMENTS}\n"
            f"Be conservative. Absence of evidence = not market-moving.")
    return (
        f"A social-media post by @{who} at {when} (New York time).\n\n"
        f'POST: "{txt}"\n\n'
        f"Classify it for a market-impact model. Judge ONLY from the text — do "
        f"not use hindsight. Most posts are routine: default to low scores.")


# ===================================================================
# API CALL (one post)
# ===================================================================
def _call(client, types_, model, mode, row, schema):
    cfg = {"temperature": 0.0, "response_mime_type": "application/json",
           "response_schema": schema}
    if mode == "grounded":
        cfg["tools"] = [types_.Tool(google_search=types_.GoogleSearch())]
    try:
        resp = client.models.generate_content(
            model=model, contents=_prompt(mode, row),
            config=types_.GenerateContentConfig(**cfg))
        return schema.model_validate_json(resp.text)
    except Exception as e:
        msg = str(e).lower()
        # 2.5-FAMILY FALLBACK. Grounding and structured output cannot be used
        # together before Gemini 3.x; the API says e.g.
        #   "Tool use with a response mime type: 'application/json' is unsupported"
        # (note: "unsupported", not "not supported" — an earlier narrower match
        # missed it and the whole model chain bailed out). Retry grounded with
        # NO schema / NO json mime and parse the JSON out of the prose.
        _incompat = any(k in msg for k in
                        ("mime", "schema", "unsupported", "not supported",
                         "tool use", "invalid_argument", "400"))
        if mode == "grounded" and _incompat:
            cfg.pop("response_schema", None)
            cfg.pop("response_mime_type", None)
            fields = json.dumps(schema.model_json_schema().get("properties", {}))
            resp = client.models.generate_content(
                model=model,
                contents=_prompt(mode, row) +
                "\n\nReply with ONLY a raw JSON object (no markdown fence) with "
                "exactly these keys:\n" + fields,
                config=types_.GenerateContentConfig(**cfg))
            import re as _re
            txt = (resp.text or "").strip()
            m = _re.search(r"\{.*\}", txt, _re.DOTALL)
            if m:
                return schema.model_validate_json(m.group(0))
            raise RuntimeError(f"grounded fallback returned no JSON: {txt[:120]}")
        raise


def _list_models(client):
    """What this API key can actually reach (names change fast)."""
    try:
        out = []
        for m in client.models.list():
            nm = getattr(m, "name", "") or ""
            acts = getattr(m, "supported_actions", None) or \
                   getattr(m, "supported_generation_methods", []) or []
            if not acts or "generateContent" in acts:
                out.append(nm.replace("models/", ""))
        return sorted(set(out))
    except Exception as e:                                     # noqa: BLE001
        print(f"  ⚠️  could not list models: {str(e)[:100]}")
        return []


def _pick_model(client, types_, candidates, mode, probe_row, schema):
    """First model the API actually accepts for this mode.
    404 = name doesn't exist. 429 = model exists but this key has no quota
    for it (common: Gemini 3.x preview on a free tier). 400 = capability
    mismatch, handled by the fallback inside _call()."""
    tried = []
    for k, m in enumerate(candidates):
        if k:
            time.sleep(8)   # probes count against the SAME per-minute quota
        try:
            _call(client, types_, m, mode, probe_row, schema)
            print(f"  ✅ model: {m}")
            return m
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                # Show the WHOLE message: it names the exact quota that was hit
                # (per-minute vs per-day, free-tier vs grounding) and usually a
                # retryDelay. Truncating it hid why a model that worked minutes
                # ago suddenly "has no quota".
                print(f"  ⚠️  {m}: QUOTA EXCEEDED — full response:\n"
                      f"      {msg[:900]}")
                why = "quota exceeded (see message above)"
            elif "404" in msg:
                print(f"  ⚠️  {m}: model name not found — trying next")
                why = "model name not found"
            else:
                print(f"  ⚠️  {m}: {msg[:200]}")
                why = msg[:80]
            tried.append(f"{m} ({why})")
    print("\n❌ No candidate model worked. Tried:")
    for t in tried:
        print(f"     - {t}")
    if any("quota" in t for t in tried):
        print(
            "\n  ⏳ These are QUOTA errors, not access errors — the same model\n"
            "     answered successfully earlier in this session. Grounded\n"
            "     (Google Search) requests have their own, much smaller free\n"
            "     allowance than plain text calls, and it is consumed per\n"
            "     minute AND per day.\n"
            "     Read the quota name in the message above:\n"
            "       *PerMinute*  -> wait a few minutes, rerun (resumes safely)\n"
            "       *PerDay*     -> resets at 00:00 Pacific; rerun tomorrow,\n"
            "                       or run the ungrounded pass meanwhile:\n"
            "                         python gemini_impact_annotator.py --mode text\n"
            "     Every post already annotated is safe in the CSV; a rerun\n"
            "     never re-bills them.")
    else:
        avail = _list_models(client)
        gens = [a for a in avail if "gemini" in a and "embedding" not in a]
        if gens:
            print("\n  Models your key CAN reach:")
            for a in gens[:25]:
                print(f"     {a}")
            print(f'\n  Pin one with:  $env:GEMINI_MODEL = "{gens[0]}"')
    sys.exit(1)


# ===================================================================
# MAIN
# ===================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Optional Gemini annotation layer (writes gemini_annotations).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mode", choices=["grounded", "text"], default="grounded",
                    help="grounded = search ON, LABEL-SIDE gl_* fields (default). "
                         "text = search OFF, feature-safe gf_* fields.")
    ap.add_argument("--table", default="training_set_HIGH_SIGNAL",
                    help="source table of posts (default: HIGH_SIGNAL only — "
                         "never ground all 70k, see cost note)")
    ap.add_argument("--limit", type=int, default=0, help="annotate at most N posts (0 = all)")
    ap.add_argument("--since", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--cost-cap", type=float, default=15.0,
                    help="refuse to start if the estimate exceeds this many USD (default 15)")
    ap.add_argument("--free-remaining", type=int, default=FREE_GROUNDED_MONTH,
                    help=f"grounded queries still free this month (default {FREE_GROUNDED_MONTH})")
    ap.add_argument("--rpm", type=float, default=8.0,
                    help="target requests/min (default 8 — free-tier grounded "
                         "calls 429 around 10/min; the pacing adapts downward "
                         "automatically on every 429)")
    ap.add_argument("--max-retries", type=int, default=6,
                    help="retries per post before leaving it PENDING for the next "
                         "run (default 6, backoff 60/120/240/480s). A post is "
                         "never skipped-and-forgotten.")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--model", default=None,
                    help="pin ONE model and skip the probe chain entirely "
                         "(each probe costs a request against the same quota)")
    ap.add_argument("--list-models", action="store_true",
                    help="print the models this API key can reach, then exit "
                         "(no posts annotated, nothing spent)")
    args = ap.parse_args()

    if args.list_models:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            sys.exit("❌ No Gemini key found (GEMINI_API_KEY / GOOGLE_API_KEY).")
        from google import genai
        for m in _list_models(genai.Client()):
            print(f"  {m}")
        return

    # ---------------------------------------------------------- source posts
    src = db.read_table(args.table)
    if src is None or not len(src):
        sys.exit(f"❌ {args.table} not found/empty — run build_final_training_set.py first.")
    for col in ("platform", "id", "text", "date"):
        if col not in src.columns:
            sys.exit(f"❌ {args.table} lacks required column '{col}'")
    if args.since:
        src = src[pd.to_datetime(src["date"], utc=True, format="mixed")
                  >= pd.Timestamp(args.since, tz="UTC")]
    src = src.sort_values("date").reset_index(drop=True)
    src["pid"] = src["platform"].astype(str) + "_" + src["id"].astype(str)

    # ---------------------------------------------------------- resume
    done = _load_done(args.mode)
    if done:
        print(f"  ♻️  {len(done)} post(s) already in {os.path.basename(ANNOT_CSV)} "
              f"(mode={args.mode}) — resuming, none re-billed")
    todo = src[~src["pid"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    n = len(todo)
    if not n:
        print("✅ Nothing to do — every selected post is already annotated.")
        return

    # ---------------------------------------------------------- cost gate
    if args.mode == "grounded":
        billable = max(0, n - args.free_remaining)
        est = billable / 1000.0 * GROUNDED_USD_PER_1K + n * TOKENS_USD_PER_CALL
        detail = (f"{n} grounded calls | {min(n, args.free_remaining)} free, "
                  f"{billable} billable @ ${GROUNDED_USD_PER_1K}/1k + tokens")
    else:
        est = n * TOKENS_USD_PER_CALL
        detail = f"{n} text-only calls (no grounding, tokens only)"
    print("=" * 70)
    print(f"  GEMINI ANNOTATION — mode={args.mode}  source={args.table}")
    print(f"  {detail}")
    print(f"  ESTIMATED COST: ${est:,.2f}   (cap ${args.cost_cap:,.2f})")
    print(f"  ETA at {args.rpm:.0f} rpm: {n / max(args.rpm, 1) / 60:.1f} h")
    print("=" * 70)
    if est > args.cost_cap:
        sys.exit(f"❌ Estimate ${est:,.2f} exceeds --cost-cap ${args.cost_cap:,.2f}. "
                 f"Lower --limit, or raise the cap deliberately.")
    if not args.yes:
        if input("  Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Aborted — nothing spent.")

    # ---------------------------------------------------------- client
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit(
            "❌ No Gemini key found (checked GEMINI_API_KEY and GOOGLE_API_KEY,\n"
            "   plus .env in DP/, the project root and the current folder).\n"
            "   Fix with ONE of:\n"
            '     $env:GEMINI_API_KEY = "your-key"        # this shell only\n'
            '     setx GEMINI_API_KEY "your-key"          # permanent, reopen shell\n'
            "     echo GEMINI_API_KEY=your-key > .env     # run from DP\\\n"
            "   Get a key at https://aistudio.google.com/apikey")
    from google import genai
    from google.genai import types as types_
    client = genai.Client()
    Grounded, TextFeat = _schemas()
    schema = Grounded if args.mode == "grounded" else TextFeat
    if args.model:
        model = args.model          # trust the pin, don't spend a probe
        print(f"  📌 model pinned: {model} (probe skipped)")
    else:
        model = _pick_model(client, types_,
                            GROUNDED_MODELS if args.mode == "grounded" else TEXT_MODELS,
                            args.mode, todo.iloc[0], schema)

    # ---------------------------------------------------------- annotate
    pfx = "gl_" if args.mode == "grounded" else "gf_"
    fieldnames = (["pid", "platform", "id", "mode", "model", "annotated_at"] +
                  [f"{pfx}{k}" for k in schema.model_fields])
    gap = 60.0 / max(args.rpm, 0.1)     # adaptive: grows on 429, decays on success
    ok = pending = 0
    t0 = time.time()

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        # RETRY UNTIL IT SUCCEEDS — a post is NEVER skipped after one 60s nap.
        # Exponential backoff 60/120/240/480s (capped), --max-retries attempts.
        # If they all fail we write NOTHING for this post, so the next run
        # picks it up again instead of silently burying it as "done".
        rec, attempt, wait = None, 0, 60.0
        while attempt <= args.max_retries:
            try:
                res = _call(client, types_, model, args.mode, row, schema)
                rec = {f"{pfx}{k}": (json.dumps(v) if isinstance(v, list) else v)
                       for k, v in res.model_dump().items()}
                gap = max(60.0 / args.rpm, gap * 0.9)   # recovered — ease off
                break
            except Exception as e:                                # noqa: BLE001
                msg = str(e)
                attempt += 1
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    gap = min(gap * 1.5, 30.0)          # throttle down, stay slower
                    print(f"  ⏳ rate limited at {i}/{n} (try {attempt}/{args.max_retries}) "
                          f"— waiting {wait:.0f}s, pacing now {60/gap:.1f} rpm")
                else:
                    print(f"  ⚠️  {i}/{n} error (try {attempt}/{args.max_retries}): {msg[:110]}")
                if attempt > args.max_retries:
                    break
                time.sleep(wait)
                wait = min(wait * 2, 480.0)

        if rec is None:
            pending += 1
            print(f"  ⏭️  {i}/{n} left PENDING (not written) — rerun will retry it")
            time.sleep(gap)
            continue

        _append_row({"pid": row["pid"], "platform": row["platform"], "id": row["id"],
                     "mode": args.mode, "model": model,
                     "annotated_at": datetime.datetime.now(
                         datetime.timezone.utc).isoformat(timespec="seconds"),
                     **rec}, fieldnames)
        ok += 1
        _print_saved(i, n, row, rec, pfx)
        _el = time.time() - t0
        print(f"  {'':<10}  [{ok} saved | {pending} pending | {_el/i:.1f}s/post | "
              f"eta {(n - i) * _el / i / 3600:.1f}h]")
        if i < n:
            time.sleep(gap)

    print(f"\n✅ {ok}/{n} annotated → {os.path.basename(ANNOT_CSV)}"
          f"{f'  ({pending} still pending — just rerun)' if pending else ''}"
          f"  [{(time.time()-t0)/60:.1f} min]")
    print(f"   Next: python build_final_training_set.py --full --use-gemini")


if __name__ == "__main__":
    main()
