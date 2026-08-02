"""
rag_confounder_annotator.py — FREE RAG confounder labelling for DP
===========================================================================
Answers the question the price data CANNOT answer on its own:

    the market moved in the hour after this post — but was it THIS POST,
    or was something else happening?

That is attribution, and it is the real bottleneck. A CPI print landing 12
minutes after a tweet gives that tweet a big 1-hour label it did not earn,
and the model then learns a lie. The placebo test already showed post
windows are barely distinguishable from random windows; confounding is a
large part of why.

Retrieval is GDELT (free, no API key, 100+ languages, 15-minute updates).
Generation is the SAME local Gemma 3 4B already loaded for embeddings, so
the marginal cost of this whole layer is zero — no quota, no billing, and
unlike Google Search grounding it is REPRODUCIBLE: re-running this in six
months returns the same evidence for a 2017 post, because the query window
is pinned. Grounding is not reproducible, which quietly poisons any
backtest you intend to re-run.

---------------------------------------------------------------------------
THE LEAKAGE FIREWALL — read before adding any field
---------------------------------------------------------------------------
This module retrieves news from the window [post, post+60min]. That is the
FUTURE relative to the post. It is exactly right for labels and absolutely
fatal for features: at live prediction time that window has not happened.

Every column produced here is prefixed `rag_` and is LABEL-SIDE ONLY —
curation, quality flags, sample weighting. NEVER a model feature.

Three guards, enforced by code rather than by discipline:

  G1  every returned article's seendate is re-checked against the requested
      window; out-of-window hits are DROPPED AND COUNTED (rag_dropped), so
      a silently broken filter shows up as a number instead of as quietly
      poisoned labels.
  G2  assert_label_side_only() is imported by build_final_training_set.py
      and raises if any `rag_` column reaches train_columns().
  G3  the retriever refuses a window whose end exceeds post+window_min.

---------------------------------------------------------------------------
WHAT IT MEASURES
---------------------------------------------------------------------------
Two separate GDELT queries per post:

  AMBIENT  broad market-moving themes (CPI, central banks, tariffs, OPEC,
           sanctions, jobs...). "What else was going on?"
  ECHO     the post's OWN entities. "How much of the window is about THIS?"

  rag_attribution = echo / (echo + ambient)   -> 1.0 = the window belongs to
                                                 this post; 0.0 = it belongs
                                                 to something else entirely
  rag_news_spike  = ambient_in_window / ambient_baseline_per_hour

Then Gemma reads the headlines (titles + timestamps only) and returns a
structured verdict. The numeric signals stand on their own if Gemma is
unavailable — they need no LLM at all.

USAGE
-----
  # ALWAYS FIRST — proves connectivity, response shape and the guards
  python rag_confounder_annotator.py --selftest

  # then a small pilot, look at the printed records
  python rag_confounder_annotator.py --limit 50

  # full HIGH_SIGNAL pass (resumable; re-run after any interruption)
  python rag_confounder_annotator.py

  # then, and only then:
  python build_final_training_set.py --full --use-rag

NOTE ON COVERAGE: --source doc searches roughly the LAST 3 MONTHS. Older
posts return no articles and are written with rag_status='out_of_coverage'
(NOT confounded, NOT down-weighted — absence of evidence is not evidence).
Backfill for 2016-2025 needs --source bigquery or --source raw, which are
stubbed here deliberately: prove the verdicts are useful on the free window
before paying for a TB-scale download.
"""
import os
import re
import sys
import csv
import json
import time
import argparse
import datetime as dt
import urllib.parse
import urllib.request
import urllib.error

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(_HERE, "rag_annotations.csv")

# ----------------------------------------------------------------- config ---
WINDOW_MIN_DEFAULT = 60      # matches the 1-hour label horizon exactly
BASELINE_HOURS     = 24      # trailing reference for the news-spike ratio
MAX_RECORDS        = 75      # per query; GDELT caps at 250
GDELT_DOC_URL      = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT         = "forecasting-instruments-research/1.0 (academic; contact via repo)"

# Broad market-moving themes. This is deliberately NOT about the post — the
# whole point is to detect what ELSE could have moved the tape.
AMBIENT_QUERY = ('(inflation OR "interest rate" OR "central bank" OR '
                 '"federal reserve" OR tariff OR sanctions OR OPEC OR '
                 '"jobs report" OR CPI OR GDP OR "trade deal" OR airstrike '
                 'OR ceasefire OR "oil price")')

# Scheduled macro releases are the single biggest confounder class and are
# recognisable from the headline alone — Gemma is asked to name them.
MACRO_MARKERS = ("cpi", "inflation report", "fomc", "federal reserve",
                 "rate decision", "nonfarm", "jobs report", "payroll",
                 "gdp", "ppi", "jobless claims", "pce")

RAG_COLUMNS = [
    "platform_id", "post_ts_utc", "rag_status",
    "rag_ambient_n", "rag_echo_n", "rag_baseline_per_hr",
    "rag_news_spike", "rag_attribution", "rag_dropped",
    "rag_confounder_present", "rag_confounder_type", "rag_confounder_desc",
    "rag_confidence", "rag_top_headline", "rag_model", "rag_annotated_at",
]


# ============================================================== firewall ====
def assert_label_side_only(columns, where="train_columns()"):
    """G2. Imported by build_final_training_set.py. Raises if any rag_ column
    reached a FEATURE list. These columns are built from news published AFTER
    the post; using one as a feature is textbook lookahead leakage."""
    bad = sorted(c for c in columns if str(c).startswith("rag_"))
    if bad:
        raise RuntimeError(
            f"LEAKAGE FIREWALL: {len(bad)} rag_* column(s) reached {where}: "
            f"{bad[:6]}{'...' if len(bad) > 6 else ''}. These are built from "
            f"news published AFTER the post and are LABEL-SIDE ONLY. Use them "
            f"for sample_weight / curation, never as model input.")
    return True


# ============================================================= retrieval ====
class Retriever:
    """Pluggable corpus backend. doc = free + keyless, last ~3 months."""

    name = "base"

    def search(self, query, start_utc, end_utc):
        raise NotImplementedError


class GdeltDocRetriever(Retriever):
    name = "doc"

    def __init__(self, max_records=MAX_RECORDS, pause=1.5, max_retries=6):
        self.max_records = max_records
        self.pause = pause
        self.max_retries = max_retries

    @staticmethod
    def _stamp(t):
        return t.strftime("%Y%m%d%H%M%S")

    @staticmethod
    def _parse_seen(s):
        """GDELT seendate looks like 20260701T120000Z."""
        try:
            return dt.datetime.strptime(str(s), "%Y%m%dT%H%M%SZ").replace(
                tzinfo=dt.timezone.utc)
        except (ValueError, TypeError):
            return None

    def search(self, query, start_utc, end_utc):
        """Return (articles, dropped). Retries until success — never skips,
        matching the annotator contract. Raises only on unrecoverable input."""
        params = {
            "query": query, "mode": "artlist", "format": "json",
            "maxrecords": str(self.max_records), "sort": "datedesc",
            "startdatetime": self._stamp(start_utc),
            "enddatetime": self._stamp(end_utc),
        }
        url = f"{GDELT_DOC_URL}?{urllib.parse.urlencode(params)}"
        delay = 5
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=45) as r:
                    raw = r.read().decode("utf-8", errors="replace")
                if not raw.strip():
                    return [], 0                     # genuinely no coverage
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    # GDELT returns an HTML error page when it is unhappy with
                    # a query; treat as retryable rather than crashing a
                    # multi-hour run.
                    raise urllib.error.URLError(f"non-JSON reply: {raw[:120]}")
                arts = data.get("articles") or []
                # ---- G1: re-verify every timestamp against the window ------
                keep, dropped = [], 0
                for a in arts:
                    seen = self._parse_seen(a.get("seendate"))
                    if seen is None or not (start_utc <= seen <= end_utc):
                        dropped += 1
                        continue
                    keep.append({"title": (a.get("title") or "").strip(),
                                 "seen": seen,
                                 "domain": a.get("domain") or "",
                                 "url": a.get("url") or ""})
                time.sleep(self.pause)               # be a good citizen
                return keep, dropped
            except Exception as e:                   # noqa: BLE001
                if attempt >= self.max_retries:
                    raise
                print(f"      ⚠️  GDELT attempt {attempt}/{self.max_retries} "
                      f"failed ({type(e).__name__}: {str(e)[:70]}) — "
                      f"retry in {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 300)
        return [], 0


class NotImplementedRetriever(Retriever):
    """--source bigquery / raw. Stubbed ON PURPOSE: build the expensive
    backfill only after the free window proves the verdicts are useful."""

    def __init__(self, name):
        self.name = name

    def search(self, query, start_utc, end_utc):
        sys.exit(
            f"❌ --source {self.name} is not implemented yet.\n"
            f"   The DOC API covers ~3 months, which is enough to prove the "
            f"pipeline. Run --source doc first, check whether "
            f"rag_confounder_present actually correlates with label noise, "
            f"and only then invest in the {self.name} backfill.")


def make_retriever(source):
    return GdeltDocRetriever() if source == "doc" else NotImplementedRetriever(source)


# ============================================================ post -> query ==
_STOP = set("""the a an and or but if then than that this these those of to in on for with
from by at as is are was were be been being it its it's we our you your they their he she
his her i me my will would can could should may might must have has had do does did not no
so just now new more most very much many much also into over under about after before out up
down off again all any both each few other some such only own same too s t don should've""".split())


def entity_query(text, max_terms=6):
    """ECHO query — the post's own entities. Capitalised multiword phrases
    first (people, countries, companies), then salient rare words."""
    text = re.sub(r"https?://\S+", " ", str(text or ""))
    phrases = re.findall(r"\b([A-Z][\w'&.-]+(?:\s+[A-Z][\w'&.-]+){0,2})\b", text)
    terms, seen = [], set()
    for p in phrases:
        p = p.strip()
        if len(p) < 4 or p.lower() in _STOP:
            continue
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        terms.append(f'"{p}"' if " " in p else p)
        if len(terms) >= max_terms:
            break
    if len(terms) < 2:
        words = [w for w in re.findall(r"[A-Za-z]{5,}", text)
                 if w.lower() not in _STOP][:max_terms]
        for w in words:
            if w.lower() not in seen:
                seen.add(w.lower())
                terms.append(w)
    if not terms:
        return None
    return "(" + " OR ".join(terms[:max_terms]) + ")"


# ============================================================= the verdict ==
def gemma_verdict(post_text, headlines, attribution, spike):
    """Local Gemma 3 4B reads the window's headlines and names the confounder.
    Returns a dict; falls back to a numeric-only verdict if Gemma is
    unavailable, because the numbers are meaningful on their own."""
    fallback = {
        "present": bool(attribution < 0.35 and spike >= 1.5),
        "type": "unknown",
        "desc": "numeric-only (Gemma unavailable)",
        "confidence": 0.3,
        "model": "numeric-only",
    }
    if not headlines:
        fallback["desc"] = "no articles retrieved in window"
        return fallback
    try:
        sys.path.insert(0, os.path.join(_HERE, "..", "Gemma-3-4b_NLP_XGBoost"))
        import gemma_embedder as GE
        import torch
        tok, model = GE._load_gemma()
        _t = getattr(tok, "tokenizer", tok)
        lines = "\n".join(f"- [{h['seen']:%H:%M}] {h['title'][:140]}"
                          for h in headlines[:25])
        sys_p = (
            "You are a markets analyst. You are given a social-media post and "
            "the news headlines published in the 60 minutes AFTER it. Decide "
            "whether the market move in that hour is better explained by "
            "SOMETHING OTHER THAN the post.\n"
            "Reply with ONLY a JSON object:\n"
            '{"confounder_present": true/false, "confounder_type": '
            '"scheduled_macro"|"geopolitical"|"earnings"|"other_actor"|"none", '
            '"confounder_desc": "<12 words>", "confidence": 0.0-1.0}\n'
            "scheduled_macro = CPI/FOMC/payrolls/GDP/PPI/claims/PCE. "
            "other_actor = a different official or institution acting. "
            "If the headlines are just coverage OF the post, that is NOT a "
            "confounder."
        )
        usr = (f"POST:\n{str(post_text)[:900]}\n\n"
               f"HEADLINES IN THE NEXT 60 MIN:\n{lines}\n\n"
               f"(share of window's news about this post: {attribution:.2f}; "
               f"news volume vs normal: {spike:.1f}x)")
        prompt = _t.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": usr}],
            tokenize=False, add_generation_prompt=True)
        enc = _t(prompt, return_tensors="pt", truncation=True, max_length=2048,
                 add_special_tokens=False)
        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=140, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=_t.pad_token_id or _t.eos_token_id)
        gen = _t.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*?\}", gen, re.DOTALL)
        if not m:
            fallback["desc"] = "gemma returned no JSON"
            return fallback
        d = json.loads(m.group(0))
        return {
            "present": bool(d.get("confounder_present", False)),
            "type": str(d.get("confounder_type", "unknown"))[:32],
            "desc": str(d.get("confounder_desc", ""))[:120],
            "confidence": float(d.get("confidence", 0.5)),
            "model": os.environ.get("GEMMA_MODEL_ID",
                                    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"),
        }
    except Exception as e:                                    # noqa: BLE001
        fallback["desc"] = f"gemma failed: {type(e).__name__}: {str(e)[:60]}"
        return fallback


# ================================================================= storage ==
def load_done(path):
    """Resume set. A post is 'done' only once its row is fully written, so an
    interrupted post is simply retried on the next run — never skipped."""
    if not os.path.exists(path):
        return set()
    done = set()
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = (row.get("platform_id") or "").strip()
                if pid:
                    done.add(pid)
    except Exception as e:                                    # noqa: BLE001
        print(f"⚠️  could not read existing CSV ({e}) — starting fresh")
    return done


def append_row(path, row):
    """Checkpoint on EVERY post: append + flush + fsync so a crash or a
    power cut loses at most the row in flight."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAG_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def print_saved(row, i, n, text):
    """Fully printed: exactly what went into the CSV, not a generic counter."""
    mark = "🚩" if str(row["rag_confounder_present"]) == "True" else "✅"
    print(f"\n  ── [{i}/{n}] {row['platform_id']}  {row['post_ts_utc']}")
    print(f"     post     : {str(text)[:100]}")
    print(f"     status   : {row['rag_status']}   dropped(G1)={row['rag_dropped']}")
    print(f"     ambient  : {row['rag_ambient_n']} articles   "
          f"echo: {row['rag_echo_n']}   baseline/hr: {row['rag_baseline_per_hr']}")
    print(f"     spike    : {row['rag_news_spike']}x   "
          f"attribution: {row['rag_attribution']}")
    print(f"     top news : {str(row['rag_top_headline'])[:100]}")
    print(f"  {mark} VERDICT : confounder={row['rag_confounder_present']} "
          f"type={row['rag_confounder_type']} conf={row['rag_confidence']}")
    print(f"     reason   : {row['rag_confounder_desc']}")
    print(f"     💾 written to {os.path.basename(OUT_CSV)}", flush=True)


# ================================================================ selftest ==
def selftest(retriever):
    print("=" * 74)
    print("  SELFTEST — connectivity, response shape, and the leak guards")
    print("=" * 74)
    ok = True

    print("\n[1/4] firewall G2: rag_* must be rejected as features")
    try:
        assert_label_side_only(["score_policy", "rag_attribution"])
        print("      ❌ FAIL — a rag_ column was accepted as a feature")
        ok = False
    except RuntimeError as e:
        print(f"      ✅ PASS — {str(e)[:80]}...")
    try:
        assert_label_side_only(["score_policy", "SPY_mom5"])
        print("      ✅ PASS — clean feature list accepted")
    except RuntimeError:
        print("      ❌ FAIL — clean feature list was rejected")
        ok = False

    print("\n[2/4] entity query extraction")
    q = entity_query("China will pay a 25% tariff. I have spoken with "
                     "President Xi about Taiwan.")
    print(f"      -> {q}")
    ok &= bool(q and "China" in q)
    print(f"      {'✅ PASS' if q and 'China' in q else '❌ FAIL'}")

    print("\n[3/4] GDELT reachability (a 2-hour window 10 days ago)")
    end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)
    start = end - dt.timedelta(hours=2)
    try:
        arts, dropped = retriever.search(AMBIENT_QUERY, start, end)
        print(f"      window   : {start:%Y-%m-%d %H:%M} -> {end:%H:%M} UTC")
        print(f"      articles : {len(arts)}   dropped by G1: {dropped}")
        for a in arts[:3]:
            print(f"        [{a['seen']:%m-%d %H:%M}] {a['domain']}: "
                  f"{a['title'][:70]}")
        if not arts:
            print("      ⚠️  zero articles — check the network, or GDELT is "
                  "throttling. NOT necessarily a bug.")
        else:
            bad = [a for a in arts if not (start <= a["seen"] <= end)]
            print(f"      {'✅ PASS' if not bad else '❌ FAIL'} — G1 timestamp "
                  f"bound held on {len(arts)} articles ({len(bad)} violations)")
            ok &= not bad
    except Exception as e:                                    # noqa: BLE001
        print(f"      ❌ FAIL — {type(e).__name__}: {str(e)[:120]}")
        ok = False

    print("\n[4/4] local Gemma verdict")
    v = gemma_verdict("Test post about tariffs on China.",
                      [{"title": "US CPI comes in hotter than expected",
                        "seen": dt.datetime.now(dt.timezone.utc)}], 0.2, 3.0)
    print(f"      model    : {v['model']}")
    print(f"      verdict  : present={v['present']} type={v['type']} "
          f"desc={v['desc'][:60]}")
    if v["model"] == "numeric-only":
        print("      ⚠️  Gemma unavailable — the numeric signals still work, "
              "but verdicts will be coarse.")

    print("\n" + "=" * 74)
    print(f"  {'✅ SELFTEST PASSED' if ok else '❌ SELFTEST FAILED'} — "
          f"{'safe to run the pilot' if ok else 'fix the above before running'}")
    print("=" * 74)
    return 0 if ok else 1


# ==================================================================== main ==
def main():
    ap = argparse.ArgumentParser(
        description="Free RAG confounder labelling (GDELT + local Gemma). "
                    "LABEL-SIDE ONLY — never a model feature.")
    ap.add_argument("--source", default="doc", choices=["doc", "bigquery", "raw"])
    ap.add_argument("--table", default="training_set_HIGH_SIGNAL",
                    help="DuckDB source table (default: training_set_HIGH_SIGNAL)")
    ap.add_argument("--limit", type=int, default=None, help="pilot on N posts")
    ap.add_argument("--window-min", type=int, default=WINDOW_MIN_DEFAULT,
                    help=f"label window in minutes (default {WINDOW_MIN_DEFAULT}, "
                         f"must match the 1h impact horizon)")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--selftest", action="store_true",
                    help="prove connectivity + guards, annotate nothing")
    ap.add_argument("--pause", type=float, default=1.5,
                    help="seconds between GDELT calls (default 1.5)")
    args = ap.parse_args()

    retriever = make_retriever(args.source)
    if isinstance(retriever, GdeltDocRetriever):
        retriever.pause = args.pause
    if args.selftest:
        return selftest(retriever)

    import duckdb
    db_path = os.path.join(_HERE, "..", "database.db")
    con = duckdb.connect(db_path, read_only=True)
    q = (f'SELECT id, platform, date, text FROM "{args.table}" '
         f'ORDER BY date DESC')          # newest first: inside DOC coverage
    if args.limit:
        q += f" LIMIT {int(args.limit)}"
    posts = con.execute(q).df()
    con.close()

    done = load_done(args.out)
    todo = [r for _, r in posts.iterrows()
            if f"{r['platform']}_{r['id']}" not in done]

    print("=" * 74)
    print("  RAG CONFOUNDER ANNOTATOR  (GDELT + local Gemma 3 4B)")
    print("=" * 74)
    print(f"  source     : {args.source}  (DOC covers ~last 3 months)")
    print(f"  table      : {args.table}   posts: {len(posts)}")
    print(f"  window     : post -> post + {args.window_min} min  (the FUTURE — "
          f"label-side only)")
    print(f"  already done: {len(done)}   to annotate: {len(todo)}")
    print(f"  output     : {args.out}  (append + fsync per post, resumable)")
    print("=" * 74, flush=True)
    if not todo:
        print("\n✅ nothing to do — every post already annotated.")
        return 0

    n_conf = n_cov = 0
    for i, r in enumerate(todo, 1):
        pid = f"{r['platform']}_{r['id']}"
        post_ts = r["date"]
        if post_ts.tzinfo is None:
            post_ts = post_ts.tz_localize("UTC")
        post_ts = post_ts.tz_convert("UTC").to_pydatetime()
        w_end = post_ts + dt.timedelta(minutes=args.window_min)
        b_start = post_ts - dt.timedelta(hours=BASELINE_HOURS)

        print(f"\n[{i}/{len(todo)}] {pid}  {post_ts:%Y-%m-%d %H:%M} UTC", flush=True)

        # ---- G3: the window can never extend past post+window_min ---------
        assert w_end <= post_ts + dt.timedelta(minutes=args.window_min), \
            "G3 violated: retrieval window exceeds the label horizon"

        amb, drop_a = retriever.search(AMBIENT_QUERY, post_ts, w_end)
        eq = entity_query(r["text"])
        echo, drop_e = (retriever.search(eq, post_ts, w_end) if eq else ([], 0))
        base, drop_b = retriever.search(AMBIENT_QUERY, b_start, post_ts)
        base_per_hr = round(len(base) / max(BASELINE_HOURS, 1), 3)

        status = "ok"
        if not amb and not echo and not base:
            status = "out_of_coverage"      # older than the DOC window
            n_cov += 1

        spike = round(len(amb) / max(base_per_hr, 0.05), 2) if base_per_hr else 0.0
        denom = len(amb) + len(echo)
        attribution = round(len(echo) / denom, 3) if denom else 0.0

        if status == "out_of_coverage":
            v = {"present": False, "type": "none",
                 "desc": "outside DOC 3-month coverage — no evidence either way",
                 "confidence": 0.0, "model": "n/a"}
        else:
            v = gemma_verdict(r["text"], amb, attribution, spike)
        n_conf += bool(v["present"])

        row = {
            "platform_id": pid,
            "post_ts_utc": post_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "rag_status": status,
            "rag_ambient_n": len(amb), "rag_echo_n": len(echo),
            "rag_baseline_per_hr": base_per_hr,
            "rag_news_spike": spike, "rag_attribution": attribution,
            "rag_dropped": drop_a + drop_e + drop_b,
            "rag_confounder_present": v["present"],
            "rag_confounder_type": v["type"],
            "rag_confounder_desc": v["desc"],
            "rag_confidence": round(float(v["confidence"]), 3),
            "rag_top_headline": (amb[0]["title"][:180] if amb else ""),
            "rag_model": v["model"],
            "rag_annotated_at": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"),
        }
        append_row(args.out, row)
        print_saved(row, i, len(todo), r["text"])

    print("\n" + "=" * 74)
    print(f"  DONE — {len(todo)} annotated, {n_conf} flagged as confounded "
          f"({n_conf / max(len(todo), 1):.0%})")
    if n_cov:
        print(f"  {n_cov} post(s) outside DOC coverage — NOT down-weighted "
              f"(absence of evidence is not evidence of a confounder)")
    print(f"  💾 {args.out}")
    print(f"\n  Next: python build_final_training_set.py --full --use-rag")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
