"""
signal_scorer.py  —  UNIFIED FEED SCORER (config-driven)
----------------------------------------------------------
Reads ALL posts from the unified_feed DuckDB table (TruthSocial posts +
English geopolitical X/Twitter posts) and writes NLP scores to posts_scored.

Rank-0 primary account (currently the US President on TruthSocial):
  entity_weight=1.0, event_weight=1.0 — highest trust, uncapped sample_weight.
Geo X/Twitter accounts are discounted:
    sample_weight_geo = base_weight × entity_weight × event_weight × SOURCE_DISCOUNT
where SOURCE_DISCOUNT=0.7 ensures geo posts never exceed 70% of the primary ceiling.

Platform is a factual string ('truthsocial' | 'x_twitter'). is_primary is a derived
boolean (platform == 'truthsocial') stored in posts_scored for XGBoost features.

The entity_weight and event_weight columns come from unified_feed (populated by
sync_unified_feed.py, which reads influence_accounts.json + events.json).

PUBLIC API (importable):
  CONFIG, FEED_TABLE, SCORED_TABLE
  extract_policy_flags(text)
  extract_ner_features(text, nlp)
  compute_composite_scores(df)
  load_spacy() / load_sbert()
  embedding_score / novelty_score / burst_position_score / relative_signal_strength
  score_single_post(text, nlp, sbert, feature_cols,
                    entity_weight, event_weight, is_primary) -> dict
  main()             — full rebuild: unified_feed → posts_scored
  score_incremental()— daily: score only new rows, append to posts_scored

CLI:
  python signal_scorer.py            → incremental (daily)
  python signal_scorer.py --full     → full rebuild

MIGRATION:
MIGRATION NOTE:
  No migrations needed — fresh database only. Old trump_truths_* tables are no longer supported.
"""

import os, re, json, sys
import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

# ============================================================== config ----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import db

FEED_TABLE        = "unified_feed"
SCORED_TABLE      = "posts_scored"
SOURCE_DISCOUNT   = 0.7      # X/Twitter geo posts: sample_weight capped at 70% of primary max
GEO_WEIGHT_CAP    = 0.70     # hard ceiling for X/Twitter geo post sample_weight
PRIMARY_WEIGHT_CAP = 1.00    # hard ceiling for rank-0 primary account sample_weight

_ENTITIES_FILE = os.path.join(_HERE, "influence_accounts.json")

def _rank0_handle() -> str:
    """Return the rank-0 TruthSocial account handle from influence_accounts.json.
    Falls back to 'us_president' if the file is missing or empty."""
    try:
        import json
        with open(_ENTITIES_FILE, encoding="utf-8") as f:
            accounts = json.load(f).get("primary_accounts", [])
        ts = sorted([a for a in accounts if a.get("platform") == "truthsocial"],
                    key=lambda a: a.get("rank", 99))
        return ts[0]["account"] if ts else "us_president"
    except Exception:
        return "us_president"


def _rank0_windows() -> dict:
    """
    {(handle_lower, platform_lower|None): (from_ts|None, to_ts|None)} for ALL
    rank-0 primary accounts. Keyed by (handle, PLATFORM) -- NOT handle alone --
    because the SAME handle can be the primary in two disjoint eras on different
    platforms: @realDonaldTrump is rank-0 on X in 2017-2021 (45th term) AND on
    Truth Social in 2024-2028 (47th term). Keying by handle alone would collapse
    the two windows (dict overwrite) and mislabel a whole presidency. The post's
    DATE + PLATFORM decide who is primary at that moment. platform=None matches
    the handle on ANY platform (defensive, for rank-0 rows with no platform set).
    """
    def _p(x):
        if x is None or str(x).strip().upper() in ("", "N/A", "NONE", "NULL"):
            return None
        try:
            return pd.Timestamp(x, tz="UTC")
        except Exception:
            return None

    try:
        import json as _json
        with open(_ENTITIES_FILE, encoding="utf-8") as f:
            accounts = _json.load(f).get("primary_accounts", [])
    except Exception:
        return {}
    out = {}
    for a in accounts:
        try:
            if int(a.get("rank", 99)) != 0:
                continue
        except (TypeError, ValueError):
            continue
        h = str(a.get("account", "")).strip().lstrip("@").lower()
        if not h:
            continue
        p = str(a.get("platform", "")).strip().lower()
        if p in ("", "n/a", "none", "null"):
            p = None
        lo = _p(a.get("active_from"))
        hi = _p(a.get("active_to")) or _p(a.get("expiration_date"))
        out[(h, p)] = (lo, hi)
    return out


def _resolve_config_path():
    env = os.environ.get("SCORER_CONFIG")
    if env:
        return env
    candidates = [
        os.path.join(_HERE, "scorer_config.json"),
        os.path.join(_HERE, "..", "DP", "scorer_config.json"),
        "scorer_config.json",
        os.path.join("..", "DP", "scorer_config.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


CONFIG_PATH = _resolve_config_path()


def load_config(path=None):
    path = path or CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ {path} not found. scorer_config.json must be in DP/."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()

POLICY_VERBS       = set(CONFIG["policy_verbs"])
GEOPOLITICAL_TERMS = set(CONFIG["geopolitical_terms"])
POLICY_FLAGS       = CONFIG["policy_flags"]
FINANCIAL_REFS     = CONFIG["financial_refs"]
NOISE_REFS         = CONFIG["noise_refs"]

# ---------------------------------------------------------------------------
# ENDORSEMENT HARD-SKIP (moved here from predict, 2026-08-02)
# ---------------------------------------------------------------------------
# Political endorsement / ceremonial posts are stuffed with policy buzzwords —
# "Strong on Crime, Borders, the Military, our Vets, Tax Cuts, the 2nd
# Amendment" — so score_policy rates them highly and NOISE_REFS (only 0.3 of
# the composite) cannot pull them back down. They were reaching training,
# the embedding cache, HIGH_SIGNAL, and the Gemma analyst run, which spends
# ~1.5s per post generating market impacts for things like:
#   ".@Troy_Balderson of Ohio is running for Congress ... He has my full and
#    total Endorsement!"  ->  GOLD +0.10%, SPY +0.20%, VIX +0.10%
# The gate existed ONLY in predict_gemma3_nlp_xgb.py, i.e. it fired after all
# of that had already happened. Scoring it here kills them at the source.
ENDORSEMENT_PATTERNS = CONFIG.get("endorsement_patterns", [])
_ENDORSEMENT_RE = [re.compile(p, re.I) for p in ENDORSEMENT_PATTERNS]
ENDORSEMENT_DAMP = float(os.environ.get("ENDORSEMENT_DAMP", "0.0"))


# ---------------------------------------------------------------------------
# COUNTRY CONTEXT — the SPEAKER x SUBJECT interaction
# ---------------------------------------------------------------------------
# "china" is not a signal by itself. The Chinese Embassy posting about China is
# routine self-promotion; the US President posting about China is cross-border
# friction. Before this, both scored the same, and the embassy's tourism and
# human-interest posts were generating real USD_CNY predictions:
#   "When a visitor's car became stuck in the mud in Xinjiang, #China, a local
#    herder stepped in to help..."            -> USD_CNY -0.30%
# The mechanism was score_embedding: financial_refs contains "tariffs on China
# will double...", so ANY China post sits close to it in SBERT space.
#
# RULE: if every country a post names is the SPEAKER'S OWN, it is self-
# reference -> damp. Name a FOREIGN country and nothing happens. A policy flag
# (tariffs, sanctions, escalation, rates) overrides the damp entirely, because
# a Chinese official announcing retaliation IS market-moving even though it is
# self-referential.
COUNTRY_TERMS   = CONFIG.get("country_terms", {})
COUNTRY_SELF_DAMP = float(CONFIG.get("country_self_damp", 0.35))
COUNTRY_POLICY_GUARD = list(CONFIG.get("country_policy_guard", []))
_COUNTRY_RE = {cc: re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)
               for cc, terms in COUNTRY_TERMS.items() if terms}


_COUNTRY_OF_HANDLE = {}


def _load_country_map():
    """{handle_lower: ISO} from influence_accounts.json. The feed carries no
    country column, so the speaker's country is resolved by handle — same
    source predict._country_for uses, so scoring and trading agree."""
    if _COUNTRY_OF_HANDLE:
        return _COUNTRY_OF_HANDLE
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "influence_accounts.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        secs = [d.get("entities", []),
                d.get("institutions", {}).get("entries", []),
                d.get("archives", {}).get("entries", []),
                d.get("primary_accounts", [])]
        for sec in secs:
            for e in (sec or []):
                h = (e.get("account") or e.get("twitter_handle") or "")
                cc = e.get("country")
                if h and cc:
                    _COUNTRY_OF_HANDLE[str(h).lstrip("@").lower()] = str(cc).upper()
    except Exception:
        pass
    return _COUNTRY_OF_HANDLE


def country_of(handle):
    return _load_country_map().get(str(handle or "").lstrip("@").lower())


def countries_mentioned(text):
    """ISO codes whose terms appear in the text."""
    t = str(text or "")
    return {cc for cc, rx in _COUNTRY_RE.items() if rx.search(t)}


def country_context_factor(text, speaker_country, feats=None, is_primary=False):
    """(factor, label). COUNTRY_SELF_DAMP when the post only talks about the
    speaker's OWN country and no policy flag fired; else 1.0.

    Rank-0 PRIMARY speakers are never damped. The damp exists to mute routine
    embassy self-promotion (a CN mission posting about CN culture). A head of
    state talking about their OWN economy — tariffs, the Fed, US energy — is
    the single highest-value signal in the feed, i.e. the exact opposite case.
    Damping it suppressed ~6.2k rank-0 posts and made them 3.9x less likely
    to survive into training."""
    if is_primary:
        return 1.0, ""
    if not speaker_country or not _COUNTRY_RE:
        return 1.0, ""
    cc = str(speaker_country).strip().upper()
    seen = countries_mentioned(text)
    if not seen or cc not in seen:
        return 1.0, ""                      # foreign subject, or none named
    foreign = seen - {cc}
    if foreign:
        return 1.0, ""                      # cross-border -> leave alone
    if feats:
        for f in COUNTRY_POLICY_GUARD:
            try:
                if float(feats.get(f) or 0.0) > 0:
                    return 1.0, ""          # real policy content -> protected
            except (TypeError, ValueError):
                pass
    return COUNTRY_SELF_DAMP, (
        f"self-referential ({cc} account talking only about {cc}, no policy flag)")


def is_endorsement(text):
    """True when the post is a political endorsement / ceremonial message."""
    t = str(text or "")
    return any(r.search(t) for r in _ENDORSEMENT_RE)


def endorsement_damp(texts):
    """Vectorised multiplier: ENDORSEMENT_DAMP for endorsements, else 1.0."""
    return np.array([ENDORSEMENT_DAMP if is_endorsement(t) else 1.0
                     for t in texts], dtype=float)
FALLBACK_FIN_KW    = CONFIG["fallback_financial_keywords"]
FALLBACK_NOISE_KW  = CONFIG["fallback_noise_keywords"]

POLICY_COLS = [f for f in POLICY_FLAGS.keys() if f != "flag_pandemic_relief"]


# ========================================================= policy flags ----
def extract_policy_flags(text):
    t = str(text).lower()
    return {flag: int(any(kw in t for kw in kws))
            for flag, kws in POLICY_FLAGS.items()}


# =========================================================== spaCy / NER ----
def load_spacy():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("  ✅ spaCy loaded (en_core_web_sm)")
            return nlp
        except OSError:
            print("  📦 Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"],
                           capture_output=True)
            return spacy.load("en_core_web_sm")
    except ImportError:
        print("  ⚠️  spaCy not installed — NER features will be zero")
        return None


_NER_KEYS = [
    "num_policy_verbs", "num_policy_nouns", "num_gpe", "num_org",
    "num_percent", "num_money", "num_date", "num_law", "num_person",
    "num_cardinal", "num_event", "num_geopolitical_terms", "num_all_caps_words",
]


def extract_ner_features(text, nlp):
    if nlp is None:
        return {k: 0 for k in _NER_KEYS}
    try:
        doc = nlp(str(text))
        return {
            "num_policy_verbs":       sum(1 for t in doc if t.pos_ == "VERB" and t.lemma_.lower() in POLICY_VERBS),
            "num_policy_nouns":       sum(1 for t in doc if t.pos_ == "NOUN" and t.lemma_.lower() in POLICY_VERBS),
            "num_gpe":                sum(1 for e in doc.ents if e.label_ == "GPE"),
            "num_org":                sum(1 for e in doc.ents if e.label_ == "ORG"),
            "num_percent":            sum(1 for e in doc.ents if e.label_ == "PERCENT"),
            "num_money":              sum(1 for e in doc.ents if e.label_ == "MONEY"),
            "num_date":               sum(1 for e in doc.ents if e.label_ == "DATE"),
            "num_law":                sum(1 for e in doc.ents if e.label_ == "LAW"),
            "num_person":             sum(1 for e in doc.ents if e.label_ == "PERSON"),
            "num_cardinal":           sum(1 for e in doc.ents if e.label_ == "CARDINAL"),
            "num_event":              sum(1 for e in doc.ents if e.label_ == "EVENT"),
            "num_geopolitical_terms": sum(1 for t in doc if t.lemma_.lower() in GEOPOLITICAL_TERMS),
            "num_all_caps_words":     sum(1 for t in doc if t.text.isupper() and len(t.text) > 2),
        }
    except Exception:
        return {k: 0 for k in _NER_KEYS}


# ===================================================== composite scores ----
def compute_composite_scores(df):
    for c in POLICY_COLS + _NER_KEYS:
        if c not in df.columns:
            df[c] = 0
    df["policy_intensity_score"] = (
        df[POLICY_COLS].sum(axis=1)
        + df["num_policy_verbs"]
        + df["num_geopolitical_terms"]
    )
    df["hawkish_risk_score"] = (
        df.get("flag_tariff_trade", 0) + df.get("flag_sanctions", 0)
        + df.get("flag_war_geopolitics", 0) + df.get("flag_supply_chain", 0)
        + df.get("flag_armed_conflict", 0)  # v2.1: invasion/incursion = war escalation
        + df["num_geopolitical_terms"]
    )
    df["growth_policy_score"] = (
        df.get("flag_stimulus", 0) + df.get("flag_energy_policy", 0)
        + df.get("flag_industrial_policy", 0) + df.get("flag_deregulation", 0)
    )
    # MACRO/NON-WAR market impact — the counterpart to hawkish_risk_score.
    # Without it, crypto/COVID/Fed/banking posts scored hawkish=0 and died at
    # every hawkish-anchored formula (sample_weight, nlp_signal, predict gate):
    # all 76 crypto-flagged posts averaged sample_weight 0.10, zero reached
    # HIGH_SIGNAL — Trump's July-2019 bitcoin tweet (BTC moved +0.73%) was
    # gated to nothing. Weights favour the single-flag domains that must be
    # able to clear the gate alone.
    # crypto/health are SINGLE binary flags while hawkish accumulates counts
    # (several flags + NER geo terms) — x4 puts one fired domain flag on the
    # same scale as an active war post, so a pure crypto/COVID bombshell can
    # clear the gate alone.
    df["macro_risk_score"] = (
        4 * df.get("flag_crypto_policy", 0) + 4 * df.get("flag_public_health", 0)
        # v2.1: exogenous shocks are single-flag risk-off domains (safe-haven /
        # supply disruption / repatriation) — x4 like crypto/health so a pure
        # disaster or terror post clears the gate ALONE instead of dying at
        # hawkish=0 (same fix the crypto/COVID note above documents).
        + 4 * df.get("flag_natural_disaster", 0) + 4 * df.get("flag_terror_attack", 0)
        + 2 * df.get("flag_interest_rate", 0) + 2 * df.get("flag_financial_system", 0)
        + df.get("flag_stimulus", 0) + df.get("flag_tax_policy", 0)
        + df.get("flag_energy_policy", 0) + df.get("flag_ai_chip_policy", 0)
    )
    return df


# ================================================== semantic embedding ----
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        print("  ✅ sentence-transformers loaded")
        return m
    except ImportError:
        print("  ⚠️  sentence-transformers not installed — keyword fallback")
        return None


def embedding_score(texts, model):
    if model is None or cosine_similarity is None:
        scores = []
        for text in texts:
            t = str(text).lower()
            fin = sum(1 for kw in FALLBACK_FIN_KW if kw in t)
            nse = sum(1 for kw in FALLBACK_NOISE_KW if kw in t)
            scores.append(np.clip((fin - nse * 0.5) / 5.0, 0.0, 1.0))
        return np.array(scores)
    te = model.encode(list(texts), batch_size=32, show_progress_bar=False)
    fe = model.encode(FINANCIAL_REFS, show_progress_bar=False)
    ne = model.encode(NOISE_REFS, show_progress_bar=False)
    return cosine_similarity(te, fe).max(axis=1) - cosine_similarity(te, ne).max(axis=1)


def novelty_score(texts, model, window=10):
    if model is None or cosine_similarity is None:
        scores = np.ones(len(texts))
        for i in range(1, len(texts)):
            start = max(0, i - window)
            curr = set(str(texts[i]).lower().split())
            mx = 0.0
            for j in range(start, i):
                prev = set(str(texts[j]).lower().split())
                u = len(curr | prev)
                if u > 0:
                    mx = max(mx, len(curr & prev) / u)
            scores[i] = 1.0 - mx
        return scores
    emb = model.encode(list(texts), batch_size=32, show_progress_bar=False)
    scores = np.ones(len(texts))
    for i in range(1, len(texts)):
        start = max(0, i - window)
        scores[i] = 1.0 - cosine_similarity(emb[i].reshape(1, -1), emb[start:i])[0].max()
    return scores


def burst_position_score(df, time_col="date", window_minutes=30):
    """Crowding penalty — fewer points when many posts cluster in the prior
    `window_minutes`. Computed in DuckDB with a RANGE window (one pass,
    O(n log n)) instead of the old O(n²) per-post pandas scan (~11 min -> <1s on
    190k). n = #posts STRICTLY before this one within the window = count over
    [t-w, t] minus the peers at exactly t. EXACT match to the old logic."""
    n = len(df)
    if n == 0:
        return np.ones(0)
    import duckdb
    con = duckdb.connect(); con.execute("SET TimeZone='UTC'")
    con.register("_t", pd.DataFrame({"rid": np.arange(n),
                                     "date": pd.to_datetime(df[time_col], utc=True)}))
    q = (f"SELECT greatest(0.05, 1.0/(1.0 + (cnt_range - cnt_same)*0.4)) AS s FROM ("
         f" SELECT rid,"
         f"  count(*) OVER (ORDER BY date RANGE BETWEEN INTERVAL '{int(window_minutes)} minutes' PRECEDING AND CURRENT ROW) AS cnt_range,"
         f"  count(*) OVER (PARTITION BY date) AS cnt_same"
         f" FROM _t) ORDER BY rid")
    out = con.execute(q).fetchnumpy()["s"]
    con.close()
    return np.asarray(out, dtype=float)


def relative_signal_strength(df, score_col="raw_score", window_hours=2):
    """Each post's raw_score vs the MEDIAN score in a ±`window_hours` window.
    Computed in DuckDB with a RANGE window median (one pass) instead of the old
    O(n²) per-post pandas mask+median. EXACT match to the old logic."""
    n = len(df)
    if n == 0:
        return np.ones(0)
    import duckdb
    con = duckdb.connect(); con.execute("SET TimeZone='UTC'")
    con.register("_t", pd.DataFrame({"rid": np.arange(n),
                                     "date": pd.to_datetime(df["date"], utc=True),
                                     "v":    df[score_col].astype(float).values}))
    q = (f"SELECT CASE WHEN med > 0.01 THEN least(5.0, greatest(0.0, v/med)) ELSE 1.0 END AS s FROM ("
         f" SELECT rid, v, median(v) OVER (ORDER BY date"
         f"   RANGE BETWEEN INTERVAL '{int(window_hours)} hours' PRECEDING"
         f"   AND INTERVAL '{int(window_hours)} hours' FOLLOWING) AS med"
         f" FROM _t) ORDER BY rid")
    out = con.execute(q).fetchnumpy()["s"]
    con.close()
    return np.asarray(out, dtype=float)


# ========================================== entity / event weight helpers ----
def _apply_entity_event_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-tier sample_weight assignment:

      rank-0 TruthSocial (is_primary=True):
          clip(base_sw, 0.05, 1.0)           — no discount, full ceiling

      secondary TruthSocial (is_primary=False, platform='truthsocial'):
          clip(base_sw × ew × evw, 0.05, 1.0) — entity_weight scales it down,
          but no SOURCE_DISCOUNT (still on the verified TruthSocial platform)

      X/Twitter geo (is_primary=False, platform='x_twitter'):
          clip(base_sw × ew × evw × SOURCE_DISCOUNT, 0.02, 0.70)
          — SOURCE_DISCOUNT=0.7 ensures geo posts never exceed 70% of primary ceiling

    Rank-0 always outweighs any other account: a perfect rank-0 post reaches 1.0,
    a perfect secondary-TS post reaches entity_weight (≤1.0), a perfect geo post 0.70.
    """
    if "is_primary" in df.columns:
        is_primary = df["is_primary"].astype(bool).values
    else:
        _ar = df["account_rank"].fillna(99).astype(float) if "account_rank" in df.columns else pd.Series(99.0, index=df.index)
        is_primary = ((df["platform"] == "truthsocial") & (_ar == 0)).values

    # secondary TruthSocial = on TS platform but NOT rank-0
    is_ts_secondary = (~is_primary) & (df["platform"].fillna("") == "truthsocial").values

    ew  = df["entity_weight"].fillna(1.0).astype(float).values
    evw = df["event_weight"].fillna(1.0).astype(float).values
    base = df["sample_weight"].values.copy()

    primary_sw  = np.clip(base,                              0.05, PRIMARY_WEIGHT_CAP)
    ts_sec_sw   = np.clip(base * ew * evw,                  0.05, PRIMARY_WEIGHT_CAP)
    geo_sw      = np.clip(base * ew * evw * SOURCE_DISCOUNT, 0.02, GEO_WEIGHT_CAP)

    df["sample_weight"] = np.where(is_primary,     primary_sw,
                          np.where(is_ts_secondary, ts_sec_sw, geo_sw))
    return df


# ========================================= single-post scoring (predict) ----
def score_single_post(text, nlp=None, sbert=None, feature_cols=None,
                      entity_weight: float = 1.0,
                      event_weight:  float = 1.0,
                      is_primary:    bool  = True,
                      account:       str   = None):
    """
    Score ONE post → dict of all numeric features.
    Used by prediction scripts (they share the exact training config).
    Context-dependent features (burst, novelty, relative) default to neutral 1.0.

    entity_weight / event_weight / is_primary let the caller supply the post's
    geo-context so sample_weight is accurate even for prediction-path geo posts.
    Defaults (1.0, 1.0, True) match rank-0 primary account (US President / TruthSocial).
    """
    norm_div = CONFIG.get("norm_divisors",
                          {"policy_intensity_score": 8.0,
                           "hawkish_risk_score": 5.0,
                           "growth_policy_score": 4.0})

    tc  = re.sub(r"https?://\S+", "", str(text)).strip()
    row = {
        "date": pd.Timestamp.now(tz="UTC"),
        "text": text,
        "favorites": 0, "retweets": 0, "replies": 0,
        "is_primary": is_primary,
        "platform":   "truthsocial" if is_primary else "x_twitter",
        "entity_weight": entity_weight,
        "event_weight":  event_weight,
    }
    row.update(extract_policy_flags(tc))
    row.update(extract_ner_features(tc, nlp))

    df = pd.DataFrame([row])
    df = compute_composite_scores(df)

    try:
        df["score_embedding"] = embedding_score([tc], sbert)
    except Exception:
        df["score_embedding"] = 0.0

    for c, default in [("score_novelty", 1.0), ("score_burst", 1.0),
                       ("score_caps", 0.0),    ("score_relative", 1.0)]:
        if c not in df.columns:
            df[c] = default

    out = df.iloc[0].to_dict()

    for base, div in norm_div.items():
        out[f"{base}_norm"] = min(out.get(base, 0.0) / div, 1.0) if div else 0.0

    pi_div = max(norm_div.get("policy_intensity_score", 8.0), 1)
    out["score_policy"] = min(out.get("policy_intensity_score", 0.0) / pi_div, 1.0)
    out["raw_score"] = float(np.clip(
        0.4 * out["score_policy"]
        + 0.3 * float(np.clip(out.get("score_embedding", 0.0), 0, 1))
        + 0.2 * out.get("score_novelty", 1.0)
        + 0.1 * out.get("score_burst",   1.0),
        0, 1))
    # COUNTRY CONTEXT — must run BEFORE the endorsement gate, exactly as in
    # score_batch, otherwise a geo post is scored one way in training and a
    # different way live. `account` resolves the speaker's country; without it
    # the damp cannot apply and the post is left undamped (same as batch when
    # the handle is unknown).
    _cf, _ = country_context_factor(
        tc, country_of(account),
        {f: out.get(f, 0.0) for f in COUNTRY_POLICY_GUARD} or None,
        is_primary=is_primary)
    out["is_self_country"] = int(_cf < 1.0)
    if _cf < 1.0:
        out["raw_score"] = out["raw_score"] * _cf

    # ENDORSEMENT HARD-SKIP — same rule as the batch path, so a live single
    # post and a bulk rescore agree.
    out["is_endorsement"] = int(is_endorsement(text))
    if out["is_endorsement"]:
        out["raw_score"] = out["raw_score"] * ENDORSEMENT_DAMP

    # domain risk = the STRONGER of war (hawkish) and non-war (macro: crypto/
    # covid/Fed/banking) impact — either domain can carry a post to full weight
    base_sw = float(np.clip(
        0.5 * out["raw_score"]
        + 0.3 * max(out.get("hawkish_risk_score_norm", 0.0),
                    out.get("macro_risk_score_norm", 0.0))
        + 0.2 * out.get("score_relative", 1.0) / 5.0,
        0.05, 1.0))

    if is_primary:
        out["sample_weight"] = float(np.clip(base_sw, 0.05, PRIMARY_WEIGHT_CAP))
    else:
        out["sample_weight"] = float(
            np.clip(base_sw * entity_weight * event_weight * SOURCE_DISCOUNT,
                    0.02, GEO_WEIGHT_CAP))

    if feature_cols:
        return {c: float(out.get(c, 0.0)) for c in feature_cols}
    return out


# ============================================================ shared helpers
def _prepare_feed_df(feed: pd.DataFrame) -> pd.DataFrame:
    """Normalise date, ensure required columns exist, add text_clean and is_primary."""
    feed = feed.copy()
    feed["id"]   = pd.to_numeric(feed["id"], errors="coerce").astype("Int64")
    feed["date"] = pd.to_datetime(feed["date"], format="mixed", utc=True).dt.tz_convert("America/New_York")
    feed = feed.sort_values("date").reset_index(drop=True)
    if "platform" not in feed.columns:
        feed["platform"] = "truthsocial"   # safe default
    # is_primary: used in _apply_entity_event_weight and stored in posts_scored for XGBoost
    if "is_primary" not in feed.columns:
        # is_primary = rank-0 PRIMARY ACCOUNT within its ACTIVE WINDOW.
        # Platform + time aware for election handovers/back-sim: Trump T2 is
        # rank-0 on TruthSocial, but Trump T1 / Biden / the next president may be
        # rank-0 on X in THEIR era. The SAME handle can recur on a different
        # platform in a disjoint era (realDonaldTrump: X 45th vs TruthSocial
        # 47th) — the post date + platform pick the primary.
        _acct_rank = feed["account_rank"].fillna(99).astype(float) if "account_rank" in feed.columns else pd.Series(99.0, index=feed.index)
        _wins = _rank0_windows()
        _acc_l = (feed["account"].fillna("").astype(str).str.lstrip("@").str.lower()
                  if "account" in feed.columns else pd.Series("", index=feed.index))
        _plat_l = (feed["platform"].fillna("").astype(str).str.strip().str.lower()
                   if "platform" in feed.columns else pd.Series("", index=feed.index))
        _in_win = pd.Series(False, index=feed.index)
        for (_h, _p), (_lo, _hi) in _wins.items():
            _m = (_acc_l == _h)
            if _p is not None:                       # same handle, different platform/era
                _m &= (_plat_l == _p)
            if not _m.any():
                continue
            if _lo is not None:
                _m &= (feed["date"] >= _lo)
            if _hi is not None:
                _m &= (feed["date"] <= _hi)
            _in_win |= _m
        if _wins:
            feed["is_primary"] = (_acct_rank == 0) & _in_win
        else:   # no windows declared — legacy behaviour
            feed["is_primary"] = (feed["platform"] == "truthsocial") & (_acct_rank == 0)
    if "entity_weight" not in feed.columns: feed["entity_weight"] = 1.0
    if "event_weight"  not in feed.columns: feed["event_weight"]  = 1.0
    feed["text_clean"] = feed["text"].fillna("").apply(
        lambda t: re.sub(r"https?://\S+", "", str(t)).strip()
    )
    return feed


def _score_batch(batch: pd.DataFrame, nlp, sbert,
                 ctx_texts: list = None, ctx_dates: list = None,
                 ctx_raw:   list = None, novelty_window: int = 10
                 ) -> pd.DataFrame:
    """
    Apply all 7 scoring layers to `batch`. ctx_* are the preceding context
    for novelty / burst / relative. If None, context = batch itself (full mode).
    Returns batch with all score columns added in-place.
    """
    norm_div = CONFIG.get("norm_divisors",
                          {"policy_intensity_score": 8.0,
                           "hawkish_risk_score": 5.0,
                           "growth_policy_score": 4.0})

    texts = batch["text_clean"].fillna("").tolist()

    # — Layer 1: Policy flags —
    flags = pd.DataFrame([extract_policy_flags(t) for t in texts], index=batch.index)
    for c in flags.columns:
        batch[c] = flags[c].values

    # — Layer 2: NER —
    ner = pd.DataFrame([extract_ner_features(t, nlp) for t in texts], index=batch.index)
    for c in ner.columns:
        batch[c] = ner[c].values

    # — Layer 3: Composites —
    batch = compute_composite_scores(batch)

    # — Layer 4: Embedding —
    batch["score_embedding"] = embedding_score(texts, sbert)

    # — Layers 5 & 6: Novelty + Burst (context-aware) —
    if ctx_texts is None:
        # Full-rebuild mode: use the batch itself as its own context
        batch["score_novelty"] = novelty_score(texts, sbert, window=novelty_window)
        batch["score_burst"]   = burst_position_score(batch)
    else:
        # Incremental mode: prepend context
        n_ctx      = len(ctx_texts)
        all_texts  = ctx_texts + texts
        all_dates  = ctx_dates + list(batch["date"])

        nov = np.ones(len(texts))
        if sbert is not None and cosine_similarity is not None:
            emb = sbert.encode(all_texts, batch_size=32, show_progress_bar=False)
            for k in range(len(texts)):
                i     = n_ctx + k
                start = max(0, i - novelty_window)
                if i > start:
                    nov[k] = 1.0 - cosine_similarity(
                        emb[i].reshape(1, -1), emb[start:i]
                    )[0].max()
        else:
            for k in range(len(texts)):
                i     = n_ctx + k
                start = max(0, i - novelty_window)
                curr  = set(str(all_texts[i]).lower().split())
                mx    = 0.0
                for j in range(start, i):
                    prev = set(str(all_texts[j]).lower().split())
                    u    = len(curr | prev)
                    if u > 0:
                        mx = max(mx, len(curr & prev) / u)
                nov[k] = 1.0 - mx
        batch["score_novelty"] = nov

        dser = pd.Series(pd.to_datetime(all_dates))
        burst = np.ones(len(texts))
        for k in range(len(texts)):
            i         = n_ctx + k
            win_start = dser.iloc[i] - pd.Timedelta(minutes=30)
            n         = ((dser >= win_start) & (dser < dser.iloc[i])).sum()
            burst[k]  = max(0.05, 1.0 / (1.0 + n * 0.4))
        batch["score_burst"] = burst

    # — caps score —
    caps_max = batch["num_all_caps_words"].max()
    batch["score_caps"] = batch["num_all_caps_words"] / caps_max if caps_max > 0 else 0.0

    # — raw_score —
    pi_div = max(norm_div.get("policy_intensity_score", 8.0), 1)
    batch["score_policy"] = np.clip(batch["policy_intensity_score"] / pi_div, 0, 1)
    batch["raw_score"]    = np.clip(
        0.4 * batch["score_policy"]
        + 0.3 * np.clip(batch["score_embedding"], 0, 1)
        + 0.2 * batch["score_novelty"]
        + 0.1 * batch["score_burst"],
        0, 1)
    # COUNTRY CONTEXT — self-referential posts (a CN account talking only
    # about CN, no policy flag) are routine, not signal. Applied before the
    # endorsement gate so both land on raw_score.
    if _COUNTRY_RE and "account" in batch:
        _accs = list(batch["account"])
        # The guard flags are written by Layer 1 above. If a future refactor
        # reorders the layers they silently vanish, `feats` goes empty, and
        # every real policy post gets damped as "self-referential". Fail loud.
        _missing = [f for f in COUNTRY_POLICY_GUARD if f not in batch]
        if COUNTRY_POLICY_GUARD and _missing:
            raise RuntimeError(
                f"country_policy_guard flags missing from batch: {_missing}. "
                "Layer 1 (policy flags) must run before the country damp, "
                "otherwise genuine policy posts are damped as self-referential.")
        # positional lookup: batch.index may not be a clean RangeIndex
        _gv = {f: pd.to_numeric(batch[f], errors="coerce").fillna(0.0).values
               for f in COUNTRY_POLICY_GUARD}
        _prim = (batch["is_primary"].astype(bool).values
                 if "is_primary" in batch else np.zeros(len(texts), dtype=bool))
        _cf, _n_self, _by = [], 0, {}
        for _i, _t in enumerate(texts):
            _cc = country_of(_accs[_i] if _i < len(_accs) else None)
            _feats = ({f: _gv[f][_i] for f in COUNTRY_POLICY_GUARD}
                      if COUNTRY_POLICY_GUARD else None)
            _f, _ = country_context_factor(_t, _cc, _feats,
                                          is_primary=bool(_prim[_i]))
            _cf.append(_f)
            if _f < 1.0:
                _n_self += 1
                _by[_cc] = _by.get(_cc, 0) + 1
        if _n_self:
            batch["raw_score"] = batch["raw_score"] * np.array(_cf)
            print(f"  🌏 {_n_self}/{len(texts)} self-referential post(s) "
                  f"×{COUNTRY_SELF_DAMP} (own country only, no policy flag): "
                  + ", ".join(f"{k}={v}" for k, v in
                              sorted(_by.items(), key=lambda x: -x[1])[:6]))
        batch["is_self_country"] = (np.array(_cf) < 1.0).astype(int)

    # ENDORSEMENT HARD-SKIP — applied to raw_score so it propagates into
    # score_relative, sample_weight and everything downstream.
    _end = endorsement_damp(texts)
    _n_end = int((_end < 1.0).sum())
    if _n_end:
        batch["raw_score"] = batch["raw_score"] * _end
        print(f"  🚫 {_n_end}/{len(texts)} endorsement/ceremonial post(s) "
              f"zeroed at scoring ({_n_end / max(len(texts), 1):.1%}) — "
              f"they never reach training, embeddings or the analyst")
    batch["is_endorsement"] = (_end < 1.0).astype(int)

    # — Layer 7: Relative signal strength —
    if ctx_raw is None:
        batch["score_relative"] = relative_signal_strength(batch)
    else:
        all_raw  = np.array(ctx_raw + list(batch["raw_score"]))
        dser_rel = pd.Series(pd.to_datetime(all_dates if ctx_texts else list(batch["date"])))
        rel      = np.ones(len(texts))
        for k in range(len(texts)):
            i  = (len(ctx_raw) if ctx_raw else 0) + k
            lo = dser_rel.iloc[i] - pd.Timedelta(hours=2)
            hi = dser_rel.iloc[i] + pd.Timedelta(hours=2)
            mask      = (dser_rel >= lo) & (dser_rel <= hi)
            neighbors = all_raw[mask.values]
            med       = np.median(neighbors) if len(neighbors) else all_raw[i]
            rel[k]    = all_raw[i] / med if med > 0.01 else 1.0
        batch["score_relative"] = np.clip(rel, 0.0, 5.0)

    # — Normalized composites (stable config divisors) —
    for c in ["policy_intensity_score", "hawkish_risk_score", "growth_policy_score",
              "macro_risk_score"]:
        div = norm_div.get(c, 0)
        if c not in batch.columns:
            batch[c] = 0
        batch[f"{c}_norm"] = np.clip(batch[c] / div, 0, 1) if div else 0.0

    # — Base sample_weight then entity/event discount —
    # domain risk = the STRONGER of war (hawkish) and non-war (macro) impact:
    # crypto/COVID/Fed/banking posts previously had hawkish=0 and could NEVER
    # earn training trust regardless of content (all 76 crypto posts sw~0.10).
    base_sw = np.clip(
        0.5 * batch["raw_score"]
        + 0.3 * np.maximum(batch["hawkish_risk_score_norm"],
                           batch["macro_risk_score_norm"])
        + 0.2 * batch["score_relative"] / 5.0,
        0.05, 1.0)
    batch["sample_weight"] = base_sw
    batch = _apply_entity_event_weight(batch)

    return batch


# ============================================================ full rebuild ----
def main():
    print("=" * 64)
    print("  UNIFIED POST SIGNAL SCORER  —  FULL REBUILD")
    print(f"  Config: {CONFIG_PATH}  (version {CONFIG.get('version', '?')})")
    print(f"  {FEED_TABLE}  →  {SCORED_TABLE}")
    print("=" * 64)


    feed = db.read_table(FEED_TABLE)
    if feed is None:
        print(f"❌ {FEED_TABLE} not found — run sync_unified_feed.py first")
        sys.exit(1)

    feed = _prepare_feed_df(feed)
    n_primary = int(feed["is_primary"].sum()) if "is_primary" in feed.columns else 0
    n_twitter = len(feed) - n_primary
    print(f"\n📂 {FEED_TABLE}: {len(feed)} posts  "
          f"(TruthSocial: {n_primary} | X/Twitter: {n_twitter})")

    print("\n📦 Loading NLP models...")
    nlp   = load_spacy()
    sbert = load_sbert()

    print("\n🚀 Scoring all layers across unified feed (chronological)...")
    feed = _score_batch(feed, nlp, sbert)   # full-rebuild: no ctx

    db.write_table(SCORED_TABLE, feed.drop(columns=["text_clean"], errors="ignore"))

    primary_df = feed[feed["is_primary"].astype(bool)] if "is_primary" in feed.columns else feed[feed["platform"] == "truthsocial"]
    geo_df     = feed[~feed["is_primary"].astype(bool)] if "is_primary" in feed.columns else feed[feed["platform"] == "x_twitter"]
    print(f"\n💾 Saved {SCORED_TABLE} ({len(feed)} rows) → {db.DB_PATH}")
    print(f"   TruthSocial  : {len(primary_df)}  "
          f"sw_mean={primary_df['sample_weight'].mean():.3f}  "
          f">0.3: {(primary_df['raw_score'] > 0.3).sum()}")
    if len(geo_df):
        print(f"   X/Twitter    : {len(geo_df)}  "
              f"sw_mean={geo_df['sample_weight'].mean():.3f}  "
              f">0.3: {(geo_df['raw_score'] > 0.3).sum()}")


# =========================================================== incremental ----
def score_incremental(context_days: int = 3, novelty_window: int = 10):
    """
    Daily mode: score only posts in unified_feed not yet in posts_scored,
    then APPEND them. Context features (novelty / burst / relative) are computed
    against the recent already-scored timeline so they're comparable to a full
    rebuild.
    """
    print("=" * 64)
    print("  UNIFIED POST SIGNAL SCORER  —  INCREMENTAL (daily)")
    print(f"  Config: {CONFIG_PATH}  (version {CONFIG.get('version', '?')})")
    print("=" * 64)


    feed = db.read_table(FEED_TABLE)
    if feed is None:
        print(f"❌ {FEED_TABLE} not found — run sync_unified_feed.py first")
        return 0

    feed = _prepare_feed_df(feed)

    scored = db.read_table(SCORED_TABLE)
    if scored is None:
        print("⚠️  No existing posts_scored — running full scoring instead.")
        return main()

    scored["id"] = pd.to_numeric(scored["id"], errors="coerce").astype("Int64")
    existing_keys = set(zip(scored["platform"], scored["id"].astype(str)))
    feed_keys = list(zip(feed["platform"], feed["id"].astype(str)))

    new = feed[[k not in existing_keys for k in feed_keys]].copy().reset_index(drop=True)

    n_primary_new = int(new["is_primary"].sum()) if (len(new) and "is_primary" in new.columns) else 0
    n_twitter_new = len(new) - n_primary_new
    print(f"\n📂 unified_feed: {len(feed)} | already scored: {len(existing_keys)} "
          f"| new: {len(new)} (TruthSocial: {n_primary_new} | X/Twitter: {n_twitter_new})")

    if new.empty:
        print("✅ posts_scored is up to date — nothing new to score.")
        return 0

    # ---- recent context from already-scored timeline ----
    scored_cols = list(scored.columns)
    ctx      = scored.copy()
    ctx["date"] = pd.to_datetime(ctx["date"], format="mixed", utc=True).dt.tz_convert("America/New_York")
    ctx      = ctx.sort_values("date").reset_index(drop=True)
    cutoff   = new["date"].min() - pd.Timedelta(days=context_days)
    ctx_recent = ctx[ctx["date"] >= cutoff]
    if len(ctx_recent) < novelty_window:
        ctx_recent = ctx.tail(novelty_window)

    ctx_texts = ctx_recent["text"].fillna("").astype(str).apply(
                    lambda t: re.sub(r"https?://\S+", "", t).strip()).tolist()
    ctx_dates = list(ctx_recent["date"])
    ctx_raw   = list(ctx_recent["raw_score"].astype(float))
    print(f"⏰  Context: {len(ctx_recent)} recent scored posts "
          f"(since {cutoff:%Y-%m-%d %H:%M})")

    print("\n📦 Loading NLP models...")
    nlp   = load_spacy()
    sbert = load_sbert()

    print("\n🚀 Scoring new posts (all 7 layers, incremental context)...")
    new = _score_batch(new, nlp, sbert,
                       ctx_texts=ctx_texts, ctx_dates=ctx_dates,
                       ctx_raw=ctx_raw, novelty_window=novelty_window)

    # Align to existing column order (fill any missing with 0) while KEEPING
    # any NEW feature columns — e.g. a flag the daily LLM added to
    # scorer_config.json this morning. db.append_table evolves the table
    # schema (ALTER ... DEFAULT 0), so every historical post reads false/0
    # for the new flag with NO full re-score.
    for c in scored_cols:
        if c not in new.columns:
            new[c] = 0
    extra = [c for c in new.columns if c not in scored_cols and c != "text_clean"]
    if extra:
        print(f"  🧬 New feature column(s) from config: {extra[:8]}"
              f"{'...' if len(extra) > 8 else ''} — old posts default to 0/false")
    out = new.reindex(columns=scored_cols + extra, fill_value=0)

    db.append_table(SCORED_TABLE, out.drop(columns=["text_clean"], errors="ignore"))

    primary_new = new[new["is_primary"].astype(bool)] if "is_primary" in new.columns else new[new["platform"] == "truthsocial"]
    geo_new     = new[~new["is_primary"].astype(bool)] if "is_primary" in new.columns else new[new["platform"] == "x_twitter"]
    total_now   = len(scored) + len(out)
    print(f"\n💾 Appended {len(out)} → {SCORED_TABLE} ({total_now} total)")
    if len(primary_new):
        print(f"   TruthSocial : {len(primary_new)}  sw_mean={primary_new['sample_weight'].mean():.3f}")
    if len(geo_new):
        print(f"   X/Twitter   : {len(geo_new)}  sw_mean={geo_new['sample_weight'].mean():.3f}")
    return len(out)


# ================================================================= CLI ----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Score unified feed (daily incremental by default).")
    ap.add_argument("--full", action="store_true",
                    help="Full rebuild: re-score ALL posts, overwrite posts_scored.")
    args = ap.parse_args()
    if args.full:
        main()
    else:
        score_incremental()