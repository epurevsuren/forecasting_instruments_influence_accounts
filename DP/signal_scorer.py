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
sync_unified_feed.py, which reads geopolitical_entities.json + events.json).

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

_ENTITIES_FILE = os.path.join(_HERE, "geopolitical_entities.json")


def _rank0_handle() -> str:
    """Return the rank-0 TruthSocial account handle from geopolitical_entities.json.
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
        + df["num_geopolitical_terms"]
    )
    df["growth_policy_score"] = (
        df.get("flag_stimulus", 0) + df.get("flag_energy_policy", 0)
        + df.get("flag_industrial_policy", 0) + df.get("flag_deregulation", 0)
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
    scores = np.ones(len(df))
    dates = pd.to_datetime(df[time_col])
    for i in range(len(df)):
        win_start = dates.iloc[i] - pd.Timedelta(minutes=window_minutes)
        n = ((dates >= win_start) & (dates < dates.iloc[i])).sum()
        scores[i] = max(0.05, 1.0 / (1.0 + n * 0.4))
    return scores


def relative_signal_strength(df, score_col="raw_score", window_hours=2):
    scores = np.ones(len(df))
    dates = pd.to_datetime(df["date"])
    vals  = df[score_col].values
    for i in range(len(df)):
        lo = dates.iloc[i] - pd.Timedelta(hours=window_hours)
        hi = dates.iloc[i] + pd.Timedelta(hours=window_hours)
        mask = (dates >= lo) & (dates <= hi)
        neighbors = vals[mask.values]
        med = np.median(neighbors) if len(neighbors) else vals[i]
        scores[i] = vals[i] / med if med > 0.01 else 1.0
    return np.clip(scores, 0.0, 5.0)


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
                      is_primary:    bool  = True):
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

    base_sw = float(np.clip(
        0.5 * out["raw_score"]
        + 0.3 * out.get("hawkish_risk_score_norm", 0.0)
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
        # is_primary = rank-0 TruthSocial ONLY (account_rank == 0).
        # Secondary TruthSocial accounts (rank > 0, e.g. a future candidate)
        # are NOT is_primary — they get entity_weight scaling without SOURCE_DISCOUNT.
        _acct_rank = feed["account_rank"].fillna(99).astype(float) if "account_rank" in feed.columns else pd.Series(99.0, index=feed.index)
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
    for c in ["policy_intensity_score", "hawkish_risk_score", "growth_policy_score"]:
        div = norm_div.get(c, 0)
        batch[f"{c}_norm"] = np.clip(batch[c] / div, 0, 1) if div else 0.0

    # — Base sample_weight then entity/event discount —
    base_sw = np.clip(
        0.5 * batch["raw_score"]
        + 0.3 * batch["hawkish_risk_score_norm"]
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

    # Align to existing column order (fill any missing with 0)
    for c in scored_cols:
        if c not in new.columns:
            new[c] = 0
    out = new.reindex(columns=scored_cols, fill_value=0)

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