"""
signal_scorer.py  —  CONFIG-DRIVEN (dynamic)
--------------------------------------------
All keyword lists, policy flags, geopolitical terms, and semantic reference
sentences now live in scorer_config.json — NOT hardcoded. Edit the JSON to
tune the scorer; no code changes needed.

CRITICAL: training and prediction both import this module, so they use the
EXACT same config → features always align. If you change scorer_config.json,
re-run scoring for both training data and predictions.

Public API (importable):
  CONFIG                      - the loaded dict
  extract_policy_flags(text)  - 14 binary flags from config
  extract_ner_features(text, nlp)
  compute_composite_scores(df)
  load_spacy() / load_sbert()
  embedding_score / novelty_score / burst_position_score / relative_signal_strength
  score_single_post(text, nlp=None, sbert=None) -> dict  (for prediction)
  main()                      - FULL batch-score trump_truths.csv -> trump_truths_scored.csv
  score_incremental()         - score only NEW posts and APPEND to trump_truths_scored.csv

CLI:
  python signal_scorer.py            -> incremental (daily mode: only new ids, append)
  python signal_scorer.py --full     -> full rebuild (rewrites trump_truths_scored.csv)

DAILY PIPELINE: run_daily_truths.bat runs daily_truths_retriever.py first (appends
new posts to trump_truths.csv), then this script in incremental mode (scores the
new posts and appends them to trump_truths_scored.csv). Context-dependent features
(novelty / burst / relative) are computed against the recent already-scored
timeline, so they match what a full rebuild would produce. Normalized scores
(score_policy, *_norm) use the fixed norm_divisors from scorer_config.json — the
same divisors the prediction path uses — so they don't drift with batch size.
"""
import os, re, json, sys
import numpy as np
import pandas as pd

# Windows: when stdout is redirected to a file (e.g. >> signal_scorer.log) Python
# defaults to cp1252, which can't encode the emoji in our log lines. Force UTF-8.
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

# ==========================================
# LOAD CONFIG (the dynamic part)
# ==========================================
# This is the ONE canonical scorer. The Finbert_NLP_XGBoost copy is now a thin
# shim that imports this file, so there is a single source of truth and the two
# folders can never drift apart again.
#
# Paths resolve relative to THIS file (not the current working dir), so the
# module behaves identically whether it's run from DP/, from
# Finbert_NLP_XGBoost/, or imported from anywhere. Override with SCORER_CONFIG.
_HERE = os.path.dirname(os.path.abspath(__file__))

def _resolve_config_path():
    env = os.environ.get("SCORER_CONFIG")
    if env:
        return env
    candidates = [
        os.path.join(_HERE, "scorer_config.json"),            # next to this file (DP/)
        os.path.join(_HERE, "..", "DP", "scorer_config.json"),# from a sibling folder
        "scorer_config.json",                                 # cwd fallback
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
            f"❌ {path} not found. The scorer is config-driven — "
            f"scorer_config.json must be in DP/ (or set SCORER_CONFIG)."
        )
    with open(path, encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()

POLICY_VERBS       = set(CONFIG["policy_verbs"])
GEOPOLITICAL_TERMS = set(CONFIG["geopolitical_terms"])
POLICY_FLAGS       = CONFIG["policy_flags"]            # dict: flag_name -> [keywords]
FINANCIAL_REFS     = CONFIG["financial_refs"]
NOISE_REFS         = CONFIG["noise_refs"]
FALLBACK_FIN_KW    = CONFIG["fallback_financial_keywords"]
FALLBACK_NOISE_KW  = CONFIG["fallback_noise_keywords"]

# Composite-score groupings (which flags roll into each composite)
POLICY_COLS = [f for f in POLICY_FLAGS.keys() if f != 'flag_covid_relief']


# ==========================================
# POLICY FLAGS — now driven by config dict
# ==========================================
def extract_policy_flags(text):
    t = str(text).lower()
    return {flag: int(any(kw in t for kw in kws))
            for flag, kws in POLICY_FLAGS.items()}


# ==========================================
# spaCy / NER
# ==========================================
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
            subprocess.run(["python","-m","spacy","download","en_core_web_sm"],
                           capture_output=True)
            return spacy.load("en_core_web_sm")
    except ImportError:
        print("  ⚠️  spaCy not installed — NER features will be zero")
        return None


_NER_KEYS = ['num_policy_verbs','num_policy_nouns','num_gpe','num_org','num_percent',
             'num_money','num_date','num_law','num_person','num_cardinal','num_event',
             'num_geopolitical_terms','num_all_caps_words']

def extract_ner_features(text, nlp):
    if nlp is None:
        return {k: 0 for k in _NER_KEYS}
    try:
        doc = nlp(str(text))
        return {
            'num_policy_verbs':       sum(1 for t in doc if t.pos_=="VERB" and t.lemma_.lower() in POLICY_VERBS),
            'num_policy_nouns':       sum(1 for t in doc if t.pos_=="NOUN" and t.lemma_.lower() in POLICY_VERBS),
            'num_gpe':                sum(1 for e in doc.ents if e.label_=="GPE"),
            'num_org':                sum(1 for e in doc.ents if e.label_=="ORG"),
            'num_percent':            sum(1 for e in doc.ents if e.label_=="PERCENT"),
            'num_money':              sum(1 for e in doc.ents if e.label_=="MONEY"),
            'num_date':               sum(1 for e in doc.ents if e.label_=="DATE"),
            'num_law':                sum(1 for e in doc.ents if e.label_=="LAW"),
            'num_person':             sum(1 for e in doc.ents if e.label_=="PERSON"),
            'num_cardinal':           sum(1 for e in doc.ents if e.label_=="CARDINAL"),
            'num_event':              sum(1 for e in doc.ents if e.label_=="EVENT"),
            'num_geopolitical_terms': sum(1 for t in doc if t.lemma_.lower() in GEOPOLITICAL_TERMS),
            'num_all_caps_words':     sum(1 for t in doc if t.text.isupper() and len(t.text)>2),
        }
    except Exception:
        return {k: 0 for k in _NER_KEYS}


# ==========================================
# COMPOSITE SCORES
# ==========================================
def compute_composite_scores(df):
    for c in POLICY_COLS + _NER_KEYS:
        if c not in df.columns:
            df[c] = 0
    df['policy_intensity_score'] = (df[POLICY_COLS].sum(axis=1)
                                     + df['num_policy_verbs'] + df['num_geopolitical_terms'])
    df['hawkish_risk_score'] = (df.get('flag_tariff_trade',0) + df.get('flag_sanctions',0)
                                + df.get('flag_war_geopolitics',0) + df.get('flag_supply_chain',0)
                                + df['num_geopolitical_terms'])
    df['growth_policy_score'] = (df.get('flag_stimulus',0) + df.get('flag_energy_policy',0)
                                 + df.get('flag_industrial_policy',0) + df.get('flag_deregulation',0))
    return df


# ==========================================
# SEMANTIC EMBEDDING (refs from config)
# ==========================================
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer('all-MiniLM-L6-v2')
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
            scores.append(np.clip((fin - nse*0.5)/5.0, 0.0, 1.0))
        return np.array(scores)
    te = model.encode(list(texts), batch_size=32, show_progress_bar=False)
    fe = model.encode(FINANCIAL_REFS, show_progress_bar=False)
    ne = model.encode(NOISE_REFS, show_progress_bar=False)
    return cosine_similarity(te, fe).max(axis=1) - cosine_similarity(te, ne).max(axis=1)

def novelty_score(texts, model, window=10):
    if model is None or cosine_similarity is None:
        scores = np.ones(len(texts))
        for i in range(1, len(texts)):
            start = max(0, i-window)
            curr = set(str(texts[i]).lower().split())
            mx = 0.0
            for j in range(start, i):
                prev = set(str(texts[j]).lower().split())
                u = len(curr | prev)
                if u > 0: mx = max(mx, len(curr & prev)/u)
            scores[i] = 1.0 - mx
        return scores
    emb = model.encode(list(texts), batch_size=32, show_progress_bar=False)
    scores = np.ones(len(texts))
    for i in range(1, len(texts)):
        start = max(0, i-window)
        scores[i] = 1.0 - cosine_similarity(emb[i].reshape(1,-1), emb[start:i])[0].max()
    return scores

def burst_position_score(df, time_col='date', window_minutes=30):
    scores = np.ones(len(df))
    dates = pd.to_datetime(df[time_col])
    for i in range(len(df)):
        win_start = dates.iloc[i] - pd.Timedelta(minutes=window_minutes)
        n = ((dates >= win_start) & (dates < dates.iloc[i])).sum()
        scores[i] = max(0.05, 1.0/(1.0 + n*0.4))
    return scores

def relative_signal_strength(df, score_col='raw_score', window_hours=2):
    scores = np.ones(len(df))
    dates = pd.to_datetime(df['date'])
    vals = df[score_col].values
    for i in range(len(df)):
        lo = dates.iloc[i] - pd.Timedelta(hours=window_hours)
        hi = dates.iloc[i] + pd.Timedelta(hours=window_hours)
        mask = (dates >= lo) & (dates <= hi)
        neighbors = vals[mask.values]
        med = np.median(neighbors) if len(neighbors) else vals[i]
        scores[i] = vals[i] / med if med > 0.01 else 1.0
    return np.clip(scores, 0.0, 5.0)


# ==========================================
# POST CLASSIFICATION
# ==========================================
def classify_post(text):
    t = str(text).strip()
    if not t:
        return 'noise'
    bare = re.sub(r'https?://\S+', '', t).strip()
    if len(bare) < 5 and re.search(r'https?://', t):
        return 'url_only'
    if re.match(r'^RT\s', t) or t.startswith('"') and t.endswith('"'):
        return 'quote_rt'
    return 'original'

def extract_comment(row):
    return str(row['text'])


# ==========================================
# SINGLE-POST SCORING (for prediction)
# ==========================================
def score_single_post(text, nlp=None, sbert=None, feature_cols=None):
    """
    Score ONE post → dict of all numeric features.
    Used by prediction scripts so they share the exact training config.
    Context-dependent features (burst, novelty, relative) default to neutral 1.0
    since a single ad-hoc post has no surrounding timeline.
    """
    tc = re.sub(r'https?://\S+', '', str(text)).strip()
    row = {'date': pd.Timestamp.now(tz='UTC'), 'text': text, 'text_clean': tc,
           'favorites': 0, 'retweets': 0, 'replies': 0}
    row.update(extract_policy_flags(tc))
    row.update(extract_ner_features(tc, nlp))

    df = pd.DataFrame([row])
    df = compute_composite_scores(df)

    # embedding score (single post)
    try:
        df['score_embedding'] = embedding_score([tc], sbert)
    except Exception:
        df['score_embedding'] = 0.0
    # neutral defaults for context features (no timeline for a single post)
    for c, default in [('score_novelty',1.0),('score_burst',1.0),
                       ('score_caps',0.0),('score_relative',1.0)]:
        if c not in df.columns:
            df[c] = default

    out = df.iloc[0].to_dict()

    # Normalized composites: divide by training-data max (saved in config).
    norm_div = CONFIG.get("norm_divisors",
                          {"policy_intensity_score": 8.0,
                           "hawkish_risk_score": 5.0,
                           "growth_policy_score": 4.0})
    for base, div in norm_div.items():
        out[f"{base}_norm"] = min(out.get(base, 0.0) / div, 1.0) if div else 0.0

    # --- ACTUALLY COMPUTE score_policy / raw_score / sample_weight ---
    # (Previously these defaulted to 0.0, so the predict-time gate always saw
    #  signal=0. Replicate main()'s formulas using the config norm_divisor for
    #  policy_intensity so a single post gets a real signal.)
    pi_div = norm_div.get("policy_intensity_score", 8.0)
    out['score_policy'] = min(out.get('policy_intensity_score', 0.0) / max(pi_div, 1), 1.0)
    out['raw_score'] = float(np.clip(
        0.4 * out['score_policy']
        + 0.3 * float(np.clip(out.get('score_embedding', 0.0), 0, 1))
        + 0.2 * out.get('score_novelty', 1.0)
        + 0.1 * out.get('score_burst', 1.0), 0, 1))
    out['sample_weight'] = float(np.clip(
        0.5 * out['raw_score']
        + 0.3 * out.get('hawkish_risk_score_norm', 0.0)
        + 0.2 * out.get('score_relative', 1.0) / 5.0, 0.05, 1.0))

    if feature_cols:
        return {c: float(out.get(c, 0.0)) for c in feature_cols}
    return out


# ==========================================
# BATCH MAIN — score trump_truths.csv
# ==========================================
def main():
    print("="*60)
    print("  TRUMP POST SIGNAL SCORER (config-driven)")
    print(f"  Config: {CONFIG_PATH}  (version {CONFIG.get('version','?')})")
    print("  Output: trump_truths_scored.csv")
    print("="*60)

    in_csv  = os.path.join(_HERE, "trump_truths.csv")
    out_csv = os.path.join(_HERE, "trump_truths_scored.csv")
    print(f"\n📂 Loading {in_csv}...")
    df = pd.read_csv(in_csv)
    df['post_type'] = df['text'].apply(classify_post)
    print(df['post_type'].value_counts().to_string())

    clean = df[df['post_type'].isin(['original','quote_rt'])].copy()
    clean['text'] = clean.apply(extract_comment, axis=1)
    clean = clean.drop(columns=['post_type'])
    clean['date'] = pd.to_datetime(clean['date'], utc=True, format='mixed').dt.tz_convert('America/New_York')
    clean = clean.sort_values('date').reset_index(drop=True)
    print(f"\n✅ Clean posts: {len(clean)}")

    clean['text_clean'] = clean['text'].apply(lambda t: re.sub(r'https?://\S+','',str(t)).strip())
    texts = clean['text_clean'].fillna("").tolist()

    print("\n📦 Loading NLP models...")
    nlp = load_spacy()
    sbert = load_sbert()

    print("\n🚩 Layer 1: Policy flags (14 categories from config)...")
    flags = pd.DataFrame([extract_policy_flags(t) for t in texts], index=clean.index)
    for c in flags.columns: clean[c] = flags[c].values

    print("🔬 Layer 2: POS + NER features...")
    ner = pd.DataFrame([extract_ner_features(t, nlp) for t in texts], index=clean.index)
    for c in ner.columns: clean[c] = ner[c].values

    print("📊 Layer 3: Composite scores...")
    clean = compute_composite_scores(clean)

    print("🧠 Layer 4: Semantic embedding score...")
    clean['score_embedding'] = embedding_score(texts, sbert)
    print("🔍 Layer 5: Novelty score...")
    clean['score_novelty'] = novelty_score(texts, sbert)
    print("⏱️  Layer 6: Burst position...")
    clean['score_burst'] = burst_position_score(clean)
    clean['score_caps'] = clean['num_all_caps_words'] / clean['num_all_caps_words'].max() if clean['num_all_caps_words'].max()>0 else 0.0

    # raw_score: blend
    clean['score_policy'] = (clean['policy_intensity_score'] /
                             max(clean['policy_intensity_score'].max(), 1))
    clean['raw_score'] = np.clip(
        0.4*clean['score_policy'] + 0.3*np.clip(clean['score_embedding'],0,1)
        + 0.2*clean['score_novelty'] + 0.1*clean['score_burst'], 0, 1)
    print("⚖️  Layer 7: Relative signal strength...")
    clean['score_relative'] = relative_signal_strength(clean)

    # normalized composites
    for c in ['policy_intensity_score','hawkish_risk_score','growth_policy_score']:
        mx = clean[c].max()
        clean[f'{c}_norm'] = clean[c]/mx if mx>0 else 0.0

    # sample_weight = blended signal
    clean['sample_weight'] = np.clip(
        0.5*clean['raw_score'] + 0.3*clean['hawkish_risk_score_norm']
        + 0.2*clean['score_relative']/5.0, 0.05, 1.0)

    clean.to_csv(out_csv, index=False)
    print(f"\n💾 Saved {out_csv} ({len(clean)} rows)")
    print(f"   raw_score mean: {clean['raw_score'].mean():.3f}, "
          f">0.3: {(clean['raw_score']>0.3).sum()} posts")


# ==========================================
# INCREMENTAL MAIN — daily mode
# ==========================================
def _load_clean_truths(in_csv):
    """trump_truths.csv -> cleaned/sorted DataFrame (same filtering as main())."""
    df = pd.read_csv(in_csv, dtype={'id': str})
    df['post_type'] = df['text'].apply(classify_post)
    clean = df[df['post_type'].isin(['original', 'quote_rt'])].copy()
    clean['text'] = clean.apply(extract_comment, axis=1)
    clean = clean.drop(columns=['post_type'])
    clean['date'] = pd.to_datetime(clean['date'], utc=True, format='mixed').dt.tz_convert('America/New_York')
    clean = clean.sort_values('date').reset_index(drop=True)
    clean['text_clean'] = clean['text'].apply(lambda t: re.sub(r'https?://\S+', '', str(t)).strip())
    return clean


def score_incremental(context_days=3, novelty_window=10):
    """
    Daily mode: score ONLY the posts in trump_truths.csv that are not yet in
    trump_truths_scored.csv, and APPEND them (existing rows are never rewritten).

    Context features use the recent already-scored timeline:
      - novelty:  compared against the previous `novelty_window` posts
      - burst:    posts in the preceding 30 minutes
      - relative: raw_score vs the ±2h neighborhood median
    Normalized scores use scorer_config.json norm_divisors (stable across runs).
    """
    in_csv  = os.path.join(_HERE, "trump_truths.csv")
    out_csv = os.path.join(_HERE, "trump_truths_scored.csv")

    print("=" * 60)
    print("  TRUMP POST SIGNAL SCORER — INCREMENTAL (daily)")
    print(f"  Config: {CONFIG_PATH}  (version {CONFIG.get('version','?')})")
    print("=" * 60)

    if not os.path.exists(out_csv):
        print("⚠️  No existing trump_truths_scored.csv — running FULL scoring instead.")
        return main()

    clean = _load_clean_truths(in_csv)
    scored = pd.read_csv(out_csv, dtype={'id': str})
    scored_cols = list(scored.columns)
    existing_ids = set(scored['id'].astype(str))

    new = clean[~clean['id'].astype(str).isin(existing_ids)].copy().reset_index(drop=True)
    print(f"\n📂 {in_csv}: {len(clean)} clean posts | already scored: {len(existing_ids)} | new: {len(new)}")
    if new.empty:
        print("✅ Nothing new to score — trump_truths_scored.csv is up to date.")
        return 0

    # ---- recent context from the already-scored timeline ----
    ctx = scored.copy()
    ctx['date'] = pd.to_datetime(ctx['date'], utc=True, format='mixed').dt.tz_convert('America/New_York')
    ctx = ctx.sort_values('date').reset_index(drop=True)
    cutoff = new['date'].min() - pd.Timedelta(days=context_days)
    ctx_recent = ctx[ctx['date'] >= cutoff]
    if len(ctx_recent) < novelty_window:               # always keep enough for novelty
        ctx_recent = ctx.tail(novelty_window)
    ctx_texts  = ctx_recent['text_clean'].fillna("").astype(str).tolist()
    ctx_dates  = list(ctx_recent['date'])
    ctx_raw    = list(ctx_recent['raw_score'].astype(float))
    n_ctx = len(ctx_recent)
    print(f"🕰️  Context: {n_ctx} recent scored posts (since {cutoff:%Y-%m-%d %H:%M})")

    texts = new['text_clean'].fillna("").tolist()

    print("\n📦 Loading NLP models...")
    nlp = load_spacy()
    sbert = load_sbert()

    print("\n🚩 Layer 1: Policy flags...")
    flags = pd.DataFrame([extract_policy_flags(t) for t in texts], index=new.index)
    for c in flags.columns: new[c] = flags[c].values

    print("🔬 Layer 2: POS + NER features...")
    ner = pd.DataFrame([extract_ner_features(t, nlp) for t in texts], index=new.index)
    for c in ner.columns: new[c] = ner[c].values

    print("📊 Layer 3: Composite scores...")
    new = compute_composite_scores(new)

    print("🧠 Layer 4: Semantic embedding score...")
    new['score_embedding'] = embedding_score(texts, sbert)

    # ---- combined timeline (context + new) for context-dependent layers ----
    all_texts = ctx_texts + texts
    all_dates = ctx_dates + list(new['date'])

    print("🔍 Layer 5: Novelty score (vs recent timeline)...")
    nov = np.ones(len(texts))
    if sbert is not None and cosine_similarity is not None:
        emb = sbert.encode(all_texts, batch_size=32, show_progress_bar=False)
        for k in range(len(texts)):
            i = n_ctx + k
            start = max(0, i - novelty_window)
            if i > start:
                nov[k] = 1.0 - cosine_similarity(emb[i].reshape(1, -1), emb[start:i])[0].max()
    else:
        for k in range(len(texts)):
            i = n_ctx + k
            start = max(0, i - novelty_window)
            curr = set(str(all_texts[i]).lower().split())
            mx = 0.0
            for j in range(start, i):
                prev = set(str(all_texts[j]).lower().split())
                u = len(curr | prev)
                if u > 0: mx = max(mx, len(curr & prev) / u)
            nov[k] = 1.0 - mx
    new['score_novelty'] = nov

    print("⏱️  Layer 6: Burst position (vs recent timeline)...")
    dser = pd.Series(pd.to_datetime(all_dates))
    burst = np.ones(len(texts))
    for k in range(len(texts)):
        i = n_ctx + k
        win_start = dser.iloc[i] - pd.Timedelta(minutes=30)
        n = ((dser >= win_start) & (dser < dser.iloc[i])).sum()
        burst[k] = max(0.05, 1.0 / (1.0 + n * 0.4))
    new['score_burst'] = burst

    # score_caps: normalize by HISTORICAL max so old rows stay comparable
    caps_max = max(float(scored['num_all_caps_words'].max()), float(new['num_all_caps_words'].max()), 0.0)
    new['score_caps'] = new['num_all_caps_words'] / caps_max if caps_max > 0 else 0.0

    # ---- raw_score with FIXED config divisor (no batch-max drift) ----
    norm_div = CONFIG.get("norm_divisors", {"policy_intensity_score": 8.0,
                                            "hawkish_risk_score": 5.0,
                                            "growth_policy_score": 4.0})
    pi_div = max(norm_div.get("policy_intensity_score", 8.0), 1)
    new['score_policy'] = np.clip(new['policy_intensity_score'] / pi_div, 0, 1)
    new['raw_score'] = np.clip(
        0.4 * new['score_policy'] + 0.3 * np.clip(new['score_embedding'], 0, 1)
        + 0.2 * new['score_novelty'] + 0.1 * new['score_burst'], 0, 1)

    print("⚖️  Layer 7: Relative signal strength (vs recent timeline)...")
    all_raw = np.array(ctx_raw + list(new['raw_score']))
    rel = np.ones(len(texts))
    for k in range(len(texts)):
        i = n_ctx + k
        lo = dser.iloc[i] - pd.Timedelta(hours=2)
        hi = dser.iloc[i] + pd.Timedelta(hours=2)
        mask = (dser >= lo) & (dser <= hi)
        neighbors = all_raw[mask.values]
        med = np.median(neighbors) if len(neighbors) else all_raw[i]
        rel[k] = all_raw[i] / med if med > 0.01 else 1.0
    new['score_relative'] = np.clip(rel, 0.0, 5.0)

    # normalized composites from config divisors (same as prediction path)
    for c in ['policy_intensity_score', 'hawkish_risk_score', 'growth_policy_score']:
        div = norm_div.get(c, 0)
        new[f'{c}_norm'] = np.clip(new[c] / div, 0, 1) if div else 0.0

    new['sample_weight'] = np.clip(
        0.5 * new['raw_score'] + 0.3 * new['hawkish_risk_score_norm']
        + 0.2 * new['score_relative'] / 5.0, 0.05, 1.0)

    # ---- append with the EXACT existing column order ----
    for c in scored_cols:
        if c not in new.columns:
            new[c] = 0
    out = new[scored_cols]
    out.to_csv(out_csv, mode='a', header=False, index=False, lineterminator='\n')
    print(f"\n💾 Appended {len(out)} new scored posts -> {out_csv} "
          f"({len(scored) + len(out)} total)")
    print(f"   new raw_score mean: {out['raw_score'].mean():.3f}, "
          f">0.3: {(out['raw_score'] > 0.3).sum()} posts")
    return len(out)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Score Trump truths (daily incremental by default).")
    ap.add_argument("--full", action="store_true",
                    help="Full rebuild: re-score ALL posts and rewrite trump_truths_scored.csv.")
    args = ap.parse_args()
    if args.full:
        main()
    else:
        score_incremental()
