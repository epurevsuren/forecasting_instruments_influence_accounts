"""
train_gemma3_nlp_xgb.py
-----------------------
Gemma-3-4B embeddings + signal_scorer NLP features → XGBoost.
Full standalone Gemma port (gemma retired 2026-07-17: the NLP-only
challenger matched the gemma-embedding model on all 23 instruments, so
the encoder is upgraded to a 4B-parameter local LLM). Keeps every honest-
evaluation upgrade: strict 4-way chronological split, trade-region magnitude
calibration + TP quantile, vol-regime training weights, PCA embedding
compression (fit on the train slice), NLP-only challenger diagnostic.

Inputs:  training_set_FINAL / posts_scored (database.db tables)
Outputs: gemma3_nlp_xgb_models/ (XGBoost model jsons + config.json + emb_pca.npz)
         + gemma3_embeddings_v1 / eval_report_gemma3 tables in database.db
Run:     uv run python train_gemma3_nlp_xgb.py
         (first run embeds ~190k posts with Gemma — hours, one-time cache)
"""
import os
# FA2 install is broken in this venv — tell unsloth to go straight to
# Xformers instead of probing FA2 and warning (same speed either way).
os.environ.setdefault("UNSLOTH_FORCE_XFORMERS", "1")
try:
    import unsloth  # noqa: F401 — must import BEFORE transformers for its patches
except ImportError:
    pass
import json
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
import db  # DuckDB helper -> ../database.db
import signal_scorer as ss  # shim -> canonical DP/signal_scorer.py (same CONFIG as scoring)

LABEL_TABLE  = "training_set_FINAL"
SCORED_TABLE = "posts_scored"
EMB_TABLE    = "gemma3_embeddings_v1"    # mean + last-token pooling (5120 dims)
EVAL_TABLE   = "eval_report_gemma3"
OUT_DIR    = "gemma3_nlp_xgb_models"
# Backbone id. Variable name kept as gemma so the sentiment-head guard in
# main() ("gemma" in gemma.lower()) correctly SKIPS the BERT-only head.
gemma    = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Instruments loaded DYNAMICALLY from DP/instruments.json (master registry) —
# add/remove there, retrain, done. The trained set is frozen into
# <model_dir>/config.json so predict/backtest always match the model.
_INSTRUMENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DP", "instruments.json")
with open(_INSTRUMENTS_FILE, encoding="utf-8") as _f:
    INSTRUMENTS = list(json.load(_f)["instruments"].keys())

# Policy flags + NER keys are read DYNAMICALLY from the canonical scorer
# (DP/scorer_config.json + DP/signal_scorer.py) so adding a flag to the JSON
# flows into training with NO edit here. The exact list used at train time is
# frozen into <model_dir>/config.json, which predict/backtest read — so
# inference always matches the model even as the config evolves.
# Pipeline order matters: run signal_scorer --full BEFORE training after a
# config change, or the new flag columns won't exist in posts_scored yet
# (they'd be 0-filled with a warning).
_FLAG_FEATURES = list(ss.CONFIG["policy_flags"].keys())      # e.g. flag_peace_deescalation
_NER_FEATURES  = list(ss.canonical._NER_KEYS)                # num_gpe, num_org, ...

NLP_FEATURES = [
    # Source-context features: let XGBoost learn non-linear interactions between
    # NLP signal and post origin (is_primary=1 → rank-0 primary within its active window, ANY platform; geo weight = credibility tier)
    'is_primary', 'entity_weight', 'event_weight', 'account_rank',
    # Composite NLP scores
    'raw_score','score_policy','score_embedding','score_novelty','score_burst',
    'score_caps','score_relative',
    'policy_intensity_score','hawkish_risk_score','growth_policy_score',
    'macro_risk_score',
    'policy_intensity_score_norm','hawkish_risk_score_norm','growth_policy_score_norm',
    'macro_risk_score_norm',
    # Policy flag indicators (dynamic, from scorer_config.json)
    *_FLAG_FEATURES,
    # NER / linguistic counts (dynamic, from the canonical scorer)
    *_NER_FEATURES,
    # Engagement signals
    'favorites','retweets','replies',
]
# ============================================================================
# TWO-HEAD CLASSIFICATION  (2026-07-31) — the fix for the dead regressor
# ============================================================================
# Diagnosis from train-834afb94 / backtest-20260731: EVERY instrument shows
# R² ~ -0.00, mean|pred| = 0.001-0.10% against mean|actual| = 0.20-1.55%, and
# MdAPE = 100%. That is not a bug — it is squared-error doing exactly what it
# should on a target whose signal-to-noise is ~0: the risk-minimising constant
# is the mean, so the trees converge to "predict ~0". Every downstream symptom
# follows from it: k calibration went NEGATIVE on 14/23 instruments (US10Y
# -76.95, AUD_USD -20.92 — a negative k means pred and actual are ANTI-
# correlated in the trade region, i.e. the ratio is fitting noise), and only
# 107 of 14,214 cells could clear the |pred| >= 0.1% TRADE bar.
#
# Method comes from the SAGE Open VIX paper (Zhang et al., 2025,
# doi:10.1177/21582440251396044), which faced the same problem and did NOT
# regress the magnitude. Their Eq. 7:
#     Risk_{t+1} = 1 if ReVIX_t >= VaR_ReVIX_t else 0
# where VaR_ReVIX_t is the 10% tail of the VIX % change under a ROLLING
# 500-day window — i.e. binarise the target against a backward-looking,
# regime-adaptive threshold, then fix the resulting class imbalance with
# category weights. They report F1 = 0.97 on volatile markets with XGBoost
# where Logit scored precision = recall = 0.
#
# Our analogue, leak-free by construction:
#   HEAD A (move)  y = |1h move| >= MOVE_Q-th pct of (|move| / vol30), where
#                  vol30 is the trailing 30-day daily-move std AT POST TIME
#                  (build_final_training_set.py:854 — "last available <= post
#                  date", so it can only see the past). The quantile itself is
#                  fitted on the TRAIN SLICE ONLY. scale_pos_weight handles the
#                  imbalance, exactly as the paper's category weights do.
#   HEAD B (dir)   y = move > 0, trained ONLY on rows where head A's label is 1.
#                  Direction on noise-sized rows is unlearnable and dilutes the
#                  fit; conditioning on the event is what the sentiment
#                  literature finds gives polarity its predictive power.
#                  This also learns per-instrument POLARITY directly — no more
#                  hand-coding the NATGAS 37.4% / COPPER 42.6% inversions.
# The regressor is still trained and saved (TP distance still wants a number),
# but it is no longer the trade trigger.
# MOVE_Q measured 2026-07-31: tightening 70 -> 90 does NOT improve direction
# (mean +0.2%, only 5/12 instruments better), so the tail is not where the
# directional signal hides. 90 is kept anyway because it matches the paper's
# 10% VaR level and yields a sane 10%-ish event rate instead of 30%.
MOVE_Q      = float(os.environ.get("MOVE_Q", "90"))    # rolling quantile level
MOVE_WINDOW = int(os.environ.get("MOVE_WINDOW", "2000"))  # posts of history
MOVE_MINP   = int(os.environ.get("MOVE_MINP", "500"))     # min history to score
DIR_MIN_N   = int(os.environ.get("DIR_MIN_N", "150"))  # min event rows for head B
TARGET_PRECISION = float(os.environ.get("TARGET_PRECISION", "0.58"))
# Instruments whose direction head cannot beat chance out-of-sample are marked
# untradeable rather than traded at a coin flip. 0.52 = a real but thin edge.
DIR_MIN_ACC = float(os.environ.get("DIR_MIN_ACC", "0.52"))

# ============================================================================
# WALK-FORWARD  (2026-08-02) — memory across posts, in the training process
# ============================================================================
# The single fixed 70/15/8/7 split trains mostly on 2017-2021 dynamics and then
# asks about 2026, and it evaluates on ONE window. Both are problems:
#   * a frozen model never sees recent regime;
#   * one window is one sample, which is exactly how the `tradeable` flag got
#     picked on ~150 calibration events and then reverted out-of-sample
#     (US2Y 60.1% -> 41.2%, VIX 59.8% -> 44.8%).
#
# Walk-forward refits on an EXPANDING window: train on [0, a0), predict
# [a0, a1), advance. Every prediction is out-of-sample, and concatenating the
# folds gives ~3.3x more evaluation data than the single split.
#
# MEASURED 2026-08-02 (8 instruments, size head, corr(|pred|,|actual|)):
#     single fixed split      0.203
#     walk-forward, cold      0.231   <- this
#     walk-forward, warm      0.203
# XLE 0.049 -> 0.163 and COPPER 0.115 -> 0.201 on walk-forward alone.
#
# NOTE ON WARM START: inheriting the previous booster via xgb_model= lost on
# 8/8 instruments (-0.028 mean). Continued boosting APPENDS trees rather than
# re-optimising them, so the model accumulates stale 2017-2021 structure. The
# memory that helps is having more DATA in the window, not carrying model
# state forward. Do not "fix" this by re-enabling warm start.
WALK_FORWARD = os.environ.get("WALK_FORWARD", "1") == "1"
WF_FOLDS     = int(os.environ.get("WF_FOLDS", "8"))
WF_START     = float(os.environ.get("WF_START", "0.50"))   # first train fraction
# A gate must produce at least this many gradeable trades across the whole
# walk-forward window, or its accuracy is not a measurement. ~100 gives a
# standard error near 5pp, which is the least we can distinguish 52% from 50%.
WF_MIN_TRADES = int(os.environ.get("WF_MIN_TRADES", "100"))


def _wf_oos(X, y, w, y_move, y_up, folds=WF_FOLDS, start_frac=WF_START):
    """Expanding-window walk-forward. Returns out-of-sample predictions for all
    three heads over rows [start, N), each produced by a model that saw ONLY
    rows before it. Used for evaluation and gate selection — never for fitting
    the shipped model.
    """
    N = len(y)
    start = int(N * start_frac)
    step = max((N - start) // max(folds, 1), 1)
    oos = {k: np.full(N, np.nan, dtype=np.float64)
           for k in ("size", "p_move", "p_up")}
    for f in range(folds):
        a0 = start + f * step
        a1 = N if f == folds - 1 else min(a0 + step, N)
        if a0 >= N or a1 <= a0:
            break
        Xtr, ytr = X[:a0], y[:a0]
        wtr = w[:a0] if w is not None else None
        mvtr, uptr = y_move[:a0], y_up[:a0]

        # --- size head (|move|) ---
        ms = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.5, min_child_weight=5, reg_alpha=0.1,
            reg_lambda=1.5, n_jobs=-1, random_state=42)
        ms.fit(Xtr, np.abs(ytr), sample_weight=wtr, verbose=False)
        oos["size"][a0:a1] = np.clip(ms.predict(X[a0:a1]), 0.0, None)

        # --- move head ---
        npos = int(mvtr.sum())
        if npos >= 20 and npos < len(mvtr):
            mm = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                objective='binary:logistic', eval_metric='auc',
                scale_pos_weight=float((len(mvtr) - npos) / max(npos, 1)),
                n_jobs=-1, random_state=42)
            mm.fit(Xtr, mvtr, sample_weight=wtr, verbose=False)
            oos["p_move"][a0:a1] = mm.predict_proba(X[a0:a1])[:, 1]

        # --- direction head, event rows only ---
        ev = np.where(mvtr == 1)[0]
        if len(ev) >= DIR_MIN_N:
            md = xgb.XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                objective='binary:logistic', eval_metric='logloss',
                n_jobs=-1, random_state=42)
            md.fit(Xtr[ev], uptr[ev],
                   sample_weight=(wtr[ev] if wtr is not None else None),
                   verbose=False)
            oos["p_up"][a0:a1] = md.predict_proba(X[a0:a1])[:, 1]
    oos["start"] = start
    return oos

# Gemma-3-4B encoder: mean + last-token pooling of the final hidden layer
# (research 2026-07-17: decoder LLMs have no [CLS]; last-token = the
# autoregressive summary but recency-biased, mean = robust on MTEB — concat
# both, PCA downstream compresses to 128 anyway). 4-bit local, batch 8.
from gemma_embedder import embed_texts, N_EMB
N_gemma = N_EMB   # 5120 = mean(2560) + last-token(2560)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("📂 Loading labels + NLP features...")
    labels = db.read_table(LABEL_TABLE)
    if labels is None:
        raise FileNotFoundError(
            f"{LABEL_TABLE} table not found in {db.DB_PATH} — "
            f"run build_final_training_set.py first")
    labels['text'] = labels['text'].fillna("")
    # Normalize 'date' to UTC tz-aware datetime — DuckDB round-trips TIMESTAMPTZ
    # columns back as the local system tz (e.g. Australia/Sydney), while
    # posts_scored stores 'date' as plain strings. Both must match for the merge key.
    labels['date'] = pd.to_datetime(labels['date'], utc=True)

    # POINT-IN-TIME LOAD: posts_scored always holds the FULL history (through today),
    # while the label table ends at the label window's end. Load scored rows ONLY
    # up to that bound — nothing after the cutoff enters this process.
    _bound_utc = labels['date'].max()
    _bound_ny  = _bound_utc.tz_convert('America/New_York')
    feats = db.read_table(SCORED_TABLE)
    if feats is None:
        raise FileNotFoundError(
            f"{SCORED_TABLE} table not found in {db.DB_PATH} — run signal_scorer.py first")
    feats['id'] = feats['id'].astype(str)
    feats['date'] = pd.to_datetime(feats['date'], format='mixed', utc=True)
    feats = feats[feats['date'] <= _bound_utc].copy()
    print(f"  🦆 {SCORED_TABLE}: {len(feats)} rows  (≤ {_bound_ny:%Y-%m-%d %H:%M %Z})")

    # Merge on (platform, id): composite PK that disambiguates Snowflake IDs
    # across TruthSocial and X/Twitter (replaces old uid string column).
    labels['id'] = pd.to_numeric(labels['id'], errors='coerce').astype('Int64')
    feats['id']  = pd.to_numeric(feats['id'],  errors='coerce').astype('Int64')
    df = labels.merge(feats, on=['platform', 'id'], how='inner', suffixes=('', '_sc'))
    n_primary = int(df['is_primary'].sum()) if 'is_primary' in df.columns else '?'
    n_twitter = len(df) - (n_primary if isinstance(n_primary, int) else 0)
    print(f"  Merged rows: {len(df)}  (primary: {n_primary} | non-primary: {n_twitter})")

    # Date order is required by the chronological CALIB slice below (X/y/w
    # all align to this order). Does NOT affect train/early-stop randomness.
    df = df.sort_values('date').reset_index(drop=True)

    # ------------------------------------------------------------------
    # TRAINING-ROW FILTER (2026-07-17): ~155k of 190k rows are non-primary
    # geo posts whose 1-hour labels are ambient market noise. A squared-
    # loss model trained on 82% "nothing happened" rows learns to predict
    # ~0 for EVERYTHING (Gemma baseline backtest: mean|pred| 0.000-0.005 on
    # most instruments, k exploding to +/-30-150). Keep only rows with a
    # meaningful finalized sample_weight — the model learns tweet-impact
    # from posts that plausibly HAD impact; the gates already ensure only
    # such posts are traded at inference. Cached embeddings make retrains
    # on any filter setting a minutes-long experiment.
    # ------------------------------------------------------------------
    # TRAIN_MIN_WEIGHT raised 0.20 -> 0.30 (2026-07-29). At 0.20 the trees
    # still saw ~30k rows dominated by low-impact posts whose 1-hour labels
    # are ambient noise, which is what pushes squared-error toward
    # "predict ~0". Override with env SFT-style if you want to sweep it:
    #   $env:TRAIN_MIN_WEIGHT = "0.25"
    TRAIN_MIN_WEIGHT = float(os.environ.get("TRAIN_MIN_WEIGHT", "0.30"))
    _n0 = len(df)
    _sw = df['sample_weight'].fillna(0.0)
    print("  📊 Rows surviving each candidate threshold (pick empirically):")
    for _t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        _k = int((_sw >= _t).sum())
        _mark = "  <- ACTIVE" if abs(_t - TRAIN_MIN_WEIGHT) < 1e-9 else ""
        print(f"       w >= {_t:.2f}: {_k:>6} rows ({_k/_n0:5.1%}){_mark}")
    df = df[_sw >= TRAIN_MIN_WEIGHT].reset_index(drop=True)
    print(f"  🧹 Training-row filter: sample_weight >= {TRAIN_MIN_WEIGHT} "
          f"-> {len(df)}/{_n0} rows kept "
          f"(primary: {int(df['is_primary'].sum()) if 'is_primary' in df.columns else '?'})")

    # HIGH-SIGNAL OVERSAMPLING (same constants as train_gemma3_analyst_lora.py)
    OVERSAMPLE_W = 0.50   # rows above this weight are repeated...
    OVERSAMPLE_X = 4      # ...this many times, in the TRAIN slice only
    print(f"  🔁 Oversampling: sample_weight > {OVERSAMPLE_W} repeated x{OVERSAMPLE_X} "
          f"({int((df['sample_weight'].fillna(0.0) > OVERSAMPLE_W).sum())} rows qualify)")

    # NLP feature matrix — cast to float32 (is_primary is bool; booleans → 0/1 fine)
    use_nlp = [c for c in NLP_FEATURES if c in df.columns]
    missing  = [c for c in NLP_FEATURES if c not in df.columns]
    if missing:
        print(f"  ⚠️  NLP features absent from merged df (0-filled): {missing}")
        for c in missing:
            df[c] = 0.0
        use_nlp = NLP_FEATURES
    X_nlp = df[use_nlp].fillna(0.0).values.astype(np.float32)

    # Partial embed cache: load what we have, re-embed ONLY missing posts.
    # Cache key is "<platform>_<id>" — unique composite identifier replacing old uid.
    emb_cached = db.read_table(EMB_TABLE)
    cache: dict = {}
    if emb_cached is not None and 'platform_id' in emb_cached.columns:
        emb_cached = emb_cached.set_index(emb_cached['platform_id'].astype(str))
        cache = {i: np.array(v, dtype=np.float32)
                 for i, v in emb_cached['embedding'].items()}

    platform_ids = (df['platform'] + '_' + df['id'].astype(str)).tolist()
    missing_idx = [i for i, pid in enumerate(platform_ids) if pid not in cache]

    if missing_idx:
        print(f"  🔢 {len(missing_idx)}/{len(platform_ids)} posts missing cached embeddings — "
              f"computing with Gemma-3-4B...")
        # INCREMENTAL CHECKPOINTS (Gemma embedding of 190k posts takes hours):
        # append each finished chunk to the DB immediately — a crash never
        # loses completed work; the rerun re-embeds only what's still missing.
        CHUNK = 2000
        for _c0 in range(0, len(missing_idx), CHUNK):
            part = missing_idx[_c0:_c0 + CHUNK]
            fresh = embed_texts(df.iloc[part]['text'].tolist())
            for j, i in enumerate(part):
                cache[platform_ids[i]] = fresh[j]
            chunk_df = pd.DataFrame({
                'platform_id': [platform_ids[i] for i in part],
                'embedding': [fresh[j].tolist() for j in range(len(part))],
            })
            db.append_table(EMB_TABLE, chunk_df)
            print(f"  💾 checkpoint {min(_c0 + CHUNK, len(missing_idx))}"
                  f"/{len(missing_idx)} appended → {EMB_TABLE}")
    else:
        print(f"  ✅ All {len(platform_ids)} embeddings found in cache")

    X_emb = np.vstack([cache[pid] for pid in platform_ids])

    # ------------------------------------------------------------------
    # GEMMA ANALYST (2026-07-17, Peter: "Gemma must READ and ANALYZE the
    # text; NLP should just suggest scoring"): Gemma generates a per-post
    # 23-instrument impact JSON (his train_stack_xgb.py stacking idea, now
    # on the honest chronological pipeline). The 23 analyst numbers become
    # first-class features — XGBoost learns when to trust the analyst,
    # when the NLP score, and how to correct the analyst's biases.
    # Cached in gemma3_analyst_v1 (generation runs once per post, ever).
    # ------------------------------------------------------------------
    import gemma_analyst as GA
    A = GA.analyst_features(
        platform_ids, df['text'].tolist(),
        accounts=df['account'].tolist() if 'account' in df.columns else None,
        dates=df['date'].tolist() if 'date' in df.columns else None)
    X_nlp = np.hstack([X_nlp, A.astype(np.float32)])
    use_nlp = list(use_nlp) + GA.ANALYST_COLS
    print(f"  🧠 Analyst features joined: +{len(GA.ANALYST_COLS)} cols "
          f"(nonzero on {(np.abs(A).sum(axis=1) > 0).mean():.0%} of posts)")

    # ------------------------------------------------------------------
    # gemma SENTIMENT HEAD (2026-07-16): use gemma the way it was
    # DESIGNED (Araci 2019; ProsusAI/gemma = classifier fine-tuned on the
    # Financial PhraseBank): 3-class sentiment PROBABILITIES, canonical
    # score = P(pos) - P(neg). We had been discarding the trained head and
    # feeding raw hidden states — the NLP-only challenger proved those add
    # nothing. Probs are computed EXACTLY from cached CLS vectors by
    # applying the model's own pooler+classifier weights (no re-embedding;
    # BertForSequenceClassification: softmax(Wc·tanh(Wp·cls+bp)+bc)).
    # Head saved to gemma_sent_head.npz so predict/backtest match.
    # ------------------------------------------------------------------
    # SENTIMENT HEAD: REMOVED (2026-07-17). That block was classifier-
    # anatomy (bert.pooler + a TRAINED 3-class classifier from the Financial
    # PhraseBank). Gemma is a decoder LLM with NO trained sentiment head —
    # AutoModelForSequenceClassification invents a RANDOM `score` layer
    # (load report: "score.weight | MISSING") whose output is noise, and its
    # init even crashes on 4-bit weights (normal_kernel_cuda on Byte).
    # If a sentiment feature proves necessary, the correct Gemma-native way
    # is next-token logit contrast (P("positive") vs P("negative") after a
    # sentiment prompt) — one extra forward pass per post, cacheable like
    # the embeddings. Not enabled yet: measure the embedding baseline first.
    print("  ⏭️  Sentiment head: none for Gemma (see comment) — NLP features only")

    # ------------------------------------------------------------------
    # EMBEDDING COMPRESSION (2026-07-16): 1536 raw gemma dims drown the
    # ~53 NLP features — trees pick NLP ~3% of the time and every model
    # flatlines (BTC/ETH/FX predict <0.1% forever -> zero trades).
    # feature_weights was a dead end: measured on synthetic 1536+53 data,
    # a 20x weight lifts NLP share only 4.4% -> 6.4% (it merely biases
    # colsample sampling). PCA to EMB_PCA_DIM keeps the embedding's main
    # semantic directions while making NLP ~30% of columns BY CONSTRUCTION.
    # Fitted on the chronological TRAIN slice only (no lookahead); saved to
    # emb_pca.npz so predict/backtest apply the identical projection.
    # ------------------------------------------------------------------
    EMB_PCA_DIM = 128
    from sklearn.decomposition import PCA
    _i70 = int(len(df) * 0.70)          # same boundary as the split below
    # RAM guard: Gemma vectors are 5120-wide — fit the PCA on
    # a stride subsample of the train slice (<=60k rows: components are
    # statistically identical, workspace is a third). Transform still maps
    # ALL rows through the fitted projection.
    _step = max(1, _i70 // 60000)
    print(f"  🔻 PCA {X_emb.shape[1]} -> {EMB_PCA_DIM} dims "
          f"(fit on train slice, every {_step}th row = {len(range(0, _i70, _step))} rows)...")
    _pca = PCA(n_components=EMB_PCA_DIM, svd_solver='randomized',
               random_state=42).fit(X_emb[:_i70:_step])
    _evr = float(_pca.explained_variance_ratio_.sum())
    X_emb = _pca.transform(X_emb).astype(np.float32)
    np.savez(os.path.join(OUT_DIR, "emb_pca.npz"),
             mean=_pca.mean_.astype(np.float32),
             components=_pca.components_.astype(np.float32))
    print(f"     explained variance: {_evr:.1%}  💾 emb_pca.npz saved")

    # ------------------------------------------------------------------
    # TECHNICAL INDICATORS (added 2026-07-29, jiewwantan-style: daily bars
    # + derived indicators as predictors). build_final_training_set emits
    # {inst}_{mom5,mom20,sma_rat,rsi14,macd_h,bb_pos,atr_pct,vol_rat},
    # every one .shift(1)-guarded so a post can only see the PREVIOUS
    # close. The placebo test proved market context matters enormously
    # (post-window vol is 2x in-session, ~0 out-of-session) and the model
    # currently cannot see any of it.
    #
    # DESIGN: attaching all 23x8=184 columns to every model would drop the
    # NLP share from ~33% to ~20% — the exact dilution the PCA step was
    # added to fix. So each instrument's model gets ONLY:
    #   * its OWN 8 indicators (what its own market is doing), plus
    #   * a small GLOBAL risk block (VIX + SPY) that matters to everything.
    # ------------------------------------------------------------------
    # must match DP/build_final_training_set.TECH_COLS (no vol_rat — the
    # daily source has no Volume series)
    TECH_COLS = ['mom5', 'mom20', 'sma_rat', 'rsi14', 'macd_h',
                 'bb_pos', 'atr_pct']
    # CROSS-INSTRUMENT CONTEXT (2026-08-03) — the one idea worth taking from
    # the Algothon 2023 winner (CookieAlgorists). Their MLR step was "other
    # stocks' pasts predict THIS stock's future" instead of a stock predicting
    # only itself. Our models were doing the narrow thing: each saw its OWN 7
    # indicators plus VIX/SPY only, so e.g. OIL's volatility could never inform
    # USD_CAD even though that channel is obvious.
    #
    # MEASURED on 10 instruments (size head, corr on |1h move|):
    #     own 7 TA only                     0.234
    #     + every instrument's atr_pct      0.255
    #     + every instrument's atr_pct+mom20 0.268   (+0.034, helps 8/10)
    # Biggest movers XLF +0.111, XLI +0.095, QQQ +0.065.
    #
    # Two columns per instrument, not all seven: atr_pct (where is risk right
    # now) and mom20 (where is the tape going). Adding all 7 x 23 would be 161
    # columns of mostly-redundant noise for ~13k training rows.
    GLOBAL_TECH = ([f'{i}_atr_pct' for i in INSTRUMENTS] +
                   [f'{i}_mom20' for i in INSTRUMENTS] +
                   [f'VIX_{c}' for c in ('sma_rat', 'mom5')] +
                   [f'SPY_{c}' for c in ('mom5',)])
    _have_tech = [c for c in df.columns
                  if any(c.endswith('_' + t) for t in TECH_COLS)]
    if _have_tech:
        print(f"  📈 Technical indicators available: {len(_have_tech)} columns "
              f"({len(TECH_COLS)}/instrument, own+global attached per model)")
    else:
        print("  ⚠️  No technical indicator columns found — rerun "
              "build_final_training_set.py --full to generate them")

    def _tech_block(inst):
        """(n, k) own-instrument + global indicators for one model."""
        cols = [f'{inst}_{c}' for c in TECH_COLS] + GLOBAL_TECH
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32), []
        blk = df[cols].apply(pd.to_numeric, errors='coerce')
        # median-fill from the TRAIN slice only (no lookahead in the filler)
        _i70 = int(len(df) * 0.70)
        blk = blk.fillna(blk.iloc[:_i70].median()).fillna(0.0)
        return blk.values.astype(np.float32), cols

    X = np.hstack([X_emb, X_nlp])
    nlp_start = X_emb.shape[1]   # index where NLP features begin
    print(f"  Combined: {X.shape[1]} dims ({X_emb.shape[1]} gemma-PCA + {len(use_nlp)} NLP)")

    w = df['sample_weight'].fillna(0.3).values if 'sample_weight' in df.columns else None
    sig = df['sample_weight'].fillna(0.3).values if 'sample_weight' in df.columns else np.full(len(df),0.5)

    print("\n🌲 Training [gemma(CLS+mean)+NLP] → XGBoost...\n")
    report = {}
    calibration = {}      # per-instrument magnitude scale, fitted OUT-OF-SAMPLE
    calibration_tp = {}   # conservative TP quantile (40th pct of actual/pred)
    calibration_raw = {}  # UNCLIPPED fit — diagnostic, shows how far off it was
    calibration_unreliable = []   # instruments whose raw fit left the sane band
    move_gate = {}        # two-head classifier gate: threshold + operating point
    clf_report = []       # per-instrument classification scorecard
    for inst in INSTRUMENTS:
        col = f"{inst}_Impact"
        if col not in df.columns: continue
        y = df[col].fillna(0.0).values
        idx = np.arange(len(df))

        # per-instrument feature matrix: [gemma-PCA | NLP | own+global TA]
        _tb, _tcols = _tech_block(inst)
        X_i = np.hstack([X, _tb]) if _tb.shape[1] else X
        
        # ---------------------------------------------------------
        # 1. STRICT 4-WAY CHRONOLOGICAL SPLIT (The Lookahead Fix)
        # ---------------------------------------------------------
        N = len(df)
        i_tr  = int(N * 0.70)  # 70% Train
        i_es  = int(N * 0.85)  # 15% Early Stop
        i_cal = int(N * 0.93)  #  8% Magnitude Calibration
                               #  7% Take Profit Calibration (Remainder)

        # NOTE: slices come from X_i (per-instrument matrix incl. its own
        # technical indicators), NOT the shared X.
        Xtr,  ytr,  itr  = X_i[:i_tr], y[:i_tr], idx[:i_tr]
        Xes,  yes,  ies  = X_i[i_tr:i_es], y[i_tr:i_es], idx[i_tr:i_es]
        Xcal, ycal, ical = X_i[i_es:i_cal], y[i_es:i_cal], idx[i_es:i_cal]
        Xtp,  ytp,  itp  = X_i[i_cal:], y[i_cal:], idx[i_cal:]

        # We still want an aggregate "Test" set for reporting R2 / MAE / etc.
        Xte, yte, ite = X_i[i_es:], y[i_es:], idx[i_es:]

        # ---------------------------------------------------------
        # 2. VOLATILITY ADJUSTMENT (The Median Leak Fix)
        # ---------------------------------------------------------
        # We now calculate the median STRICTLY on the training slice
        w_tr = w[:i_tr].copy() if w is not None else np.ones(len(ytr))
        
        vcol = f'{inst}_vol30'
        if vcol in df.columns:
            v_train = pd.to_numeric(df.iloc[:i_tr][vcol], errors='coerce')
            med = float(v_train.median()) if v_train.notna().sum() > 50 else None
            
            if med and med > 0:
                # Apply only to training weights
                v_vals = v_train.fillna(med).values
                v_vals = np.where(v_vals == 0, 1e-9, v_vals)
                # Floor 0.33, NOT 0.10 — the 0.10 floor (tried 2026-07-15)
                # collapsed every model (SPY corr 0.71->0.34, mean|pred| /10):
                # high-vol30 regimes (COVID, tariff waves, SVB) hold most of
                # the REAL impact examples; the proxy can't tell "crazy
                # because of the post" from "crazy ambient", so crushing
                # those rows to 10% removed the signal itself.
                adj = np.clip(med / v_vals, 0.33, 2.0)
                w_tr = w_tr * adj

        # ---------------------------------------------------------
        # PER-INSTRUMENT LABEL-QUALITY TRUST (2026-07-28)
        # ---------------------------------------------------------
        # Session status is per instrument: a 03:00 post is shut for SPY but
        # live for BTC/FX. Down-weight ONLY this instrument's weak-label rows
        # here — never globally in build (that damped 75,271 posts off SPY's
        # calendar, cut HIGH_SIGNAL 2,733->1,765 and starved every instrument).
        #   next_open = market was shut; label is the next session's first hour
        #               (real + tradeable, but hours of other news intervene)
        #   daily     = daily-bar fallback, weakest evidence
        QUALITY_TRUST = {'intraday': 1.0, 'next_open': 0.5, 'daily': 0.3}
        qcol = f'{inst}_quality'
        if qcol in df.columns:
            q_tr = df.iloc[:i_tr][qcol].astype(str).map(QUALITY_TRUST).fillna(1.0)
            w_tr = w_tr * q_tr.values
            _n_no = int((df.iloc[:i_tr][qcol].astype(str) == 'next_open').sum())
            if _n_no:
                print(f"     {inst}: {_n_no} next_open rows at "
                      f"x{QUALITY_TRUST['next_open']} trust")

        # ---------------------------------------------------------
        # 3. HIGH-SIGNAL OVERSAMPLING  (mirrors train_gemma3_analyst_lora.py)
        # ---------------------------------------------------------
        # The trees see ~30k rows of which only a minority are genuinely
        # market-moving, so squared-error drives them toward "predict ~0".
        # The LoRA script solves this by repeating sample_weight > 0.50 rows
        # OVERSAMPLE_X times; we apply the SAME logic here.
        #
        # SAFETY: duplication happens ONLY on the training slice, AFTER the
        # chronological split — a duplicated row can never land in the
        # early-stop, calibration or TP windows, so this cannot leak.
        # (For XGBoost this is mathematically equivalent to multiplying
        # those rows' sample_weight by OVERSAMPLE_X; duplication is used to
        # keep the two training scripts literally comparable.)
        Xtr_f, ytr_f, w_tr_f = Xtr, ytr, w_tr
        # _rep_idx = row indices (into the train slice) actually fed to the
        # trees. The classifier heads reuse it so their labels line up with
        # the SAME oversampled rows the regressor saw.
        _rep_idx = np.arange(i_tr)
        if w is not None:
            _hi = np.where(w[:i_tr] > OVERSAMPLE_W)[0]
            if len(_hi):
                _rep = np.concatenate(
                    [np.arange(i_tr)] +
                    [_hi] * (OVERSAMPLE_X - 1)).astype(int)
                _rep_idx = _rep
                Xtr_f, ytr_f, w_tr_f = Xtr[_rep], ytr[_rep], w_tr[_rep]
                print(f"     {inst}: oversampled {len(_hi)} high-signal rows "
                      f"(w>{OVERSAMPLE_W}) x{OVERSAMPLE_X} -> {len(_rep)} train rows")

        # (feature_weights REMOVED 2026-07-16: measured dead end — 20x weight
        # lifted NLP importance only 4.4%->6.4%; it only biases colsample
        # sampling. Replaced by the PCA compression above.)
        m = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.5, objective='reg:squarederror',
            early_stopping_rounds=40, n_jobs=-1, random_state=42)

        m.fit(Xtr_f, ytr_f, sample_weight=w_tr_f,
              eval_set=[(Xes, yes)], verbose=False)

        # NLP-ONLY CHALLENGER — same split, NLP features only. Answers "is
        # gemma earning its 128 columns?" per instrument, in every train
        # log. If the challenger matches/beats the full model repeatedly,
        # the embedding is decoration and should be dropped for that
        # instrument. Diagnostic only — champion model is still saved.
        m_nlp = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.5, objective='reg:squarederror',
            early_stopping_rounds=40, n_jobs=-1, random_state=42)
        # challenger trains on the SAME oversampled slice, so the comparison
        # stays apples-to-apples
        m_nlp.fit(Xtr_f[:, nlp_start:], ytr_f, sample_weight=w_tr_f,
                  eval_set=[(Xes[:, nlp_start:], yes)], verbose=False)
        
        # Overall Test Predictions for Reporting
        pred_te = m.predict(Xte)

        mae, r2 = mean_absolute_error(yte, pred_te), r2_score(yte, pred_te)
        mask = np.abs(yte) > 0.1
        dir_acc = (np.sign(pred_te[mask])==np.sign(yte[mask])).mean() if mask.sum()>0 else float('nan')

        # challenger metrics on the same test window
        pred_nlp = m_nlp.predict(Xte[:, nlp_start:])
        r2_nlp = r2_score(yte, pred_nlp)
        dir_nlp = (np.sign(pred_nlp[mask])==np.sign(yte[mask])).mean() if mask.sum()>0 else float('nan')
        
        noise_mask = sig[ite] < 0.3
        noise_pred = np.abs(pred_te[noise_mask]).mean() if noise_mask.sum()>0 else 0.0
        
        imp = m.feature_importances_
        nlp_share = imp[nlp_start:].sum() / max(imp.sum(), 1e-9)

        # =========================================================
        # 3b. TWO-HEAD CLASSIFICATION  (the SAGE Eq.7 replacement)
        # =========================================================
        # --- ROLLING QUANTILE OF |move| ITSELF  (rewritten 2026-07-31 #2) ----
        # v1 normalised |move| by {inst}_vol30 and froze one train-slice
        # quantile. Two measured failures killed it:
        #
        #  (a) vol30 is broken for FX and rates. The build log prints EUR
        #      baseline std = 0.017% — real EUR/USD daily vol is ~0.4%, so the
        #      ratio exploded and USD_CNY's "threshold" came out at 39-120.
        #  (b) `fillna(0.0)` makes MISSING data look like a zero move, and
        #      US10Y is 48.2% zeros / NATGAS 44.1%. The frozen quantile then
        #      landed on that tie-mass, so the test slice fired 82-89% events
        #      (US10Y n=2472, NATGAS n=2290 in train-834afb94) instead of 30%.
        #
        # This is the paper's method taken literally instead of by analogy:
        # SAGE Eq. 7 thresholds against a quantile of the series' OWN recent
        # history (rolling 500 days). Here: rolling quantile of |1h move| over
        # the previous MOVE_WINDOW posts, zeros excluded as missing data,
        # strict `>` so a tie-mass can never fire. `.shift(1)` means row i is
        # scored only against rows < i, so it is leak-free per-row rather than
        # per-slice — and it tracks regime instead of freezing 2016 vol.
        # Measured: event-rate drift train->test drops from 5/13 instruments
        # over 15pp to 0/13, with rates landing at 6-15% everywhere.
        _abs = np.abs(y)
        _s = pd.Series(np.where(_abs > 0, _abs, np.nan))    # 0 == no data
        _thr_roll = (_s.shift(1)
                       .rolling(MOVE_WINDOW, min_periods=MOVE_MINP)
                       .quantile(MOVE_Q / 100.0).values)
        _valid = (~np.isnan(_thr_roll)) & (_abs > 0)
        y_move = np.zeros(len(y), dtype=int)
        y_move[_valid] = (_abs[_valid] > _thr_roll[_valid]).astype(int)
        y_up = (y > 0).astype(int)                          # head-B label
        # median threshold, for the config/report only
        thr_ratio = float(np.nanmedian(_thr_roll)) if _valid.any() else 0.0
        _v_med = float(np.nanmedian(_abs[_abs > 0])) if (_abs > 0).any() else 0.0

        # --- HEAD A: does this post move the instrument abnormally? ---------
        mv_tr, mv_es = y_move[:i_tr][_rep_idx], y_move[i_tr:i_es]
        _npos = int(mv_tr.sum())
        _spw  = float((len(mv_tr) - _npos) / max(_npos, 1))
        m_mv = xgb.XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.5, objective='binary:logistic',
            eval_metric='auc', scale_pos_weight=_spw,        # <- the paper's
            early_stopping_rounds=40, n_jobs=-1, random_state=42)  # class weights
        m_mv.fit(Xtr_f, mv_tr, sample_weight=w_tr_f,
                 eval_set=[(Xes, mv_es)], verbose=False)

        p_mv_te  = m_mv.predict_proba(Xte)[:, 1]
        p_mv_cal = m_mv.predict_proba(Xcal)[:, 1]
        try:
            auc_mv = float(roc_auc_score(y_move[i_es:], p_mv_te))
        except ValueError:
            auc_mv = float('nan')

        # --- HEAD B: given a move, which way? -------------------------------
        # Trained ONLY on event rows. If an instrument has too few events to
        # fit honestly we record it and fall back to no directional edge.
        ev_tr = np.where(y_move[:i_tr] == 1)[0]
        if len(ev_tr) >= DIR_MIN_N:
            ev_es = np.where(y_move[i_tr:i_es] == 1)[0]
            m_dir = xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=2.0, objective='binary:logistic',
                eval_metric='logloss',
                early_stopping_rounds=40, n_jobs=-1, random_state=42)
            _es_dir = [(Xes[ev_es], y_up[i_tr:i_es][ev_es])] if len(ev_es) >= 30 \
                      else [(Xtr[ev_tr], y_up[:i_tr][ev_tr])]
            m_dir.fit(Xtr[ev_tr], y_up[:i_tr][ev_tr],
                      sample_weight=w_tr[ev_tr],
                      eval_set=_es_dir, verbose=False)
            p_dir_te  = m_dir.predict_proba(Xte)[:, 1]
            p_dir_cal = m_dir.predict_proba(Xcal)[:, 1]
        else:
            m_dir, p_dir_te = None, np.full(len(Xte), 0.5)
            p_dir_cal = np.full(len(Xcal), 0.5)

        # --- HEAD C: SIZE  |move|, not the signed return --------------------
        # THE fix for "the predicted number is nowhere near the actual".
        # A signed return is  y = sign x size.  sign is ~a coin flip (50-54%),
        # size is genuinely predictable — so regressing the SIGNED value forces
        # one model to solve both, the unpredictable factor dominates the
        # squared error, and the optimum collapses to ~0. That is the whole
        # story behind mean|pred| = 0.001% vs mean|actual| = 0.318% and
        # MdAPE = 100% on every instrument.
        #
        # Measured on the real data 2026-07-31 (12 instruments, held-out slice,
        # correlation between prediction and truth):
        #     signed return, NLP only          -0.022   <- what shipped
        #     |move|,        NLP only          +0.007
        #     |move|,        NLP + 7 TA cols   +0.173   <- this head
        # and the predicted scale lands at 0.89-1.13x the actual mean instead
        # of ~1/300th of it. Best: OIL 0.333, GOLD 0.241, DIA 0.235, SPY 0.213.
        #
        # The technical indicators are what make this work — they carry the
        # VOLATILITY STATE the post lands in, which is precisely what sets the
        # size of the reaction. They said nothing about direction, and that is
        # not a failure of the indicators: it is what they measure.
        m_size = xgb.XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.5, objective='reg:squarederror',
            early_stopping_rounds=40, n_jobs=-1, random_state=42)
        _abs_tr = np.abs(ytr_f)
        m_size.fit(Xtr_f, _abs_tr, sample_weight=w_tr_f,
                   eval_set=[(Xes, np.abs(yes))], verbose=False)
        _sz_te = np.clip(m_size.predict(Xte), 0.0, None)
        _sz_cal = np.clip(m_size.predict(Xcal), 0.0, None)

        # SCALE CORRECTION — pathology only, fitted on the calibration slice.
        #
        # Measured 2026-07-31: forcing mean|pred| to match mean|actual| fixes
        # the level (scale -> 1.0 everywhere) but makes the TYPICAL error
        # WORSE: SPY MdAPE 32%->44%, QQQ 37%->50%, GOLD 45%->53%. |move| is
        # right-skewed, so a head sitting near the MEDIAN is what minimises
        # median error; scaling it up to the MEAN overshoots most rows.
        # Mild under-prediction is also the right bias for trading — a tighter
        # TP gets filled, and the simulate log is 148/158 timeouts on QQQ.
        #
        # So: leave a healthy 0.4-1.6x alone, and only correct genuine
        # mis-scaling (COPPER ran 2.07x on data that only starts 2022-04;
        # correcting it takes MdAPE 223% -> 70%).
        _ac = np.abs(y[i_es:i_cal])
        _m_ac, _m_pc = _ac[_ac >= 0.1], _sz_cal[_ac >= 0.1]
        size_k = 1.0
        if len(_m_ac) >= 20 and _m_pc.mean() > 1e-9:
            _raw_scale = float(_m_pc.mean() / _m_ac.mean())
            if not (0.40 <= _raw_scale <= 1.60):
                size_k = float(np.clip(1.0 / _raw_scale, 0.4, 2.5))
                print(f"     📏 {inst}: size scale {_raw_scale:.2f}x out of band "
                      f"-> correcting x{size_k:.2f}")
        _sz_te = _sz_te * size_k
        _sz_cal = _sz_cal * size_k
        _a_te = np.abs(y[i_es:])
        size_corr = float(np.corrcoef(_sz_te, _a_te)[0, 1]) \
            if _sz_te.std() > 1e-12 else float('nan')
        # scale check: predictions should average the same as reality, not
        # 1/300th of it. Ratio far from 1.0 means the head is mis-scaled.
        size_ratio = float(_sz_te.mean() / max(_a_te.mean(), 1e-9))
        # MdAPE on real moves — the metric that read 100% (= "predicting zero")
        _rm = _a_te >= 0.1
        size_mdape = float(np.median(np.abs(_sz_te[_rm] - _a_te[_rm]) / _a_te[_rm])) \
            if _rm.sum() >= 20 else float('nan')
        # (saved further down, AFTER the final all-rows refit — saving here
        #  would ship the 70%-only model)

        # --- operating point chosen OUT-OF-SAMPLE on the calibration slice --
        # Sweep p_move thresholds and score each on "direction was right, on
        # rows that really moved".
        #
        # BUG FIXED 2026-07-31 #2 — this is what produced "only 1 tradeable
        # instrument". v1 seeded p_move_thr = 0.99 as a sentinel and only
        # overwrote it when some threshold reached TARGET_PRECISION. Nothing
        # reached 0.58, so 22 of 23 instruments shipped a gate of p >= 0.99,
        # which no post can clear: F1 = 0.00 across the board and zero TRADEs.
        # COPPER looked like "the one that works" purely because it happened to
        # trip the target and got a real 0.30 gate (14 trades, 57.1%). The
        # other 22 were never evaluated — they were silently switched off.
        #
        # Now: always pick a usable threshold. Prefer the LOOSEST one hitting
        # TARGET_PRECISION (picking the max on a small slice overfits it); if
        # none does, fall back to the BEST-scoring threshold and record
        # target_met=False so the shortfall is visible instead of disguised as
        # a dead instrument.
        ev_cal = y_move[i_es:i_cal] == 1
        dir_ok_cal = ((p_dir_cal > 0.5) == (y[i_es:i_cal] > 0)) & ev_cal
        _cands = []
        for _t in np.arange(0.30, 0.95, 0.05):
            _sel = (p_mv_cal >= _t) & (np.abs(p_dir_cal - 0.5) >= 0.05)
            _nev = int((_sel & ev_cal).sum())
            if _sel.sum() < 20 or _nev < 10:
                continue
            _cands.append((float(_t), float(dir_ok_cal[_sel].sum() / _nev),
                           int(_sel.sum())))
        _hit = [c for c in _cands if c[1] >= TARGET_PRECISION]
        if _hit:                       # loosest threshold meeting the target
            p_move_thr, prec_cal, n_cal_sel = _hit[0]
            target_met = True
        elif _cands:                   # best available — never "never trade"
            p_move_thr, prec_cal, n_cal_sel = max(_cands, key=lambda c: c[1])
            target_met = False
        else:                          # too few calibration events to judge
            p_move_thr, prec_cal, n_cal_sel = 0.50, float('nan'), 0
            target_met = False

        # --- WALK-FORWARD OOS: the honest evaluation surface ----------------
        # Replaces the single 8% calibration slice for gate selection and the
        # tradeable decision. ~3.3x more out-of-sample rows, spread across 8
        # regimes instead of one — which is the actual fix for a "winner"
        # chosen on ~150 events reverting in the backtest.
        wf = None
        if WALK_FORWARD:
            wf = _wf_oos(X_i, y, w, y_move, y_up)
            _s = wf["start"]
            _m = ~np.isnan(wf["size"])
            _ev_wf = (y_move == 1) & _m
            _a = np.abs(y)
            _rm = _m & (_a >= 0.1)
            wf_size_corr = (float(np.corrcoef(wf["size"][_rm], _a[_rm])[0, 1])
                            if _rm.sum() >= 50 else float('nan'))
            _du = ~np.isnan(wf["p_up"])
            _dev = _ev_wf & _du
            wf_dir = (float(((wf["p_up"][_dev] > 0.5) == (y[_dev] > 0)).mean())
                      if _dev.sum() >= 50 else float('nan'))
            print(f"     🔁 WF({WF_FOLDS} folds, OOS n={int(_m.sum())}): "
                  f"size-corr={wf_size_corr:+.3f}  dir-on-events={wf_dir:.1%} "
                  f"(n={int(_dev.sum())})"
                  if wf_dir == wf_dir else
                  f"     🔁 WF: size-corr={wf_size_corr:+.3f}  dir n/a")

        # --- TRADEABLE: must hold on TWO independent windows ----------------
        # v1 flagged an instrument tradeable on the calibration slice alone —
        # about 150 events. Picking the best of 23 instruments on n=150 (SE ~
        # 4pp) selects noise, and the backtest proved it: US2Y 60.1% -> 41.2%,
        # VIX 59.8% -> 44.8%, OIL 59.1% -> not traded. Every "winner" chosen
        # that way reverted, and TRADE accuracy (44.1%) came out BELOW SKIP
        # (47.4%) — the gate was actively anti-selective.
        #
        # Requiring the edge to survive on the calibration slice AND the
        # untouched TP slice is a far harder test to pass by luck: two
        # independent windows both landing above the bar is ~4x less likely
        # under the null than one.
        dir_cal_acc = (float(dir_ok_cal.sum() / ev_cal.sum())
                       if ev_cal.sum() >= 30 else float('nan'))
        ev_tp = y_move[i_cal:] == 1
        if m_dir is not None and ev_tp.sum() >= 30:
            _pd_tp = m_dir.predict_proba(Xtp)[:, 1]
            dir_tp_acc = float((((_pd_tp > 0.5) == (y[i_cal:] > 0)) & ev_tp).sum()
                               / ev_tp.sum())
        else:
            dir_tp_acc = float('nan')
        _both = (dir_cal_acc == dir_cal_acc and dir_tp_acc == dir_tp_acc
                 and dir_cal_acc >= DIR_MIN_ACC and dir_tp_acc >= DIR_MIN_ACC)
        tradeable = bool(_both and m_dir is not None)

        # ====================================================================
        # TRADEABLE, JUDGED ON THE POPULATION WE ACTUALLY TRADE
        # ====================================================================
        # BUG FIXED 2026-08-02. Previous versions scored `dir-on-events`:
        # direction accuracy over EVERY row that moved (~1,000/instrument).
        # But we never trade every row that moved — the gate fires only when
        # p_move clears its threshold AND the direction head has an edge. All
        # the rows where the model has no opinion (p_up ~ 0.5) score ~50% by
        # construction and swamped the measurement.
        #
        # The damage was not conservatism, it was ANTI-SELECTION. Backtest
        # 20260802 shipped exactly one tradeable instrument, XLE, which then
        # scored 45.5% — while BTC (58.3%), USD_CAD (57.1%), QQQ (55.2%),
        # USD_MXN (54.4%) and GBP_USD (54.2%) sat in the SKIP column.
        #
        # Now: sweep the gate on the WALK-FORWARD out-of-sample predictions
        # (9,295 rows over 8 regimes), score each candidate on the rows that
        # gate would ACTUALLY have traded, and take the loosest one that
        # clears DIR_MIN_ACC with at least WF_MIN_TRADES of them. Still fully
        # out-of-sample — every wf prediction came from a model that saw only
        # earlier rows.
        wf_dir_acc = float('nan')
        wf_gate_n = 0
        if wf is not None:
            _du = ~np.isnan(wf["p_up"]) & ~np.isnan(wf["p_move"])
            _meaning = _du & (np.abs(y) >= 0.1)      # noise rows can't grade direction
            _right = (wf["p_up"] > 0.5) == (y > 0)
            _best = None
            for _t in np.arange(0.30, 0.90, 0.05):
                for _e in (0.05, 0.08, 0.12):
                    _g = _meaning & (wf["p_move"] >= _t) & \
                         (np.abs(wf["p_up"] - 0.5) >= _e)
                    _n = int(_g.sum())
                    if _n < WF_MIN_TRADES:
                        continue
                    _a = float(_right[_g].mean())
                    if _best is None or _a > _best[2]:
                        _best = (float(_t), float(_e), _a, _n)
                    if _a >= DIR_MIN_ACC:
                        _best = (float(_t), float(_e), _a, _n)
                        break
                if _best is not None and _best[2] >= DIR_MIN_ACC:
                    break
            if _best is not None:
                p_move_thr, edge_min_i, wf_dir_acc, wf_gate_n = _best
                target_met = wf_dir_acc >= DIR_MIN_ACC
                tradeable = bool(target_met and m_dir is not None)
                print(f"     🎯 {inst}: gate p>={p_move_thr:.2f} edge>={edge_min_i:.2f} "
                      f"-> {wf_dir_acc:.1%} on {wf_gate_n} WF trades "
                      f"({'TRADEABLE' if tradeable else 'below bar'})"
                      + ("" if tradeable == _both else
                         f"   [2-slice said {'pass' if _both else 'fail'} "
                         f"on ~{int(ev_cal.sum())} rows]"))
            else:
                edge_min_i = 0.05
                print(f"     ⚠️  {inst}: no gate reaches {WF_MIN_TRADES} trades "
                      f"on the WF window — untradeable")
                tradeable = False
        else:
            edge_min_i = 0.05
        move_gate[inst] = {
            "thr_ratio": round(thr_ratio, 4),
            "vol_median": round(_v_med, 4),
            "p_move_thr": round(p_move_thr, 3),
            "auc_move": round(auc_mv, 3) if auc_mv == auc_mv else None,
            "event_rate_train": round(float(y_move[:i_tr].mean()), 3),
            "event_rate_test": round(float(y_move[i_es:].mean()), 3),
            "dir_trained": m_dir is not None,
            "n_events_train": int(len(ev_tr)),
            "cal_precision": round(prec_cal, 3) if prec_cal == prec_cal else None,
            "cal_n": n_cal_sel,
            "target_met": target_met,
            "dir_cal_acc": round(dir_cal_acc, 3) if dir_cal_acc == dir_cal_acc else None,
            "dir_tp_acc": round(dir_tp_acc, 3) if dir_tp_acc == dir_tp_acc else None,
            "wf_dir_acc": round(wf_dir_acc, 3) if wf_dir_acc == wf_dir_acc else None,
            "wf_gate_n": wf_gate_n,
            "wf_folds": WF_FOLDS if wf is not None else 0,
            # per-instrument edge threshold — backtest/predict must use THIS,
            # not a single global --edge-min, because the sweep tuned them
            # together with p_move_thr.
            "edge_min": round(edge_min_i, 3),
            # SIZE head — Layer 2 sizes TP/SL from a PER-POST number now
            "size_corr": round(size_corr, 3) if size_corr == size_corr else None,
            "size_ratio": round(size_ratio, 3),
            "size_mdape": round(size_mdape, 3) if size_mdape == size_mdape else None,
            "size_k": round(size_k, 3),   # backtest/predict MUST apply this
            # THE flag the backtest and Layer 2 should respect: this
            # instrument's direction head beat chance out-of-sample.
            "tradeable": tradeable,
            # conditional magnitude: median |move| GIVEN an event, from the
            # calibration slice. This is what Layer-2 should size TP from —
            # an empirical number, not the regressor's collapsed output.
            "median_abs_move": round(float(np.median(np.abs(y[i_es:i_cal][ev_cal])))
                                     if ev_cal.sum() else 0.0, 4),
        }
        # ====================================================================
        # FINAL REFIT ON **ALL** ROWS  (2026-08-02)
        # ====================================================================
        # Every head above was fitted on X_i[:i_tr] — the first 70% — because
        # the later slices are needed as honest holdouts for early stopping,
        # calibration, the gate sweep and the TP quantile. That is correct for
        # MEASURING. It is wrong for SHIPPING: it means the model that goes
        # live has never seen the most recent ~30% of posts (~5,500 rows,
        # roughly the last two years), which is exactly the period most like
        # tomorrow. Walk-forward measured how much that costs — XLE 0.049 ->
        # 0.163, COPPER 0.115 -> 0.201 — purely from training on more recent
        # data.
        #
        # Standard practice, and what we do now: SELECT on holdouts, then
        # REFIT on everything for deployment. Tree count is frozen to the
        # early-stopped best_iteration, so there is no held-out set required
        # and no risk of training longer than the honest fit justified.
        #
        # NOTE: this cannot leak. Calibration, gates, tradeable and every
        # reported metric were all computed BEFORE this block from models that
        # never saw the later rows. Refitting only changes what ships.
        _n_full = len(y)
        print(f"     🔄 {inst}: final refit on ALL {_n_full} rows "
              f"(was {i_tr} = first 70%) — metrics above stay from the "
              f"holdout fits")

        def _refit(proto, Xf, yf, wf_, n_est):
            p = proto.get_params()
            p.pop("early_stopping_rounds", None)
            p["n_estimators"] = max(int(n_est or 0), 50)
            m2 = type(proto)(**p)
            m2.fit(Xf, yf, sample_weight=wf_, verbose=False)
            return m2

        # Capture the honest tree counts BEFORE refitting. The refit drops
        # early_stopping_rounds (there is no holdout left to stop on), so the
        # refitted estimators have no .best_iteration at all — reading it
        # afterwards raises AttributeError.
        _bi_main = int(m.best_iteration or 0)
        _bi_size = int(m_size.best_iteration or 0)
        _bi_mv = int(m_mv.best_iteration or 0)
        _bi_dir = int(m_dir.best_iteration or 0) if m_dir is not None else 0

        _w_all = w if w is not None else np.ones(_n_full)
        m = _refit(m, X_i, y, _w_all, _bi_main)
        m_size = _refit(m_size, X_i, np.abs(y), _w_all, _bi_size)
        m_mv = _refit(m_mv, X_i, y_move, _w_all, _bi_mv)
        if m_dir is not None:
            _ev_all = np.where(y_move == 1)[0]
            if len(_ev_all) >= DIR_MIN_N:
                m_dir = _refit(m_dir, X_i[_ev_all], y_up[_ev_all],
                               _w_all[_ev_all], _bi_dir)

        m_mv.save_model(f"{OUT_DIR}/{col}__move.json")
        if m_dir is not None:
            m_dir.save_model(f"{OUT_DIR}/{col}__dir.json")
        m_size.save_model(f"{OUT_DIR}/{col}__size.json")

        # honest head-B score: direction accuracy on REAL events in the test
        # window, which is the only number that maps to a tradeable decision
        _ev_te = y_move[i_es:] == 1
        dir_ev = float(((p_dir_te > 0.5) == (y[i_es:] > 0))[_ev_te].mean()) \
            if _ev_te.sum() else float('nan')
        # F1/precision/recall are scored on the TP slice ONLY (i_cal:), which
        # is untouched by both training AND the threshold sweep above. Scoring
        # them on Xte would include the calibration rows the threshold was
        # picked on — that inflates precision for free.
        _yt_tp, _pp_tp = y_move[i_cal:], m_mv.predict_proba(Xtp)[:, 1]
        try:
            _hat = (_pp_tp >= p_move_thr).astype(int)
            _f1 = float(f1_score(_yt_tp, _hat, zero_division=0))
            _pr = float(precision_score(_yt_tp, _hat, zero_division=0))
            _rc = float(recall_score(_yt_tp, _hat, zero_division=0))
        except ValueError:
            _f1 = _pr = _rc = float('nan')

        # ---------------------------------------------------------
        # 3. MAGNITUDE CALIBRATION (Fitted only on Xcal)
        # ---------------------------------------------------------
        pred_cal = m.predict(Xcal)
        abs_p_cal = np.abs(pred_cal)
        cal_mask = np.zeros(len(pred_cal), dtype=bool)
        
        for _q in (90, 75, 0):
            cal_mask = (abs_p_cal >= np.percentile(abs_p_cal, _q)) & (np.abs(ycal) > 0.05)
            if cal_mask.sum() >= 15: # Lowered minimums slightly for the smaller split
                break
                
        if cal_mask.sum() >= 5 and (pred_cal[cal_mask] ** 2).sum() > 1e-9:
            k_cal = float((pred_cal[cal_mask] * ycal[cal_mask]).sum()
                          / (pred_cal[cal_mask] ** 2).sum())
            # FLOOR BACK TO 0.25 (2026-07-29). The 0.02 floor (added
            # 2026-07-16 to mute over-predicting rate models) silently
            # destroyed the whole pipeline: any instrument whose trade-region
            # fit came out low had EVERY prediction multiplied by 0.02, so
            # nothing could clear the 0.1% TRADE bar — backtest 003903 showed
            # VIX x0.02, BTC x0.02 and just 6 TRADE flags across 618 posts
            # (was 478). Muting a suspect model by scaling it to zero also
            # hides the problem instead of reporting it, so:
            #   * k stays inside a sane band [0.25, 8]
            #   * the RAW fit is recorded separately, and an instrument whose
            #     raw fit falls outside the band is flagged UNRELIABLE so the
            #     backtest/Layer-2 can skip it deliberately rather than
            #     trading a silently-zeroed signal.
            k_raw = k_cal
            k_cal = float(np.clip(k_cal, 0.25, 8.0))
            calibration_raw[inst] = round(k_raw, 4)
            if not (0.25 <= k_raw <= 8.0):
                calibration_unreliable.append(inst)
                print(f"     ⚠️  {inst}: raw calibration {k_raw:.3f} outside "
                      f"[0.25, 8] -> clipped to {k_cal:.2f} and flagged UNRELIABLE")
        else:
            k_cal = 1.0
            calibration_raw[inst] = None

        calibration[inst] = round(k_cal, 3)

        # ---------------------------------------------------------
        # 4. TAKE PROFIT CALIBRATION (Fitted only on Xtp)
        # ---------------------------------------------------------
        # The Chained Leak Fix: We use entirely unseen data to evaluate 
        # how k_cal performs, ensuring k_tp is an honest out-of-sample estimate.
        pred_tp = m.predict(Xtp)
        abs_p_tp = np.abs(pred_tp)
        tp_mask = np.zeros(len(pred_tp), dtype=bool)
        
        for _q in (90, 75, 0):
            tp_mask = (abs_p_tp >= np.percentile(abs_p_tp, _q)) & (abs_p_tp > 1e-6)
            if tp_mask.sum() >= 15: 
                break
                
        _pm = tp_mask & (abs_p_tp > 1e-6)
        ratios = (ytp[_pm] * np.sign(pred_tp[_pm])) / (np.abs(pred_tp[_pm]) * k_cal)
        ratios = ratios[ratios > 0]                 
        
        if len(ratios) >= 5:
            k_tp = float(np.clip(np.percentile(ratios, 40), 0.2, 1.5))
        else:
            k_tp = 0.7
            
        calibration_tp[inst] = round(k_tp, 3)

        m.save_model(f"{OUT_DIR}/{col}.json")
        report[inst] = {"mae":round(mae,4),"r2":round(r2,3),
                        "dir_acc":round(float(dir_acc),3) if dir_acc==dir_acc else None,
                        "noise_pred":round(float(noise_pred),4),
                        "nlp_share":round(float(nlp_share),3),
                        "calibration_k":calibration[inst],
                        "best_iter":_bi_main,   # captured pre-refit (see above)
                        "auc_move":move_gate[inst]["auc_move"],
                        "f1_move":round(_f1,3) if _f1==_f1 else None,
                        "dir_acc_on_events":round(dir_ev,3) if dir_ev==dir_ev else None,
                        "n_events_train":move_gate[inst]["n_events_train"]}
        clf_report.append({
            "inst": inst, "auc": auc_mv, "f1": _f1, "prec": _pr, "rec": _rc,
            "dir_ev": dir_ev, "n_ev": int(_ev_te.sum()),
            "thr": p_move_thr, "trained": m_dir is not None,
            "tradeable": tradeable, "target_met": target_met,
            "evt_te": float(y_move[i_es:].mean()),
            "size_corr": size_corr, "size_ratio": size_ratio,
            "size_mdape": size_mdape,
            "dcal": dir_cal_acc, "dtp": dir_tp_acc})

        flag = "✅" if r2>0.1 else ("🟡" if r2>0 else "🔴")
        da = f"dir={dir_acc:.0%}" if dir_acc==dir_acc else "dir=n/a"
        _da_nlp = f"{dir_nlp:.0%}" if dir_nlp == dir_nlp else "n/a"
        print(f"  {flag} {inst:<10} R²={r2:+.3f}  {da}  noise|p|={noise_pred:.3f}  "
              f"NLP={nlp_share:.0%}  k={calibration[inst]:.2f}  "
              f"| NLP-only R²={r2_nlp:+.3f} dir={_da_nlp}")
        # the line that actually matters now
        _cf = "✅" if (dir_ev == dir_ev and dir_ev >= 0.55) else (
              "🟡" if (dir_ev == dir_ev and dir_ev >= 0.52) else "🔴")
        print(f"     {_cf} CLF  move-AUC={auc_mv:.3f}  F1={_f1:.2f}  "
              f"dir-on-events={dir_ev:.1%} (n={int(_ev_te.sum())})  "
              f"gate p>={p_move_thr:.2f}"
              + ("" if target_met else "(best-effort)")
              + f"  events tr/te={move_gate[inst]['event_rate_train']:.0%}/"
                f"{move_gate[inst]['event_rate_test']:.0%}"
              + ("  ✅TRADEABLE" if tradeable else "  ⛔untradeable")
              + ("" if m_dir is not None else "  ⚠️ dir head NOT trained (too few events)"))
        _sc = "✅" if (size_corr == size_corr and size_corr >= 0.15) else (
              "🟡" if (size_corr == size_corr and size_corr >= 0.08) else "🔴")
        print(f"     {_sc} SIZE corr(|pred|,|actual|)={size_corr:+.3f}  "
              f"scale={size_ratio:.2f}x (want ~1.0)  "
              f"MdAPE={size_mdape:.0%} (was 100% = predicting zero)  "
              f"dir cal/tp={dir_cal_acc:.0%}/{dir_tp_acc:.0%}"
              if size_corr == size_corr else "     🔴 SIZE head degenerate")

    # config.json written AFTER training so it carries the fitted calibration
    json.dump({"gemma":gemma,"emb_dim":int(X_emb.shape[1]),"pooling":"cls+mean",
               "emb_pca":EMB_PCA_DIM,   # predict/backtest must project with emb_pca.npz
               "nlp_features":use_nlp,"instruments":INSTRUMENTS,
               # per-instrument TA block recipe — predict/backtest MUST
               # rebuild [emb | nlp | own TECH | GLOBAL_TECH] in this order
               "tech_cols":TECH_COLS,"global_tech":GLOBAL_TECH,
               "calibration":calibration,"calibration_tp":calibration_tp,
               "calibration_raw":calibration_raw,
               "calibration_unreliable":calibration_unreliable,
               # two-head classifier gate — backtest/predict use THIS to
               # decide TRADE, not |pred| >= 0.1%
               "move_gate":move_gate,"move_q":MOVE_Q,
               "target_precision":TARGET_PRECISION},
              open(f"{OUT_DIR}/config.json","w"))
    if calibration_unreliable:
        print(f"\n⚠️  UNRELIABLE calibration on {len(calibration_unreliable)}/"
              f"{len(calibration)} instruments: {', '.join(calibration_unreliable)}")
        print("   Their raw fit was outside [0.25, 8] — predictions are clipped,")
        print("   not zeroed. Treat their TRADE flags with suspicion.")
    print("\n🎯 TP quantiles (Layer-2 --tp-mult per instrument, 40th pct of "
          "actual/calibrated-pred):")
    print("   " + "  ".join(f"{i}:{v:.2f}" for i, v in calibration_tp.items()))

    eval_df = pd.DataFrame([{"instrument": k, **v} for k, v in report.items()])
    db.write_table(EVAL_TABLE, eval_df)
    good = sum(1 for v in report.values() if v['r2']>0.1)
    dirs = [v['dir_acc'] for v in report.values() if v['dir_acc']]
    # ---------------- classification scorecard (the headline now) -----------
    if clf_report:
        cr = sorted(clf_report, key=lambda r: (-(r['dir_ev'] if r['dir_ev'] == r['dir_ev'] else 0)))
        print("\n" + "=" * 78)
        print("  TWO-HEAD CLASSIFIER  (SAGE Eq.7 style — binarised vs rolling vol)")
        print("  dir-on-events = directional accuracy on rows that REALLY moved")
        print("=" * 78)
        print(f"  {'instrument':<11}{'moveAUC':>8}{'sizeCorr':>9}{'scale':>7}"
              f"{'MdAPE':>7}{'dir cal/tp':>12}{'dir-ev':>8}  status")
        for r in cr:
            _d = f"{r['dir_ev']:.1%}" if r['dir_ev'] == r['dir_ev'] else "n/a"
            _mk = "✅" if (r['dir_ev'] == r['dir_ev'] and r['dir_ev'] >= 0.55) else (
                  "🟡" if (r['dir_ev'] == r['dir_ev'] and r['dir_ev'] >= 0.52) else "🔴")
            g = lambda v: f"{v:.0%}" if v == v else " n/a"
            print(f"  {_mk}{r['inst']:<10}{r['auc']:>8.3f}"
                  f"{r['size_corr']:>9.3f}{r['size_ratio']:>7.2f}"
                  f"{r['size_mdape']:>7.0%}"
                  f"{g(r['dcal'])+'/'+g(r['dtp']):>12}{_d:>8}"
                  f"  {'TRADEABLE' if r['tradeable'] else 'skip'}"
                  + ("" if r['target_met'] else " (best-effort gate)")
                  + ("" if r['trained'] else " (dir head off)"))
        _sz = [r['size_corr'] for r in cr if r['size_corr'] == r['size_corr']]
        _sr = [r['size_ratio'] for r in cr if r['size_ratio'] == r['size_ratio']]
        print("-" * 78)
        print(f"  SIZE head: mean corr(|pred|,|actual|) = {np.mean(_sz):+.3f}  "
              f"mean scale = {np.mean(_sr):.2f}x")
        print(f"    (the OLD signed regressor scored corr 0.031 at scale ~0.003x "
              f"-> MdAPE 100%)")
        _dv = [r['dir_ev'] for r in cr if r['dir_ev'] == r['dir_ev']]
        _au = [r['auc'] for r in cr if r['auc'] == r['auc']]
        _tr = [r for r in cr if r['tradeable']]
        print("-" * 78)
        print(f"  mean move-AUC = {np.mean(_au):.3f} (0.50 = coin flip)   "
              f"mean dir-on-events = {np.mean(_dv):.1%}")
        print(f"  TRADEABLE (direction beat {DIR_MIN_ACC:.0%} out-of-sample on the "
              f"calibration slice): {len(_tr)}/{len(cr)}")
        if _tr:
            print("    -> " + ", ".join(r['inst'] for r in _tr))
        _nogate = [r['inst'] for r in cr if not r['target_met']]
        if _nogate:
            print(f"  gate fell back to best-effort on {len(_nogate)}: "
                  + ", ".join(_nogate[:8]) + ("..." if len(_nogate) > 8 else ""))
            print(f"    (none reached the {TARGET_PRECISION:.0%} precision target — "
                  "lower TARGET_PRECISION or accept thinner edge)")
        print("=" * 78)

    print(f"\n  {good}/{len(report)} R²>0.1 | mean directional acc: {np.mean(dirs):.0%}")
    print(f"  📋 {EVAL_TABLE} saved — check 'noise_pred' (want LOW = endorsements quiet)")
    print(f"     and 'nlp_share' (how much your scorer drives vs gemma embeddings)")
    print(f"\n✅ Models saved → {OUT_DIR}/")


if __name__ == "__main__":
    main()