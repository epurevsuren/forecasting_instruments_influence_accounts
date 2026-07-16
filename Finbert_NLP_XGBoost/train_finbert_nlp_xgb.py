"""
train_finbert_nlp_xgb.py  —  IMPROVED
-------------------------------------
FinBERT embeddings + signal_scorer NLP features → XGBoost, with real upgrades:

  1. CLS + mean pooling concatenated (better sentence representation than mean alone)
  2. Early stopping (prevents overfit on 768+43 features / ~13 k rows)
  3. Directional accuracy metric (does it get UP vs DOWN right? — matters more
     than R² for a trading signal)
  4. Noise-baseline check: prints mean |prediction| on low-signal posts, so you
     can SEE if endorsements are being correctly predicted near-zero
  5. Feature importance: reports whether NLP flags or FinBERT dims drive each
     instrument (tells you if the NLP scorer is actually contributing)
  6. Partial embed cache: re-embeds ONLY posts not in finbert_embeddings_v2,
     so adding geo tweets doesn't force a full 13k-post re-embed each run.

Inputs:  training_set_FINAL / posts_scored (database.db tables)
         unified_feed labels include TruthSocial posts + English X/Twitter geo posts (13 k+ rows)
Outputs: finbert_nlp_xgb_models/ (XGBoost model jsons + config.json)
         + finbert_embeddings_v2 / eval_report tables in database.db
Run:     uv run python train_finbert_nlp_xgb.py
"""
import os, json
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import db  # DuckDB helper -> ../database.db
import signal_scorer as ss  # shim -> canonical DP/signal_scorer.py (same CONFIG as scoring)

LABEL_TABLE  = "training_set_FINAL"
SCORED_TABLE = "posts_scored"
EMB_TABLE    = "finbert_embeddings_v2"   # v2 = CLS+mean pooling
EVAL_TABLE   = "eval_report"
OUT_DIR    = "finbert_nlp_xgb_models"
FINBERT    = "ProsusAI/finbert"
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
N_FINBERT = 1536  # CLS(768) + mean(768)


_tok_model = [None, None]   # [tokenizer, model] — loaded once


def _load_finbert():
    if _tok_model[0] is None:
        print(f"📥 Loading FinBERT on {DEVICE}...")
        _tok_model[0] = AutoTokenizer.from_pretrained(FINBERT)
        _tok_model[1] = AutoModelForSequenceClassification.from_pretrained(
            FINBERT, output_hidden_states=True).to(DEVICE).eval()
    return _tok_model[0], _tok_model[1]


def embed_texts(texts):
    tok, model = _load_finbert()
    embs, B = [], 32
    print(f"🔢 Embedding {len(texts)} posts (CLS + mean pooling)...")
    for i in range(0, len(texts), B):
        batch = [str(t)[:512] for t in texts[i:i+B]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        last = out.hidden_states[-1]
        cls  = last[:, 0, :]                              # [CLS] token
        mask = enc['attention_mask'].unsqueeze(-1).float()
        mean = (last*mask).sum(1)/mask.sum(1).clamp(min=1e-9)  # mean pool
        embs.append(torch.cat([cls, mean], dim=1).cpu().numpy())
        done = i + len(batch)
        if done % 320 == 0 or done == len(texts):
            print(f"   {done}/{len(texts)}")
    return np.vstack(embs)


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
              f"computing with FinBERT...")
        fresh = embed_texts(df.iloc[missing_idx]['text'].tolist())
        for j, i in enumerate(missing_idx):
            cache[platform_ids[i]] = fresh[j]
        # Persist full updated cache
        emb_df = pd.DataFrame({
            'platform_id': list(cache.keys()),
            'embedding': [v.tolist() for v in cache.values()],
        })
        db.write_table(EMB_TABLE, emb_df)
        print(f"  💾 Cached {len(emb_df)} embeddings → {EMB_TABLE}")
    else:
        print(f"  ✅ All {len(platform_ids)} embeddings found in cache")

    X_emb = np.vstack([cache[pid] for pid in platform_ids])

    # ------------------------------------------------------------------
    # EMBEDDING COMPRESSION (2026-07-16): 1536 raw FinBERT dims drown the
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
    print(f"  🔻 PCA {X_emb.shape[1]} -> {EMB_PCA_DIM} dims (fit on first 70% = train slice)...")
    _pca = PCA(n_components=EMB_PCA_DIM, svd_solver='randomized',
               random_state=42).fit(X_emb[:_i70])
    _evr = float(_pca.explained_variance_ratio_.sum())
    X_emb = _pca.transform(X_emb).astype(np.float32)
    np.savez(os.path.join(OUT_DIR, "emb_pca.npz"),
             mean=_pca.mean_.astype(np.float32),
             components=_pca.components_.astype(np.float32))
    print(f"     explained variance: {_evr:.1%}  💾 emb_pca.npz saved")

    X = np.hstack([X_emb, X_nlp])
    nlp_start = X_emb.shape[1]   # index where NLP features begin
    print(f"  Combined: {X.shape[1]} dims ({X_emb.shape[1]} FinBERT-PCA + {len(use_nlp)} NLP)")

    w = df['sample_weight'].fillna(0.3).values if 'sample_weight' in df.columns else None
    sig = df['sample_weight'].fillna(0.3).values if 'sample_weight' in df.columns else np.full(len(df),0.5)

    print("\n🌲 Training [FinBERT(CLS+mean)+NLP] → XGBoost...\n")
    report = {}
    calibration = {}      # per-instrument magnitude scale, fitted OUT-OF-SAMPLE
    calibration_tp = {}   # conservative TP quantile (40th pct of actual/pred)
    for inst in INSTRUMENTS:
        col = f"{inst}_Impact"
        if col not in df.columns: continue
        y = df[col].fillna(0.0).values
        idx = np.arange(len(df))
        
        # ---------------------------------------------------------
        # 1. STRICT 4-WAY CHRONOLOGICAL SPLIT (The Lookahead Fix)
        # ---------------------------------------------------------
        N = len(df)
        i_tr  = int(N * 0.70)  # 70% Train
        i_es  = int(N * 0.85)  # 15% Early Stop
        i_cal = int(N * 0.93)  #  8% Magnitude Calibration
                               #  7% Take Profit Calibration (Remainder)

        Xtr,  ytr,  itr  = X[:i_tr], y[:i_tr], idx[:i_tr]
        Xes,  yes,  ies  = X[i_tr:i_es], y[i_tr:i_es], idx[i_tr:i_es]
        Xcal, ycal, ical = X[i_es:i_cal], y[i_es:i_cal], idx[i_es:i_cal]
        Xtp,  ytp,  itp  = X[i_cal:], y[i_cal:], idx[i_cal:]
        
        # We still want an aggregate "Test" set for reporting R2 / MAE / etc.
        Xte, yte, ite = X[i_es:], y[i_es:], idx[i_es:]

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

        # (feature_weights REMOVED 2026-07-16: measured dead end — 20x weight
        # lifted NLP importance only 4.4%->6.4%; it only biases colsample
        # sampling. Replaced by the PCA compression above.)
        m = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.5, objective='reg:squarederror',
            early_stopping_rounds=40, n_jobs=-1, random_state=42)

        m.fit(Xtr, ytr, sample_weight=w_tr, eval_set=[(Xes, yes)], verbose=False)

        # NLP-ONLY CHALLENGER — same split, NLP features only. Answers "is
        # FinBERT earning its 128 columns?" per instrument, in every train
        # log. If the challenger matches/beats the full model repeatedly,
        # the embedding is decoration and should be dropped for that
        # instrument. Diagnostic only — champion model is still saved.
        m_nlp = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.5, objective='reg:squarederror',
            early_stopping_rounds=40, n_jobs=-1, random_state=42)
        m_nlp.fit(Xtr[:, nlp_start:], ytr, sample_weight=w_tr,
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
            k_cal = float(np.clip(k_cal, 0.25, 20.0))
        else:
            k_cal = 1.0
            
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
                        "best_iter":int(m.best_iteration or 0)}
        
        flag = "✅" if r2>0.1 else ("🟡" if r2>0 else "🔴")
        da = f"dir={dir_acc:.0%}" if dir_acc==dir_acc else "dir=n/a"
        _da_nlp = f"{dir_nlp:.0%}" if dir_nlp == dir_nlp else "n/a"
        print(f"  {flag} {inst:<10} R²={r2:+.3f}  {da}  noise|p|={noise_pred:.3f}  "
              f"NLP={nlp_share:.0%}  k={calibration[inst]:.2f}  "
              f"| NLP-only R²={r2_nlp:+.3f} dir={_da_nlp}")

    # config.json written AFTER training so it carries the fitted calibration
    json.dump({"finbert":FINBERT,"emb_dim":int(X_emb.shape[1]),"pooling":"cls+mean",
               "emb_pca":EMB_PCA_DIM,   # predict/backtest must project with emb_pca.npz
               "nlp_features":use_nlp,"instruments":INSTRUMENTS,
               "calibration":calibration,"calibration_tp":calibration_tp},
              open(f"{OUT_DIR}/config.json","w"))
    print("\n🎯 TP quantiles (Layer-2 --tp-mult per instrument, 40th pct of "
          "actual/calibrated-pred):")
    print("   " + "  ".join(f"{i}:{v:.2f}" for i, v in calibration_tp.items()))

    eval_df = pd.DataFrame([{"instrument": k, **v} for k, v in report.items()])
    db.write_table(EVAL_TABLE, eval_df)
    good = sum(1 for v in report.values() if v['r2']>0.1)
    dirs = [v['dir_acc'] for v in report.values() if v['dir_acc']]
    print(f"\n  {good}/{len(report)} R²>0.1 | mean directional acc: {np.mean(dirs):.0%}")
    print(f"  📋 {EVAL_TABLE} saved — check 'noise_pred' (want LOW = endorsements quiet)")
    print(f"     and 'nlp_share' (how much your scorer drives vs FinBERT embeddings)")
    print(f"\n✅ Models saved → {OUT_DIR}/")


if __name__ == "__main__":
    main()