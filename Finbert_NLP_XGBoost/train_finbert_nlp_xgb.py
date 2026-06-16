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

Inputs:  truth_training_set_FINAL / posts_scored (database.db tables)
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

LABEL_TABLE  = "truth_training_set_FINAL"
SCORED_TABLE = "posts_scored"
EMB_TABLE    = "finbert_embeddings_v2"   # v2 = CLS+mean pooling
EVAL_TABLE   = "eval_report"
OUT_DIR    = "finbert_nlp_xgb_models"
FINBERT    = "ProsusAI/finbert"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUMENTS = [
    'SPY','QQQ','DIA','XLI','XLF','XLE','VIX','OIL','GOLD','COPPER','NATGAS',
    'EUR_USD','USD_JPY','GBP_USD','USD_CNY','USD_CAD','USD_MXN','USD_CHF','AUD_USD',
    'US10Y','US2Y','BTC','ETH',
]

NLP_FEATURES = [
    # Source-context features: let XGBoost learn non-linear interactions between
    # NLP signal and post origin (is_primary=1 → rank-0 TruthSocial; geo weight = credibility tier)
    'is_primary', 'entity_weight', 'event_weight', 'account_rank',
    # Composite NLP scores
    'raw_score','score_policy','score_embedding','score_novelty','score_burst',
    'score_caps','score_relative',
    'policy_intensity_score','hawkish_risk_score','growth_policy_score',
    'policy_intensity_score_norm','hawkish_risk_score_norm','growth_policy_score_norm',
    # Policy flag indicators
    'flag_stimulus','flag_tariff_trade','flag_interest_rate','flag_tax_policy',
    'flag_sanctions','flag_war_geopolitics','flag_energy_policy','flag_immigration_policy',
    'flag_industrial_policy','flag_deregulation','flag_financial_system','flag_supply_chain',
    'flag_ai_chip_policy','flag_pandemic_relief',
    # NER / linguistic counts
    'num_policy_verbs','num_policy_nouns','num_gpe','num_org','num_percent','num_money',
    'num_date','num_law','num_person','num_cardinal','num_event',
    'num_geopolitical_terms','num_all_caps_words',
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

    # Merge on uid: globally unique key "<source>_<id>" prevents cross-platform
    # Snowflake ID collisions between TruthSocial and X/Twitter posts.
    df = labels.merge(feats, on='uid', how='inner', suffixes=('', '_sc'))
    n_primary = int(df['is_primary'].sum()) if 'is_primary' in df.columns else '?'
    n_twitter = len(df) - (n_primary if isinstance(n_primary, int) else 0)
    print(f"  Merged rows: {len(df)}  (TruthSocial: {n_primary} | X/Twitter: {n_twitter})")

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
    # This avoids a full 13k-post re-embed when only geo tweets are new.
    emb_cached = db.read_table(EMB_TABLE)
    cache: dict = {}
    if emb_cached is not None and 'uid' in emb_cached.columns:
        emb_cached = emb_cached.set_index(emb_cached['uid'].astype(str))
        cache = {i: np.array(v, dtype=np.float32)
                 for i, v in emb_cached['embedding'].items()}

    uids = df['uid'].astype(str).tolist() if 'uid' in df.columns else []
    missing_idx = [i for i, uid in enumerate(uids) if uid not in cache]

    if missing_idx:
        print(f"  🔢 {len(missing_idx)}/{len(uids)} posts missing cached embeddings — "
              f"computing with FinBERT...")
        fresh = embed_texts(df.iloc[missing_idx]['text'].tolist())
        for j, i in enumerate(missing_idx):
            cache[uids[i]] = fresh[j]
        # Persist full updated cache
        if uids:
            emb_df = pd.DataFrame({
                'uid': list(cache.keys()),
                'embedding': [v.tolist() for v in cache.values()],
            })
            db.write_table(EMB_TABLE, emb_df)
            print(f"  💾 Cached {len(emb_df)} embeddings → {EMB_TABLE}")
    else:
        print(f"  ✅ All {len(uids)} embeddings found in cache")

    if uids:
        X_emb = np.vstack([cache[uid] for uid in uids])
    else:
        # Fallback: embed everything (no id column — shouldn't happen with current schema)
        print("  ⚠️  no 'id' column in merged df — embeddings not cached")
        X_emb = embed_texts(df['text'].tolist())

    X = np.hstack([X_emb, X_nlp])
    nlp_start = X_emb.shape[1]   # index where NLP features begin
    print(f"  Combined: {X.shape[1]} dims ({X_emb.shape[1]} FinBERT + {len(use_nlp)} NLP)")

    w = df['sample_weight'].fillna(0.3).values if 'sample_weight' in df.columns else None
    sig = df['sample_weight'].fillna(0.3).values if 'sample_weight' in df.columns else np.full(len(df),0.5)

    json.dump({"finbert":FINBERT,"emb_dim":int(X_emb.shape[1]),"pooling":"cls+mean",
               "nlp_features":use_nlp,"instruments":INSTRUMENTS},
              open(f"{OUT_DIR}/config.json","w"))

    print("\n🌲 Training [FinBERT(CLS+mean)+NLP] → XGBoost...\n")
    report = {}
    for inst in INSTRUMENTS:
        col = f"{inst}_Impact"
        if col not in df.columns: continue
        y = df[col].fillna(0.0).values
        idx = np.arange(len(df))
        Xtr,Xte,ytr,yte,wtr,_,itr,ite = train_test_split(
            X,y,w,idx,test_size=0.15,random_state=42)

        m = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.5, objective='reg:squarederror',
            early_stopping_rounds=40, n_jobs=-1, random_state=42)
        m.fit(Xtr,ytr,sample_weight=wtr,eval_set=[(Xte,yte)],verbose=False)
        pred = m.predict(Xte)

        mae,r2 = mean_absolute_error(yte,pred), r2_score(yte,pred)
        # directional accuracy on meaningful moves (|true|>0.1)
        mask = np.abs(yte) > 0.1
        dir_acc = (np.sign(pred[mask])==np.sign(yte[mask])).mean() if mask.sum()>0 else float('nan')
        # noise check: mean |pred| on low-signal test posts (should be small)
        noise_mask = sig[ite] < 0.3
        noise_pred = np.abs(pred[noise_mask]).mean() if noise_mask.sum()>0 else 0.0
        # feature importance: NLP share
        imp = m.feature_importances_
        nlp_share = imp[nlp_start:].sum() / max(imp.sum(),1e-9)

        m.save_model(f"{OUT_DIR}/{col}.json")
        report[inst] = {"mae":round(mae,4),"r2":round(r2,3),
                        "dir_acc":round(float(dir_acc),3) if dir_acc==dir_acc else None,
                        "noise_pred":round(float(noise_pred),4),
                        "nlp_share":round(float(nlp_share),3),
                        "best_iter":int(m.best_iteration or 0)}
        flag = "✅" if r2>0.1 else ("🟡" if r2>0 else "🔴")
        da = f"dir={dir_acc:.0%}" if dir_acc==dir_acc else "dir=n/a"
        print(f"  {flag} {inst:<10} R²={r2:+.3f}  {da}  noise|p|={noise_pred:.3f}  NLP={nlp_share:.0%}")

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