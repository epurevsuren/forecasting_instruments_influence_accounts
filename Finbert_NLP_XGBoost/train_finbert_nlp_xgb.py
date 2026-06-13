"""
train_finbert_nlp_xgb.py  —  IMPROVED
-------------------------------------
FinBERT embeddings + signal_scorer NLP features → XGBoost, with real upgrades:

  1. CLS + mean pooling concatenated (better sentence representation than mean alone)
  2. Early stopping (prevents overfit on 768+43 features / 4973 rows)
  3. Directional accuracy metric (does it get UP vs DOWN right? — matters more
     than R² for a trading signal)
  4. Noise-baseline check: prints mean |prediction| on low-signal posts, so you
     can SEE if endorsements are being correctly predicted near-zero
  5. Feature importance: reports whether NLP flags or FinBERT dims drive each
     instrument (tells you if the NLP scorer is actually contributing)

Inputs:  truth_training_set_FINAL.csv, trump_truths_scored.csv
Outputs: finbert_nlp_xgb_models/  (+ eval_report.json)
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

LABEL_CSV  = "../DP/truth_training_set_TEST.csv"
SCORED_CSV = "../DP/trump_truths_scored.csv"
OUT_DIR    = "finbert_nlp_xgb_models"
EMB_CACHE  = "finbert_embeddings_v2.npy"   # v2 = CLS+mean pooling
FINBERT    = "ProsusAI/finbert"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUMENTS = [
    'SPY','QQQ','DIA','XLI','XLF','XLE','VIX','OIL','GOLD','COPPER','NATGAS',
    'EUR_USD','USD_JPY','GBP_USD','USD_CNY','USD_CAD','USD_MXN','USD_CHF','AUD_USD',
    'US10Y','US2Y','BTC','ETH',
]

NLP_FEATURES = [
    'raw_score','score_policy','score_embedding','score_novelty','score_burst',
    'score_caps','score_relative',
    'policy_intensity_score','hawkish_risk_score','growth_policy_score',
    'policy_intensity_score_norm','hawkish_risk_score_norm','growth_policy_score_norm',
    'flag_stimulus','flag_tariff_trade','flag_interest_rate','flag_tax_policy',
    'flag_sanctions','flag_war_geopolitics','flag_energy_policy','flag_immigration_policy',
    'flag_industrial_policy','flag_deregulation','flag_financial_system','flag_supply_chain',
    'flag_ai_chip_policy','flag_covid_relief',
    'num_policy_verbs','num_policy_nouns','num_gpe','num_org','num_percent','num_money',
    'num_date','num_law','num_person','num_cardinal','num_event',
    'num_geopolitical_terms','num_all_caps_words',
    'favorites','retweets','replies',
]
N_FINBERT = 1536  # CLS(768) + mean(768)


def embed_texts(texts):
    print(f"📥 Loading FinBERT on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(FINBERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT, output_hidden_states=True).to(DEVICE).eval()
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
        if (i//B) % 10 == 0: print(f"   {i+len(batch)}/{len(texts)}")
    return np.vstack(embs)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("📂 Loading labels + NLP features...")
    labels = pd.read_csv(LABEL_CSV); labels['text'] = labels['text'].fillna("")
    # POINT-IN-TIME LOAD: trump_truths_scored.csv always holds the FULL history
    # (through today), while the label csv ends at the backtest cutoff. Load
    # scored rows ONLY up to the label window's end via duckdb — the file no
    # longer needs to be cut manually before the test post (e.g. the 0.78-NLP
    # Emir post), and nothing after the cutoff enters this process.
    _bound_utc = pd.to_datetime(labels['date'], format='mixed', utc=True).max()
    _bound_ny  = _bound_utc.tz_convert('America/New_York')
    try:
        import duckdb
        feats = duckdb.sql(
            f"SELECT * FROM read_csv_auto('{SCORED_CSV}', types={{'date': 'VARCHAR', 'id': 'VARCHAR'}}) "
            f"WHERE CAST(date AS TIMESTAMPTZ) <= TIMESTAMPTZ '{_bound_utc.isoformat()}'").df()
        print(f"  🦆 duckdb: scored csv loaded WHERE date <= {_bound_ny:%Y-%m-%d %H:%M %Z} "
              f"(last labeled post; {len(feats)} rows)")
    except ImportError:
        feats = pd.read_csv(SCORED_CSV)
        _d = pd.to_datetime(feats['date'], format='mixed', utc=True)
        feats = feats[_d <= _bound_utc].copy()
        print(f"  ⚠️  duckdb not installed (uv add duckdb) — pandas fallback, "
              f"rows <= {_bound_ny:%Y-%m-%d %H:%M %Z}: {len(feats)}")
    key = ['date','text'] if 'text' in feats.columns else ['date']
    df = labels.merge(feats, on=key, how='inner', suffixes=('','_sc'))
    print(f"  Merged rows: {len(df)}")

    use_nlp = [c for c in NLP_FEATURES if c in df.columns]
    X_nlp = df[use_nlp].fillna(0.0).values

    if os.path.exists(EMB_CACHE) and len(np.load(EMB_CACHE)) == len(df):
        print("💾 Loading cached embeddings (v2)...")
        X_emb = np.load(EMB_CACHE)
    else:
        X_emb = embed_texts(df['text'].tolist()); np.save(EMB_CACHE, X_emb)
        print(f"💾 Cached → {EMB_CACHE}")

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

    json.dump(report, open(f"{OUT_DIR}/eval_report.json","w"), indent=2)
    good = sum(1 for v in report.values() if v['r2']>0.1)
    dirs = [v['dir_acc'] for v in report.values() if v['dir_acc']]
    print(f"\n  {good}/{len(report)} R²>0.1 | mean directional acc: {np.mean(dirs):.0%}")
    print(f"  📋 eval_report.json saved — check 'noise_pred' (want LOW = endorsements quiet)")
    print(f"     and 'nlp_share' (how much your scorer drives each instrument)")


if __name__ == '__main__':
    main()
