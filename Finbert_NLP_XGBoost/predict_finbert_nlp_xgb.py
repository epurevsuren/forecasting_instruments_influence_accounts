"""
predict_finbert_nlp_xgb.py
--------------------------
Predict: post → [FinBERT embedding + NLP scorer features] → XGBoost → impacts.
Plus an optional NLP gate that damps obvious-noise posts (low NLP signal),
fixing FinBERT's endorsement-hallucination (Rick Allen → NatGas +5% nonsense).

Requires signal_scorer.py + scorer_config.json in this folder.
Run:  uv run python predict_finbert_nlp_xgb.py
"""
import os, json
import numpy as np
import torch
import xgboost as xgb
import signal_scorer as ss
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUT_DIR = "finbert_nlp_xgb_models"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# NLP gate: damps low-signal (noise) posts via a STEEP sigmoid.
# Below ~0.3 signal → crushed to near-zero (endorsements); above ~0.5 → near-full
# (real policy/geopolitics). Sharp transition around mid=0.45.
# Set GATE_ENABLED=False to see raw combined-model output.
GATE_ENABLED = True
GATE_K       = 16.0   # steepness — higher = sharper noise/signal cutoff
GATE_MID     = 0.45   # signal midpoint where gate = 0.5

LABELS = [
    ('SPY','📈','S&P 500'),('QQQ','💻','Nasdaq'),('DIA','🏭','Dow'),
    ('XLI','🔧','Industrials'),('XLF','🏦','Financials'),('XLE','⛽','Energy'),
    ('VIX','😨','VIX Fear'),
    ('OIL','🛢️','WTI Oil'),('GOLD','🥇','Gold'),('COPPER','🟫','Copper'),('NATGAS','🔥','NatGas'),
    ('EUR_USD','💶','EUR/USD'),('USD_JPY','💴','USD/JPY'),('GBP_USD','💷','GBP/USD'),
    ('USD_CNY','🇨🇳','USD/CNY'),('USD_CAD','🇨🇦','USD/CAD'),('USD_MXN','🇲🇽','USD/MXN'),
    ('USD_CHF','🇨🇭','USD/CHF'),('AUD_USD','🇦🇺','AUD/USD'),
    ('US10Y','📜','US 10Y'),('US2Y','📃','US 2Y'),
    ('BTC','₿','Bitcoin'),('ETH','Ξ','Ethereum'),
]

_tok = _bert = None

def load():
    global _tok, _bert
    cfg = json.load(open(f"{OUT_DIR}/config.json"))
    print(f"📥 Loading FinBERT on {DEVICE}...")
    _tok = AutoTokenizer.from_pretrained(cfg['finbert'])
    _bert = AutoModelForSequenceClassification.from_pretrained(
        cfg['finbert'], output_hidden_states=True).to(DEVICE).eval()
    models = {}
    for inst,_,_ in LABELS:
        p = f"{OUT_DIR}/{inst}_Impact.json"
        if os.path.exists(p):
            m = xgb.XGBRegressor(); m.load_model(p); models[inst]=m
    nlp   = ss.load_spacy()
    sbert = ss.load_sbert()
    print(f"✅ FinBERT + NLP scorer + {len(models)} XGBoost models\n")
    return cfg, models, nlp, sbert

def finbert_embed(text):
    """CLS + mean pooling — must match training v2."""
    enc = _tok([str(text)[:512]], return_tensors="pt", padding=True,
               truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = _bert(**enc)
    last = out.hidden_states[-1]
    cls  = last[:, 0, :]
    mask = enc['attention_mask'].unsqueeze(-1).float()
    mean = (last*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
    return torch.cat([cls, mean], dim=1).cpu().numpy()

def predict(text, cfg, models, nlp, sbert):
    feats = ss.score_single_post(text, nlp=nlp, sbert=sbert)
    nlp_vec = np.array([[float(feats.get(c,0.0)) for c in cfg['nlp_features']]])
    X = np.hstack([finbert_embed(text), nlp_vec])

    # NLP gate as a TRADE DECISION (not a position scaler):
    #   gate >= 0.5  (signal >= GATE_MID) -> real post: take the FULL move (×1.0)
    #   gate <  0.5                       -> noise: apply the tiny gate so the
    #                                        damped move is too small to trade
    import math
    # Gate signal = the SAME formula as the nlp_signal/sample_weight column in
    # truth_training_set_FINAL/TEST (build_final_training_set.compute_nlp_signal):
    #   mean( policy_intensity/8 capped, hawkish_risk/5 capped, scorer sample_weight )
    # The raw scorer sample_weight alone is on a much harsher scale (policy/25,
    # hawkish/11) and was crushing genuinely hawkish posts that training trusted.
    parts = []
    if feats.get('policy_intensity_score') is not None:
        parts.append(min(float(feats['policy_intensity_score']) / 8.0, 1.0))
    if feats.get('hawkish_risk_score') is not None:
        parts.append(min(float(feats['hawkish_risk_score']) / 5.0, 1.0))
    if feats.get('sample_weight') is not None:
        parts.append(float(feats['sample_weight']))
    signal = float(np.mean(parts)) if parts else float(feats.get('raw_score', 0.5))
    gate = 1.0/(1.0+math.exp(-GATE_K*(signal-GATE_MID))) if GATE_ENABLED else 1.0
    mult = 1.0 if gate >= 0.5 else gate

    out = {}
    for inst,_,_ in LABELS:
        if inst in models:
            raw = float(models[inst].predict(X)[0])
            out[inst] = raw * mult
    return out, signal, gate

def show(text, r, signal, gate):
    print("\n" + "─"*64)
    print(f"📝 {text[:120]}{'...' if len(text)>120 else ''}")
    print(f"   NLP signal={signal:.3f}  gate×{gate:.2f}" +
          ("  (real signal — FULL move)" if gate >= 0.5 else "  (noise — damped)"))
    print("─"*64)
    print("📊 PREDICTED 1-HOUR MARKET IMPACT (FinBERT+NLP→XGBoost):")
    for inst, emoji, name in LABELS:
        v = r.get(inst, 0.0)
        arrow = "▲" if v>0 else ("▼" if v<0 else "─")
        bar = "█"*min(int(abs(v)*4),20)
        print(f"  {emoji}  {name:<12} {arrow} {v:+.4f}%  {bar}")
    print("─"*64+"\n")

def main():
    cfg, models, nlp, sbert = load()
    print("="*64)
    print("  🤖 FinBERT + 📊 NLP + 🌲 XGBoost — 23 instruments")
    print(f"  NLP gate: {'ON' if GATE_ENABLED else 'OFF'}   Type 'quit' to exit")
    print("="*64+"\n")
    while True:
        try:
            t = input("📝 Enter post: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not t: continue
        if t.lower() in ('quit','exit','q'): break
        r, sig, gate = predict(t, cfg, models, nlp, sbert)
        show(t, r, sig, gate)


if __name__ == '__main__':
    main()
