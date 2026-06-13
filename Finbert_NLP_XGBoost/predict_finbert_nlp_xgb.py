"""
predict_finbert_nlp_xgb.py
--------------------------
Predict: post → [FinBERT embedding + NLP scorer features] → XGBoost → impacts.
Plus an optional NLP gate that damps obvious-noise posts (low NLP signal),
fixing FinBERT's endorsement-hallucination (Rick Allen → NatGas +5% nonsense).

Requires signal_scorer.py + scorer_config.json in this folder.
Run:  uv run python predict_finbert_nlp_xgb.py
"""
import os, re, json
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

# --------------------------------------------------------------- temporal gate
# PREDICT-TIME ONLY — no training/CSV changes, no new columns. Based on the
# "tense & horizon" finance-NLP finding that future-tense news produces a
# market reaction ~4x larger than past-tense news (already-known/priced-in),
# and that when a post mixes a past event with a future-looking action, the
# future-looking part is the new information and dominates.
#
# Priority order:
#   1. explicit FUTURE time/intent phrases -> full move (new information)
#   2. explicit PAST time phrases, no future cue -> damp (stale/priced-in)
#   3. anything else (incl. "breaking"/present-perfect announcements like
#      "I have just been informed") -> full move (default)
TEMPORAL_PAST_PHRASES = [
    r"last night", r"yesterday", r"this morning", r"earlier today",
    r"overnight", r"earlier this week", r"last week", r"last month",
    r"a (?:few|couple of) (?:hours|days) ago", r"this past (?:weekend|week)",
]
TEMPORAL_FUTURE_PHRASES = [
    r"tomorrow", r"tonight", r"next week", r"next month", r"next year",
    r"this weekend", r"later today", r"this afternoon", r"this evening",
    r"soon", r"shortly", r"upcoming", r"in the coming (?:days|weeks|months)",
]
TEMPORAL_FUTURE_INTENT = [
    r"\bwill\b", r"\bgoing to\b", r"\bplan(?:s|ning)? to\b", r"\bset to\b",
    r"\babout to\b", r"\bexpected to\b", r"\bscheduled to\b", r"\bintend(?:s)? to\b",
    r"\bmust\b.{0,20}\brespond", r"\bgoing to respond\b",
]
TEMPORAL_BREAKING = [
    r"\bbreaking\b", r"\bjust announced\b", r"\bmoments ago\b", r"\bright now\b",
    r"\bhappening now\b", r"\bas we speak\b",
]
TEMPORAL_DAMP = 0.15   # multiplier applied when a post is judged stale/priced-in


def temporal_factor(text):
    """Return (factor, label) for the predict-time temporal gate."""
    t = text.lower()
    has_future_time   = any(re.search(p, t) for p in TEMPORAL_FUTURE_PHRASES)
    has_future_intent = any(re.search(p, t) for p in TEMPORAL_FUTURE_INTENT)
    has_breaking      = any(re.search(p, t) for p in TEMPORAL_BREAKING)
    has_past_time     = any(re.search(p, t) for p in TEMPORAL_PAST_PHRASES)

    if has_future_time or has_future_intent or has_breaking:
        return 1.0, "future/new-info"
    if has_past_time:
        return TEMPORAL_DAMP, "past/stale"
    return 1.0, "neutral"


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
    # Gate signal = the SAME formula as the nlp_sig