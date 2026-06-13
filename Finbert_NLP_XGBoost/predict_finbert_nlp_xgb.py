"""
predict_finbert_nlp_xgb.py
--------------------------
Predict: post → [FinBERT embedding + NLP scorer features] → XGBoost → impacts.
Plus an optional NLP gate that damps obvious-noise posts (low NLP signal),
fixing FinBERT's endorsement-hallucination (Rick Allen → NatGas +5% nonsense).

Requires signal_scorer.py + scorer_config.json in this folder.
Run:  uv run python predict_finbert_nlp_xgb.py
"""
import os, re, json, argparse, datetime
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import signal_scorer as ss
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NY = 'America/New_York'

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
#   1. explicit PAST time phrases ("last night", "yesterday", ...) -> damp
#      (stale/priced-in) UNLESS a concrete dated FUTURE phrase or a breaking/
#      just-now marker is also present (case 2 below wins).
#   2. explicit dated FUTURE phrases ("tomorrow", "next week", ...) or
#      BREAKING/just-now markers -> full move (new information)
#   3. AMBIGUOUS time-of-day phrases ("this morning"/"tonight"/etc.) -> resolved
#      using --time (the post's own timestamp), if given; if "now" has already
#      passed that part of the day relative to the post, it's past/stale,
#      otherwise it's future/neutral. Without --time, treated as neutral.
#   4. anything else -> full move (default, neutral)
#
# NOTE: vague modal/intent phrases ("will", "must respond", "going to", etc.)
# are deliberately NOT treated as overriding signals on their own — this kind
# of rhetorical language is extremely common boilerplate and is not reliable
# evidence of genuinely new, dated information. Only an explicit dated future
# reference or a breaking/just-now marker overrides a past-time phrase.
TEMPORAL_PAST_PHRASES = [
    r"last night", r"yesterday", r"earlier today",
    r"overnight", r"earlier this week", r"last week", r"last month",
    r"a (?:few|couple of) (?:hours|days) ago", r"this past (?:weekend|week)",
]
TEMPORAL_FUTURE_PHRASES = [
    r"tomorrow", r"next week", r"next month", r"next year",
    r"this weekend", r"later today",
    r"soon", r"shortly", r"upcoming", r"in the coming (?:days|weeks|months)",
]
TEMPORAL_BREAKING = [
    r"\bbreaking\b", r"\bjust announced\b", r"\bmoments ago\b", r"\bright now\b",
    r"\bhappening now\b", r"\bas we speak\b",
]
# phrase -> (start_hour, end_hour) of the day it refers to (24h, local post time)
TEMPORAL_AMBIGUOUS = {
    r"this morning":   (5, 12),
    r"this afternoon": (12, 18),
    r"this evening":   (18, 24),
    r"tonight":        (18, 24),
}
TEMPORAL_DAMP = 0.15   # multiplier applied when a post is judged stale/priced-in


def temporal_factor(text, post_hour=None):
    """Return (factor, label) for the predict-time temporal gate.

    post_hour: 0-23 local hour the post was made (from --time), or None if
    unknown. Only used to resolve TEMPORAL_AMBIGUOUS phrases.
    """
    t = text.lower()
    has_future_time = any(re.search(p, t) for p in TEMPORAL_FUTURE_PHRASES)
    has_breaking     = any(re.search(p, t) for p in TEMPORAL_BREAKING)
    has_past_time    = any(re.search(p, t) for p in TEMPORAL_PAST_PHRASES)

    if has_past_time and not (has_future_time or has_breaking):
        return TEMPORAL_DAMP, "past/stale"
    if has_future_time or has_breaking:
        return 1.0, "future/new-info"

    if post_hour is not None:
        for phrase, (start, end) in TEMPORAL_AMBIGUOUS.items():
            if re.search(phrase, t):
                if post_hour >= end:      # that part of the day has already passed
                    return TEMPORAL_DAMP, f"past/stale ('{phrase.strip()}' already over at post time)"
                return 1.0, f"future/ongoing ('{phrase.strip()}' still ahead at post time)"

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
            m