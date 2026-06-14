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
TEMPORAL_DAMP = 0.15   # default multiplier when a post is judged stale/priced-in
                       # (used when post_hour is unknown, i.e. no --time given)

# Graduated damping for explicit PAST-time phrases ("last night", "overnight",
# etc.), based on WHEN (NY local, from --time) the post lands relative to the
# US cash-equity session (9:30-16:00 ET). Trump often sleeps through overnight
# events and only posts hours later -- how stale that is depends on whether the
# market has already had a chance to trade on it:
#   - before the open  -> not yet digested by cash equities; premarket/futures
#                          may still react -> partial damp
#   - during the session -> already traded through the morning -> fully stale
#   - after the close    -> cash closed, but FX/crypto/futures trade overnight
#                          -> partial damp
TEMPORAL_PAST_DAMP_PREMARKET  = 0.5    # post_hour < 9   (before the open)
TEMPORAL_PAST_DAMP_SESSION    = 0.15   # 9 <= post_hour < 16 (market open)
TEMPORAL_PAST_DAMP_AFTERHOURS = 0.3    # post_hour >= 16 (after the close)


def temporal_factor(text, post_hour=None):
    """Return (factor, label) for the predict-time temporal gate.

    post_hour: 0-23 local hour the post was made (from --time), or None if
    unknown. Used to resolve TEMPORAL_AMBIGUOUS phrases, and to grade how
    stale an explicit PAST-time phrase is relative to market hours.
    """
    t = text.lower()
    has_future_time = any(re.search(p, t) for p in TEMPORAL_FUTURE_PHRASES)
    has_breaking     = any(re.search(p, t) for p in TEMPORAL_BREAKING)
    has_past_time    = any(re.search(p, t) for p in TEMPORAL_PAST_PHRASES)

    if has_past_time and not (has_future_time or has_breaking):
        if post_hour is None:
            return TEMPORAL_DAMP, "past/stale"
        if post_hour < 9:
            return TEMPORAL_PAST_DAMP_PREMARKET, "past/stale (posted premarket -- not yet traded by cash equities)"
        if post_hour < 16:
            return TEMPORAL_PAST_DAMP_SESSION, "past/stale (posted during session -- already traded)"
        return TEMPORAL_PAST_DAMP_AFTERHOURS, "past/stale (posted after close -- only FX/crypto/futures react overnight)"
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

def load(model_dir=None):
    global _tok, _bert
    model_dir = model_dir or OUT_DIR
    cfg = json.load(open(f"{model_dir}/config.json"))
    print(f"📥 Loading FinBERT on {DEVICE}...")
    _tok = AutoTokenizer.from_pretrained(cfg['finbert'])
    _bert = AutoModelForSequenceClassification.from_pretrained(
        cfg['finbert'], output_hidden_states=True).to(DEVICE).eval()
    models = {}
    for inst,_,_ in LABELS:
        p = f"{model_dir}/{inst}_Impact.json"
        if os.path.exists(p):
            m = xgb.XGBRegressor(); m.load_model(p); models[inst]=m
    if model_dir != OUT_DIR:
        print(f"  📂 Models loaded from {model_dir} (non-default --model-dir)")
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

def parse_stamp(s):
    """yyyymmddhhmm, NY local time -> tz-aware Timestamp."""
    return pd.Timestamp(datetime.datetime.strptime(s.strip(), "%Y%m%d%H%M"), tz=NY)


def predict(text, cfg, models, nlp, sbert, post_ts=None):
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

    # Predict-time temporal gate (see TEMPORAL_* above): stale/already-priced
    # posts get damped further, regardless of the NLP gate result.
    post_hour = post_ts.hour if post_ts is not None else None
    tfactor, tlabel = temporal_factor(text, post_hour=post_hour)
    mult *= tfactor

    out = {}
    for inst,_,_ in LABELS:
        if inst in models:
            raw = float(models[inst].predict(X)[0])
            out[inst] = raw * mult
    return out, signal, gate, tfactor, tlabel

def show(text, r, signal, gate, tfactor, tlabel):
    print("\n" + "─"*64)
    print(f"📝 {text[:120]}{'...' if len(text)>120 else ''}")
    print(f"   NLP signal={signal:.3f}  gate×{gate:.2f}" +
          ("  (real signal — FULL move)" if gate >= 0.5 else "  (noise — damped)"))
    print(f"   temporal={tlabel}  factor×{tfactor:.2f}" +
          ("  (last night/past, no future cue -> damped)" if tfactor < 1.0 else
           "  (future/breaking/neutral -> no extra damping)"))
    total_mult = gate * tfactor if gate < 0.5 else tfactor
    if tfactor < 1.0:
        print(f"   ⚠️  TOTAL ×{total_mult:.2f} — STALE / ALREADY PRICED IN — DON'T TRADE")
    elif gate < 0.5:
        print(f"   ⚠️  TOTAL ×{total_mult:.2f} — LOW-SIGNAL NOISE — DON'T TRADE")
    else:
        print(f"   ✅ TOTAL ×{total_mult:.2f} — FULL MOVE — TRADEABLE")
    print("─"*64)
    print("📊 PREDICTED 1-HOUR MARKET IMPACT (FinBERT+NLP→XGBoost):")
    for inst, emoji, name in LABELS:
        v = r.get(inst, 0.0)
        arrow = "▲" if v>0 else ("▼" if v<0 else "─")
        bar = "█"*min(int(abs(v)*4),20)
        print(f"  {emoji}  {name:<12} {arrow} {v:+.4f}%  {bar}")
    print("─"*64+"\n")

def main():
    ap = argparse.ArgumentParser(description="Interactive FinBERT+NLP+XGBoost predictor.")
    ap.add_argument("--time", metavar="yyyymmddhhmm",
                    help="NY local time the post was made; only used to resolve "
                         "ambiguous time-of-day phrases (this morning/tonight/etc.) "
                         "in the temporal gate. Optional.")
    ap.add_argument("--model-dir", default=None,
                    help=f"Directory with <INST>_Impact.json + config.json "
                         f"(default {OUT_DIR}). Point this at "
                         f"finbert_nlp_xgb_models_live to use models fine-tuned "
                         f"by backtest_simulator.py --fine-tune.")
    args = ap.parse_args()
    post_ts = parse_stamp(args.time) if args.time else None

    cfg, models, nlp, sbert = load(args.model_dir)
    print("="*64)
    print("  🤖 FinBERT + 📊 NLP + 🌲 XGBoost — 23 instruments")
    print(f"  NLP gate: {'ON' if GATE_ENABLED else 'OFF'}   Type 'quit' to exit")
    if post_ts is not None:
        print(f"  Post time: {post_ts} (used for ambiguous time-of-day phrases)")
    print("="*64+"\n")
    while True:
        try:
            t = input("📝 Enter post: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not t: continue
        if t.lower() in ('quit','exit','q'): break
        r, sig, gate, tfactor, tlabel = predict(t, cfg, models, nlp, sbert, post_ts=post_ts)
        show(t, r, sig, gate, tfactor, tlabel)


if __name__ == '__main__':
    main()
