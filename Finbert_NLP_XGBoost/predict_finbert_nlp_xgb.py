"""
predict_finbert_nlp_xgb.py
--------------------------
Predict: post -> [FinBERT embedding + NLP scorer features] -> XGBoost -> impacts.
Plus an optional NLP gate that damps obvious-noise posts (low NLP signal),
fixing FinBERT's endorsement-hallucination (Rick Allen -> NatGas +5% nonsense).

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

GATE_ENABLED = True
GATE_K       = 16.0
GATE_MID     = 0.45

TRADE_THRESHOLD = 0.1

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
    r"\bhappening now\b", r"\bas we speak\b", r"\bat this (?:very )?moment\b",
    r"\bat my direction\b", r"\bhereby announc(?:e|ing|ed)\b",
    r"\bwe got (?:him|her|them)\b",
    r"\bi (?:have |'ve |am |will |just |now |already |strongly |"
    r"formally |personally |immediately )*(?:ordered|directed|authorized|"
    r"launched|requested|demanded|instructed|called|spoke|"
    r"warned|signed|imposed|decided|asked|urged|pressed|"
    r"communicated|made clear|made it clear|finalized|concluded|reached a|"
    r"secured|completed|confirmed|announced)\b",
    r"\ba (?:letter|statement|message) (?:sent|addressed|delivered|issued)\b",
    r"\bnot anymore\b",
]
TEMPORAL_AMBIGUOUS = {
    r"this morning":   (5, 12),
    r"this afternoon": (12, 18),
    r"this evening":   (18, 24),
    r"tonight":        (18, 24),
}
TEMPORAL_DAMP = 0.15

TEMPORAL_PAST_DAMP_PREMARKET  = 0.5
TEMPORAL_PAST_DAMP_SESSION    = 0.15
TEMPORAL_PAST_DAMP_AFTERHOURS = 0.3


def temporal_factor(text, post_hour=None):
    """Return (factor, label) for the predict-time temporal gate."""
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
                if post_hour >= end:
                    return TEMPORAL_DAMP, f"past/stale ('{phrase.strip()}' already over at post time)"
                return 1.0, f"future/ongoing ('{phrase.strip()}' still ahead at post time)"

    return 1.0, "neutral"


LABELS = [
    ('SPY','\U0001f4c8','S&P 500'),('QQQ','\U0001f4bb','Nasdaq'),('DIA','\U0001f3ed','Dow'),
    ('XLI','\U0001f527','Industrials'),('XLF','\U0001f3e6','Financials'),('XLE','⛽','Energy'),
    ('VIX','\U0001f628','VIX Fear'),
    ('OIL','\U0001f6e2️','WTI Oil'),('GOLD','\U0001f947','Gold'),('COPPER','\U0001f7eb','Copper'),('NATGAS','\U0001f525','NatGas'),
    ('EUR_USD','\U0001f4b6','EUR/USD'),('USD_JPY','\U0001f4b4','USD/JPY'),('GBP_USD','\U0001f4b7','GBP/USD'),
    ('USD_CNY','\U0001f1e8\U0001f1f3','USD/CNY'),('USD_CAD','\U0001f1e8\U0001f1e6','USD/CAD'),('USD_MXN','\U0001f1f2\U0001f1fd','USD/MXN'),
    ('USD_CHF','\U0001f1e8\U0001f1ed','USD/CHF'),('AUD_USD','\U0001f1e6\U0001f1fa','AUD/USD'),
    ('US10Y','\U0001f4dc','US 10Y'),('US2Y','\U0001f4c3','US 2Y'),
    ('BTC','₿','Bitcoin'),('ETH','Ξ','Ethereum'),
]

_tok = _bert = None

TRADE_ACCURACY = {}

def load(model_dir=None):
    global _tok, _bert, TRADE_ACCURACY
    model_dir = model_dir or OUT_DIR
    cfg = json.load(open(f"{model_dir}/config.json"))

    acc_path = f"{model_dir}/trade_accuracy.json"
    if os.path.exists(acc_path):
        with open(acc_path, encoding='utf-8') as f:
            acc_report = json.load(f)
        TRADE_ACCURACY = acc_report.get('instruments', {})
        win = acc_report.get('window', {})
        print(f"  \U0001f4ca Loaded TRADE/SKIP filtered accuracy from {acc_path} "
              f"(backtest window {win.get('since','?')} -> {win.get('until','?')})")
    else:
        TRADE_ACCURACY = {}

    print(f"\U0001f4e5 Loading FinBERT on {DEVICE}...")
    _tok = AutoTokenizer.from_pretrained(cfg['finbert'])
    _bert = AutoModelForSequenceClassification.from_pretrained(
        cfg['finbert'], output_hidden_states=True).to(DEVICE).eval()
    models = {}
    for inst,_,_ in LABELS:
        p = f"{model_dir}/{inst}_Impact.json"
        if os.path.exists(p):
            m = xgb.XGBRegressor(); m.load_model(p); models[inst]=m
    if model_dir != OUT_DIR:
        print(f"  \U0001f4c2 Models loaded from {model_dir} (non-default --model-dir)")
    nlp   = ss.load_spacy()
    sbert = ss.load_sbert()
    print(f"✅ FinBERT + NLP scorer + {len(models)} XGBoost models\n")
    return cfg, models, nlp, sbert

def finbert_embed(text):
    """CLS + mean pooling -- must match training v2."""
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


def predict(text, cfg, models, nlp, sbert, post_ts=None,
            entity_weight: float = 1.0,
            event_weight:  float = 1.0,
            is_primary:    bool  = True,
            is_trump:      bool  = None):   # legacy alias for is_primary
    """Predict 1-hour market impacts for a post.

    entity_weight / event_weight / is_primary: supply these for geo-account posts
    so the NLP scorer applies the correct source-context weight (the same values
    used during training). Defaults (1.0, 1.0, True) match rank-0 TruthSocial behaviour.
    is_trump is a legacy alias for is_primary and takes precedence if supplied.
    """
    if is_trump is not None:
        is_primary = bool(is_trump)
    feats = ss.score_single_post(
        text, nlp=nlp, sbert=sbert,
        feature_cols=cfg['nlp_features'],
        entity_weight=entity_weight,
        event_weight=event_weight,
        is_primary=is_primary,
    )
    nlp_vec = np.array([[float(feats.get(c, 0.0)) for c in cfg['nlp_features']]])
    X = np.hstack([finbert_embed(text), nlp_vec])

    import math
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

    post_hour = post_ts.hour if post_ts is not None else None
    tfactor, tlabel = temporal_factor(text, post_hour=post_hour)
    mult *= tfactor

    out = {}
    for inst,_,_ in LABELS:
        if inst in models:
            raw = float(models[inst].predict(X)[0])
            out[inst] = raw * mult
    return out, signal, gate, tfactor, tlabel


def show(text, r, signal, gate, tfactor, tlabel,
         account=None, account_name=None, is_primary=True, is_trump=None,
         entity_weight=1.0, event_weight=1.0):
    if is_trump is not None:
        is_primary = bool(is_trump)
    print("\n" + "-"*64)
    # Source attribution line
    if is_primary:
        print("@realDonaldTrump - Donald J. Trump  (TruthSocial)")
    else:
        handle = account or "unknown"
        aname  = account_name or handle
        ew_str = f"entity_w={entity_weight:.2f}  event_w={event_weight:.2f}"
        print(f"@{handle} - {aname}  (X/Twitter)   {ew_str}")
    print(f"POST: {text[:120]}{'...' if len(text)>120 else ''}")
    print(f"   NLP signal={signal:.3f}  gate x{gate:.2f}" +
          ("  (real signal - FULL move)" if gate >= 0.5 else "  (noise - damped)"))
    print(f"   temporal={tlabel}  factor x{tfactor:.2f}" +
          ("  (past, no future cue -> damped)" if tfactor < 1.0 else
           "  (future/breaking/neutral -> no extra damping)"))
    total_mult = gate * tfactor if gate < 0.5 else tfactor
    if tfactor < 1.0:
        print(f"   TOTAL x{total_mult:.2f} - STALE / ALREADY PRICED IN - DON'T TRADE")
    elif gate < 0.5:
        print(f"   TOTAL x{total_mult:.2f} - LOW-SIGNAL NOISE - DON'T TRADE")
    else:
        print(f"   TOTAL x{total_mult:.2f} - FULL MOVE - TRADEABLE")
    print("-"*64)
    damped = total_mult != 1.0
    print("PREDICTED 1-HOUR MARKET IMPACT (FinBERT+NLP->XGBoost):")
    for inst, emoji, name in LABELS:
        v = r.get(inst, 0.0)
        arrow = "^" if v>0 else ("v" if v<0 else "-")
        bar = "#"*min(int(abs(v)*4),20)
        decision = "SKIP" if (damped or abs(v) < TRADE_THRESHOLD) else "TRADE"
        hist = TRADE_ACCURACY.get(inst)
        hist_str = ""
        if hist:
            if decision == "TRADE":
                if hist.get('trade_meaningful_acc') is not None:
                    hist_str = (f"  (filtered acc {hist['trade_meaningful_acc']*100:.1f}%, "
                                 f"n={hist['trade_meaningful_n']})")
                elif hist.get('trade_acc') is not None:
                    hist_str = f"  (hist TRADE acc {hist['trade_acc']*100:.1f}%, n={hist['trade_n']})"
            elif decision == "SKIP" and hist.get('skip_acc') is not None:
                hist_str = f"  (hist SKIP acc {hist['skip_acc']*100:.1f}%, n={hist['skip_n']})"
        print(f"  {emoji}  {name:<12} {arrow} {v:+.4f}%  {bar}  {decision}{hist_str}")
    print("-"*64+"\n")


def main():
    ap = argparse.ArgumentParser(description="Interactive FinBERT+NLP+XGBoost predictor.")
    ap.add_argument("--time", metavar="yyyymmddhhmm",
                    help="NY local time the post was made; only used to resolve "
                         "ambiguous time-of-day phrases in the temporal gate. Optional.")
    ap.add_argument("--model-dir", default=None,
                    help=f"Directory with <INST>_Impact.json + config.json "
                         f"(default {OUT_DIR}). Point at finbert_nlp_xgb_models_live "
                         f"to use fine-tuned models from backtest_simulator.py --fine-tune.")
    ap.add_argument("--entity-weight", type=float, default=1.0,
                    help="Entity weight for the post account (0-1). "
                         "Default 1.0 (Trump). For geo accounts: mention_count/28 "
                         "for leaders, or fixed institution weight (e.g. IDF=0.70).")
    ap.add_argument("--event-weight", type=float, default=1.0,
                    help="Event weight for the post account (0-1.3). "
                         "Default 1.0 (Trump / no active events). "
                         "Use EventManager.get_account_multiplier(handle) for geo posts.")
    ap.add_argument("--no-trump", dest="is_primary", action="store_false", default=True,
                    help="Pass when predicting for a geo X/Twitter account post. "
                         "Tells the NLP scorer is_primary=False so sample_weight "
                         "applies the SOURCE_DISCOUNT correctly.")
    ap.add_argument("--account", default=None, metavar="HANDLE",
                    help="Twitter/X handle of the posting account (e.g. ZelenskyyUa). "
                         "Display only -- shown in the source attribution line.")
    ap.add_argument("--account-name", default=None, metavar="NAME",
                    help="Display name of the posting account (e.g. 'Volodymyr Zelenskyy'). "
                         "Display only -- shown in the source attribution line.")
    args = ap.parse_args()
    post_ts = parse_stamp(args.time) if args.time else None

    cfg, models, nlp, sbert = load(args.model_dir)
    print("="*64)
    print("="*64)
    print("  FinBERT + NLP + XGBoost -- 23 instruments")
    print(f"  NLP gate: {'ON' if GATE_ENABLED else 'OFF'}   Type 'quit' to exit")
    if post_ts is not None:
        print(f"  Post time: {post_ts} (used for ambiguous time-of-day phrases)")
    if not args.is_primary:
        handle = args.account or "unknown"
        aname  = args.account_name or handle
        print(f"  Source: @{handle} - {aname}  "
              f"(entity_w={args.entity_weight:.2f}  event_w={args.event_weight:.2f})")
    print("="*64+"\n")
    while True:
        try:
            t = input("Enter post: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not t: continue
        if t.lower() in ('quit','exit','q'): break
        r, sig, gate, tfactor, tlabel = predict(
            t, cfg, models, nlp, sbert, post_ts=post_ts,
            entity_weight=args.entity_weight,
            event_weight=args.event_weight,
            is_primary=args.is_primary,
        )
        show(t, r, sig, gate, tfactor, tlabel,
             account=args.account, account_name=args.account_name,
             is_primary=args.is_primary)


if __name__ == "__main__":
    main()