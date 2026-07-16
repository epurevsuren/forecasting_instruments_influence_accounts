"""
gemma_analyst.py — Gemma 3 4B as the ANALYST (reads text, outputs impacts)
---------------------------------------------------------------------------
Peter's stacking architecture (from his train_stack_xgb.py), now on the
honest pipeline: Gemma READS each post with a financial-analyst prompt and
GENERATES a 23-instrument JSON of predicted 1-hour %% impacts. Those 23
numbers become first-class features next to the NLP scorer's — XGBoost is
the portfolio manager that learns when to trust the analyst, when to trust
the scorer, and how to correct the analyst's systematic biases.

Generation costs ~1-2s/post on an 8GB GPU, so results are CACHED in DuckDB
(gemma3_analyst_v1, keyed platform_id) with incremental checkpoints — the
big run happens once, exactly like the embedding cache.

Set GEMMA_ANALYST_LORA to a PEFT adapter dir (e.g. the old fine-tuned
./gemma3_truth_predictor) to layer the fine-tune on top of the base model.
"""
import os
import re
import json
import numpy as np
import pandas as pd

import db                      # folder-local shim -> ../database.db
import gemma_embedder as GE    # shares the loaded 4-bit model

ANALYST_TABLE = "gemma3_analyst_v1"
MAX_NEW_TOKENS = 320
GEN_BATCH = 4
CHECKPOINT_EVERY = 100

INSTRUMENTS = [
    'SPY','QQQ','DIA','XLI','XLF','XLE','VIX',
    'OIL','GOLD','COPPER','NATGAS',
    'EUR_USD','USD_JPY','GBP_USD','USD_CNY','USD_CAD','USD_MXN','USD_CHF','AUD_USD',
    'US10Y','US2Y','BTC','ETH',
]
ANALYST_COLS = [f"analyst_{i}" for i in INSTRUMENTS]
_KEYS = [f"{i}_Impact" for i in INSTRUMENTS]
# sanity caps per instrument class (same spirit as the training IMPACT_CAP)
_CAPS = {**{i: 3.0 for i in ('SPY','QQQ','DIA','XLI','XLF','XLE')},
         'VIX': 20.0, 'OIL': 8.0, 'GOLD': 5.0, 'COPPER': 5.0, 'NATGAS': 10.0,
         **{i: 2.5 for i in INSTRUMENTS if '_' in i},
         'US10Y': 8.0, 'US2Y': 8.0, 'BTC': 10.0, 'ETH': 12.0}

SYSTEM_PROMPT = """You are an expert financial analyst who predicts how political/geopolitical social-media posts move markets within one hour of posting.

You predict moves for 23 instruments:
- US equities/sectors: SPY (broad), QQQ (tech), DIA (Dow), XLI (industrials), XLF (financials), XLE (energy)
- Volatility: VIX (fear gauge — rises on uncertainty)
- Commodities: OIL, GOLD, COPPER, NATGAS
- FX (USD vs each): EUR, JPY, GBP, CNY (China), CAD (Canada), MXN (Mexico), CHF, AUD
- Bonds: US10Y, US2Y (yields rise when bonds sell off)
- Crypto: BTC, ETH

Cause-and-effect patterns:
- Military escalation / war threats -> VIX up, oil up, gold up, stocks down
- De-escalation / peace deals / averted attacks -> VIX down, oil down, stocks up (relief rally)
- Tariffs on a country -> that country's currency moves vs USD (China tariffs -> USD_CNY up, Mexico tariffs -> USD_MXN up), US stocks down, VIX up
- Trade deals / tariff pauses -> stocks up, VIX down
- Pro-crypto statements -> BTC and ETH up; sanctions/capital-control news can also move crypto (backdoor asset transfer)
- Fed pressure / rate-cut demands -> stocks up, gold up, yields (US10Y/US2Y) down
- Drilling / energy expansion -> oil down, XLE mixed
- Routine politics, endorsements, personal attacks, rally announcements -> near-zero across all

Read the post's MEANING, not just keywords. A post about CANCELLING an attack is de-escalation (calming), even though it mentions "attack" and "military". Distinguish genuine policy actions from political noise. Most routine posts should be near 0 for every instrument.

Respond with ONLY a JSON object (no other text) giving the predicted percentage move for each instrument, using these exact keys:
{"SPY_Impact":0.0,"QQQ_Impact":0.0,"DIA_Impact":0.0,"XLI_Impact":0.0,"XLF_Impact":0.0,"XLE_Impact":0.0,"VIX_Impact":0.0,"OIL_Impact":0.0,"GOLD_Impact":0.0,"COPPER_Impact":0.0,"NATGAS_Impact":0.0,"EUR_USD_Impact":0.0,"USD_JPY_Impact":0.0,"GBP_USD_Impact":0.0,"USD_CNY_Impact":0.0,"USD_CAD_Impact":0.0,"USD_MXN_Impact":0.0,"USD_CHF_Impact":0.0,"AUD_USD_Impact":0.0,"US10Y_Impact":0.0,"US2Y_Impact":0.0,"BTC_Impact":0.0,"ETH_Impact":0.0}

Positive = price up, negative = price down. Typical ranges: stocks ±2%, VIX ±15%, oil ±5%, gold ±3%, FX ±1.5%, yields ±6%, crypto ±8%. Routine posts ≈ 0."""


def _get_model():
    tok, model = GE._load_gemma()
    lora = os.environ.get("GEMMA_ANALYST_LORA")
    if lora and not getattr(model, "_analyst_lora_loaded", False):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora)
            model._analyst_lora_loaded = True
            print(f"  🎓 Analyst LoRA loaded from {lora}")
        except Exception as e:                            # noqa: BLE001
            print(f"  ⚠️  LoRA load failed ({e}) — using base model")
    return tok, model


def _parse_json(text):
    """Extract the impact JSON from a generation. Returns (23,) float32 or None."""
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = np.zeros(len(_KEYS), dtype=np.float32)
    for j, (k, inst) in enumerate(zip(_KEYS, INSTRUMENTS)):
        try:
            v = float(d.get(k, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        cap = _CAPS.get(inst, 5.0)
        out[j] = float(np.clip(v, -cap, cap))
    return out


def analyze_texts(texts):
    """Gemma READS each post and returns (n, 23) predicted impacts."""
    import torch
    tok, model = _get_model()
    dev = next(model.parameters()).device
    _pad_side = tok.padding_side
    tok.padding_side = "left"   # decoder batch generation REQUIRES left padding
    results = []
    for i in range(0, len(texts), GEN_BATCH):
        batch = [str(t)[:2000] for t in texts[i:i + GEN_BATCH]]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=True) for t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(dev)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for b in range(len(batch)):
            gen = tok.decode(out[b][enc['input_ids'].shape[1]:],
                             skip_special_tokens=True)
            vec = _parse_json(gen)
            results.append(vec if vec is not None else np.zeros(len(_KEYS), dtype=np.float32))
    tok.padding_side = _pad_side
    return np.vstack(results)


def analyst_features(platform_ids, texts):
    """(n, 23) analyst impact features, cache-first with incremental
    checkpoints (a crash never loses generated work)."""
    cached = db.read_table(ANALYST_TABLE)
    cache = {}
    if cached is not None and 'platform_id' in cached.columns:
        for _, r in cached.iterrows():
            cache[str(r['platform_id'])] = np.array(
                [r.get(c, 0.0) for c in ANALYST_COLS], dtype=np.float32)

    missing = [i for i, pid in enumerate(platform_ids) if pid not in cache]
    if missing:
        print(f"  🧠 Gemma ANALYST: {len(missing)}/{len(platform_ids)} posts to "
              f"analyze (generation ~1-2s/post, cached forever)...")
        for c0 in range(0, len(missing), CHECKPOINT_EVERY):
            part = missing[c0:c0 + CHECKPOINT_EVERY]
            preds = analyze_texts([texts[i] for i in part])
            rows = []
            for j, i in enumerate(part):
                cache[platform_ids[i]] = preds[j]
                rows.append({'platform_id': platform_ids[i],
                             **{c: float(preds[j][k]) for k, c in enumerate(ANALYST_COLS)}})
            db.append_table(ANALYST_TABLE, pd.DataFrame(rows))
            done = min(c0 + CHECKPOINT_EVERY, len(missing))
            print(f"  💾 analyst checkpoint {done}/{len(missing)} → {ANALYST_TABLE}")
    else:
        print(f"  ✅ All {len(platform_ids)} analyst rows found in {ANALYST_TABLE}")

    return np.vstack([cache[pid] for pid in platform_ids])
