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
# Full speed by default (the Razer handles its own thermals; the earlier
# freeze was Razer Synapse, not heat). Env knobs remain for emergencies:
# GEMMA_ANALYST_THROTTLE=0.5 adds a cooldown between batches.
GEN_BATCH  = int(os.environ.get("GEMMA_ANALYST_BATCH", "4"))
THROTTLE_S = float(os.environ.get("GEMMA_ANALYST_THROTTLE", "0"))
CHECKPOINT_EVERY = 100

# Instruments + caps come from THE MASTER REGISTRY (DP/instruments.json) —
# never hardcoded. Registry order defines the ANALYST_COLS feature order,
# identical everywhere (train loads INSTRUMENTS from the same file, so the
# columns always line up). impact_cap is Peter's per-instrument label cap —
# the analyst's JSON output is clipped to the same bounds as training
# labels, so an LLM hallucination ("BTC +50%") can't poison a feature.
# NOTE for cTrader integration: caps/impacts here are PERCENT moves; the
# pip/lot conversion happens ONLY in Layer 3 via the registry's "ctrader"
# blocks (pip_position, lot_size, leverage) + ctrader_bridge — FX percent
# moves are small but at 30:1 class leverage a 0.2% move is a 6% margin
# return, which is why FX must never be filtered or capped by raw percent
# thresholds borrowed from equities.
_INSTRUMENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DP", "instruments.json")
with open(_INSTRUMENTS_FILE, encoding="utf-8") as _f:
    _REG = json.load(_f)["instruments"]
INSTRUMENTS  = list(_REG.keys())
ANALYST_COLS = [f"analyst_{i}" for i in INSTRUMENTS]
_KEYS = [f"{i}_Impact" for i in INSTRUMENTS]
_CAPS = {i: float(v.get("impact_cap", 5.0)) for i, v in _REG.items()}
# cTrader class leverage per instrument (registry) — the live display ranks
# calls by MARGIN-RELATIVE return (|pred%| x leverage): a 0.2% EUR_USD move
# at 30:1 (=6% margin) outranks a 3% VIX move at low class leverage. Same
# ranking Layer 3 uses for burst allocation.
_LEV = {i: float(v.get("ctrader", {}).get("leverage", 10)) for i, v in _REG.items()}

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
__JSON_TEMPLATE__

Positive = price up, negative = price down. Typical ranges: stocks ±2%, VIX ±15%, oil ±5%, gold ±3%, FX ±1.5%, yields ±6%, crypto ±8%. Routine posts ≈ 0."""

# exact-keys template generated FROM THE REGISTRY so adding an instrument to
# instruments.json flows into the analyst prompt with no code edit
SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "__JSON_TEMPLATE__",
    "{" + ",".join(f'"{k}":0.0' for k in _KEYS) + "}")


def _get_model():
    tok, model = GE._load_gemma()
    # unsloth fast-inference kernels for generation (~2x) — no-op elsewhere
    if not getattr(model, "_fast_inference_on", False):
        try:
            from unsloth import FastModel
            FastModel.for_inference(model)
            model._fast_inference_on = True
            print("  ⚡ unsloth fast-inference enabled for generation")
        except Exception:
            pass
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


def _gen_batch(texts, tok, model, dev):
    """One padded generation batch -> list of parsed (23,) vectors."""
    import torch
    # BYPASS the (unsloth-patched, multimodal) processor: its __call__ drops
    # the padding request with left padding -> ragged tensors ("expected
    # sequence of length 762 ... got 761"). The underlying TEXT tokenizer
    # pads reliably and model.generate takes input_ids/attention_mask.
    _t = getattr(tok, "tokenizer", tok)
    prompts = [_t.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": t}],
        tokenize=False, add_generation_prompt=True) for t in texts]
    _side = _t.padding_side
    _t.padding_side = "left"    # decoder batch generation REQUIRES left padding
    try:
        enc = _t(prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=1024,
                 add_special_tokens=False)   # chat template already added <bos>
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=_t.pad_token_id or _t.eos_token_id)
    finally:
        _t.padding_side = _side
    vecs = []
    for b in range(len(texts)):
        gen = _t.decode(out[b][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        v = _parse_json(gen)
        vecs.append(v if v is not None else np.zeros(len(_KEYS), dtype=np.float32))
    return vecs


def analyze_texts(texts, meta=None):
    """Gemma READS each post and returns (n, len(INSTRUMENTS)) impacts.
    `meta` (optional): account names aligned with `texts`, for the live log.
    Falls back to one-post-at-a-time if a batch fails — a single bad post
    can't kill an hours-long cached run."""
    import time
    tok, model = _get_model()
    dev = next(model.parameters()).device
    results = []
    for i in range(0, len(texts), GEN_BATCH):
        if THROTTLE_S > 0 and i > 0:
            time.sleep(THROTTLE_S)   # thermal cooldown between batches
        batch = [str(t)[:2000] for t in texts[i:i + GEN_BATCH]]
        try:
            vecs = _gen_batch(batch, tok, model, dev)
            # LIVE: account + FULL post text, calls ranked by MARGIN-RELATIVE
            # return (|pred%| x class leverage from the registry) — what is
            # most TRADEABLE, not what has the biggest raw percent.
            for _k, (_t, _v) in enumerate(zip(batch, vecs)):
                _mret = np.array([abs(_v[o]) * _LEV.get(INSTRUMENTS[o], 10.0)
                                  for o in range(len(_v))])
                _top = np.argsort(-_mret)[:3]
                _call = ", ".join(
                    f"{INSTRUMENTS[o]} {_v[o]:+.2f}% (≈{_v[o] * _LEV.get(INSTRUMENTS[o], 10.0):+.0f}% margin)"
                    for o in _top if _mret[o] >= 1.0) or "flat"
                _who = f"@{meta[i + _k]}  " if meta is not None and meta[i + _k] else ""
                _txt = re.sub(r"\s+", " ", str(_t)).strip()
                print(f"    🧠 {_who}{_txt}")
                print(f"       → {_call}")
            results.extend(vecs)
        except Exception as e:                            # noqa: BLE001
            print(f"  ⚠️  batch generate failed ({type(e).__name__}: {str(e)[:80]}) "
                  f"— retrying one-by-one")
            for t in batch:
                try:
                    results.extend(_gen_batch([t], tok, model, dev))
                except Exception as e2:                   # noqa: BLE001
                    print(f"  ⚠️  single-post generate failed ({str(e2)[:60]}) — zeros")
                    results.append(np.zeros(len(_KEYS), dtype=np.float32))
    return np.vstack(results)


def analyst_features(platform_ids, texts, accounts=None):
    """(n, len(INSTRUMENTS)) analyst impact features, cache-first with
    incremental checkpoints (a crash never loses generated work).
    `accounts` (optional): account names for the live log."""
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
            print(f"  🗂️  analyzing posts {c0 + 1}-{min(c0 + len(part), len(missing))} "
                  f"of {len(missing)}...")
            preds = analyze_texts(
                [texts[i] for i in part],
                meta=[accounts[i] for i in part] if accounts is not None else None)
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
