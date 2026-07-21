"""
gemma_embedder.py — Gemma 3 4B as a local text encoder for the NLP+XGBoost pipeline
------------------------------------------------------------------------------------
Replaces gemma's CLS+mean embedding with Gemma-3-4B hidden-state pooling.

DESIGN (research-backed, 2026-07-17):
  * Decoder-only LLMs have no [CLS]; the two viable poolings are LAST-TOKEN
    (the autoregressive summary position) and MEAN over tokens. Literature is
    split: last-token wins on some retrieval suites but suffers recency bias /
    "oversquashing"; mean pooling is more robust on MTEB averages. We CONCAT
    BOTH (mean 2560 + last-token 2560 = 5120 dims), mirroring the CLS+mean
    concat the gemma pipeline used — downstream PCA (fit on the train slice
    in the canonical trainer) compresses to 128 dims anyway, so carrying both
    views costs nothing at model time.
  * 4-bit quantization (the unsloth pre-quantized bnb checkpoint Peter already
    has locally) so the 4B model fits an 8GB GPU with headroom for batching.
  * Single forward pass per post — NO text generation. Peter's previous
    attempt fine-tuned Gemma to EMIT JSON impact numbers; that failed on raw
    data and is also ~1000x slower per post than encoding. Here Gemma is the
    reader, XGBoost stays the forecaster (same division of labor as gemma).
  * torch.inference_mode + right-truncation at 256 tokens (Truth/X posts are
    short; 256 covers >99% of them, matching the gemma path).

The fine-tuned LoRA from the earlier experiment (./gemma3_truth_predictor in
the old attempt) can be layered on later via PEFT — start with the base
instruct model so results are comparable to the gemma baseline.
"""
import os
import numpy as np

# Prefer Peter's local pre-quantized checkpoint (fits 8GB VRAM); fall back to
# the official weights if unsloth's mirror is unavailable.
GEMMA_ID_4BIT = os.environ.get("GEMMA_MODEL_ID", "unsloth/gemma-3-4b-it-unsloth-bnb-4bit")
GEMMA_ID_FULL = "unsloth/gemma-3-4b-it"   # ungated mirror (google/gemma-3-4b-it needs HF auth)
HIDDEN        = 2560                    # Gemma-3-4B hidden size
N_EMB         = HIDDEN * 2              # mean(2560) + last-token(2560)
EMB_TABLE     = "gemma3_embeddings_v1"  # DuckDB cache
MAX_LEN       = 256
BATCH         = 8                       # 4B model — smaller batches than gemma's 32

_tok_model = [None, None]


def _probe(tok, model):
    """One tiny forward: hidden states must come back real and finite.
    Catches broken quant states / patched forwards BEFORE the 190k-post run."""
    import torch
    enc = tok(["probe"], return_tensors="pt")
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    if hs is None or not torch.isfinite(hs[-1]).all():
        raise RuntimeError("probe forward returned no/NaN hidden states")


def _try_unsloth():
    """Peter's unsloth-studio venv: FastModel is THE loader that understands
    the unsloth dynamic-4bit checkpoint layout (vanilla transformers leaves
    LinearFP4 layers without quant_state -> AssertionError in bitsandbytes
    fix_4bit_weight_quant_state_from_module). His old train_gemma3.py loaded
    exactly this way, on this box, successfully."""
    from unsloth import FastModel
    print(f"📥 Loading {GEMMA_ID_4BIT} via unsloth FastModel ...")
    model, tok = FastModel.from_pretrained(
        model_name     = GEMMA_ID_4BIT,
        max_seq_length = MAX_LEN,
        load_in_4bit   = True,
        full_finetuning= False,
    )
    model.eval()
    return tok, model


def _try_transformers_nf4():
    """Standard path: quantize the FULL (ungated unsloth mirror) checkpoint
    on load with plain bitsandbytes NF4 — a layout vanilla transformers
    fully owns, so every layer gets a proper quant_state."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    mid = GEMMA_ID_FULL
    print(f"📥 Loading {mid} with BitsAndBytesConfig NF4 (quantize on load)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, quantization_config=bnb, device_map={"": 0})
    model.eval()
    return tok, model


def _try_cpu():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"📥 Loading {GEMMA_ID_FULL} on CPU (last resort — SLOW)...")
    tok = AutoTokenizer.from_pretrained(GEMMA_ID_FULL)
    model = AutoModelForCausalLM.from_pretrained(GEMMA_ID_FULL,
                                                 torch_dtype=torch.float32)
    model.eval()
    return tok, model


def _load_gemma():
    if _tok_model[0] is None:
        import torch
        loaders = ([_try_unsloth, _try_transformers_nf4, _try_cpu]
                   if torch.cuda.is_available() else [_try_cpu])
        last_err = None
        for fn in loaders:
            try:
                tok, model = fn()
                _probe(tok, model)
                _tok_model[0], _tok_model[1] = tok, model
                print("  ✅ probe forward OK — hidden states are real")
                break
            except Exception as e:                        # noqa: BLE001
                last_err = e
                print(f"  ⚠️  {fn.__name__} failed ({type(e).__name__}: {e}) — next loader")
                _tok_model[0] = _tok_model[1] = None
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        if _tok_model[1] is None:
            raise RuntimeError(f"No Gemma loader succeeded (last: {last_err})")
    return _tok_model


def embed_texts(texts):
    """Batch-encode texts -> (n, 5120) float32: [mean-pool | last-token] of the
    final hidden layer. Drop-in replacement for the gemma embed_texts."""
    import torch
    tok, model = _load_gemma()
    dev = next(model.parameters()).device
    embs = []
    print(f"🔢 Gemma-embedding {len(texts)} posts (mean + last-token pooling)...")
    for i in range(0, len(texts), BATCH):
        batch = [str(t)[:2000] for t in texts[i:i + BATCH]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN)
        enc = {k: v.to(dev) for k, v in enc.items()}
        # embeddings must ALWAYS come from the BASE model (they must match
        # the gemma3_embeddings_v1 cache) — if an analyst LoRA is attached
        # to the shared model, disable it for this forward pass.
        import contextlib
        _ctx = (model.disable_adapter() if hasattr(model, "disable_adapter")
                else contextlib.nullcontext())
        with _ctx, torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        last = out.hidden_states[-1].float()               # (B, T, 2560)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        mean = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        # last REAL token per row (padding-aware)
        idx = enc['attention_mask'].sum(1) - 1             # (B,)
        lastvec = last[torch.arange(last.shape[0]), idx]   # (B, 2560)
        embs.append(torch.cat([mean, lastvec], dim=1).cpu().numpy().astype(np.float32))
        done = i + len(batch)
        if done % 200 == 0 or done == len(texts):
            print(f"   {done}/{len(texts)}")
    return np.vstack(embs)


def embed_one(text):
    """(1, 5120) embedding for a single post — predict-time replacement for
    gemma_embed()."""
    return embed_texts([text])
