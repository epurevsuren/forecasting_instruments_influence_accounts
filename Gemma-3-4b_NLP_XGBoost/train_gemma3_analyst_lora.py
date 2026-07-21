"""
train_gemma3_analyst_lora.py — TEACH the analyst what actually moves markets
-----------------------------------------------------------------------------
Peter's original idea (train_gemma3.py, Unsloth LoRA-SFT), rebuilt on the
honest pipeline. The zero-shot analyst was measured at corr ±0.04 vs actual
1-hour moves — it reads well but was never TAUGHT market outcomes. This
fine-tunes Gemma-3-4B on (post -> ACTUAL 23-instrument impacts):

  * data from DuckDB training_set_FINAL (not the old CSVs)
  * sample_weight >= 0.2 row filter (same as the XGB trainer — no noise rows)
  * CHRONOLOGICAL 85% cut: the last 15% of the decade is NEVER seen in
    fine-tuning, so backtests on it stay honest
  * high-signal oversampling (weight > 0.5 -> x4), replacing per-row weights
    (Peter's own fix — custom weighted trainers broke unsloth's loss path)
  * targets built from THE REGISTRY (same keys/caps as gemma_analyst.py)
  * resumable: checkpoints every 200 steps, auto-resume if interrupted

After training:
  $env:GEMMA_ANALYST_LORA = "D:\\...\\Gemma-3-4b_NLP_XGBoost\\gemma3_analyst_lora"
  python train_gemma3_nlp_xgb.py     # regenerates analyst cache (table gemma3_analyst_ft1)
  python backtest_simulator.py ...   # fine-tuned analyst calls in every decision

Run:  & "$env:USERPROFILE\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe" train_gemma3_analyst_lora.py
"""
import os
os.environ.setdefault("UNSLOTH_FORCE_XFORMERS", "1")
# RTX 5070 Laptop: the compiled FlexAttention BACKWARD kernel needs 114,688B
# of shared memory but this GPU's hardware limit is 101,376B ->
# InductorError OutOfMemoryError at step 0. The flex-disable knob alone was
# NOT honored by unsloth 2026.7.3, so ALL escape hatches are on: no flex,
# no torch.compile (~25% slower, but it trains instead of crashing).
os.environ.setdefault("UNSLOTH_DISABLE_FLEX_ATTENTION", "1")
os.environ.setdefault("UNSLOTH_ENABLE_FLEX_ATTENTION", "0")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
# expandable_segments reduces fragmentation — on 8GB the training run sat
# right at the VRAM edge; fragmentation after the step-200 checkpoint
# pushed it into Windows sysmem paging (23 s/step -> 450 s/step).
os.environ.setdefault("PYTORCH_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:512")

from unsloth import FastModel                       # BEFORE transformers
from unsloth.chat_templates import train_on_responses_only

import sys
import json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.normpath(os.path.join(_HERE, "..", "DP"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import db
from gemma_analyst import SYSTEM_PROMPT, INSTRUMENTS, _KEYS, _CAPS
from gemma_embedder import GEMMA_ID_4BIT

SAVE_PATH        = os.path.join(_HERE, "gemma3_analyst_lora")
MAX_SEQ_LENGTH   = 1024
TRAIN_MIN_WEIGHT = 0.20   # same noise filter as the XGB trainer
TRAIN_FRAC       = 0.85   # chronological cut — last 15% NEVER seen (honest OOS)
OVERSAMPLE_W     = 0.50   # rows above this weight are repeated...
OVERSAMPLE_X     = 4      # ...this many times (replaces per-row loss weights)


def build_dataset(tok):
    print("📂 Loading training_set_FINAL from DuckDB...")
    df = db.read_table("training_set_FINAL")
    if df is None:
        sys.exit("❌ training_set_FINAL missing — run build_final_training_set.py first")
    df['text'] = df['text'].fillna("")
    df['date'] = pd.to_datetime(df['date'], utc=True, format='mixed')
    df = df.sort_values('date').reset_index(drop=True)

    n0 = len(df)
    df = df[pd.to_numeric(df['sample_weight'], errors='coerce').fillna(0) >= TRAIN_MIN_WEIGHT]
    df = df.reset_index(drop=True)
    cut = int(len(df) * TRAIN_FRAC)
    holdout_start = df['date'].iloc[cut] if cut < len(df) else None
    df = df.iloc[:cut]
    print(f"  🧹 filter w>={TRAIN_MIN_WEIGHT}: {len(df)}/{n0} rows | "
          f"chronological cut at {holdout_start} (last 15% NEVER trained on)")

    # oversample high-signal rows (Peter's fix for broken weighted trainers)
    hi = df[pd.to_numeric(df['sample_weight'], errors='coerce') > OVERSAMPLE_W]
    df = pd.concat([df] + [hi] * (OVERSAMPLE_X - 1), ignore_index=True)
    df = df.sample(frac=1.0, random_state=3407).reset_index(drop=True)  # shuffle WITHIN the train window
    print(f"  🔁 oversampled {len(hi)} high-signal rows x{OVERSAMPLE_X} -> {len(df)} examples")

    def to_target(row):
        d = {}
        for inst, key in zip(INSTRUMENTS, _KEYS):
            v = pd.to_numeric(row.get(key), errors='coerce')
            cap = _CAPS.get(inst, 5.0)
            d[key] = round(float(np.clip(0.0 if pd.isna(v) else v, -cap, cap)), 2)
        return json.dumps(d, separators=(',', ':'))

    texts = []
    for _, row in df.iterrows():
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(row['text'])[:2000]},
                {"role": "assistant", "content": to_target(row)}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False))
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def main():
    print(f"📥 Loading {GEMMA_ID_4BIT} via unsloth...")
    model, tok = FastModel.from_pretrained(
        model_name      = GEMMA_ID_4BIT,
        max_seq_length  = MAX_SEQ_LENGTH,
        load_in_4bit    = True,
        full_finetuning = False,
    )
    model = FastModel.get_peft_model(
        model,
        r                          = 8,
        lora_alpha                 = 16,
        lora_dropout               = 0,
        bias                       = "none",
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth",
        random_state               = 3407,
    )

    ds = build_dataset(tok)

    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tok,
        train_dataset = ds,
        args = SFTConfig(
            # batch 1 + accum 8 = SAME effective batch (8) with roughly half
            # the activation VRAM — keeps the run off the Windows sysmem-
            # paging cliff that collapsed 23 s/step to 450 s/step at 8GB.
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            # First adapter doesn't need the full epoch (~28h healthy): loss
            # plateaus early; ~1500 steps (~12k examples incl. oversampled
            # high-signal) tests the hypothesis in ~10-12h. Override:
            # $env:SFT_MAX_STEPS = "-1"  (full epoch)
            max_steps                   = int(os.environ.get("SFT_MAX_STEPS", "1500")),
            num_train_epochs            = 1,
            learning_rate               = 2e-4,
            lr_scheduler_type           = "cosine",
            warmup_ratio                = 0.03,
            logging_steps               = 10,
            save_steps                  = 200,      # resumable checkpoints
            save_total_limit            = 2,
            output_dir                  = SAVE_PATH + "_ckpt",
            report_to                   = "none",
            seed                        = 3407,
        ),
    )
    # loss only on the ANALYST'S ANSWER (the JSON), not the prompt — the
    # native unsloth path (custom weighted trainers broke gradients before)
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part    = "<start_of_turn>model\n",
    )

    _resume = os.path.isdir(SAVE_PATH + "_ckpt") and any(
        d.startswith("checkpoint") for d in os.listdir(SAVE_PATH + "_ckpt"))
    print(f"🚀 SFT starting ({'resuming' if _resume else 'fresh'})...")
    trainer.train(resume_from_checkpoint=_resume or None)

    model.save_pretrained(SAVE_PATH)
    tok.save_pretrained(SAVE_PATH)
    print(f"\n✅ Analyst LoRA saved → {SAVE_PATH}")
    print("Next steps:")
    print(f'  $env:GEMMA_ANALYST_LORA = "{SAVE_PATH}"')
    print("  re-run train_gemma3_nlp_xgb.py  (analyst cache regenerates into gemma3_analyst_ft1)")
    print("  re-run backtest_simulator.py    (fine-tuned analyst in every decision)")


if __name__ == "__main__":
    main()
