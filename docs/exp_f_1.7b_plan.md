# Experiment F — Qwen3-TTS-1.7B Turkish LoRA Plan

## Why

Stage 2 step 2000 on 0.6B is confirmed as the ceiling (F9). Foreign accent and C→K persist
and cannot be resolved with LoRA rank-16 or partial full fine-tuning at 0.6B scale.

Qwen3-TTS-12Hz-1.7B-Base is the next available model in the family. More parameters → more
capacity for Turkish phoneme/acoustic shift.

---

## Critical Pre-Training Checks

These must be done BEFORE starting training. Run `check_1.7b_config.py` on the server.

### 1. Dataset compatibility — HIGHEST RISK

The existing ISSAI tokens at `/home/hcfk/datasets/issai_tokens/` were tokenized using the
0.6B model's codec. The 1.7B model is named "12Hz" which may indicate a different codec:

| Question | What to check |
|----------|--------------|
| Same codec? | `config.json` → `speech_tokenizer_config` or vocoder config |
| Same vocab size? | 0.6B codec = 2048 codes. Check 1.7B. |
| Same num codebooks? | 0.6B = 16. train.py hardcodes `range(1, 16)`. |
| Same frame rate? | 0.6B ≈ 12.5Hz. 1.7B = 12Hz. ~4% difference, likely compatible. |

**If codec is the same or compatible:** existing issai_tokens are usable → start training immediately.

**If codec is different:** must re-tokenize. Run `prepare_dataset.py` with the 1.7B model.
This takes ~hours on the full ISSAI dataset.

### 2. Codec token IDs

CODEC_PAD, CODEC_BOS, CODEC_EOS, CODEC_THINK may differ in 1.7B. `check_1.7b_config.py`
will print the exact values. If different, pass as CLI args to `train.py`:

```bash
--codec_pad X --codec_bos Y --codec_eos Z --codec_think A --codec_think_bos B --codec_think_eos C
```

`train.py` now auto-reads from `config.json` first; CLI args override.

### 3. Turkish language ID

`check_1.7b_config.py` will list `talker_config.codec_language_id`. If 1.7B already has
Turkish registered, use that ID. If not, `train.py` will auto-assign the next available ID.
Verify with `--turkish_lang_id` override if needed.

### 4. Number of codebooks

If 1.7B uses N ≠ 16 codebooks, `build_inputs_and_labels` in `train.py` will fail.
`check_1.7b_config.py` will print the code_predictor structure. If N ≠ 16, `train.py`
needs a `--num_codebooks` arg change (not yet implemented; add only if needed).

---

## Training Strategy

Replicate the 0.6B path that worked. Stage 1 first, evaluate, then Stage 2.

### Stage 1 — Attention+MLP LoRA

Config: `configs/exp_f_1.7b_stage1.yaml`

```bash
python3 /home/hcfk/train.py \
  --model_dir  /home/hcfk/models/Qwen3-TTS-1.7B-Base \
  --data_dir   /home/hcfk/datasets/issai_tokens \
  --output_dir /home/hcfk/checkpoints/exp_f_1.7b_stage1 \
  --lora_rank 16 --lora_alpha 32 \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lr 5e-7 --cp_lr 0 \
  --scheduler constant --warmup_steps 100 \
  --max_steps 1000 --sample_every 200 \
  --save_at_steps "500,800,1000" \
  --grad_accum 4
```

**Decision gates:**

| Step | Check | Action |
|------|-------|--------|
| 200 | Turkish intelligible? No metallic noise? | If bad audio → stop, investigate codec/LR |
| 500 | Accent less than 0.6B best_perceptual? | If yes → continue; if flat → try lr=1e-6 |
| 1000 | Accent clearly less → proceed to Stage 2 | If not → diagnose before Stage 2 |

### Stage 2 — Freeze MLP LoRA, Attention-Only

Config: `configs/exp_f_1.7b_stage2.yaml`

```bash
python3 /home/hcfk/train.py \
  --model_dir      /home/hcfk/models/Qwen3-TTS-1.7B-Base \
  --data_dir       /home/hcfk/datasets/issai_tokens \
  --output_dir     /home/hcfk/checkpoints/exp_f_1.7b_stage2 \
  --resume_from    /home/hcfk/checkpoints/exp_f_1.7b_stage1/step_001000 \
  --freeze_mlp_lora \
  --lr 1e-7 --cp_lr 0 \
  --scheduler constant --warmup_steps 0 \
  --max_steps 2000 --sample_every 500 \
  --save_at_steps "500,1000,1500,2000"
```

**Decision gates:**

| Step | Check | Action |
|------|-------|--------|
| 500 | Better than Stage 1 step 1000? | If degrading → stop, take stage1/step_001000 |
| 1000 | C→K reduced vs 0.6B best? | Track trend |
| 1500 | Still improving? | If not → select step 1000 as best |
| 2000 | Final perceptual check | Promote best step to `best_perceptual_1.7b` |

---

## Carry-Over Rules (from 0.6B findings)

| Rule | Reason |
|------|--------|
| `cp_lr = 0` always | CP training degrades audio (F3) |
| MLP LoRA ≤1000 steps | Longer degrades acoustic prior (F6) |
| Select checkpoint by listening | Eval loss ≠ perceptual peak (F8) |
| `--save_at_steps` mandatory in Stage 2 | Peak may not be at final step (F8) |
| `--scheduler constant` | Cosine killed effective LR mid-run on 0.6B |

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| C→K substitution | Eliminated or clearly reduced vs 0.6B |
| Foreign accent | Perceptibly less than 0.6B best_perceptual |
| All 5 test sentences | Intelligible with clean EOS |
| No regression on s1/s3/s4/s5 | Basic Turkish quality maintained |

---

## Files Created for This Experiment

| File | Purpose |
|------|---------|
| `check_1.7b_config.py` | Pre-training model inspection — run first |
| `configs/exp_f_1.7b_stage1.yaml` | Stage 1 run spec |
| `configs/exp_f_1.7b_stage2.yaml` | Stage 2 run spec |
| `docs/exp_f_1.7b_plan.md` | This document |

`train.py` changes: added `resolve_model_constants()` that auto-reads codec constants from
`config.json` and supports `--codec_*` / `--turkish_lang_id` CLI overrides.
