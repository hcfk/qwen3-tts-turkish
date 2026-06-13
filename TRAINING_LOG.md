# Training Log

Chronological notes on actual training runs — what we used, what broke, and what we changed.

---

## Sub Loss Diagnostic Gate

**This is the primary gate before any training decision.** Check sub loss trend in the first 500–1000 steps.

Random baseline: `log(2048) = 7.62`

| Sub loss result | Diagnosis | Next action |
|----------------|-----------|-------------|
| **stays ~7.6** | target/codebook alignment broken — LR is not the issue | Stop tuning LR. Debug: sub_targets shape, sub_logits shape, codebook order, target shift, EOS/PAD mask, vocab range 0–2047 |
| **drops below 7.2** | code_predictor CAN learn, but LoRA hidden drift interferes | Joint training: `lr_lora=5e-7–1e-6`, `cp_lr=2e-5–5e-5`, `scheduler=constant`, `grad_clip=1.0` |
| **reaches 6.x band** | CP-only is healthy | Stage training: (1) 500–1000 steps CP-only, (2) unfreeze LoRA at very low LR, (3) reduce CP LR slightly, continue jointly |

**Sample quality gate** (after sub loss improves):

| Sample output | Diagnosis | Next action |
|--------------|-----------|-------------|
| sub drops but sample still noisy | decode/reconstruction problem | Check inference codebook order, decode path |
| sub drops and sample ends cleanly | training path correct | Proceed to quality/accent fine-tune phase |

**CP-only isolation test** (`--train_code_predictor_only`): the most critical diagnostic.
- Target: `sub_loss < 7.2` at 2000 steps
- If sub stays 7.4–7.6: wiring/alignment is broken — do not continue training without fixing it first

---

## Epoch 1 — `issai_run1`

**Command:**
```bash
python3 train.py \
    --model_dir   /home/hcfk/models/Qwen3-TTS-0.6B-Base \
    --data_dir    /home/hcfk/datasets/issai_tokens \
    --output_dir  /home/hcfk/checkpoints/issai_run1 \
    --epochs      1
```

**Hyperparameters (all defaults):**
- LR: `5e-6`
- LoRA rank: `64`, alpha: `128`
- Max steps: no limit (ran full dataset)
- Grad accum: `4`

**Hardware:** NVIDIA GB10, CUDA 13.0

**Result:**
- ~44,780 steps, 280 min
- Final loss ~5.4 (main ~1.5, sub ~7.6)
- Checkpoints saved: `issai_run1/best`, `issai_run1/final`
- Eval: 5 Turkish sentences synthesized via `eval_epoch1.py` — output sounded reasonable

**Notes:**
- `issai_run1/best` = lowest eval loss checkpoint during the run — this is the one to use
- `issai_run1/final` = last step — slightly worse than `best`, do not use as a resume point

---

## Epoch 2 — `issai_run2` (attempt 1, discarded)

**What went wrong:** Launched with default `--max_steps 1000`. Only ran 1000 steps (7.3 min) instead of the full ~45,000. LR cosine schedule collapsed to 0 within those 1000 steps. Eval loss went up to 5.79 — worse than epoch 1.

**Also wrong:** Used `--resume_from issai_run1/final` instead of `issai_run1/best`.

**Discarded.** `issai_run2` output from this attempt should be ignored.

---

## Epoch 2 — `issai_run2` (attempt 2, current)

**Command:**
```bash
python3 train.py \
    --model_dir   /home/hcfk/models/Qwen3-TTS-0.6B-Base \
    --data_dir    /home/hcfk/datasets/issai_tokens \
    --output_dir  /home/hcfk/checkpoints/issai_run2 \
    --resume_from /home/hcfk/checkpoints/issai_run1/final \
    --lr          2e-6 \
    --max_steps   45000 \
    --epochs      1
```

**Changes from epoch 1:**
- Resume from `issai_run1/final` — chosen after listening to eval WAVs, quality was good
- LR halved: `5e-6` → `2e-6` (standard practice for continued fine-tuning)
- `--max_steps 45000` explicitly set to match epoch 1's full dataset pass

**Status:** Stopped at step ~700. See findings below.

**What happened:**
- Sub loss (code predictor) rose from ~7.6 (epoch 1 end) to ~7.8 — directly causing audio to get noisier
- Eval loss at step 700 was 5.76, higher than epoch 1's final ~5.4
- Continued training on the same data caused overfitting and generalization loss

**Conclusion: Epoch 2 made audio quality worse, not better. Do not continue from `issai_run1/final`.**

---

## Deep Diagnostic — Sub Loss Root Cause Analysis

After epoch 1 and 2 runs, audio quality degraded and `sub` loss stayed stuck at ~7.7 throughout all training.

### Observation
- `sub ≈ 7.7` throughout all training steps, never meaningfully decreasing
- `log(2048) = 7.624` — codec codebook vocabulary size is 2048
- **7.7 ≈ random baseline** → `code_predictor` was not learning

### Architecture
Qwen3-TTS has two prediction components inside the talker:
- **Main talker LM** (`talker.model`) — predicts codebook 0 autoregressively via cross-entropy on `labels`
- **Code predictor** (`talker.code_predictor`) — takes talker hidden states + cb0 and predicts codebooks 1–15 via `forward_sub_talker_finetune()`

Sub loss = code predictor loss over codebooks 1–15 (teacher-forced). This is the acoustic detail layer — without it, the decoder produces noise.

### What was ruled out
```
code_predictor trainable params:  141,570,304  (all trainable)
code_predictor frozen params:     0
code_predictor params in optimizer: 141,570,304  (all included)
```
Code predictor is NOT frozen, IS in the optimizer. Problem is not missing gradients.

### Gradient check (base model, 1 batch)
```
loss_main = 3.2175   sub_loss = 9.8336   (fresh base model on Turkish)
code_predictor grad abs mean: 1.07e-03   max: 4.73e-03
LoRA grad abs mean:           1.80e-03   max: 1.20e-02
```
Gradients DO flow to code_predictor. But sub_loss went from 9.83 (base) → 7.7 (trained) — barely moved.

### Root cause: LR imbalance + hidden state drift
1. **Same LR for LoRA and code_predictor (`5e-6`)** — LoRA has ~18M params with concentrated updates; code_predictor has 141M params needing a higher LR to adapt
2. **Hidden state distribution drift** — LoRA shifts talker hidden states away from what the pre-trained code_predictor expects. Code_predictor sees out-of-distribution inputs but adapts too slowly at `5e-6`

### Fix required in `train.py`
Use separate LR per parameter group:
```python
lora_params = [p for n, p in talker.named_parameters() if p.requires_grad and "code_predictor" not in n]
cp_params    = [p for n, p in talker.code_predictor.named_parameters() if p.requires_grad]

optimizer = AdamW([
    {"params": lora_params, "lr": 1e-6},
    {"params": cp_params,   "lr": 1e-5},
], weight_decay=0.01)
```

**Expected behavior after fix:** sub loss should drop from 7.7 → 6.x range within first 500–1000 steps. If it doesn't, audio reconstruction will remain noisy regardless of main loss.

---

## LR / Scheduler Diagnostic Runs

After the dual-LR fix, a series of short tests were run to find the right CP LR. All used `--train_code_predictor_only` (LoRA frozen) unless noted.

| Run | lr_lora | cp_lr | scheduler | steps | sub result | verdict |
|-----|---------|-------|-----------|-------|------------|---------|
| overfit_test | 5e-6 | 1e-5 | cosine | 500 | 7.65 | barely moved |
| overfit_test2 | 5e-6 | 1e-5 | cosine | 5000 | 7.58 | cosine killed CP LR by mid-run |
| test_constant_lr | 1e-6 | 2e-5 | constant | 2000 | 7.42 | slow improvement, no breakthrough |
| cp_only_test | 0 | 5e-5 | constant | 2000 | **6.45** @ step 1900, then 7.51 | learns but unstable at 5e-5 |
| cp_stage1 | 0 | 1e-5 | constant | 2000 | 7.5x | 1e-5 too slow, no breakthrough |

**Key finding: sub loss and sample quality are NOT directly correlated.**

`issai_run1/final` produced good audio with `sub ≈ 7.7` throughout. The CP-only test that got sub to 6.45 used frozen LoRA — meaning those hidden states had never been adapted for Turkish. Sub can drop without producing better Turkish TTS.

This invalidates the assumption that "sub must reach 6.x for good audio."

---

## Strategic Pivot — B Path: Sample-Preserving Continuation

**Decision:** Stop chasing sub loss. Focus on why `issai_run1/final` worked.

Most likely causes of epoch 2 degradation:
1. LoRA over-learned, degraded base speech prior
2. Same small dataset → overfit on 2nd pass
3. EOS/duration behavior broke with more training
4. LR still too high for epoch 2 continuation

**Strategy:** Ultra-low LR continuation from `issai_run1/final`. Goal is accent reduction without breaking audio quality.

### Run B1 — LoRA only, CP frozen

```bash
python3 train.py \
    --resume_from /home/hcfk/checkpoints/issai_run1/final \
    --lr 1e-7 --cp_lr 0 \
    --max_steps 500 --sample_every 100 \
    --scheduler constant --warmup_steps 20 \
    --grad_accum 4
```

Stop at first sign of noise. Listen to samples at every 100-step checkpoint.

### Run B2 (if B1 stable) — tiny joint update

```bash
--lr 2e-7 --cp_lr 5e-6 --max_steps 500
```

### Decision rule

| B1 sample result | Action |
|-----------------|--------|
| Same or better | Continue to B2 |
| Worse / noisy | `issai_run1/final` is optimal — stop training |

**`issai_run1/final` samples archived at:** `/home/hcfk/eval_archive_issai_run1_final/`

---

## Final Model

**`issai_run1/best`** is the best checkpoint. It has the lowest eval loss from epoch 1 and produced clean, intelligible Turkish TTS output.

Use this for inference:
```bash
python3 eval_epoch1.py \
    --model_dir   /home/hcfk/models/Qwen3-TTS-0.6B-Base \
    --adapter_dir /home/hcfk/checkpoints/issai_run1/best \
    --output_dir  ./eval_output
```

---

## Lessons Learned

| # | Lesson |
|---|--------|
| 1 | Always resume from `best`, not `final` — `best` has the lowest eval loss |
| 2 | Always set `--max_steps` explicitly when resuming — the default (1000) is only for quick smoke tests |
| 3 | Lower LR on resumed runs: epoch 1 used `5e-6`, epoch 2+ should use `2e-6` or lower |
| 4 | Data dir is `issai_tokens/` (pre-tokenized), not `ISSAI/ISSAI_TSC_218/` (raw audio) |
| 5 | `eval_epoch1.py` requires `--model_dir` and `--adapter_dir` — no defaults |
| 6 | Epoch 2 on the same dataset caused overfitting — sub loss rose, audio got noisier. One full epoch was sufficient for this dataset size |
| 7 | `issai_run1/best` is the final production model — do not train further on ISSAI without new/different data |
| 8 | `sub ≈ log(codebook_size)` means code_predictor is at random baseline — not learning anything |
| 9 | code_predictor (141M) needs higher LR than LoRA (18M) — use separate param groups: LoRA `1e-6`, code_predictor `1e-5` |
| 10 | Always check sub loss trend early: if sub stays flat in first 500 steps, stop and fix LR before wasting compute |
