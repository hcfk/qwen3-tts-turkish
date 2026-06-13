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

### Run B1 — Results

| Step | eval loss | sub loss | note |
|------|-----------|----------|------|
| 100  | 6.454     | ~9.2     | |
| 200  | 6.438     | ~9.2     | |
| 300  | 6.424     | ~9.0     | |
| 400  | 6.409     | ~9.3     | |
| 500  | **6.397** | ~9.1     | best overall |

Sub loss higher than issai_run1/final (9.x vs 7.7) because CP is frozen and tiny LoRA updates shift hidden states away from CP's training distribution. This is expected at cp_lr=0.

**Sample verdict:** Step 500 sounds noticeably more Turkish than issai_run1/final. Trailing silence present (inference max_new_tokens too generous, not a training issue). **Selected as `best_perceptual`.**

---

### Run B2 — REJECTED (cp_lr degrades audio)

```bash
--resume_from run_b1/best --lr 2e-7 --cp_lr 5e-6 --max_steps 500
```

| Step | eval loss | sub loss |
|------|-----------|----------|
| 100  | 5.752     | 7.76     |
| 200  | 5.716     | 7.69     |
| 300  | 5.703     | 7.66     |
| 400  | 5.698     | 7.65     |
| 500  | **5.692** | 7.65     |

Metrics looked good but **perceptual sample quality degraded**. CP adapting to updated LoRA hidden states broke acoustic detail.

**Verdict: Rejected. Marked `REJECTED_cp_lr_degrades_audio` on server.**

---

### Run B3 — REJECTED (cp_lr degrades audio)

```bash
--resume_from run_b2/best --lr 3e-7 --cp_lr 1e-5 --max_steps 500
```

| Step | eval loss | sub loss |
|------|-----------|----------|
| 100  | 5.672     | 7.63     |
| 200  | 5.661     | 7.60     |
| 300  | 5.653     | 7.59     |
| 400  | 5.645     | 7.58     |
| 500  | **5.642** | **7.47** |

Sub loss dropped below random baseline (7.47 < 7.62) for the first time. Eval loss also steadily improved. Despite this, **perceptual sample quality was worse than B1**.

**Critical finding:** Lower sub loss did NOT correlate with better audio quality. Training the code_predictor reduced sub loss but degraded acoustic quality. The best checkpoint was selected by perceptual sample quality, not validation/sub loss.

**Verdict: Rejected. Marked `REJECTED_cp_lr_degrades_audio` on server.**

---

## Experiment C — Fresh Start, Attention + MLP LoRA

**Motivation:** run_b1/best sounds like a foreigner speaking Turkish. Attention-only LoRA adapts token routing but Turkish phoneme/prosody mapping lives in FFN layers (gate/up/down_proj). Expanding LoRA targets to include MLP may reduce the accent.

Cannot resume from run_b1/best (rank-64 attention LoRA already applied — adding different rank MLP LoRA on top would corrupt the adapter config). Must start fresh.

CP stays frozen. Sub loss is not a decision signal. Judge only by perceptual samples.

**Command:**
```bash
python3 train.py \
    --model_dir  /home/hcfk/models/Qwen3-TTS-0.6B-Base \
    --data_dir   /home/hcfk/datasets/issai_tokens \
    --output_dir /home/hcfk/checkpoints/exp_c \
    --lora_targets "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_rank  16 --lora_alpha 32 \
    --lr 5e-7 --cp_lr 0 \
    --scheduler constant --warmup_steps 200 \
    --grad_accum 4 --sample_every 2000 --epochs 1
```

**Hyperparameters and rationale:**
- `rank=16` (was 64): smaller rank = less risk of destroying acoustic priors; MLP has many more params than attention, rank 16 is conservative
- `lr=5e-7` (was 5e-6 in epoch 1): ultra-low to preserve base acoustic quality
- `cp_lr=0`: CP training degrades audio regardless of sub loss improvement (proven in B2/B3)
- `sample_every=2000`: full epoch is ~45K steps, check quality at 2K/4K/... intervals

**Decision rule:**

| Sample result vs run_b1/best | Action |
|-----------------------------|--------|
| Accent reduced, no new noise | Experiment C is the new main path |
| Metallic/noisy | Reduce rank or LR |
| No perceptual difference | Attention+MLP LoRA insufficient; base model limit reached |
| Degraded | run_b1/best stays final |

**Note on `--max_steps`:** First launch forgot `--max_steps`, defaulted to 1000. Ran only 1000 steps (6.5 min). Same bug as epoch 2 attempt 1 — always set `--max_steps` explicitly.

**Step 1000 metrics (from the short run):**

| Step | eval loss | sub loss |
|------|-----------|----------|
| 100  | 7.782     | ~9.8     |
| 500  | 7.501     | ~10.4    |
| 1000 | **7.062** | ~9.0     |

Eval loss dropped 0.72 in just 1000 steps — steeper than epoch 1 at the same point. Attention+MLP LoRA is learning faster (more params touching phoneme/prosody layers).

**Step 1000 sample verdict:** TBD — downloaded to `eval_output/exp_c/`. Listen before deciding to continue.

### Experiment C Continuation

```bash
python3 train.py \
    --model_dir  /home/hcfk/models/Qwen3-TTS-0.6B-Base \
    --data_dir   /home/hcfk/datasets/issai_tokens \
    --output_dir /home/hcfk/checkpoints/exp_c_continue \
    --resume_from /home/hcfk/checkpoints/exp_c/final \
    --lr 5e-7 --cp_lr 0 \
    --max_steps 10000 --sample_every 1000 \
    --scheduler constant --warmup_steps 0 --grad_accum 4
```

Monitor at every 1000 steps. Compare each sample against run_b1/best. Stop immediately if noise or metallic artefacts appear.

**Status:** Running (PID 642277). Results TBD.

---

## Final Checkpoint — `best_perceptual`

**`/home/hcfk/checkpoints/best_perceptual`** = copy of `run_b1/best`

Selected by perceptual listening, not by lowest validation or sub loss.

### Rules established from B-path experiments

| Rule | Rationale |
|------|-----------|
| `cp_lr` must stay 0 in continuation runs | Any CP LR degrades acoustic quality, even when sub loss improves |
| Sub loss is NOT a proxy for audio quality | B3 had sub=7.47 (best ever) but worst audio |
| LoRA-only continuation is safe at ultra-low LR | B1 (lr=1e-7, cp frozen) improved Turkish accent without breaking quality |
| Stop at first sign of sample degradation | Do not rely on loss curves alone |

### If further training is attempted

Only LoRA-only, ultra-low LR:

```bash
--resume_from /home/hcfk/checkpoints/best_perceptual \
--lr 5e-8  --cp_lr 0 \
--max_steps 300 --sample_every 100 \
--scheduler constant --warmup_steps 10 --grad_accum 4
```

Accept only if step 100/200/300 samples are perceptually better than B1. Otherwise B1 is final.

---

## Final Model

**`/home/hcfk/checkpoints/best_perceptual`** is the production checkpoint. Selected by perceptual listening — more Turkish accent than issai_run1/best, no acoustic degradation.

Use this for inference:
```bash
python3 inference.py \
    --model_dir   /home/hcfk/models/Qwen3-TTS-0.6B-Base \
    --adapter_dir /home/hcfk/checkpoints/best_perceptual \
    --text        "Merhaba, bu bir test cümlesidir." \
    --output      output.wav
```

---

## SSH / Inference Notes

**PuTTY plink mangles Turkish characters.** Passing Turkish text via `plink ... "python3 inference.py --text 'Türkçe metin'"` causes `ü → ?`, `ş → ?` etc. The model receives corrupted input and produces wrong phonemes.

**Fix:** Use `run_inference_test.py` — transfer with pscp (binary, encoding-safe), run with plink. Turkish text is hardcoded in the Python file as UTF-8. This is the correct way to test inference with Turkish characters.

**Always use `run_inference_test.py` for quality evaluation**, not inline `--text` arguments over SSH.

**Numbers:** Pass numbers as Turkish words, not digits. `1923` → model falls back to Chinese phoneme prior. `bin dokuz yüz yirmi üç` → model can pronounce correctly if trained on spelled-out forms.

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
