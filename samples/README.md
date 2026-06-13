# Audio Samples

Organized by training stage. All samples use the same two test sentences unless noted.

- **s1:** "Bugün hava çok güzel."
- **s2:** "Türkiye Cumhuriyeti 1923 yılında kuruldu." (digits — may trigger non-Turkish phonemes)

For full 5-sentence evaluation see `run_inference_test.py`.

---

## 01 — Epoch 1 Baseline (`issai_run1/best`)

Attention-only LoRA rank 64, ~45K steps, lr=5e-6.
5 different Turkish sentences synthesized after epoch 1.
Clearly Turkish but foreign accent audible.

## 02 — Epoch 2 Degradation

Samples from epoch 2 training at steps 1400 and 5600.
Audio quality degrades progressively — overfitting on second pass of same data.
Documented to show what "too much training" sounds like.

## 03 — B1: LoRA-only Accepted (`best_perceptual`)

lr=1e-7, cp_lr=0 (CP frozen), 500 steps from issai_run1/final.
More Turkish than epoch 1 baseline. Trailing silence present (inference parameter, not training).
**This is the current best_perceptual checkpoint.**

## 04 — B2: CP LR Rejected

lr=2e-7, cp_lr=5e-6, 500 steps.
Sub loss improved (7.65) but perceptual quality degraded.
Demonstrates that sub loss ≠ audio quality.

## 05 — B3: CP LR Rejected

lr=3e-7, cp_lr=1e-5, 500 steps.
Sub loss reached 7.47 (below random baseline) — best metric result.
Worst perceptual result. Strong evidence against CP training.

## 06 — Final (B1 `best_perceptual` + silence trim)

`best_perceptual` checkpoint with trailing silence trimmed in inference.
Two sentences: "Bugün hava çok güzel." and "Türkiye Cumhuriyeti 1923 yılında kuruldu."

## 07 — Exp C Step 1000

`exp_c/final`: fresh base model, attention+MLP LoRA rank 16, lr=5e-7, 1000 steps, cp_lr=0.
Full 5-sentence standard test with automatic number normalization.
Perceptually better than B1. Was best_perceptual until Stage 2 recovery.

## 08 — Exp D Stage 2 (steps 1000–5000)

Stage 2 from exp_c step 1000: MLP LoRA frozen, attention-only lr=1e-7.
Steps 1000/2000/3000/4000/5000 train.py SAMPLE_SENTENCES samples.
Note: s2 still used digit "1923" in this run (fixed in subsequent runs).
Step 2000 was perceptual peak but checkpoint not saved — triggered recovery run.

## 09 — Exp D2 Stage 2 Recovery (steps 1000, 2000)

Re-run of Stage 2 with `--save_at_steps 1000,1500,2000` and SAMPLE_SENTENCES fix.
s2 now uses "bin dokuz yüz yirmi üç" (spelled out). Step 2000 confirmed better than 1000.

## 10 — Exp D2 Step 2000 — `best_perceptual` ✅

Full 5-sentence `run_inference_test.py` evaluation from Stage 2 step 2000.
Number normalizer active. Best perceptual result across all experiments.
**This is the final best_perceptual checkpoint for Qwen3-TTS-0.6B.**

## 14 — Exp F 1.7B Stage 2 Step 1500 — `best_perceptual_1.7b` ✅

Full 5-sentence `run_inference_test.py` on Stage 2 step 1500.
Audio noticeably cleaner than 0.6B, EOS more stable.
C→K, Ç, Ü phoneme errors still present — same as 0.6B.
This is the final checkpoint for the 1.7B LoRA path (v0.2-experimental).

## 13 — Exp F 1.7B Stage 2 (steps 500–2000)

Stage 2 from Stage 1 step 1000. MLP LoRA frozen, attention-only lr=1e-7.
Samples at steps 500/1000/1500/2000. Step 2000 slightly worse than 1500 perceptually.

## 12 — Exp F 1.7B Stage 1 (steps 200–1000)

`Qwen3-TTS-12Hz-1.7B-Base`, attention+MLP LoRA rank 16, lr=5e-7, max_steps=1000.
Samples at steps 200/400/600/800/1000. Eval loss 7.79→7.14 — 2.3× faster than 0.6B.
Stage 2 now running. Compare step 1000 s2 to 0.6B best_perceptual (sample 10) for accent reduction.

## 11 — Exp E Partial FT (last 2 layers)

Partial full fine-tune of last 2 transformer layers (no LoRA), lr=1e-6, max_steps=500.
s1 and s2 at step 500. Perceptual quality WORSE than Stage 2 step 2000.
Confirms F9: 0.6B ceiling reached. Partial FT does not outperform LoRA here.
