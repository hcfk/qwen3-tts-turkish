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

## 07 — Exp C Step 1000 — New `best_perceptual` ✅

`exp_c/final`: fresh base model, attention+MLP LoRA rank 16, lr=5e-7, 1000 steps, cp_lr=0.
Full 5-sentence standard test with automatic number normalization.
Perceptually better than B1 (less foreign accent). Numbers correct (normalizer active).
**This is the current best_perceptual checkpoint.**
