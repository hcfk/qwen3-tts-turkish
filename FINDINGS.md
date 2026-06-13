# Key Findings — Qwen3-TTS Turkish Fine-tuning

Empirical findings from fine-tuning Qwen3-TTS-0.6B-Base for Turkish. Each finding is backed by a training run; see `TRAINING_LOG.md` for full details.

---

## F1 — Turkish TTS is achievable but native accent is not yet reached

Qwen3-TTS-0.6B-Base can be adapted to synthesize understandable Turkish speech. The adapted model correctly handles Turkish text and produces speech that is clearly Turkish, but retains a perceptible foreign accent.

Root cause: the base model's acoustic priors are trained on Mandarin and English. LoRA on attention layers shifts text→codec routing but cannot fully override the deep phoneme/prosody representations learned from non-Turkish data.

---

## F2 — Sub loss is not a reliable perceptual quality metric

The code predictor's cross-entropy loss (sub loss) over codebooks 1–15 is commonly expected to track audio quality. Our experiments show this is false in the continuation fine-tuning regime:

- **Experiment B3**: sub loss reached 7.47 — below the random baseline of log(2048) = 7.62 — the best sub loss result in the entire project. Yet perceptual audio quality was **worse** than runs where sub loss stayed at 9.x.
- **Run B1**: sub loss stayed at ~9.1 throughout (CP frozen). Audio quality was **better** than any joint CP+LoRA run.

**Conclusion:** Do not use sub loss as a proxy for audio quality. Judge checkpoints by perceptual listening only.

---

## F3 — Training code_predictor degrades acoustic quality

In runs B2 and B3, the code predictor was trained jointly with LoRA (cp_lr = 5e-6 and 1e-5 respectively). Despite improving sub loss, both runs produced **worse audio** than B1 where CP was frozen.

Hypothesis: the pre-trained code predictor encodes a strong acoustic prior. When LoRA shifts the hidden state distribution, the CP adapts to the shifted distribution — but this adaptation moves away from the base acoustic manifold and introduces artefacts.

**Rule:** `cp_lr = 0` in all continuation and fine-tuning runs. Do not train the code predictor.

---

## F4 — Ultra-low LR LoRA-only continuation is the safest quality improvement path

Run B1 (lr = 1e-7, cp frozen, 500 steps from issai_run1/final) produced the best perceptual result of all continuation experiments: noticeably more Turkish accent than the epoch 1 baseline, with no acoustic degradation.

The selected production checkpoint (`best_perceptual`) is from this run.

---

## F5 — Attention-only LoRA (rank 64) hits a ceiling on accent reduction

After epoch 1 (issai_run1, ~45K steps, attention-only rank 64) the model had a clear foreign accent. Continued training with the same architecture did not significantly reduce this accent; it caused overfitting and quality degradation instead.

The ceiling is architectural: `q/k/v/o_proj` adaptation shifts token routing and attention patterns, but Turkish phoneme-to-codec mappings likely live in the FFN layers (`gate/up/down_proj`), which were not part of the LoRA target.

---

## F6 — MLP LoRA helps only in a very short training window

Experiment C (attention + MLP LoRA, rank 16, lr = 5e-7, cp frozen, fresh base model) produced better perceptual audio than B1 at **step 1000**, but degraded progressively after that:

| Total steps | Perceptual result |
|-------------|------------------|
| 1.000 | Better than B1 — new best_perceptual |
| 2.000 | Slightly worse |
| 3.000 | Noticeably worse |
| 5.000 | Metallic and noisy — rejected |

**Conclusion:** MLP LoRA is useful only with very early stopping (≤ 1K steps from base). Beyond that it destroys the acoustic prior, similar to what CP training did in Exp B.

**Rule:** If using MLP LoRA, set `max_steps ≤ 1000` from base model. Do not continue past the first quality check.

**Current best_perceptual:** `exp_c/final` (step 1000, attention+MLP rank 16, lr=5e-7, cp frozen).

---

## F8 — Stage 2 (frozen MLP LoRA + attention-only refinement) improves quality further

Experiment D (Stage 2) started from exp_c step 1000 (`best_perceptual`), froze all MLP LoRA weights, and continued training only the attention LoRA at lr=1e-7. Result:

- Stage 2 step 1000 already sounded better than exp_c step 2000
- Stage 2 step 2000 was the perceptual peak — better than exp_c step 1000 (`best_perceptual`)
- Stage 2 step 3000–5000: quality held but did not improve further

Eval loss decreased monotonically from 7.059 (step 100) to 6.831 (step 5000), confirming that eval loss alone is not sufficient to select the best checkpoint.

**Conclusion:** The staged approach works. MLP LoRA gave the initial phoneme/prosody shift; attention-only refinement on top improved accent further without degrading acoustic texture.

**Critical operational lesson:** When eval loss is monotonically decreasing, the `best` checkpoint (saved by eval loss) equals the final step — not the perceptual optimum. Always use `--save_at_steps` to snapshot candidate checkpoints for perceptual evaluation.

**Stage 2 checkpoint status:** Step 2000 was identified as perceptual best but the checkpoint was overwritten by continued training. Step 2000 weights are not recoverable from this run.

---

## F9 — Qwen3-TTS-0.6B has reached its practical Turkish adaptation ceiling

**Confirmed by exp_e (partial full fine-tune, last 2 transformer layers).**

After 5 experiments (exp_a, run_b1–b3, exp_c, exp_d Stage 2, exp_e partial FT), a persistent foreign accent and C→K phoneme substitution remain and could not be eliminated at this model scale.

**Exp E results — all learning rate variants failed to outperform Stage 2:**

| Run | Method | LR | Steps | Eval loss drop | Outcome |
|-----|--------|----|-------|---------------|---------|
| E1 | Partial FT last 2 layers | 1e-7 | 3000 | ~0.000 (flat) | 0 learning |
| E2 | Partial FT last 2 layers | 5e-7 | 600 | 0.006 | 45× slower than LoRA |
| E3 | Partial FT last 2 layers | 1e-6 | 500 | 0.023 | Perceptual worse than Stage 2 |

LoRA at lr=5e-7 achieved 0.28 eval loss drop in 500 steps — 12× faster than partial FT at the same LR.

**Conclusion:** The remaining accent and C→K issues are not expressible through either LoRA rank-16 updates or direct weight training of the last 2 layers. The 0.6B model capacity is the binding constraint.

**Next direction:** Qwen3-TTS-1.7B or a TTS base with native Turkish phoneme coverage.

---

## F7 — SSH encoding corrupts Turkish characters; use pscp for test scripts

Passing Turkish text inline via PuTTY plink (`--text "Türkçe"`) silently corrupts non-ASCII characters to `?`. The model receives broken input and produces wrong phonemes.

**Fix:** Hardcode Turkish test sentences in a Python script, transfer via `pscp` (binary, encoding-safe), execute via plink. See `run_inference_test.py`.

---

## Summary Table

| Finding | Implication |
|---------|-------------|
| F1: Turkish achievable, accent remains | Experimental quality, not production |
| F2: Sub loss ≠ audio quality | Always evaluate by listening |
| F3: CP training degrades audio | Keep cp_lr = 0 forever |
| F4: LoRA-only ultra-low LR is safest | B1 is the production baseline |
| F5: Attention-only LoRA hits ceiling | Rank 64 attn-only cannot reach native accent |
| F6: MLP LoRA only helps at ≤1K steps | Early stopping mandatory; later steps degrade audio |
| F7: SSH breaks Turkish chars | Use pscp + hardcoded test script |
| F8: Staged LoRA works; eval loss ≠ perceptual peak | Always snapshot at candidate steps |
| F9: 0.6B ceiling confirmed — partial FT also failed | Next direction: Qwen3-TTS-1.7B or Turkish-native base model |
