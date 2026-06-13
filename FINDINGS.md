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

## F6 — Expanding LoRA to FFN layers (Experiment C) is the correct next step

Experiment C trains LoRA on all 7 projection types (attention + MLP), rank 16, lr = 5e-7, cp frozen, from base model.

Early results (3K steps): eval loss drops faster than epoch 1 at the same step count, suggesting the FFN LoRA provides additional signal. Perceptual improvement is gradual — meaningful comparison against `best_perceptual` expected around 10K steps.

**Status:** In progress. This section will be updated after 10K and full-epoch evaluation.

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
| F6: FFN LoRA is the next hypothesis | Experiment C in progress |
| F7: SSH breaks Turkish chars | Use pscp + hardcoded test script |
