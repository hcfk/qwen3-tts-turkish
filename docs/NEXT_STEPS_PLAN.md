# Qwen3-TTS Turkish — Next Step Decision Plan

## Current Status

We tested several LoRA-based adaptation paths for Turkish TTS using Qwen3-TTS-0.6B-Base.

The overall progression was:

```text
exp_a → exp_c → exp_d
```

Each stage improved Turkish intelligibility to some degree, but the following issues still remain:

```text
- foreign accent
- C → K substitution
- imperfect Turkish phoneme/prosody mapping
- unstable quality when training deeper acoustic-related components
```

The best perceptual result so far appears to be:

```text
Stage 2 step 2000
```

However, the exact checkpoint was not saved. This creates an immediate recovery question before moving to larger experiments.

---

# Key Findings

## F1 — LoRA improves Turkish but may be near its ceiling

Attention-only LoRA improved Turkish intelligibility but retained a foreign accent.

Adding MLP LoRA helped at short training windows, but longer training degraded perceptual audio quality.

This suggests that LoRA can adapt the model toward Turkish, but may not fully overcome the base model's non-Turkish acoustic priors.

## F2 — Code predictor training is not useful for perceptual quality

Runs with `cp_lr > 0` reduced sub loss in some cases, but produced worse audio.

Conclusion:

```text
code_predictor should remain frozen
cp_lr = 0
```

## F3 — MLP LoRA is useful only with early stopping

Experiment C showed that attention + MLP LoRA can improve over attention-only LoRA early in training.

However, quality degraded after longer continuation.

Current interpretation:

```text
MLP LoRA can provide useful early alignment,
but it is highly sensitive and can damage the acoustic prior.
```

## F4 — Stage 2 is promising

Stage 2 starts from the MLP-aligned checkpoint and freezes MLP LoRA while continuing attention LoRA only at ultra-low LR.

Hypothesis:

```text
MLP LoRA gives the initial phoneme/prosody shift.
Attention LoRA can then refine Turkish mapping without further disturbing the acoustic prior.
```

Stage 2 step 2000 currently appears to be the best perceptual point, but the checkpoint was not saved.

---

# Open Decisions

## Option 1 — Re-run Stage 2 and recover step 2000

### Description

Re-run Stage 2 from the same `exp_c/final` starting point with deterministic settings and save the step 2000 checkpoint explicitly.

Suggested settings:

```text
resume_from: exp_c/final
trainable: attention LoRA only
frozen: MLP LoRA, code_predictor, base weights
lr_lora: 1e-7
cp_lr: 0
scheduler: constant
grad_accum: 4
max_steps: 2000
save_at_steps: 1000, 1500, 2000
sample_every: 500 or 1000
```

### Pros

```text
- Lowest risk
- Directly targets the best observed perceptual result
- Cheap and fast (~13 minutes)
- Preserves current experiment path
- Gives a clean checkpoint for HF/GitHub release
```

### Cons

```text
- May not reproduce exactly if training is not deterministic
- Still likely limited by LoRA ceiling
- Foreign accent and C→K may remain
```

### Decision

**This is the immediate next step.**

---

## Option 2 — Experiment E: Partial full fine-tune

### Description

Instead of LoRA-only adaptation, partially unfreeze the main model weights.

Suggested first experiment:

```text
Model: Qwen3-TTS-0.6B-Base
Trainable:
- last 2 transformer layers full weights

Frozen:
- code_predictor
- speech tokenizer / decoder
- early transformer layers

LR:
- 5e-8 to 1e-7

Max steps:
- 3000 initial test
```

### Pros

```text
- More powerful than LoRA
- May reduce foreign accent more effectively
- Can modify deeper phoneme/acoustic mapping
```

### Cons

```text
- Higher risk of catastrophic forgetting
- Higher risk of metallic/noisy audio
- Requires train.py changes (--partial_ft_layers, --save_at_steps)
- Larger checkpoints (not a lightweight adapter)
- Harder to package for HuggingFace
```

### Decision

Run only after Stage 2 step 2000 is recovered and confirmed as best.

---

## Option 3 — Move to Qwen3-TTS-1.7B

### Description

Repeat the best known safe training strategy on the larger Qwen3-TTS-1.7B-Base model.

Initial strategy:

```text
attention-only LoRA first
then short attention + MLP LoRA
cp_lr = 0
perceptual checkpoint selection
```

### Pros

```text
- More model capacity
- Better chance of learning Turkish phoneme mapping
- May reduce C→K and foreign accent better than 0.6B
```

### Cons

```text
- More expensive (compute, disk, memory)
- Longer training
- Same base acoustic prior problem may remain
- Larger model does not guarantee native Turkish accent
```

### Decision

Consider after exhausting the best 0.6B path, especially if Stage 2 recovery confirms that LoRA has reached its ceiling.

---

# Recommended Order

```text
1. Re-run Stage 2 with max_steps=2000 and explicit checkpoint saving.
2. Evaluate step 1000, 1500, and 2000 perceptually.
3. If step 2000 is confirmed best:
   - promote it to best_perceptual
   - upload to Hugging Face as v0.1-experimental
   - document limitations clearly
4. If Stage 2 still leaves unacceptable C→K / foreign accent:
   - run Experiment E partial full fine-tune (last 2 layers)
5. If Experiment E fails or remains limited:
   - move to Qwen3-TTS-1.7B or another Turkish-capable base model
```

---

# Current Preferred Decision

The immediate next step should be:

```text
Re-run Stage 2 and save step 2000.
```

Reason:

```text
Stage 2 step 2000 is currently the best perceptual point, but the checkpoint was not saved.
Before starting riskier experiments, we need to recover and preserve that checkpoint.
```

---

# Release Positioning

The current model should be positioned as:

```text
Qwen3-TTS Turkish Experimental LoRA
```

Quality statement:

```text
The model can synthesize understandable Turkish speech, but it may retain a foreign accent
due to the base model's acoustic priors and limited adaptation of deeper acoustic/prosodic layers.
```

Limitations:

```text
- not native-quality Turkish
- foreign accent remains
- C→K substitution may occur
- not production-ready
- not intended for impersonation or non-consensual voice cloning
```
