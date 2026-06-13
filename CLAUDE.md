# CLAUDE.md — QwenTR Project Rules

Guidelines for working on this project.

---

## Commit Rules

- No `Co-Authored-By` lines in commit messages
- Commit messages must explain **why**, not just what changed
- Reference `TRAINING_LOG.md` when a commit fixes something diagnosed there (e.g. "See TRAINING_LOG.md 'Deep Diagnostic' section")

---

## Training Log Rules

Every training decision must be documented in `TRAINING_LOG.md`:

- **Before a run:** command, hyperparameters, and why those values were chosen
- **After a run:** result, what the losses did, and whether audio quality improved
- **If something is wrong:** root cause analysis with evidence (loss values, gradient checks, etc.)
- **If something is changed:** what changed, why, and what the expected effect is

Do not leave a run undocumented. Even failed/discarded runs get a section.

---

## Diagnostic Rules

Before concluding a training run is "broken":
1. Check `sub` loss — if `sub ≈ log(codebook_size)`, code_predictor is at random baseline
2. Check gradients on `code_predictor` — are they flowing?
3. Check if `code_predictor` params are in the optimizer
4. Compare `sub` loss trend: is it flat or declining?

A declining `sub` (even slowly) is healthy. Flat `sub` = something is wrong with the training objective or LR.

---

## LR Tuning Rules

This project has two separate param groups:

| Component | Default LR | Notes |
|-----------|-----------|-------|
| LoRA (`talker.model`) | `5e-6` | Lower = less hidden state drift |
| Code predictor | `1e-5` | Needs higher LR to adapt to shifting hidden states |

If `sub` stays above 7.5 at step 1000–1500:
- Try LoRA LR: `1e-6` / `2e-6`
- Try CP LR: `2e-5`
- Rationale: lower LoRA LR reduces hidden state drift so code_predictor has a more stable target

---

## Remote Server

- Host: `10.20.20.9`, user: `hcfk`
- Model: `/home/hcfk/models/Qwen3-TTS-0.6B-Base`
- Dataset (tokenized): `/home/hcfk/datasets/issai_tokens/`
- Checkpoints: `/home/hcfk/checkpoints/`
- Training log: `~/train_epoch*.log`, `~/train_overfit*.log`
- SSH via PuTTY plink: `"C:\Program Files\PuTTY\plink.exe" -ssh -l hcfk -pw "..." -batch 10.20.20.9 "..."`
- File transfer: `"C:\Program Files\PuTTY\pscp.exe"`

---

## Key Constants (from model config)

```python
TURKISH_LANG_ID = 2072      # assigned, not in base model
CODEC_EOS       = 2150
CODEC_PAD       = 2148
CODEC_BOS       = 2149
CODEC_THINK     = 2154
```

Codec vocab size = 2048 → random CE baseline = `log(2048) ≈ 7.62`

---

## Checkpoints

| Checkpoint | Status | Notes |
|-----------|--------|-------|
| `issai_run1/best` | ✅ Best so far | Lowest eval loss from epoch 1 |
| `issai_run1/final` | ⚠️ Use with caution | Last step, slightly worse than best |
| `issai_run2/` | ❌ Discarded | Ran with wrong LR setup, sub loss not learning |
