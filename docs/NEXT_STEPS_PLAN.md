# Qwen3-TTS Turkish — Roadmap

## 0.6B Status — CLOSED

All 0.6B adaptation paths have been exhausted. Stage 2 step 2000 is the final checkpoint.

| Experiment | Result |
|-----------|--------|
| Exp A — attention LoRA rank 64, lr=5e-6 | Understandable Turkish, strong accent |
| Exp B1 — LoRA-only, CP frozen | Improved; earlier best_perceptual |
| Exp B2/B3 — CP training | Rejected; audio degraded |
| Exp C — attention+MLP LoRA rank 16 | Best at step 1K; degrades after |
| Exp D Stage 2 — freeze MLP, attn-only lr=1e-7 | **Final best_perceptual at step 2000** |
| Exp E — partial FT last 2 layers | Did NOT outperform Stage 2 (F9 confirmed) |

**Release:** Stage 2 step 2000 → `best_perceptual` → v0.1-experimental on HuggingFace.

---

## Next Direction — Qwen3-TTS-1.7B

### Motivation

The 0.6B model has a practical adaptation ceiling (F9). Remaining issues (foreign accent, C→K) cannot be resolved with LoRA rank-16/64 or partial FT. The 1.7B model has more capacity for the phoneme/acoustic shift.

### Initial Strategy

Replicate the best known 0.6B path on the 1.7B model:

```text
Stage 1: attention+MLP LoRA rank 16, lr=5e-7, cp_lr=0
         max_steps=1000, sample_every=200
         → perceptual check at step 1000 (expected to be the best early window)

Stage 2: freeze MLP LoRA, attention-only lr=1e-7, cp_lr=0
         max_steps=2000, save_at_steps=500,1000,1500,2000
         → perceptual check at each snapshot
```

### Key Rules (carry over from 0.6B)

| Rule | Reason |
|------|--------|
| cp_lr = 0 always | CP training degrades audio (F3) |
| MLP LoRA ≤1K steps only | Longer degrades acoustic prior (F6) |
| Checkpoint selection by listening | Eval loss != perceptual peak (F8) |
| --save_at_steps mandatory in Stage 2 | Perceptual peak may not be at final step |
| scheduler = constant | Cosine kills effective CP LR mid-run |

### Success Criteria

Stage 2 step 2000 on 1.7B should be evaluated against the 0.6B best_perceptual on the same 5 test sentences. Specific targets:

- C→K substitution reduced or eliminated
- Foreign accent clearly less than 0.6B best_perceptual
- All 5 test sentences intelligible with clean EOS

### If 1.7B Still Insufficient

Consider:
- Turkish-capable TTS base model (e.g., one with native multilingual phoneme coverage)
- Data augmentation (Turkish G2P pre-processing, phoneme-level supervision)
- Larger dataset (ISSAI is ~179K utterances; more data may help unlock deeper adaptation)
