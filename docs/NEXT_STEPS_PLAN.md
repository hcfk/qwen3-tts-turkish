# Qwen3-TTS Turkish — Roadmap

## Status Summary

### 0.6B — Closed (v0.1-experimental)

Stage 2 step 2000 is the final checkpoint. LoRA adaptation ceiling reached (F9).

### 1.7B — Closed (v0.2-experimental)

Stage 2 step 1500 is the final checkpoint. Model-size scaling did not fix phoneme errors (F11).

**Key finding across both models:** Audio quality and convergence speed improve with scale,
but C→K, Ç, Ü, Ö, Ş phoneme mapping errors persist. Root cause: base model's
Mandarin-dominant acoustic prior has no Turkish phoneme paths. LoRA cannot add
new phoneme paths — it can only modify existing ones.

---

## What NOT to do next

| Approach | Why not |
|----------|---------|
| More LoRA steps (Stage 3) | Loss improves but phoneme errors don't — already confirmed |
| Higher LoRA rank | Same prior problem, just more parameters in the wrong space |
| CP training (cp_lr > 0) | Degrades audio every time (F3) |
| Partial FT last N layers | Tried on 0.6B — did not help (F9) |
| 3B or larger base LoRA | F11 predicts same result — size isn't the constraint |

---

## Next Direction — Experiment G: G2P/Pseudo-phoneme Preprocessing

### Hypothesis

Turkish phoneme errors may be fixable at the **input** level, not the weight level.
The base model handles Latin/English/German phoneme sequences. If Turkish characters
are substituted with ASCII approximations that map to the correct phoneme paths,
the model may produce correct Turkish sounds without any additional training.

### Level 1 — Inference-only spelling hack (no training needed)

Test 3 substitution schemas on all 5 fixed sentences using `g2p_spelling_test.py`:

```bash
python3 g2p_spelling_test.py \
    --model_dir   /home/hcfk/models/Qwen3-TTS-1.7B-Base \
    --adapter_dir /home/hcfk/checkpoints/best_perceptual_1.7b \
    --output_dir  /home/hcfk/eval_g2p_test
```

**Schemas to test:**

| Schema | Key substitutions | Rationale |
|--------|-------------------|-----------|
| baseline | none | reference |
| schema_a | c→dj, ç→tsch, ş→sch, ü→ue, ö→oe | German digraphs (model has German) |
| schema_b | c→j, ç→ch, ş→sh, ü→yu, ö→ur | English phoneme approximations |
| schema_c | c→j, ç→ch only | Conservative — fix worst offenders only |

**Decision:** If any schema fixes C/Ç without breaking other phonemes → proceed to Level 2.
If no schema works → the prior cannot be routed around; need different base model.

### Level 2 — Pseudo-phoneme dataset + fine-tune (if Level 1 shows promise)

If a good schema is found:
1. Apply schema to all ISSAI training texts
2. Short fine-tune: 500–1000 steps, attention+MLP LoRA, lr=5e-7
3. Evaluate whether phoneme corrections hold

### Level 3 — If pseudo-phoneme fails

Consider:
- Turkish-native TTS base model (e.g., a model pretrained on Turkish speech)
- G2P model (proper phoneme converter like `espeak-ng -v tr`) feeding IPA to a phoneme-aware TTS
- Different architecture altogether

---

## Checkpoints

| Checkpoint | Version | Notes |
|-----------|---------|-------|
| `best_perceptual` | 0.6B v0.1 | 0.6B Stage 2 step 2000 |
| `best_perceptual_1.7b` | 1.7B v0.2 | 1.7B Stage 2 step 1500 |
