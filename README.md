---
language:
  - tr
license: mit
base_model:
  - Qwen/Qwen3-TTS-0.6B-Base
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base
tags:
  - text-to-speech
  - turkish
  - lora
  - peft
  - qwen3-tts
datasets:
  - issai/Turkish_Speech_Corpus
---

# Qwen3-TTS Turkish Fine-tuning

Fine-tuning Qwen3-TTS models for Turkish language TTS using staged LoRA on the [ISSAI Turkish Speech Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus).

**0.6B status:** Released as v0.1-experimental. Stage 2 step 2000 = `best_perceptual`. See [MODEL_CARD.md](MODEL_CARD.md).

**1.7B status:** Released as v0.2-experimental. Stage 2 step 1500 = best_perceptual_1.7b. Audio cleaner than 0.6B, same Turkish phoneme errors remain (C→K, Ç, Ü). Model-size scaling confirmed insufficient — next: G2P preprocessing experiment.

---

## Repository Structure

```
README.md             — this file
MODEL_CARD.md         — 0.6B release card, intended use, limitations
FINDINGS.md           — empirical findings from all experiments
TRAINING_LOG.md       — chronological run-by-run notes
docs/
  exp_f_1.7b_plan.md  — 1.7B experiment plan and decision gates
  NEXT_STEPS_PLAN.md  — roadmap

configs/
  exp_a_attention_rank64.yaml
  exp_b_lora_only.yaml
  exp_c_attention_mlp_rank16.yaml
  exp_d_stage2_attention_from_expc.yaml
  exp_f_1.7b_stage1.yaml
  exp_f_1.7b_stage2.yaml

train.py              — LoRA fine-tuning script (supports 0.6B and 1.7B)
inference.py          — single-text synthesis
run_inference_test.py — 5-sentence standard evaluation (UTF-8 safe)
prepare_dataset.py    — ISSAI corpus tokenization
check_1.7b_config.py  — model inspection script

samples/
  01–11/              — 0.6B experiment samples
  12_exp_f_1.7b_stage1/ — 1.7B Stage 1 samples (steps 200–1000)
```

---

## Experiments

### 0.6B — Closed (v0.1-experimental)

| Experiment | Method | Perceptual result |
|-----------|--------|------------------|
| exp_a | Attention LoRA rank 64, lr=5e-6 | Turkish intelligible, strong accent |
| run_b1 | LoRA-only, cp_lr=0 | Improved accent |
| run_b2/b3 | cp_lr > 0 | Rejected — CP training degrades audio |
| exp_c step 1000 | Attention+MLP LoRA rank 16, lr=5e-7 | Best early; degrades past 1K |
| exp_d Stage 2 step 2000 ✅ | Freeze MLP LoRA, attn-only lr=1e-7 | **Final best_perceptual** |
| exp_e partial FT | Last 2 layers, lr=1e-6 | Rejected — worse than Stage 2 |

**0.6B best_perceptual:** `exp_d2/step_002000` — Stage 2 attention-only, MLP frozen.

### 1.7B — In Progress

| Experiment | Method | Status |
|-----------|--------|--------|
| exp_f Stage 1 | Attention+MLP LoRA rank 16, lr=5e-7 | ✅ Done — eval 7.79→7.14 in 1000 steps |
| exp_f Stage 2 step 1500 ✅ | Freeze MLP LoRA, attn-only lr=1e-7 | **best_perceptual_1.7b** — cleaner audio, same phoneme errors |

---

## Training Method

Staged LoRA on the 28-layer talker backbone. Key rules carried across all experiments:

- `cp_lr = 0` always — training the code predictor degrades perceptual audio (F3)
- MLP LoRA ≤1000 steps from base — longer degrades acoustic prior (F6)
- Checkpoint selection by perceptual listening — eval loss ≠ perceptual peak (F8)
- `--save_at_steps` mandatory in Stage 2 — perceptual peak may not be at final step
- `--scheduler constant` — cosine annealing killed effective LR mid-run

---

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU with at least 16GB VRAM (24GB+ for 1.7B).

---

## Dataset

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="issai/Turkish_Speech_Corpus", repo_type="dataset",
                  local_dir="./ISSAI_TSC_218")
```

```bash
python prepare_dataset.py \
    --dataset_dir ./ISSAI_TSC_218 \
    --output_dir  ./issai_tokens \
    --model_dir   ./Qwen3-TTS-0.6B-Base
```

The same pre-tokenized ISSAI dataset is compatible with both 0.6B and 1.7B — they share an identical speech tokenizer.

---

## Fine-tune (0.6B — reproduce best result)

Stage 1:
```bash
python train.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --data_dir    ./issai_tokens \
    --output_dir  ./checkpoints/exp_c \
    --lora_targets "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_rank 16 --lora_alpha 32 \
    --lr 5e-7 --cp_lr 0 \
    --scheduler constant --warmup_steps 100 --grad_accum 4 \
    --max_steps 1000 --save_at_steps "1000"
```

Stage 2 (from Stage 1 checkpoint):
```bash
python train.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --data_dir    ./issai_tokens \
    --output_dir  ./checkpoints/exp_d \
    --resume_from ./checkpoints/exp_c/step_001000 \
    --freeze_mlp_lora \
    --lr 1e-7 --cp_lr 0 \
    --scheduler constant --warmup_steps 0 --grad_accum 4 \
    --max_steps 2000 --save_at_steps "500,1000,1500,2000" --sample_every 500
```

---

## Inference

```bash
python inference.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --adapter_dir ./checkpoints/exp_d/step_002000 \
    --text        "Bugün hava çok güzel." \
    --output      output.wav
```

Write numbers as words. Raw digits trigger non-Turkish phoneme behavior.

## Evaluate

```bash
python run_inference_test.py \
    --adapter_dir ./checkpoints/exp_d/step_002000 \
    --output_dir  ./eval_output
```

---

## Audio Samples

`samples/` contains output WAVs at each training stage. Current 0.6B best: `samples/10_exp_d2_step2000_best_perceptual/`. 1.7B Stage 1 samples: `samples/12_exp_f_1.7b_stage1/`.

---

## Key Findings

See [FINDINGS.md](FINDINGS.md) for the full list. Short version:

- Sub loss is not a reliable perceptual quality metric (F2)
- CP training always degrades audio — keep `cp_lr=0` (F3)
- MLP LoRA helps only in a short early window (F6)
- Staged LoRA (Stage 2) outperformed all other 0.6B approaches (F8)
- 0.6B has a practical adaptation ceiling — 1.7B experiment now running (F9)
- 1.7B learns significantly faster than 0.6B at same LR (2× eval loss drop in Stage 1) (F10)
- Model-size scaling improves audio clarity but does NOT fix Turkish phoneme errors (F11)
- Next approach: G2P/pseudo-phoneme input preprocessing (experiment-g branch)
