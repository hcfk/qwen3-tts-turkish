---
language:
  - tr
license: mit
base_model: Qwen/Qwen3-TTS-0.6B-Base
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

Fine-tuning [Qwen3-TTS-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base) for Turkish language TTS using LoRA.

**Status:** Experimental — understandable Turkish with a foreign accent. Not production-ready.

> The model produces intelligible Turkish speech but retains a foreign accent due to the base model's non-Turkish acoustic priors. Checkpoint selection is based on perceptual listening, not validation loss alone.

---

## Repository Structure

```
README.md             — this file
MODEL_CARD.md         — model details, intended use, limitations
FINDINGS.md           — empirical findings from all experiments
TRAINING_LOG.md       — chronological run-by-run notes
docs/
  NEXT_STEPS_PLAN.md  — decision framework for next experiments

configs/
  exp_a_attention_rank64.yaml
  exp_b_lora_only.yaml
  exp_c_attention_mlp_rank16.yaml
  exp_d_stage2_attention_from_expc.yaml

train.py              — LoRA fine-tuning script
inference.py          — single-text synthesis
run_inference_test.py — 5-sentence standard evaluation (UTF-8 safe)
prepare_dataset.py    — ISSAI corpus tokenization

samples/
  01_epoch1_baseline/
  02_epoch2_degradation/
  03_run_b1_lora_only/
  04_run_b2_rejected/
  05_run_b3_rejected/
  06_final_silence_trim/
  07_exp_c_step1000_best_perceptual/   ← current best_perceptual
  08_exp_d_stage2/
```

---

## Method

Fine-tuning uses LoRA on the 28-layer talker backbone of Qwen3-TTS-0.6B-Base, applied to the ISSAI Turkish Speech Corpus. The training strategy evolved across four experiments:

- **exp_a:** Attention-only LoRA (q/k/v/o_proj, rank 64) for one full epoch — establishes Turkish intelligibility but leaves a foreign accent
- **exp_c:** Attention + MLP LoRA (rank 16) from base — better Turkish accent at step 1000, degrades after
- **exp_d (Stage 2):** Resume from exp_c step 1000, freeze MLP LoRA, continue attention-only at lr=1e-7 — best result so far

**Critical rules established from experiments:**
- `cp_lr` must stay `0` — training the code predictor degrades perceptual audio quality even when sub loss improves
- MLP LoRA is effective only within ~1000 steps of the base model — longer runs degrade acoustic texture
- Checkpoint selection must be perceptual — eval loss and sub loss do not track audio quality reliably

---

## Current Best Checkpoint

`best_perceptual` = `exp_c/final` (step 1000, attention+MLP LoRA rank 16, lr=5e-7, cp_lr=0)

Stage 2 (exp_d) step 2000 was perceptually better but the checkpoint was not saved (monotonic eval loss issue). A recovery run is in progress. See `FINDINGS.md` F8 and `TRAINING_LOG.md`.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU with at least 16GB VRAM.

---

## Dataset

Download the ISSAI Turkish Speech Corpus from HuggingFace:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="issai/Turkish_Speech_Corpus", repo_type="dataset",
                  local_dir="./ISSAI_TSC_218")
```

## Prepare Dataset

```bash
python prepare_dataset.py \
    --dataset_dir ./ISSAI_TSC_218 \
    --output_dir  ./issai_tokens \
    --model_dir   ./Qwen3-TTS-0.6B-Base
```

## Fine-tune

Recommended starting command (exp_a equivalent — safe full epoch):

```bash
python train.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --data_dir    ./issai_tokens \
    --output_dir  ./checkpoints/exp_a \
    --lora_targets "q_proj,k_proj,v_proj,o_proj" \
    --lora_rank   64 --lora_alpha 128 \
    --lr 5e-6 --cp_lr 0 \
    --scheduler constant --grad_accum 4 \
    --epochs 1
```

Attention + MLP LoRA (exp_c equivalent — better accent, must stop at 1000 steps):

```bash
python train.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --data_dir    ./issai_tokens \
    --output_dir  ./checkpoints/exp_c \
    --lora_targets "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_rank   16 --lora_alpha 32 \
    --lr 5e-7 --cp_lr 0 \
    --scheduler constant --warmup_steps 200 --grad_accum 4 \
    --max_steps 1000 --sample_every 1000
```

Stage 2 — freeze MLP LoRA, continue attention-only (requires exp_c checkpoint):

```bash
python train.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --data_dir    ./issai_tokens \
    --output_dir  ./checkpoints/exp_d \
    --resume_from ./checkpoints/exp_c/final \
    --freeze_mlp_lora \
    --lora_targets "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_rank 16 --lora_alpha 32 \
    --lr 1e-7 --cp_lr 0 \
    --scheduler constant --warmup_steps 100 --grad_accum 4 \
    --max_steps 2000 --save_at_steps "1000,1500,2000" --sample_every 1000
```

## Inference

```bash
python inference.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --adapter_dir ./checkpoints/exp_c/final \
    --text        "Merhaba, bu bir test cümlesidir." \
    --output      output.wav
```

Numbers are automatically normalized to Turkish words at inference time (`1923` → `bin dokuz yüz yirmi üç`). Avoid passing raw digits — the base model may read them in Chinese or English phonemes.

## Evaluate

```bash
python run_inference_test.py \
    --adapter_dir ./checkpoints/exp_c/final \
    --output_dir  ./eval_output/exp_c
```

---

## Experiments

| Experiment | Key settings | Perceptual result |
|-----------|-------------|------------------|
| exp_a — Epoch 1 | Attention LoRA rank 64, lr=5e-6, ~45K steps | Turkish but foreign accent |
| run_b1 — LoRA-only | lr=1e-7, cp_lr=0, 500 steps from exp_a | More Turkish, previous best |
| run_b2 / run_b3 | cp_lr > 0 | Rejected — CP training degraded audio |
| exp_c — step 1000 ✅ | Attention+MLP rank 16, lr=5e-7, fresh base | Best LoRA result; degrades past 1K |
| exp_d Stage 2 | Freeze MLP LoRA, attention-only lr=1e-7 from exp_c | Step 2000 best (checkpoint recovery in progress) |

See `FINDINGS.md` for key insights and `TRAINING_LOG.md` for run-by-run details.

---

## Audio Samples

Listen to the progression across experiments in `samples/`. Each subdirectory corresponds to one training stage. The current best_perceptual samples are in `samples/07_exp_c_step1000_best_perceptual/`.

---

## Known Limitations

- Foreign accent remains on all checkpoints
- C→K phoneme substitution ("Cumhuriyeti" may sound like "Kumhuriyeti")
- Codec vocabulary does not include Turkish natively — accent reduction requires deep adaptation
- Numbers must be written as Turkish words, or passed through the built-in normalizer

---

## Ethical Use

- Not intended for impersonation or non-consensual voice cloning
- Experimental quality only — do not use in production voice applications
- Base model (Qwen3-TTS) and ISSAI dataset are subject to their own licenses

---

## HuggingFace

Model adapter weights: [huggingface.co/hcfk/qwen3-tts-turkish](https://huggingface.co/hcfk/qwen3-tts-turkish)

---

## Acknowledgements

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
- [ISSAI Turkish Speech Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus) by ISSAI, Nazarbayev University

## License

MIT — see [LICENSE](LICENSE). Base model and dataset are subject to their own licenses.
