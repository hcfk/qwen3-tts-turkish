# Qwen3-TTS Turkish Fine-tuning

Fine-tuning [Qwen3-TTS-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base) for Turkish language TTS using LoRA.

**Status:** Experimental — understandable Turkish with a foreign accent. Not production-ready.

> The model can synthesize understandable Turkish speech, but it may retain a foreign accent due to the base model's acoustic priors and limited adaptation of deeper acoustic/prosodic layers.

---

## Repository Structure

```
qwen3-tts-turkish/
  README.md             — this file
  MODEL_CARD.md         — model details, intended use, limitations
  FINDINGS.md           — empirical findings from all experiments
  TRAINING_LOG.md       — chronological run-by-run notes
  LICENSE

  configs/
    exp_a_attention_rank64.yaml       — Epoch 1: attention LoRA rank 64
    exp_b_lora_only.yaml              — B-path: ultra-low LR continuation
    exp_c_attention_mlp_rank16.yaml   — Experiment C: attention+MLP rank 16

  train.py              — LoRA fine-tuning script
  inference.py          — single-text synthesis
  run_inference_test.py — 5-sentence standard evaluation (UTF-8 safe)
  prepare_dataset.py    — ISSAI corpus tokenization

  samples/
    README.md
    01_epoch1_baseline/
    02_epoch2_degradation/
    03_run_b1_lora_only/   ← current best_perceptual
    04_run_b2_rejected/
    05_run_b3_rejected/
    06_final/
```

---

## Method

- **Base model:** Qwen3-TTS-0.6B-Base
- **Fine-tuning:** LoRA (rank=64, alpha=128) on `q/k/v/o_proj` of the 28-layer talker backbone
- **Dataset:** ISSAI Turkish Speech Corpus — 179,258 train utterances, 24kHz
- **Language token:** Turkish assigned codec vocab ID `2072`
- **Key finding:** `code_predictor` must stay frozen — training it reduces sub loss but degrades perceptual audio quality

---

## Current Best Checkpoint

`best_perceptual` = Run B1, step 500 (LoRA-only, cp_lr=0, lr=1e-7, 500 steps from issai_run1/final)

Selected by perceptual listening, not by lowest validation loss. See `FINDINGS.md` for why.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU.

## Dataset

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

```bash
python train.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --data_dir    ./issai_tokens \
    --output_dir  ./checkpoints/run1 \
    --lora_targets "q_proj,k_proj,v_proj,o_proj" \
    --lora_rank   64 --lora_alpha 128 \
    --lr          5e-6 --cp_lr 0 \
    --scheduler   constant --grad_accum 4 \
    --epochs      1
```

## Inference

```bash
python inference.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --adapter_dir ./checkpoints/run1/best \
    --text        "Merhaba, bu bir test cümlesidir." \
    --output      output.wav
```

**Note:** Write numbers as words — `bin dokuz yüz yirmi üç`, not `1923`.

## Evaluate

```bash
python run_inference_test.py \
    --adapter_dir ./checkpoints/run1/best \
    --output_dir  ./eval_output/run1
```

---

## Audio Samples

Listen to the training progression in `samples/`. The B1 accepted checkpoint is in `samples/03_run_b1_lora_only/` and the final trimmed output in `samples/06_final/`.

---

## Experiments

| Experiment | Config | Result |
|-----------|--------|--------|
| Exp A (epoch 1) | `configs/exp_a_attention_rank64.yaml` | Foreign accent, clean audio |
| Exp B (B-path) | `configs/exp_b_lora_only.yaml` | More Turkish, CP frozen |
| Exp C (in progress) | `configs/exp_c_attention_mlp_rank16.yaml` | Attention+MLP, fresh start |

See `FINDINGS.md` for key insights and `TRAINING_LOG.md` for run-by-run details.

---

## Ethical Use

- Not intended for impersonation or non-consensual voice cloning
- Experimental quality only — do not use in production voice applications
- Base model (Qwen3-TTS) and ISSAI dataset are subject to their own licenses

## Acknowledgements

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
- [ISSAI Turkish Speech Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus) by ISSAI, Nazarbayev University

## License

MIT — see [LICENSE](LICENSE). Base model and dataset subject to their own licenses.
