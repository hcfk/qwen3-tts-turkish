# Qwen3-TTS Turkish Fine-tuning

Fine-tuning [Qwen3-TTS-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base) for Turkish language TTS using LoRA.

Qwen3-TTS does not natively support Turkish. This project adds Turkish support by fine-tuning on the [ISSAI Turkish Speech Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus), which contains ~180K utterances of studio-quality Turkish speech.

> **Status:** Training in progress. Fine-tuned adapter weights will be released on HuggingFace after evaluation.

---

## Method

- **Base model:** Qwen3-TTS-0.6B-Base (905M parameter talker LM + 141M code predictor)
- **Fine-tuning:** LoRA (rank=64, alpha=128) on the 28-layer talker backbone (`q/k/v/o_proj`)
- **Sub-talker:** `code_predictor` trained jointly to predict all 16 codebooks
- **Dataset:** ISSAI Turkish Speech Corpus — 179,258 train + 5,861 test WAV/text pairs, 16kHz mono, resampled to 24kHz
- **Input format:** Exactly matches Qwen3-TTS `non_streaming_mode` generate() embedding structure
- **Language token:** Turkish is assigned codec vocab ID `2072` (new token, not present in base model)
- **Hardware:** NVIDIA GB10 (128GB unified VRAM)

## Pipeline

```
ISSAI_TSC_218/          ──► prepare_dataset.py ──► tsc_tokens/
  Train/*.wav                 (speech tokenizer)      Train/*.npy
  Train/*.txt                                         Train_metadata.json

tsc_tokens/ + Qwen3-TTS ──► train.py ──► checkpoints/
                              (LoRA)        best/   (adapter weights)
                                            final/

checkpoints/final/ ──► inference.py ──► output.wav
```

## Setup

```bash
pip install -r requirements.txt
```

**Note:** Requires a CUDA GPU. Tested on NVIDIA GB10 (aarch64) with CUDA 13.0.

## Dataset

Download [ISSAI Turkish Speech Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus) from HuggingFace:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="issai/Turkish_Speech_Corpus", repo_type="dataset",
                  local_dir="./ISSAI_TSC_218")
```

## Step 1: Prepare Dataset

Encodes all WAV files using the Qwen3-TTS speech tokenizer (12.5 Hz, 16 RVQ codebooks) and saves codec token arrays.

```bash
python prepare_dataset.py \
    --dataset_dir ./ISSAI_TSC_218 \
    --output_dir  ./tsc_tokens \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --splits Train Test
```

Runtime: ~2 hours for 179K files on a single GPU.

## Step 2: Fine-tune

```bash
python train.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --data_dir    ./tsc_tokens \
    --output_dir  ./checkpoints/qwen3-tts-turkish \
    --epochs      3 \
    --lora_rank   64 \
    --lora_alpha  128 \
    --lr          5e-5
```

Runtime: ~6 hours for 3 epochs on NVIDIA GB10.

## Step 3: Inference

```bash
python inference.py \
    --model_dir   ./Qwen3-TTS-0.6B-Base \
    --adapter_dir ./checkpoints/qwen3-tts-turkish/final \
    --text        "Merhaba, bu bir test cümlesidir." \
    --output      output.wav
```

## Pre-trained Adapter

*(Coming soon — will be released on HuggingFace after evaluation)*

```python
from peft import PeftModel
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-0.6B-Base", device_map="cuda")
tts.model.talker.model = PeftModel.from_pretrained(
    tts.model.talker.model, "hcfk/qwen3-tts-turkish-lora"
)
```

## Acknowledgements

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
- [ISSAI Turkish Speech Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus) by Institute of Smart Systems and Artificial Intelligence, Nazarbayev University

## License

MIT — see [LICENSE](LICENSE).

The base model (Qwen3-TTS) is subject to its own license. The ISSAI dataset is subject to its own terms of use.
