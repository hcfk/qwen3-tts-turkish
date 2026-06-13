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
  - experimental
datasets:
  - issai/Turkish_Speech_Corpus
---

# Qwen3-TTS Turkish Experimental LoRA

> ⚠️ **Experimental research checkpoint. Not a native-quality Turkish TTS model yet.**

This is an experimental Turkish LoRA adaptation of [Qwen3-TTS-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base).

The model can synthesize understandable Turkish speech, but it may retain a foreign accent due to the base model's acoustic priors and limited adaptation of deeper acoustic/prosodic layers.

**This model is not production-ready and should not be used for impersonation, fraud, non-consensual voice cloning, or misleading synthetic media.**

---

## Status

| Property | Value |
|----------|-------|
| Version | v0.2-experimental |
| Base model | Qwen/Qwen3-TTS-0.6B-Base |
| Language target | Turkish (tr) |
| Current quality | Understandable Turkish with a foreign accent |
| Production-ready | No |
| Primary checkpoint | `adapter/` (exp_c step 1000 — attention+MLP LoRA, selected by perceptual quality) |

---

## Usage

This adapter requires the Qwen3-TTS base model, which you should download separately from the [Qwen repository](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base). Do not re-upload the base model weights.

```python
import torch
from peft import PeftModel
from qwen_tts import Qwen3TTSModel
from huggingface_hub import snapshot_download

# Download base model from Qwen
base_dir = snapshot_download("Qwen/Qwen3-TTS-0.6B-Base")

# Download this adapter
adapter_dir = snapshot_download("hcfk/qwen3-tts-turkish", subfolder="adapter")

# Load
tts = Qwen3TTSModel.from_pretrained(base_dir, device_map="cuda", dtype=torch.bfloat16)
tts.model.talker.model = PeftModel.from_pretrained(tts.model.talker.model, adapter_dir)
tts.model.talker.model.eval()
```

**Note:** Write numbers as words — `bin dokuz yüz yirmi üç`, not `1923`. The base model was not trained on Turkish digits and may produce non-Turkish phonemes for numeral input.

For a full inference example see the [GitHub repository](https://github.com/hcfk/qwen3-tts-turkish).

---

## Training Data

- **Dataset:** [ISSAI Turkish Speech Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
- **Source:** HuggingFace dataset repository
- **License:** MIT (as shown on dataset page)
- **Size:** ~179,258 training utterances, 24 kHz, studio quality
- **Attribution:** Provided in reference to ISSAI / TurkicASR affiliated work

---

## Training Method

- **Fine-tuning:** LoRA (PEFT) on the talker backbone
- **LoRA targets:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (attention + MLP)
- **LoRA rank:** 16, alpha: 32
- **Training steps:** 1000 (from base model) — early stopping is critical
- **Code predictor:** Kept frozen throughout — see important finding below
- **Number normalization:** Digits are automatically converted to Turkish words at inference time

---

## Important Finding

> Lower sub loss did not necessarily correlate with better perceptual audio quality. Training the code predictor reduced sub loss in some experiments, but degraded acoustic quality. Similarly, MLP LoRA produced better audio only at step 1000 — continuing to 2K-5K introduced metallic artifacts. The current best checkpoint is selected by perceptual sample quality, not by sub loss or validation loss alone.

This is a non-obvious result: in experiments B2 and B3, sub loss reached below the random baseline (7.47 < log(2048) = 7.62), yet audio quality was perceptually worse. The selected checkpoint (run_b1) was chosen because it sounded better despite having a higher sub loss.

---

## Experiments Summary

| Experiment | Config | Perceptual Result |
|-----------|--------|------------------|
| Exp A — Epoch 1 | Attention LoRA rank 64, lr=5e-6, 45K steps | Understandable Turkish, foreign accent |
| Exp B1 — B-path | LoRA-only lr=1e-7, CP frozen, 500 steps | More Turkish, good audio |
| Exp B2 — Rejected | Joint CP training cp_lr=5e-6 | Degraded audio despite lower loss |
| Exp B3 — Rejected | Joint CP training cp_lr=1e-5 | Best sub loss, worst audio |
| Exp C — step 1000 ✅ | Attention+MLP LoRA rank 16, lr=5e-7, **1K steps only** | Best result — current checkpoint |
| Exp C — step 2K-5K ❌ | Same config, continued | Metallic artifacts, degraded quality |

Full experiment log: [TRAINING_LOG.md](https://github.com/hcfk/qwen3-tts-turkish/blob/main/TRAINING_LOG.md)  
Key findings: [FINDINGS.md](https://github.com/hcfk/qwen3-tts-turkish/blob/main/FINDINGS.md)

---

## Audio Samples

Sample audio outputs from each training stage are available in the [GitHub repository samples/](https://github.com/hcfk/qwen3-tts-turkish/tree/main/samples) directory.

---

## Limitations

- Foreign accent audible on most utterances
- Digits may trigger Chinese/English phoneme patterns — write numbers as Turkish words
- Not tested for all Turkish phonemes and edge cases
- No speaker control or style transfer

---

## Ethical Use

This model is released for research and educational purposes only.

- ❌ Do not use for impersonation or identity fraud
- ❌ Do not use for non-consensual voice cloning
- ❌ Do not use for generating misleading or deceptive synthetic media
- ✅ Research, prototyping, academic study of multilingual TTS

---

## Links

- GitHub: [hcfk/qwen3-tts-turkish](https://github.com/hcfk/qwen3-tts-turkish)
- Base model: [Qwen/Qwen3-TTS-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base)
- Dataset: [issai/Turkish_Speech_Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)

---

## Citation

```bibtex
@misc{qwen3tts-turkish,
  title  = {Qwen3-TTS Turkish Experimental LoRA},
  author = {Fatih Küçükpetek},
  year   = {2025},
  url    = {https://huggingface.co/hcfk/qwen3-tts-turkish}
}
```

## License

MIT. The base model (Qwen3-TTS) and the ISSAI dataset are subject to their own respective licenses.
