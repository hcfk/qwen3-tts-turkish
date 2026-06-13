# Qwen3-TTS Turkish Experimental LoRA

⚠️ Experimental research checkpoint. Not a native-quality Turkish TTS model.

This is an experimental Turkish LoRA adaptation of [Qwen/Qwen3-TTS-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base).

The model can synthesize understandable Turkish speech, but it may retain a foreign accent due to the base model's acoustic priors and the practical adaptation ceiling observed with Qwen3-TTS-0.6B.

**This model is not production-ready and should not be used for impersonation, fraud, non-consensual voice cloning, or misleading synthetic media.**

---

## Status

| Property | Value |
|----------|-------|
| Version | v0.1-experimental |
| Base model | Qwen/Qwen3-TTS-0.6B-Base |
| Language target | Turkish (tr) |
| Current quality | Understandable Turkish with foreign accent |
| Production-ready | No |
| Primary checkpoint | Stage 2 step 2000 |
| Selection method | Perceptual listening quality and EOS stability |

---

## Usage

This adapter requires the Qwen3-TTS base model, which should be downloaded separately from the Qwen repository.
Do not re-upload the base model weights.

```python
import torch
from peft import PeftModel
from qwen_tts import Qwen3TTSModel
from huggingface_hub import snapshot_download

# Download base model from Qwen
base_dir = snapshot_download("Qwen/Qwen3-TTS-0.6B-Base")

# Download this adapter
adapter_dir = snapshot_download("hcfk/qwen3-tts-turkish", subfolder="adapter")

# Load base model
tts = Qwen3TTSModel.from_pretrained(
    base_dir,
    device_map="cuda",
    dtype=torch.bfloat16
)

# Load Turkish LoRA adapter
tts.model.talker.model = PeftModel.from_pretrained(
    tts.model.talker.model,
    adapter_dir
)

tts.model.talker.model.eval()
```

For a full inference example, see the [GitHub repository](https://github.com/hcfk/qwen3-tts-turkish).

---

## Important Input Note

Write numbers, dates, units, and abbreviations as Turkish words before inference.

Example:

```
Correct: Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu.
Avoid:   Türkiye Cumhuriyeti 1923 yılında kuruldu.
```

The base model was not trained natively on Turkish text patterns, and raw digits may trigger non-Turkish phoneme behavior.

---

## Training Data

| Property | Value |
|----------|-------|
| Dataset | issai/Turkish_Speech_Corpus |
| Source | Hugging Face dataset repository |
| License | MIT, as shown on the dataset page |
| Size | ~179K training utterances |
| Audio | Resampled to 24 kHz |
| Attribution | ISSAI / TurkicASR affiliated work |

---

## Training Method

| Property | Value |
|----------|-------|
| Fine-tuning method | LoRA / PEFT |
| Base model | Qwen3-TTS-0.6B-Base |
| Final selected checkpoint | Stage 2 step 2000 |
| Code predictor | Frozen in final successful runs |
| Final selection metric | Perceptual quality, not loss alone |

The final selected 0.6B checkpoint was produced through staged LoRA adaptation and selected by perceptual listening quality and EOS stability.

---

## Important Findings

### F1 — Sub loss did not reliably predict audio quality

Lower sub loss did not necessarily correlate with better perceptual audio quality.

In some experiments, training the code predictor reduced sub loss, but degraded acoustic quality. Therefore, the final checkpoint was selected by listening tests and EOS stability, not by sub loss alone.

### F2 — Code predictor training degraded perceptual quality

Experiments with `cp_lr > 0` reduced sub loss in some cases, but produced worse audio.

Final conclusion:
- `code_predictor` should remain frozen
- `cp_lr = 0`

### F3 — MLP LoRA helped only with early stopping

Attention + MLP LoRA improved quality during the early training window, but longer training degraded audio and introduced artifacts.

### F4 — Stage 2 step 2000 was the best 0.6B checkpoint

Stage 2 continued from the best early attention+MLP checkpoint, froze MLP LoRA, and trained attention LoRA only at ultra-low learning rate.

The best perceptual result for Qwen3-TTS-0.6B was: **Stage 2 step 2000**.

### F5 — Partial full fine-tuning did not beat LoRA

Partial full fine-tuning of the last 2 transformer layers was tested, but did not outperform the best Stage 2 LoRA checkpoint.

### F6 — Qwen3-TTS-0.6B appears to have reached its practical Turkish adaptation ceiling

Remaining issues include:
- foreign accent
- C→K substitution
- imperfect Turkish phoneme/prosody mapping

These issues were not fully resolved with LoRA or partial full fine-tuning on the 0.6B model.

---

## Experiments Summary

| Experiment | Config | Result |
|-----------|--------|--------|
| Exp A | Attention LoRA rank 64 | Understandable Turkish, foreign accent |
| Exp B1 | Attention-only LoRA, CP frozen | Improved perceptual quality |
| Exp B2 | Joint CP training | Rejected; audio degraded |
| Exp B3 | Higher CP LR | Rejected; lower sub loss but worse audio |
| Exp C | Attention + MLP LoRA | Improved early, degraded with longer training |
| Exp D / Stage 2 | MLP frozen, attention-only continuation | Best 0.6B perceptual result at step 2000 |
| Exp E | Partial full fine-tune, last 2 layers | Did not outperform Stage 2 |

Full experiment log: [TRAINING_LOG.md](https://github.com/hcfk/qwen3-tts-turkish/blob/master/TRAINING_LOG.md)
Key findings: [FINDINGS.md](https://github.com/hcfk/qwen3-tts-turkish/blob/master/FINDINGS.md)

---

## Audio Samples

Sample audio outputs from each training stage are available in the [GitHub repository samples/ directory](https://github.com/hcfk/qwen3-tts-turkish/tree/master/samples).

---

## Known Limitations

- Not native-quality Turkish.
- Foreign accent remains.
- C→K substitution may occur.
- Raw digits may trigger non-Turkish phoneme patterns.
- Numbers, dates, abbreviations, and units should be normalized before inference.
- Not tested for all Turkish phonemes and edge cases.
- No speaker control.
- No style transfer.
- Not production-ready.

---

## Ethical Use

This model is released for research and educational purposes only.

Do not use this model for:
- impersonation
- identity fraud
- non-consensual voice cloning
- misleading or deceptive synthetic media

Acceptable use examples:
- research
- prototyping
- academic study
- multilingual TTS adaptation experiments

---

## Links

- GitHub: https://github.com/hcfk/qwen3-tts-turkish
- Base model: https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base
- Dataset: https://huggingface.co/datasets/issai/Turkish_Speech_Corpus

---

## Citation

```bibtex
@misc{qwen3tts-turkish,
  title  = {Qwen3-TTS Turkish Experimental LoRA},
  author = {Fatih Küçükpetek},
  year   = {2026},
  url    = {https://huggingface.co/hcfk/qwen3-tts-turkish}
}
```

---

## License

MIT.

The base model [Qwen/Qwen3-TTS-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base) and the dataset [issai/Turkish_Speech_Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus) are subject to their own respective licenses and terms of use.
