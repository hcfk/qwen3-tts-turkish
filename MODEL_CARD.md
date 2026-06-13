# Model Card — Qwen3-TTS Turkish LoRA

## Model Description

This is an experimental LoRA adapter for [Qwen3-TTS-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base) that adds Turkish language support via fine-tuning on the [ISSAI Turkish Speech Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus).

**Status:** Experimental. Not production-ready.

| Property | Value |
|----------|-------|
| Base model | Qwen/Qwen3-TTS-0.6B-Base |
| Language target | Turkish (tr) |
| Fine-tuning method | LoRA (PEFT) |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj` |
| LoRA rank | 64 |
| Training data | ISSAI Turkish Speech Corpus (~179K utterances) |
| Current quality | Understandable Turkish with a foreign accent |

---

## Intended Use

- Research and prototyping of Turkish TTS
- Academic study of multilingual TTS adaptation
- Internal voice interface testing

## Out-of-Scope Use

- Production voice applications
- Impersonation or non-consensual voice cloning
- Any use requiring native-quality Turkish pronunciation

---

## Quality Assessment

The model can synthesize understandable Turkish speech, but it may retain a foreign accent due to the base model's acoustic priors and limited adaptation of deeper acoustic/prosodic layers.

Known limitations:
- **Foreign accent:** Phonemes not present in Mandarin/English (ğ, ı, ö, ü, ş, ç) may be mispronounced
- **Number reading:** Digits (e.g. "1923") may trigger non-Turkish phoneme patterns; write numbers as words
- **Prosody:** Turkish sentence stress and intonation may not match native patterns

---

## Training Details

See `TRAINING_LOG.md` and `FINDINGS.md` for full experimental history.

Key decisions:
- `code_predictor` kept frozen — training it reduced sub loss but degraded perceptual audio quality (see Finding F2, F3)
- Selected checkpoint (`best_perceptual`) chosen by perceptual listening, not by validation loss
- Ultra-low learning rate (1e-7) for continuation to avoid acoustic prior degradation

---

## Evaluation

Evaluated perceptually on 5 fixed test sentences covering:
- Basic Turkish phonemes
- Turkish-specific characters (ğ, ı, ö, ü, ş, ç)
- Numbers written as words
- Complex consonant clusters
- Sentence-final intonation

See `samples/` for audio outputs at each training stage.

---

## Ethical Considerations

This model is derived from Qwen3-TTS which is subject to its own license and usage terms. The ISSAI dataset is subject to its own terms of use. Do not use this model for voice cloning without explicit consent from the voice owner.

---

## Citation

```bibtex
@misc{qwen3tts-turkish,
  title  = {Qwen3-TTS Turkish LoRA Fine-tuning},
  author = {Fatih Küçükpetek},
  year   = {2025},
  url    = {https://github.com/hcfk/qwen3-tts-turkish}
}
```
