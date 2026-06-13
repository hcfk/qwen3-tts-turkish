"""
Inference: Generate Turkish speech with the fine-tuned LoRA adapter.

Usage:
    python inference.py \
        --model_dir   /path/to/Qwen3-TTS-0.6B-Base \
        --adapter_dir /path/to/checkpoints/qwen3-tts-turkish/final \
        --text        "Merhaba, bu bir test cümlesidir." \
        --output      output.wav
"""
import argparse
import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from qwen_tts import Qwen3TTSModel

TURKISH_LANG_ID = 2072


def load_model(model_dir: str, adapter_dir: str):
    print("Loading base model...")
    tts_wrapper = Qwen3TTSModel.from_pretrained(
        model_dir, device_map="cuda", dtype=torch.bfloat16
    )
    print("Loading LoRA adapter...")
    tts_wrapper.model.talker.model = PeftModel.from_pretrained(
        tts_wrapper.model.talker.model, adapter_dir
    )
    tts_wrapper.model.talker.model.eval()
    return tts_wrapper


def generate(tts_wrapper, text: str, output_path: str):
    # Register Turkish language ID in the model config so generate() accepts it
    tts_wrapper.model.config.talker_config.codec_language_id["turkish"] = TURKISH_LANG_ID
    tts_wrapper.model.supported_languages = list(
        tts_wrapper.model.supported_languages
    ) + ["turkish"]

    print(f"Generating speech for: {text!r}")
    input_ids = [tts_wrapper.processor(
        text=f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt",
    )["input_ids"].to(tts_wrapper.device)]

    with torch.inference_mode():
        talker_codes, talker_hidden = tts_wrapper.model.generate(
            input_ids=input_ids,
            languages=["turkish"],
            non_streaming_mode=True,
        )

    # Decode speech tokens to waveform
    speech_tok = tts_wrapper.model.speech_tokenizer.model
    codes_tensor = talker_codes[0].unsqueeze(0).cuda()  # [1, T, 16]
    with torch.inference_mode():
        wav = speech_tok.decode(codes_tensor)
    wav_np = wav.audio_values[0].cpu().float().numpy()

    # Trim trailing silence: keep 0.15s after last active sample
    active = np.where(np.abs(wav_np) > 0.005)[0]
    if len(active) > 0:
        wav_np = wav_np[:active[-1] + int(0.15 * 24000)]

    sf.write(output_path, wav_np, 24000)
    print(f"Saved → {output_path}  ({len(wav_np)/24000:.2f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",   required=True,
                        help="Path to Qwen3-TTS-0.6B-Base")
    parser.add_argument("--adapter_dir", required=True,
                        help="Path to fine-tuned LoRA adapter (final/ or best/)")
    parser.add_argument("--text",        required=True,
                        help="Turkish text to synthesize")
    parser.add_argument("--output",      default="output.wav")
    args = parser.parse_args()

    tts_wrapper = load_model(args.model_dir, args.adapter_dir)
    generate(tts_wrapper, args.text, args.output)


if __name__ == "__main__":
    main()
