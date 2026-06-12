"""
Quick evaluation after epoch 1: synthesize 5 Turkish sentences and save WAVs.

Usage:
    python eval_epoch1.py \
        --model_dir   /path/to/Qwen3-TTS-0.6B-Base \
        --adapter_dir /path/to/checkpoints/qwen3-tts-turkish/best \
        --output_dir  ./eval_output
"""
import argparse
import os
import torch
import soundfile as sf
from peft import PeftModel
from qwen_tts import Qwen3TTSModel

TURKISH_LANG_ID = 2072

SENTENCES = [
    "Bugün İstanbul'da hava çok güzel.",
    "Çalışma saatimiz sabah sekiz buçukta başlıyor.",
    "Türkiye Cumhuriyeti 1923 yılında kuruldu.",
    "Öğrenciler ölçüm sonuçlarını değerlendirdi.",
    "Şirket yüzde otuz beş büyüme açıkladı.",
]


def load_model(model_dir, adapter_dir):
    print("Loading base model...")
    tts = Qwen3TTSModel.from_pretrained(model_dir, device_map="cuda", dtype=torch.bfloat16)
    print("Loading LoRA adapter...")
    tts.model.talker.model = PeftModel.from_pretrained(tts.model.talker.model, adapter_dir)
    tts.model.talker.model.eval()
    tts.model.config.talker_config.codec_language_id["turkish"] = TURKISH_LANG_ID
    tts.model.supported_languages = list(tts.model.supported_languages) + ["turkish"]
    return tts


def synthesize(tts, text, output_path):
    input_ids = [tts.processor(
        text=f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt",
    )["input_ids"].to(tts.device)]

    with torch.inference_mode():
        talker_codes, _ = tts.model.generate(
            input_ids=input_ids,
            languages=["turkish"],
            non_streaming_mode=True,
        )

    speech_tok = tts.model.speech_tokenizer.model
    codes_tensor = talker_codes[0].unsqueeze(0).cuda()
    with torch.inference_mode():
        wav = speech_tok.decode(codes_tensor.permute(0, 2, 1))

    wav_np = wav.squeeze().cpu().float().numpy()
    sf.write(output_path, wav_np, 24000)
    duration = len(wav_np) / 24000
    print(f"  [{duration:.1f}s] → {output_path}")
    return duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",   required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir",  default="./eval_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tts = load_model(args.model_dir, args.adapter_dir)

    print(f"\nSynthesizing {len(SENTENCES)} sentences...\n")
    for i, text in enumerate(SENTENCES, 1):
        print(f"[{i}/{len(SENTENCES)}] {text}")
        out = os.path.join(args.output_dir, f"sent_{i:02d}.wav")
        synthesize(tts, text, out)

    print(f"\nDone. WAV files saved to: {args.output_dir}")
    print("Copy to local machine with:")
    print(f"  scp -r hcfk@192.0.0.131:{args.output_dir} C:\\dev\\QwenTR\\eval_output")


if __name__ == "__main__":
    main()
