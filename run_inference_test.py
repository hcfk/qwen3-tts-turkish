"""
Standard inference test — fixed sentence set for consistent quality evaluation.
Transfer via pscp (not plink) to preserve UTF-8 Turkish characters.

Usage:
    python3 run_inference_test.py --adapter_dir /path/to/checkpoint --output_dir /path/to/out
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from inference import load_model, generate, normalize_numbers

MODEL_DIR = "/home/hcfk/models/Qwen3-TTS-0.6B-Base"

# Fixed test set — do not change between evaluations
SENTENCES = [
    ("s1", "Bugün hava çok güzel."),
    ("s2", "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."),
    ("s3", "Çocuklar çiçek, şeker ve üzüm yedi."),
    ("s4", "Öğrenciler ölçüm sonuçlarını değerlendirdi."),
    ("s5", "Şirket yüzde otuz beş büyüme açıkladı."),
]

# Evaluation checklist (perceptual, not metric-based):
#   C/K confusion       — "Cumhuriyeti" → "Kumhuriyeti"?
#   Number reading      — "bin dokuz yüz" correct or Chinese/English phonemes?
#   Turkish phonemes    — ı, ğ, ü, ö, ş, ç rendered correctly?
#   EOS / trailing silence — clean ending or long silence?
#   Metallic / noise    — any degradation vs previous checkpoint?

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--model_dir",   default=MODEL_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tts = load_model(args.model_dir, args.adapter_dir)

    for tag, text in SENTENCES:
        out = os.path.join(args.output_dir, f"{tag}.wav")
        generate(tts, normalize_numbers(text), out)

if __name__ == "__main__":
    main()
