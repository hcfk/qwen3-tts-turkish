"""
Quick inference test script — run directly on server.
Contains Turkish text hardcoded to avoid SSH encoding issues.

Usage:
    python3 run_inference_test.py --adapter_dir /path/to/checkpoint --output_dir /path/to/out
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from inference import load_model, generate

MODEL_DIR = "/home/hcfk/models/Qwen3-TTS-0.6B-Base"

SENTENCES = [
    ("s1", "Bugün hava çok güzel."),
    ("s2", "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."),
    ("s3", "Merhaba, nasılsınız?"),
    ("s4", "İstanbul, Türkiye'nin en büyük şehridir."),
]

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
        generate(tts, text, out)

if __name__ == "__main__":
    main()
