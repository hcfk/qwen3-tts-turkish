"""
Experiment G — Turkish pseudo-phoneme spelling hack test.

Hypothesis: Turkish phoneme errors (C→K, Ç, Ü, Ö, Ş, Ğ) may be fixable at the INPUT
level. The base model's Mandarin-dominant prior has no Turkish phoneme paths, but it
DOES handle Latin/English phoneme sequences. Substituting Turkish chars with ASCII
approximations may route through existing phoneme paths correctly.

Usage (inference-only, no training):
    python3 g2p_spelling_test.py \
        --model_dir   /home/hcfk/models/Qwen3-TTS-1.7B-Base \
        --adapter_dir /home/hcfk/checkpoints/best_perceptual_1.7b \
        --output_dir  /home/hcfk/eval_g2p_test

Tests 4 substitution schemas on 5 fixed sentences. Listen and compare to baseline.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from inference import load_model, generate

# Fixed test sentences
SENTENCES = [
    ("s1", "Bugün hava çok güzel."),
    ("s2", "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."),
    ("s3", "Çocuklar çiçek, şeker ve üzüm yedi."),
    ("s4", "Öğrenciler ölçüm sonuçlarını değerlendirdi."),
    ("s5", "Şirket yüzde otuz beş büyüme açıkladı."),
]

# Substitution schemas to test
# Goal: find which schema best handles C/Ç/Ş/Ğ/Ü/Ö/I without training
SCHEMAS = {
    "baseline": {},  # no substitution — original Turkish chars

    "schema_a": {
        # Simple German-style digraphs — model has German data
        "ç": "tsch", "Ç": "Tsch",
        "ş": "sch",  "Ş": "Sch",
        "ğ": "",      # silent/lengthening — just drop it
        "ü": "ue",   "Ü": "Ue",
        "ö": "oe",   "Ö": "Oe",
        "ı": "i",
        "c": "dj",   "C": "Dj",   # C in Turkish = /dʒ/
    },

    "schema_b": {
        # English phoneme approximations
        "ç": "ch",   "Ç": "Ch",
        "ş": "sh",   "Ş": "Sh",
        "ğ": "",
        "ü": "yu",   "Ü": "Yu",
        "ö": "ur",   "Ö": "Ur",
        "ı": "uh",
        "c": "j",    "C": "J",    # C in Turkish = /dʒ/
    },

    "schema_c": {
        # Conservative: only fix the worst offenders (C and Ç)
        "ç": "ch",   "Ç": "Ch",
        "c": "j",    "C": "J",
        # leave ş, ğ, ü, ö, ı as-is
    },
}


def apply_schema(text: str, schema: dict) -> str:
    for src, dst in schema.items():
        text = text.replace(src, dst)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",   required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir",  required=True)
    args = parser.parse_args()

    tts = load_model(args.model_dir, args.adapter_dir)

    for schema_name, schema in SCHEMAS.items():
        schema_dir = os.path.join(args.output_dir, schema_name)
        os.makedirs(schema_dir, exist_ok=True)
        print(f"\n=== Schema: {schema_name} ===")

        for tag, original in SENTENCES:
            transformed = apply_schema(original, schema)
            if transformed != original:
                print(f"  {tag}: {original!r}")
                print(f"      → {transformed!r}")
            else:
                print(f"  {tag}: (unchanged) {original!r}")
            out = os.path.join(schema_dir, f"{tag}.wav")
            generate(tts, transformed, out)

    print(f"\nAll schemas done. Listen and compare at: {args.output_dir}/")
    print("Evaluation checklist:")
    print("  - Does C sound like /dʒ/ (Turkish C) or /k/ (Mandarin mapping)?")
    print("  - Does Ç sound like /tʃ/ or still wrong?")
    print("  - Does Ü sound like /y/ or still /u/?")
    print("  - Which schema is closest to native Turkish?")


if __name__ == "__main__":
    main()
