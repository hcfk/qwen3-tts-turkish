"""
ISSAI Turkish Speech Corpus dataset preparation.

1. Gerekirse ISSAI_TSC_218.tar.gz'yi extract eder
2. Train ve Test split'lerini speech tokenizer ile encode eder
3. Train_metadata.json ve Test_metadata.json oluşturur

Usage:
    python prepare_dataset.py \
        --dataset_dir /home/hcfk/datasets/ISSAI \
        --output_dir  /home/hcfk/datasets/issai_tokens \
        --model_dir   /home/hcfk/models/Qwen3-TTS-0.6B-Base
"""
import argparse
import json
import os
import tarfile

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from pathlib import Path
from tqdm import tqdm

TARGET_SR = 24000


def extract_if_needed(dataset_dir: str) -> str:
    """ISSAI_TSC_218.tar.gz varsa extract et, klasör yolunu döndür."""
    tar_path  = Path(dataset_dir) / "ISSAI_TSC_218.tar.gz"
    extracted = Path(dataset_dir) / "ISSAI_TSC_218"

    if extracted.exists():
        print(f"Zaten extract edilmiş: {extracted}")
        return str(extracted)

    if not tar_path.exists():
        candidates = list(Path(dataset_dir).rglob("ISSAI_TSC_218.tar.gz"))
        if candidates:
            tar_path = candidates[0]
        else:
            raise FileNotFoundError(f"ISSAI_TSC_218.tar.gz bulunamadı: {dataset_dir}")

    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dataset_dir)
    print(f"Extract tamamlandı → {extracted}")
    return str(extracted)


def encode_audio(wav_path: str, speech_tok) -> np.ndarray:
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_t = torch.from_numpy(audio).float()
    if sr != TARGET_SR:
        audio_t = F.resample(audio_t, sr, TARGET_SR)
    audio_t = audio_t.unsqueeze(0).cuda()
    mask = torch.ones(audio_t.shape, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        out = speech_tok.encode(audio_t, padding_mask=mask)
    return out.audio_codes[0].cpu().numpy()  # [T, 16]


def process_split(split: str, issai_dir: str, output_dir: str, speech_tok):
    split_dir = Path(issai_dir) / split
    out_dir   = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(split_dir.glob("*.wav"))
    print(f"\n{split}: {len(wav_files)} dosya")

    metadata = []
    for wav_path in tqdm(wav_files, desc=split):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        try:
            codes = encode_audio(str(wav_path), speech_tok)
        except Exception as e:
            print(f"  SKIP {wav_path.name}: {e}")
            continue

        npy_path = out_dir / (wav_path.stem + ".npy")
        np.save(str(npy_path), codes)

        metadata.append({
            "token_file": str(npy_path),
            "text":       text,
            "n_frames":   codes.shape[0],
        })

    meta_path = Path(output_dir) / f"{split}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Kaydedildi: {len(metadata)} örnek → {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="ISSAI_TSC_218.tar.gz'nin bulunduğu dizin")
    parser.add_argument("--output_dir",  required=True,
                        help="Token .npy ve metadata JSON çıktı dizini")
    parser.add_argument("--model_dir",   required=True,
                        help="Qwen3-TTS-0.6B-Base model dizini")
    parser.add_argument("--splits",      nargs="+", default=["Train", "Test"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    issai_dir = extract_if_needed(args.dataset_dir)

    print("\nSpeech tokenizer yükleniyor...")
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    speech_tok = Qwen3TTSTokenizerV2Model.from_pretrained(
        os.path.join(args.model_dir, "speech_tokenizer")
    ).cuda().eval()
    print("Tokenizer hazır")

    for split in args.splits:
        process_split(split, issai_dir, args.output_dir, speech_tok)

    print(f"\n=== Tamamlandı === Token dizini: {args.output_dir}")


if __name__ == "__main__":
    main()
