"""
Step 1: Convert Turkish Speech Corpus audio files to Qwen3-TTS speech tokens.

Encodes each WAV file using the Qwen3-TTS speech tokenizer (12.5 Hz, 16 codebooks)
and saves the resulting token arrays as .npy files alongside a metadata JSON.

Usage:
    python prepare_dataset.py \
        --dataset_dir /path/to/ISSAI_TSC_218 \
        --output_dir  /path/to/tsc_tokens \
        --model_dir   /path/to/Qwen3-TTS-0.6B-Base \
        --splits Train Test
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from qwen_tts import Qwen3TTSTokenizer

TARGET_SR = 24000  # Qwen3-TTS speech tokenizer input sample rate


def encode_audio(wav_path: str, speech_tok) -> np.ndarray | None:
    try:
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_t = torch.from_numpy(audio).float()
        if sr != TARGET_SR:
            audio_t = torchaudio.functional.resample(audio_t, sr, TARGET_SR)
        audio_t = audio_t.unsqueeze(0).cuda()
        mask = torch.ones(audio_t.shape, dtype=torch.bool, device="cuda")
        with torch.no_grad():
            out = speech_tok.encode(audio_t, padding_mask=mask)
        # audio_codes: list[Tensor[T, n_codebooks]], one entry per batch item
        return out.audio_codes[0].cpu().numpy()  # [T, 16]
    except Exception as e:
        print(f"Error encoding {wav_path}: {e}")
        return None


def process_split(split: str, dataset_dir: Path, output_dir: Path, speech_tok):
    split_dir = dataset_dir / split
    out_dir = output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(split_dir.glob("*.wav"))
    print(f"\nProcessing {split}: {len(wav_files)} files")

    metadata = []
    for wav_path in tqdm(wav_files):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        codes = encode_audio(str(wav_path), speech_tok)
        if codes is None:
            continue

        token_path = out_dir / (wav_path.stem + ".npy")
        np.save(str(token_path), codes)
        metadata.append({
            "id": wav_path.stem,
            "text": text,
            "token_file": str(token_path),
            "n_frames": codes.shape[0],  # T (time steps at 12.5 Hz)
            "n_codes": codes.shape[1],   # 16 codebooks
        })

    meta_path = output_dir / f"{split}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(metadata)} samples → {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="Root of ISSAI_TSC_218 dataset (contains Train/ and Test/ subdirs)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write .npy token files and metadata JSON")
    parser.add_argument("--model_dir", required=True,
                        help="Path to Qwen3-TTS-0.6B-Base model directory")
    parser.add_argument("--splits", nargs="+", default=["Train", "Test"])
    args = parser.parse_args()

    speech_tok_path = os.path.join(args.model_dir, "speech_tokenizer")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading speech tokenizer...")
    tok_wrapper = Qwen3TTSTokenizer.from_pretrained(speech_tok_path)
    speech_tok = tok_wrapper.model.cuda().eval()
    print(f"Tokenizer ready — downsample rate: {speech_tok.get_encode_downsample_rate()}")

    for split in args.splits:
        process_split(split, Path(args.dataset_dir), Path(args.output_dir), speech_tok)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
