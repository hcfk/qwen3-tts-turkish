"""
Tuncer dataset tokenization.

Reads Tuncer_dataset/metadata.json, encodes each WAV with the Qwen3-TTS
speech tokenizer, saves (T,16) codec arrays as .npy files, and writes
Train_metadata.json / Test_metadata.json for train.py.

Usage:
    python prepare_tuncer_tokens.py \
        --dataset_dir /home/hcfk/Tuncer_dataset \
        --model_dir   /home/hcfk/models/Qwen3-TTS-0.6B-Base \
        --test_ratio  0.1
"""
import argparse
import json
import os
import random

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from tqdm import tqdm

TARGET_SR = 24000


def encode_audio(wav_path, speech_tok):
    audio, sr = sf.read(wav_path)
    audio_t = torch.from_numpy(audio).float()
    if audio_t.dim() > 1:
        audio_t = audio_t.mean(dim=-1)
    if sr != TARGET_SR:
        audio_t = F.resample(audio_t, sr, TARGET_SR)
    audio_t = audio_t.unsqueeze(0).cuda()
    mask = torch.ones(audio_t.shape, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        out = speech_tok.encode(audio_t, padding_mask=mask)
    return out.audio_codes[0].cpu().numpy()  # [T, 16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_dir",   required=True)
    parser.add_argument("--test_ratio",  type=float, default=0.1)
    args = parser.parse_args()

    token_dir = os.path.join(args.dataset_dir, "tokens")
    os.makedirs(token_dir, exist_ok=True)

    # Load Whisper metadata
    meta_path = os.path.join(args.dataset_dir, "metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")

    # Load speech tokenizer (float32, separate from talker)
    print("Loading speech tokenizer...")
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    speech_tok = Qwen3TTSTokenizerV2Model.from_pretrained(
        os.path.join(args.model_dir, "speech_tokenizer")
    ).cuda()
    print("Tokenizer ready")

    # Encode all WAVs
    encoded = []
    for s in tqdm(samples, desc="Encoding"):
        wav_path = s["wav_path"]
        npy_name = os.path.splitext(os.path.basename(wav_path))[0] + ".npy"
        npy_path = os.path.join(token_dir, npy_name)

        try:
            codes = encode_audio(wav_path, speech_tok)
            np.save(npy_path, codes)
            encoded.append({
                "token_file": npy_path,
                "text":       s["text"],
                "n_frames":   codes.shape[0],
                "source":     s.get("source", ""),
            })
        except Exception as e:
            print(f"  SKIP {wav_path}: {e}")

    # Train / test split
    random.seed(42)
    random.shuffle(encoded)
    n_test  = max(1, int(len(encoded) * args.test_ratio))
    test    = encoded[:n_test]
    train   = encoded[n_test:]

    train_meta = os.path.join(args.dataset_dir, "Train_metadata.json")
    test_meta  = os.path.join(args.dataset_dir, "Test_metadata.json")
    with open(train_meta, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(test_meta, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    total_dur = sum(s["n_frames"] for s in encoded) / 12.5
    print(f"\n=== Done ===")
    print(f"  Encoded   : {len(encoded)} / {len(samples)} samples")
    print(f"  Train     : {len(train)} samples")
    print(f"  Test      : {len(test)} samples")
    print(f"  Duration  : {total_dur/60:.1f} min")
    print(f"  Tokens    : {token_dir}")


if __name__ == "__main__":
    main()
