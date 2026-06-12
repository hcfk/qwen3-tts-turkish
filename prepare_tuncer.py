"""
Tuncer speaker dataset preparation — kaliteli versiyon.

Filtreler:
  - Süre: 3-12 saniye arası
  - Baş/son sessizlik trim (RMS tabanlı)
  - Gürültü tespiti: Whisper no_speech_prob + avg_logprob
  - Minimum kelime sayısı: 4
  - Sıkıştırma oranı kontrolü (tekrar eden ses tespiti)

Usage:
    python prepare_tuncer.py \
        --input_dir  /home/hcfk/Tuncer \
        --output_dir /home/hcfk/Tuncer_dataset
"""
import argparse
import json
import os
import re
import subprocess

import numpy as np
import torch
import soundfile as sf
import whisper
from tqdm import tqdm

TARGET_SR    = 24000
MIN_SEC      = 3.0
MAX_SEC      = 12.0
MIN_WORDS    = 4
MAX_NO_SPEECH = 0.25      # Whisper gürültü olasılığı eşiği
MIN_LOGPROB  = -0.6       # Whisper transkript güven eşiği
MAX_COMPRESS = 2.4        # Tekrarlayan ses tespiti (yüksekse döngüsel ses)
SILENCE_RMS  = 0.008      # Sessizlik eşiği (RMS)
SILENCE_FRAME = 512       # Sessizlik kontrol frame boyutu


def load_audio_ffmpeg(path: str):
    """ffmpeg ile MP3/WAV yükle, 24kHz mono float32 tensor döndür."""
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR),
        "-loglevel", "error", "pipe:1",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    return torch.from_numpy(audio).unsqueeze(0)  # [1, T]


def trim_silence(waveform: torch.Tensor, threshold=SILENCE_RMS, frame=SILENCE_FRAME):
    """Baş ve son sessizliği kırp."""
    hop = frame // 2
    frames = waveform.squeeze(0).unfold(0, frame, hop)
    rms = frames.pow(2).mean(-1).sqrt()
    above = (rms > threshold).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return waveform
    start = max(0, above[0].item() * hop - frame)
    end   = min(waveform.shape[-1], (above[-1].item() + 1) * hop + frame)
    return waveform[:, start:end]


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^[\s\-–—]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def word_count(text: str) -> int:
    return len(text.split())


def process_file(mp3_path, whisper_model, output_wav_dir):
    print(f"\n→ {os.path.basename(mp3_path)}")

    # Ses yükle (tüm dosya)
    try:
        full_audio = load_audio_ffmpeg(mp3_path)
    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        return []

    # Transkript
    result = whisper_model.transcribe(mp3_path, language="tr", beam_size=5, verbose=False)
    segs   = result["segments"]
    print(f"  Whisper segments: {len(segs)}")

    base    = os.path.splitext(os.path.basename(mp3_path))[0]
    results = []
    skipped = {"süre": 0, "gürültü": 0, "güven": 0, "kelime": 0, "sıkışma": 0, "sessiz": 0}

    for i, seg in enumerate(segs):
        raw_dur = seg["end"] - seg["start"]
        text    = clean_text(seg["text"])

        # 1. Ham süre filtresi (trim öncesi)
        if raw_dur < (MIN_SEC - 0.5) or raw_dur > (MAX_SEC + 2):
            skipped["süre"] += 1
            continue

        # 2. Gürültü filtresi
        if seg.get("no_speech_prob", 0) > MAX_NO_SPEECH:
            skipped["gürültü"] += 1
            continue

        # 3. Transkript güven filtresi
        if seg.get("avg_logprob", 0) < MIN_LOGPROB:
            skipped["güven"] += 1
            continue

        # 4. Tekrarlayan ses filtresi (compression_ratio)
        if seg.get("compression_ratio", 0) > MAX_COMPRESS:
            skipped["sıkışma"] += 1
            continue

        # 5. Minimum kelime
        if word_count(text) < MIN_WORDS:
            skipped["kelime"] += 1
            continue

        # Ses dilimi al
        start_s = int(seg["start"] * TARGET_SR)
        end_s   = int(seg["end"]   * TARGET_SR)
        clip    = full_audio[:, start_s:end_s]

        # 6. Sessizlik trim
        clip_trimmed = trim_silence(clip)
        trimmed_dur  = clip_trimmed.shape[-1] / TARGET_SR

        # Trim sonrası süre kontrolü
        if trimmed_dur < MIN_SEC or trimmed_dur > MAX_SEC:
            skipped["sessiz"] += 1
            continue

        # WAV kaydet
        wav_name = f"{base}_seg{i:04d}.wav"
        wav_path = os.path.join(output_wav_dir, wav_name)
        sf.write(wav_path, clip_trimmed.squeeze(0).numpy(), TARGET_SR)

        results.append({
            "wav_path":       wav_path,
            "text":           text,
            "duration":       round(trimmed_dur, 3),
            "source":         os.path.basename(mp3_path),
            "seg_idx":        i,
            "no_speech_prob": round(seg.get("no_speech_prob", 0), 4),
            "avg_logprob":    round(seg.get("avg_logprob", 0), 4),
        })

    print(f"  Kabul: {len(results)} | Reddedilen: {skipped}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_size", default="large-v3")
    args = parser.parse_args()

    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"Whisper {args.model_size} yükleniyor...")
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    wmodel  = whisper.load_model(args.model_size, device=device)

    mp3s = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".mp3")
    ])
    print(f"{len(mp3s)} MP3 dosyası bulundu\n")

    all_samples = []
    for mp3 in tqdm(mp3s, desc="Dosyalar"):
        try:
            samples = process_file(mp3, wmodel, wav_dir)
            all_samples.extend(samples)
        except Exception as e:
            print(f"  HATA: {e}")

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    total_dur = sum(s["duration"] for s in all_samples)
    print(f"\n=== Tamamlandı ===")
    print(f"  Toplam segment : {len(all_samples)}")
    print(f"  Toplam süre    : {total_dur/60:.1f} dakika")
    print(f"  Metadata       : {meta_path}")

    # Süre dağılımı
    durs = [s["duration"] for s in all_samples]
    if durs:
        print(f"  Süre (ort/min/max): {np.mean(durs):.1f}s / {np.min(durs):.1f}s / {np.max(durs):.1f}s")


if __name__ == "__main__":
    main()
