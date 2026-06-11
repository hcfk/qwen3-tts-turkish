"""
Mini pipeline test — 50 audio files, 20 training steps (~5-10 min).
Uses the exact same embedding structure as Qwen3TTS non_streaming_mode generate.
"""
import os, json, torch, numpy as np, soundfile as sf, torchaudio
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR      = "/home/hcfk/models/Qwen3-TTS-0.6B-Base"
DATA_SRC       = "/home/hcfk/datasets/Turkish_Speech_Corpus/ISSAI_TSC_218/Train"
OUT_TOK        = "/tmp/qwen_tr_test/tokens"
OUT_CKPT       = "/tmp/qwen_tr_test/checkpoint"
N_SAMPLES      = 50
TARGET_SR      = 24000
TURKISH_LANG_ID = 2072   # new codec vocab token for Turkish

# Token IDs (from model config)
TTS_BOS_TID    = 151672  # text vocab
TTS_EOS_TID    = 151673
TTS_PAD_TID    = 151671
CODEC_THINK    = 2154    # codec vocab
CODEC_THINK_BOS= 2156
CODEC_THINK_EOS= 2157
CODEC_PAD      = 2148
CODEC_BOS      = 2149
CODEC_EOS      = 2150

os.makedirs(OUT_TOK, exist_ok=True)
os.makedirs(OUT_CKPT, exist_ok=True)

# ── 1. Encode audio files ─────────────────────────────────────────────────────
print(f"\n=== STEP 1: Encoding {N_SAMPLES} audio files ===")
tok_wrapper = Qwen3TTSTokenizer.from_pretrained(f"{MODEL_DIR}/speech_tokenizer")
speech_tok  = tok_wrapper.model.cuda().eval()

wav_files = sorted(Path(DATA_SRC).glob("*.wav"))[:N_SAMPLES]
metadata  = []
for wav_path in tqdm(wav_files):
    txt_path = wav_path.with_suffix(".txt")
    if not txt_path.exists(): continue
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text: continue

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1: audio = audio.mean(axis=1)
    audio_t = torch.from_numpy(audio).float()
    if sr != TARGET_SR:
        audio_t = torchaudio.functional.resample(audio_t, sr, TARGET_SR)
    audio_t = audio_t.unsqueeze(0).cuda()
    mask    = torch.ones(audio_t.shape, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        out = speech_tok.encode(audio_t, padding_mask=mask)
    codes = out.audio_codes[0].cpu().numpy()  # (T, 16)
    token_path = Path(OUT_TOK) / (wav_path.stem + ".npy")
    np.save(str(token_path), codes)
    metadata.append({"id": wav_path.stem, "text": text,
                     "token_file": str(token_path), "n_frames": codes.shape[0]})

meta_path = Path(OUT_TOK) / "metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"Encoded {len(metadata)} samples — shape example: {codes.shape}")

# ── 2. Load TTS model ─────────────────────────────────────────────────────────
print("\n=== STEP 2: Loading model ===")
tts_wrapper = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda",
                                             torch_dtype=torch.bfloat16)
inner_model  = tts_wrapper.model   # Qwen3TTSForConditionalGeneration
talker       = inner_model.talker  # Qwen3TTSTalkerForConditionalGeneration
processor    = tts_wrapper.processor
dev          = talker.device
dtype        = torch.bfloat16

# Apply LoRA to talker backbone (28-layer transformer)
lora_cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
talker.model = get_peft_model(talker.model, lora_cfg)
talker.model.print_trainable_parameters()

# ── Precompute fixed embeddings ───────────────────────────────────────────────
@torch.no_grad()
def _text_proj_single(token_id: int):
    ids = torch.tensor([[token_id]], device=dev, dtype=torch.long)
    return talker.text_projection(talker.get_text_embeddings()(ids))  # [1, 1, 1024]

@torch.no_grad()
def _codec_embed_single(token_id: int):
    ids = torch.tensor([[token_id]], device=dev, dtype=torch.long)
    return talker.get_input_embeddings()(ids)  # [1, 1, 1024]

# ── Training forward pass ─────────────────────────────────────────────────────
def build_inputs_and_labels(text: str, codec_codes: torch.Tensor):
    """
    Replicates non_streaming_mode embedding construction from generate().
    text: raw Turkish string
    codec_codes: [T, 16] int64 codec tokens on GPU
    Returns: inputs_embeds [1, T_total, 1024], labels [1, T_total]
    """
    T = codec_codes.shape[0]

    # Format text like the model expects: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    full_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = processor(text=full_text, return_tensors="pt")["input_ids"].to(dev)  # [1, N]
    # input_ids[:, :3] = role prefix; input_ids[:, 3:-5] = text content; input_ids[:, -5:] = suffix

    with torch.no_grad():
        # Precompute tts special embeds (via text_projection)
        special = torch.tensor([[TTS_BOS_TID, TTS_EOS_TID, TTS_PAD_TID]], device=dev)
        tts_bos_e, tts_eos_e, tts_pad_e = talker.text_projection(
            talker.get_text_embeddings()(special)
        ).chunk(3, dim=1)  # each [1, 1, 1024]

        # Role prefix embed (3 tokens)
        role_e = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))  # [1, 3, 1024]

        # Codec prefix: [think, think_bos, lang_id, think_eos, codec_pad, codec_bos]
        codec_prefix_ids = torch.tensor([[CODEC_THINK, CODEC_THINK_BOS, TURKISH_LANG_ID,
                                          CODEC_THINK_EOS]], device=dev)
        codec_suffix_ids = torch.tensor([[CODEC_PAD, CODEC_BOS]], device=dev)
        codec_all = torch.cat([
            talker.get_input_embeddings()(codec_prefix_ids),  # [1, 4, 1024]
            talker.get_input_embeddings()(codec_suffix_ids),  # [1, 2, 1024]
        ], dim=1)  # [1, 6, 1024]

        # _talker_embed = [tts_pad×4, tts_bos] + codec_all[:, :-1]  [1, 5, 1024]
        tts_pad_4 = tts_pad_e.expand(1, 4, -1)
        _talker_embed = torch.cat([tts_pad_4, tts_bos_e], dim=1) + codec_all[:, :-1]

        # non_streaming_mode text content: input_ids[:, 3:-5]
        T_text = input_ids.shape[1] - 8  # remove 3 prefix + 5 suffix
        text_content_ids = input_ids[:, 3:-5]  # [1, T_text]

        codec_pad_expand = talker.get_input_embeddings()(
            torch.full((1, T_text + 1), CODEC_PAD, device=dev)
        )  # [1, T_text+1, 1024]
        text_proj = talker.text_projection(talker.get_text_embeddings()(text_content_ids))  # [1, T_text, 1024]
        text_eos_e = torch.cat([text_proj, tts_eos_e], dim=1)  # [1, T_text+1, 1024]
        text_combined = text_eos_e + codec_pad_expand  # [1, T_text+1, 1024]

        # tts_pad + codec_bos  [1, 1, 1024]
        codec_bos_e = talker.get_input_embeddings()(torch.tensor([[CODEC_BOS]], device=dev))
        pad_bos = tts_pad_e + codec_bos_e  # [1, 1, 1024]

        prefill = torch.cat([role_e, _talker_embed, text_combined, pad_bos], dim=1)
        T_prefix = prefill.shape[1]  # T_text + 10

    # Codec teacher-forcing embeddings: sum of all 16 codebook embeds + tts_pad
    codes_dev = codec_codes.long().unsqueeze(0)  # [1, T, 16] — keep as long, never cast to float
    codec_sum_e = talker.get_input_embeddings()(codes_dev[:, :, 0])  # [1, T, 1024]
    for i in range(1, 16):
        codec_sum_e = codec_sum_e + talker.code_predictor.get_input_embeddings()[i-1](codes_dev[:, :, i])
    codec_input_e = codec_sum_e + tts_pad_e.expand(1, T, -1)  # [1, T, 1024]

    # Full inputs: [prefill, codec[0]+pad, ..., codec[T-1]+pad]
    inputs_embeds = torch.cat([prefill, codec_input_e], dim=1).to(dtype)  # [1, T_prefix+T, 1024]

    # Labels: -100 for prefill, first codebook targets for codec positions
    # HF internal shift: loss at pos i trains to predict labels[i+1]
    # → label at pos T_prefix-1 trains to predict codec[0], T_prefix → codec[1], ...
    first_cb = codes_dev[:, :, 0]  # [1, T]
    labels = torch.cat([
        torch.full((1, T_prefix), -100, dtype=torch.long, device=dev),
        first_cb,
    ], dim=1)  # [1, T_prefix+T]

    return inputs_embeds, labels, T_prefix

def training_step(text: str, codec_codes: torch.Tensor, optimizer):
    talker.train()
    optimizer.zero_grad()

    inputs_embeds, labels, T_prefix = build_inputs_and_labels(text, codec_codes)

    # Run through talker backbone
    outputs = talker.model(inputs_embeds=inputs_embeds)
    hidden_states = outputs.last_hidden_state  # [1, T_total, 1024]

    # First codebook loss (via codec_head)
    codec_logits = talker.codec_head(hidden_states)  # [1, T_total, 3072]
    loss_main = talker.loss_function(logits=codec_logits, labels=labels,
                                     vocab_size=talker.config.vocab_size)

    # Sub-talker loss (remaining 15 codebooks, batched over all time steps)
    T = codec_codes.shape[0]
    codes_dev = codec_codes.long().unsqueeze(0)  # [1, T, 16]
    codec_hidden = hidden_states[:, T_prefix:, :]  # [1, T, 1024]

    # Reshape to [T, 16] and [T, 1024] for sub-talker
    h_flat     = codec_hidden.squeeze(0)          # [T, 1024]
    codes_flat = codes_dev.squeeze(0)             # [T, 16]
    _, sub_loss = talker.forward_sub_talker_finetune(codes_flat, h_flat)

    loss = loss_main + 0.5 * sub_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(talker.parameters(), 1.0)
    optimizer.step()

    return loss.item(), loss_main.item(), sub_loss.item()

# ── 3. Mini training run ──────────────────────────────────────────────────────
print("\n=== STEP 3: Mini training (20 steps) ===")

optimizer = torch.optim.AdamW(
    [p for p in talker.parameters() if p.requires_grad],
    lr=5e-5, weight_decay=0.01
)

MAX_T = 128  # cap frames for test speed
for step in range(20):
    sample = metadata[step % len(metadata)]
    codes  = torch.from_numpy(np.load(sample["token_file"])).long()[:MAX_T].cuda()
    loss, l_main, l_sub = training_step(sample["text"], codes, optimizer)
    if (step + 1) % 5 == 0:
        print(f"  Step {step+1:3d} | loss={loss:.4f}  main={l_main:.4f}  sub={l_sub:.4f}")

print("\n=== PIPELINE TEST PASSED ✓ ===")
