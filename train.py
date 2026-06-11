"""
Step 2: Fine-tune Qwen3-TTS-0.6B-Base on Turkish Speech Corpus using LoRA.

Adds Turkish language support by training the talker LM and code predictor
to generate speech tokens from Turkish text input.

Architecture:
  - LoRA adapters on the 28-layer talker backbone (q/k/v/o projections)
  - Sub-talker (code_predictor) fine-tuned for all 16 codebooks
  - Input embeddings follow Qwen3-TTS non_streaming_mode format exactly

Usage:
    python train.py \
        --model_dir   /path/to/Qwen3-TTS-0.6B-Base \
        --data_dir    /path/to/tsc_tokens \
        --output_dir  /path/to/checkpoints/qwen3-tts-turkish
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from qwen_tts import Qwen3TTSModel

# Token IDs (fixed by Qwen3-TTS-0.6B-Base model config)
TTS_BOS_TID     = 151672  # text vocab: <|tts_bos|>
TTS_EOS_TID     = 151673  # text vocab: <|tts_eos|>
TTS_PAD_TID     = 151671  # text vocab: <|tts_pad|>
CODEC_THINK     = 2154    # codec vocab: think token
CODEC_THINK_BOS = 2156    # codec vocab: think begin
CODEC_THINK_EOS = 2157    # codec vocab: think end
CODEC_PAD       = 2148    # codec vocab: pad
CODEC_BOS       = 2149    # codec vocab: begin-of-speech
CODEC_EOS       = 2150    # codec vocab: end-of-speech
TURKISH_LANG_ID = 2072    # new language token added to codec vocab for Turkish


class TSCDataset(Dataset):
    def __init__(self, meta_path: str, max_t: int):
        with open(meta_path, encoding="utf-8") as f:
            samples = json.load(f)
        self.samples = [s for s in samples if 0 < s["n_frames"] <= max_t]
        print(f"Loaded {len(self.samples)} samples from {meta_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        codes = torch.from_numpy(np.load(s["token_file"])).long()  # [T, 16]
        return {"text": s["text"], "codes": codes}


def collate_single(batch):
    return batch[0]


def build_inputs_and_labels(text, codec_codes, talker, processor, dev, dtype):
    """
    Builds inputs_embeds and labels replicating Qwen3-TTS non_streaming_mode generate().

    The input sequence structure (all embeddings summed/concatenated):
      [role_prefix(3) | codec_think_prefix(5) | text+eos+codec_pad(T_text+1) | pad+codec_bos(1)]
      + [codec_sum_embed[t] + tts_pad for t in 0..T-1]

    Labels: -100 for the prefix, first-codebook token IDs for codec positions.
    HF-style internal shift means position T_prefix-1 trains to predict codec[0].
    """
    T = codec_codes.shape[0]
    full_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = processor(text=full_text, return_tensors="pt")["input_ids"].to(dev)

    with torch.no_grad():
        special = torch.tensor([[TTS_BOS_TID, TTS_EOS_TID, TTS_PAD_TID]], device=dev)
        tts_bos_e, tts_eos_e, tts_pad_e = talker.text_projection(
            talker.get_text_embeddings()(special)
        ).chunk(3, dim=1)

        role_e = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))

        cp_ids = torch.tensor([[CODEC_THINK, CODEC_THINK_BOS, TURKISH_LANG_ID, CODEC_THINK_EOS]], device=dev)
        cs_ids = torch.tensor([[CODEC_PAD, CODEC_BOS]], device=dev)
        codec_all_e = torch.cat([
            talker.get_input_embeddings()(cp_ids),
            talker.get_input_embeddings()(cs_ids),
        ], dim=1)

        _talker_embed = (
            torch.cat([tts_pad_e.expand(1, 4, -1), tts_bos_e], dim=1) + codec_all_e[:, :-1]
        )

        T_text = input_ids.shape[1] - 8
        text_content_ids = input_ids[:, 3:-5]
        codec_pad_expand = talker.get_input_embeddings()(
            torch.full((1, T_text + 1), CODEC_PAD, device=dev)
        )
        text_proj_e = talker.text_projection(talker.get_text_embeddings()(text_content_ids))
        text_combined = torch.cat([text_proj_e, tts_eos_e], dim=1) + codec_pad_expand

        pad_bos = tts_pad_e + talker.get_input_embeddings()(torch.tensor([[CODEC_BOS]], device=dev))
        prefill = torch.cat([role_e, _talker_embed, text_combined, pad_bos], dim=1)
        T_prefix = prefill.shape[1]

    codes_dev = codec_codes.long().unsqueeze(0)  # [1, T, 16] — never cast to float
    codec_sum_e = talker.get_input_embeddings()(codes_dev[:, :, 0])
    for i in range(1, 16):
        codec_sum_e = codec_sum_e + talker.code_predictor.get_input_embeddings()[i-1](
            codes_dev[:, :, i]
        )
    codec_input_e = codec_sum_e + tts_pad_e.expand(1, T, -1)

    inputs_embeds = torch.cat([prefill, codec_input_e], dim=1).to(dtype)

    first_cb = codes_dev[:, :, 0]
    labels = torch.cat([
        torch.full((1, T_prefix), -100, dtype=torch.long, device=dev),
        first_cb,
    ], dim=1)

    return inputs_embeds, labels, T_prefix


def compute_loss(text, codec_codes, talker, processor, dev, dtype, sub_batch):
    inputs_embeds, labels, T_prefix = build_inputs_and_labels(
        text, codec_codes, talker, processor, dev, dtype
    )

    outputs = talker.model(inputs_embeds=inputs_embeds)
    hidden_states = outputs.last_hidden_state

    codec_logits = talker.codec_head(hidden_states)
    loss_main = talker.loss_function(
        logits=codec_logits, labels=labels, vocab_size=talker.config.vocab_size
    )

    # Sub-talker: train code_predictor to predict codebooks 1-15 at each time step
    T = codec_codes.shape[0]
    codec_hidden = hidden_states[:, T_prefix:, :].squeeze(0)  # [T, H]
    codes_flat = codec_codes.long()                           # [T, 16]

    total_sub = torch.tensor(0.0, device=dev, dtype=dtype)
    for i in range(0, T, sub_batch):
        h_chunk = codec_hidden[i:i + sub_batch]
        c_chunk = codes_flat[i:i + sub_batch]
        _, sl = talker.forward_sub_talker_finetune(c_chunk, h_chunk)
        total_sub = total_sub + sl * h_chunk.shape[0]

    return loss_main + 0.5 * (total_sub / T), loss_main, total_sub / T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  required=True)
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--max_t",        type=int,   default=512,
                        help="Max codec frames per sample (~41 sec at 12.5 Hz)")
    parser.add_argument("--lora_rank",    type=int,   default=64)
    parser.add_argument("--lora_alpha",   type=int,   default=128)
    parser.add_argument("--warmup_steps", type=int,   default=200)
    parser.add_argument("--log_steps",    type=int,   default=50)
    parser.add_argument("--save_steps",   type=int,   default=500)
    parser.add_argument("--sub_batch",    type=int,   default=256,
                        help="Sub-batch size for code_predictor forward")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.bfloat16

    print("Loading TTS model...")
    tts_wrapper = Qwen3TTSModel.from_pretrained(
        args.model_dir, device_map="cuda", dtype=dtype
    )
    inner_model = tts_wrapper.model
    talker = inner_model.talker
    processor = tts_wrapper.processor
    dev = talker.device

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    talker.model = get_peft_model(talker.model, lora_cfg)
    talker.model.print_trainable_parameters()

    train_ds = TSCDataset(f"{args.data_dir}/Train_metadata.json", args.max_t)
    eval_ds  = TSCDataset(f"{args.data_dir}/Test_metadata.json",  max_t=128)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          collate_fn=collate_single, num_workers=0)
    eval_dl  = DataLoader(eval_ds,  batch_size=1, shuffle=False,
                          collate_fn=collate_single, num_workers=0)

    total_steps = len(train_dl) * args.epochs
    optimizer = AdamW(
        [p for p in talker.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps)

    print(f"\nTraining: {len(train_ds)} samples × {args.epochs} epochs = {total_steps} steps")

    global_step = 0
    best_eval = float("inf")
    log_loss = 0.0

    for epoch in range(args.epochs):
        talker.train()
        t0 = time.time()

        for sample in train_dl:
            codes = sample["codes"].cuda()[:args.max_t]
            if codes.shape[0] < 2:
                continue

            optimizer.zero_grad()
            loss, l_main, l_sub = compute_loss(
                sample["text"], codes, talker, processor, dev, dtype, args.sub_batch
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(talker.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            if global_step <= args.warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * global_step / args.warmup_steps
            else:
                scheduler.step()

            log_loss += loss.item()

            if global_step % args.log_steps == 0:
                avg = log_loss / args.log_steps
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch+1} Step {global_step:6d} | "
                      f"loss={avg:.4f}  main={l_main.item():.4f}  "
                      f"sub={l_sub.item():.4f}  lr={lr:.2e}")
                log_loss = 0.0

            if global_step % args.save_steps == 0:
                talker.eval()
                eval_losses = []
                with torch.no_grad():
                    for i, s in enumerate(eval_dl):
                        if i >= 50:
                            break
                        c = s["codes"].cuda()[:args.max_t]
                        if c.shape[0] < 2:
                            continue
                        el, _, _ = compute_loss(
                            s["text"], c, talker, processor, dev, dtype, args.sub_batch
                        )
                        eval_losses.append(el.item())
                eval_loss = float(np.mean(eval_losses)) if eval_losses else float("inf")
                print(f"  *** Eval loss: {eval_loss:.4f} (step {global_step}) ***")
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    ckpt = os.path.join(args.output_dir, "best")
                    os.makedirs(ckpt, exist_ok=True)
                    talker.model.save_pretrained(ckpt)
                    print(f"  Saved best checkpoint → {ckpt}")
                talker.train()

        print(f"\nEpoch {epoch+1} done in {(time.time()-t0)/60:.1f} min\n")

    final = os.path.join(args.output_dir, "final")
    os.makedirs(final, exist_ok=True)
    talker.model.save_pretrained(final)
    print(f"Training complete. Final adapter → {final}")


if __name__ == "__main__":
    main()
