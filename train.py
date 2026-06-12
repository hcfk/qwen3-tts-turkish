"""
Step 2: Fine-tune Qwen3-TTS-0.6B-Base on Turkish Speech Corpus using LoRA.

Usage:
    # Fresh start
    python train.py --model_dir ... --data_dir ... --output_dir ...

    # Resume from checkpoint with lower LR
    python train.py --model_dir ... --data_dir ... --output_dir ... \
        --resume_from ./checkpoints/best --lr 2e-6 --max_steps 20000 --sample_every 5000
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from qwen_tts import Qwen3TTSModel

TTS_BOS_TID     = 151672
TTS_EOS_TID     = 151673
TTS_PAD_TID     = 151671
CODEC_THINK     = 2154
CODEC_THINK_BOS = 2156
CODEC_THINK_EOS = 2157
CODEC_PAD       = 2148
CODEC_BOS       = 2149
CODEC_EOS       = 2150
TURKISH_LANG_ID = 2072

SAMPLE_SENTENCES = [
    "Bugün hava çok güzel.",
    "Türkiye Cumhuriyeti 1923 yılında kuruldu.",
]


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
        codes = torch.from_numpy(np.load(s["token_file"])).long()
        return {"text": s["text"], "codes": codes}


def collate_single(batch):
    return batch[0]


def build_inputs_and_labels(text, codec_codes, talker, processor, dev, dtype):
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

        # EOS position embedding
        eos_e = tts_pad_e + talker.get_input_embeddings()(torch.tensor([[CODEC_EOS]], device=dev))

    codes_dev = codec_codes.long().unsqueeze(0)  # [1, T, 16] — never cast to float
    codec_sum_e = talker.get_input_embeddings()(codes_dev[:, :, 0])
    for i in range(1, 16):
        codec_sum_e = codec_sum_e + talker.code_predictor.get_input_embeddings()[i-1](
            codes_dev[:, :, i]
        )
    codec_input_e = codec_sum_e + tts_pad_e.expand(1, T, -1)

    # Append EOS embedding so model learns when to stop
    inputs_embeds = torch.cat([prefill, codec_input_e, eos_e], dim=1).to(dtype)

    first_cb = codes_dev[:, :, 0]
    eos_label = torch.tensor([[CODEC_EOS]], dtype=torch.long, device=dev)
    labels = torch.cat([
        torch.full((1, T_prefix), -100, dtype=torch.long, device=dev),
        first_cb,
        eos_label,
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

    # Sub-talker: only on codec positions, not EOS position
    T = codec_codes.shape[0]
    codec_hidden = hidden_states[:, T_prefix:T_prefix + T, :].squeeze(0)  # [T, H]
    codes_flat = codec_codes.long()

    total_sub = torch.tensor(0.0, device=dev, dtype=dtype)
    for i in range(0, T, sub_batch):
        h_chunk = codec_hidden[i:i + sub_batch]
        c_chunk = codes_flat[i:i + sub_batch]
        _, sl = talker.forward_sub_talker_finetune(c_chunk, h_chunk)
        total_sub = total_sub + sl * h_chunk.shape[0]

    return loss_main + 0.5 * (total_sub / T), loss_main, total_sub / T


def generate_sample(tts_wrapper, step, output_dir):
    speech_tok = tts_wrapper.model.speech_tokenizer.model
    for i, text in enumerate(SAMPLE_SENTENCES):
        input_ids = [tts_wrapper.processor(
            text=f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
            return_tensors="pt",
        )["input_ids"].to(tts_wrapper.device)]
        with torch.inference_mode():
            talker_codes, _ = tts_wrapper.model.generate(
                input_ids=input_ids,
                languages=["turkish"],
                non_streaming_mode=True,
                max_new_tokens=300,
            )
        codes_tensor = talker_codes[0].unsqueeze(0).cuda()
        with torch.inference_mode():
            wav = speech_tok.decode(codes_tensor)
        wav_np = wav.audio_values[0].cpu().float().numpy()
        out = os.path.join(output_dir, f"sample_step{step:06d}_s{i+1}.wav")
        sf.write(out, wav_np, 24000)
        print(f"  [sample] {text!r} → {len(wav_np)/24000:.1f}s → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--resume_from",  default=None,
                        help="Path to existing LoRA adapter to resume from")
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--max_steps",     type=int,   default=1000,
                        help="Stop after this many steps (0 = full epochs)")
    parser.add_argument("--lr",            type=float, default=5e-6)
    parser.add_argument("--max_t",         type=int,   default=150,
                        help="Max codec frames (~12s at 12.5Hz)")
    parser.add_argument("--lora_rank",     type=int,   default=64)
    parser.add_argument("--lora_alpha",    type=int,   default=128)
    parser.add_argument("--warmup_steps",  type=int,   default=50)
    parser.add_argument("--log_steps",     type=int,   default=10)
    parser.add_argument("--save_steps",    type=int,   default=100)
    parser.add_argument("--sample_every",  type=int,   default=100,
                        help="Generate audio samples every N steps (0 = disabled)")
    parser.add_argument("--early_stop",    type=int,   default=3,
                        help="Stop if eval loss rises N consecutive evals (0 = disabled)")
    parser.add_argument("--sub_batch",     type=int,   default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    if args.sample_every > 0:
        os.makedirs(samples_dir, exist_ok=True)
    dtype = torch.bfloat16

    print("Loading TTS model...")
    tts_wrapper = Qwen3TTSModel.from_pretrained(
        args.model_dir, device_map="cuda", dtype=dtype
    )
    inner_model = tts_wrapper.model
    talker = inner_model.talker
    processor = tts_wrapper.processor
    dev = talker.device

    # Register Turkish so generate() accepts it
    inner_model.config.talker_config.codec_language_id["turkish"] = TURKISH_LANG_ID
    inner_model.supported_languages = list(inner_model.supported_languages) + ["turkish"]

    if args.resume_from:
        print(f"Resuming from {args.resume_from} ...")
        talker.model = PeftModel.from_pretrained(talker.model, args.resume_from, is_trainable=True)
    else:
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
    eval_ds  = TSCDataset(f"{args.data_dir}/Test_metadata.json",  max_t=args.max_t)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          collate_fn=collate_single, num_workers=0)
    eval_dl  = DataLoader(eval_ds,  batch_size=1, shuffle=False,
                          collate_fn=collate_single, num_workers=0)

    total_steps = args.max_steps if args.max_steps > 0 else len(train_dl) * args.epochs
    optimizer = AdamW(
        [p for p in talker.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - args.warmup_steps))

    print(f"\nTraining: lr={args.lr:.1e}  max_steps={total_steps}"
          f"  resume={'yes' if args.resume_from else 'no'}")

    global_step    = 0
    best_eval      = float("inf")
    eval_rises     = 0
    prev_eval_loss = float("inf")
    log_loss       = 0.0
    done           = False

    for epoch in range(args.epochs):
        if done:
            break
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
                lr  = optimizer.param_groups[0]["lr"]
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
                        el, _, _ = compute_loss(s["text"], c, talker, processor, dev, dtype, args.sub_batch)
                        eval_losses.append(el.item())
                eval_loss = float(np.mean(eval_losses)) if eval_losses else float("inf")
                print(f"  *** Eval loss: {eval_loss:.4f} (step {global_step}) ***")
                if eval_loss < best_eval:
                    best_eval  = eval_loss
                    eval_rises = 0
                    ckpt = os.path.join(args.output_dir, "best")
                    os.makedirs(ckpt, exist_ok=True)
                    talker.model.save_pretrained(ckpt)
                    print(f"  Saved best checkpoint → {ckpt}")
                else:
                    if eval_loss > prev_eval_loss:
                        eval_rises += 1
                        print(f"  ↑ Eval loss yükseliyor ({eval_rises}/{args.early_stop})")
                        if args.early_stop > 0 and eval_rises >= args.early_stop:
                            print(f"  Early stop — eval loss {args.early_stop} kez üst üste yükseldi.")
                            done = True
                prev_eval_loss = eval_loss
                talker.train()

            if args.sample_every > 0 and global_step % args.sample_every == 0:
                talker.eval()
                print(f"\n  --- Generating samples at step {global_step} ---")
                generate_sample(tts_wrapper, global_step, samples_dir)
                talker.train()

            if args.max_steps > 0 and global_step >= args.max_steps:
                done = True
                break

        print(f"\nEpoch {epoch+1} done in {(time.time()-t0)/60:.1f} min\n")

    final = os.path.join(args.output_dir, "final")
    os.makedirs(final, exist_ok=True)
    talker.model.save_pretrained(final)
    print(f"Training complete. Final adapter → {final}")


if __name__ == "__main__":
    main()
