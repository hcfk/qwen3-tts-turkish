"""
Run on server after download to extract 1.7B model constants.
python3 /home/hcfk/check_1.7b_config.py
"""
import json, os, sys
import torch

MODEL_DIR = "/home/hcfk/models/Qwen3-TTS-1.7B-Base"

print("=" * 60)
print("Qwen3-TTS-1.7B-Base — Config Inspection")
print("=" * 60)

# 1. Raw config
cfg_path = os.path.join(MODEL_DIR, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)

print("\n[config.json top-level keys]")
for k, v in cfg.items():
    if not isinstance(v, dict):
        print(f"  {k}: {v}")

# Talker sub-config if present
if "talker_config" in cfg:
    print("\n[talker_config]")
    for k, v in cfg["talker_config"].items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

# 2. Tokenizer vocab — find language and codec tokens
tok_path = os.path.join(MODEL_DIR, "tokenizer.json")
if os.path.exists(tok_path):
    with open(tok_path) as f:
        tok = json.load(f)
    vocab = tok.get("model", {}).get("vocab", {})
    if not vocab:
        vocab = tok.get("added_tokens_decoder", {})
        vocab = {v.get("content", ""): int(k) for k, v in vocab.items()}

    print("\n[Tokenizer — language and codec tokens]")
    for name, tid in sorted(vocab.items(), key=lambda x: x[1]):
        if any(x in name.lower() for x in ["turkish", "tr>", "<tr", "eos", "pad", "bos", "think", "codec", "lang", "2148", "2149", "2150", "2154"]):
            print(f"  {repr(name)}: {tid}")

    # Print tokens around expected codec range
    print("\n[Tokens with ID 2140-2160 (codec range on 0.6B)]")
    reverse = {v: k for k, v in vocab.items()}
    for i in range(2140, 2161):
        if i in reverse:
            print(f"  {i}: {repr(reverse[i])}")

# 3. Generation config
gen_cfg_path = os.path.join(MODEL_DIR, "generation_config.json")
if os.path.exists(gen_cfg_path):
    with open(gen_cfg_path) as f:
        gen_cfg = json.load(f)
    print("\n[generation_config.json]")
    for k, v in gen_cfg.items():
        print(f"  {k}: {v}")

# 4. Model structure (load skeleton only)
print("\n[Loading model skeleton to inspect structure...]")
try:
    from transformers import AutoModelForCausalLM, AutoConfig
    auto_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    print(f"  AutoConfig type: {type(auto_cfg).__name__}")
    print(f"  Model type: {getattr(auto_cfg, 'model_type', 'unknown')}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"  Top-level class: {type(model).__name__}")
    for name, child in model.named_children():
        print(f"  .{name}: {type(child).__name__}")
        for n2, c2 in child.named_children():
            print(f"    .{n2}: {type(c2).__name__}")
            for n3, c3 in c2.named_children():
                print(f"      .{n3}: {type(c3).__name__}")
                break  # just first level of each sub
            break
except Exception as e:
    print(f"  AutoModel load failed: {e}")
    print("  Trying manual inspection...")
    safe_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors") or f.endswith(".bin")]
    print(f"  Weight files: {safe_files}")

# 5. Codec vocab size estimate
print("\n[Checking for codec/vocoder config]")
for fname in os.listdir(MODEL_DIR):
    if "vocoder" in fname.lower() or "codec" in fname.lower() or "decoder" in fname.lower():
        full = os.path.join(MODEL_DIR, fname)
        print(f"  {fname}: {os.path.getsize(full)//1024} KB")
        if fname.endswith(".json"):
            with open(full) as f:
                d = json.load(f)
            for k, v in d.items():
                if not isinstance(v, dict):
                    print(f"    {k}: {v}")

print("\n[File listing]")
for fname in sorted(os.listdir(MODEL_DIR)):
    size = os.path.getsize(os.path.join(MODEL_DIR, fname)) // (1024*1024)
    print(f"  {fname:50s} {size:6d} MB")

print("\nDone.")
