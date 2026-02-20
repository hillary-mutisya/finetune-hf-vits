"""
Save V4 200ep checkpoint-153500 as a proper HuggingFace VitsModel.

Steps:
1. Load training checkpoint (has weight_g/weight_v from weight_norm)
2. Convert weight_norm pairs: weight = weight_g * weight_v / ||weight_v||
3. Load into VitsModel
4. Save locally + push to HuggingFace Hub
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from dotenv import load_dotenv

load_dotenv("/home/hillary/trainingTTS/.env")

# ── Paths ──
CHECKPOINT_DIR = "./outputs/vits_finetuned_luo_sharonm_v4/checkpoint-153500"
SAVE_DIR = "./outputs/vits_luo_26_05_f_v4_200ep_hf"
HUB_MODEL_ID = "mutisya/vits_luo_26_05_f_v4_200ep"

sys.path.insert(0, ".")
from utils.modeling_vits_training import VitsModelForPreTraining

from transformers import VitsModel, VitsTokenizer, AutoConfig

# ── 1. Load checkpoint weights ──
print(f"Loading checkpoint from: {CHECKPOINT_DIR}")
ckpt_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
ckpt_weights = load_file(ckpt_path)
print(f"  Checkpoint has {len(ckpt_weights)} keys")

# ── 2. Load config ──
# The checkpoint config has vocab_size=38 (from training data tokenizer) but the
# actual text_encoder weights are [28, 192] because text_encoder was frozen.
# Use the v4_50ep config which has the correct vocab_size=28.
config = AutoConfig.from_pretrained("mutisya/vits_luo_26_05_f_v4")
print(f"  Config loaded: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")

# ── 3. Convert weight_norm pairs ──
# Find all weight_v keys (each has a matching weight_g)
weight_v_keys = [k for k in ckpt_weights if k.endswith(".weight_v")]
print(f"  Found {len(weight_v_keys)} weight_norm pairs to convert")

converted_weights = {}
converted_count = 0

for key, tensor in ckpt_weights.items():
    if key.endswith(".weight_v"):
        # Convert: weight = weight_g * weight_v / ||weight_v||
        base = key[:-len(".weight_v")]
        g_key = base + ".weight_g"
        w_key = base + ".weight"
        
        weight_v = tensor
        weight_g = ckpt_weights[g_key]
        
        # Normalize: ||weight_v|| computed over all dims except first
        norm_dims = list(range(1, weight_v.dim()))
        norm = weight_v.norm(dim=norm_dims, keepdim=True)
        weight = weight_g * weight_v / norm
        
        converted_weights[w_key] = weight
        converted_count += 1
    elif key.endswith(".weight_g"):
        # Skip - already handled with weight_v
        continue
    else:
        # Direct copy
        converted_weights[key] = tensor

print(f"  Converted {converted_count} weight_norm pairs")
print(f"  Final state_dict has {len(converted_weights)} keys")

# ── 4. Filter out discriminator keys ──
disc_keys = [k for k in converted_weights if k.startswith("discriminator.")]
for k in disc_keys:
    del converted_weights[k]
print(f"  Removed {len(disc_keys)} discriminator keys")
print(f"  Generator state_dict has {len(converted_weights)} keys")

# ── 5. Load into VitsModel ──
print("\nCreating VitsModel and loading converted weights...")
model = VitsModel(config)
expected_keys = set(model.state_dict().keys())
provided_keys = set(converted_weights.keys())

missing = expected_keys - provided_keys
unexpected = provided_keys - expected_keys

if missing:
    print(f"  WARNING: Missing keys: {len(missing)}")
    for k in sorted(missing)[:5]:
        print(f"    {k}")
if unexpected:
    print(f"  WARNING: Unexpected keys: {len(unexpected)}")
    for k in sorted(unexpected)[:5]:
        print(f"    {k}")

result = model.load_state_dict(converted_weights, strict=False)
matched = len(expected_keys) - len(result.missing_keys)
print(f"  Loaded {matched}/{len(expected_keys)} weights")
assert matched == len(expected_keys), f"Weight loading incomplete! Missing: {result.missing_keys}"
print("  ✓ All weights loaded successfully!")

# ── 6. Quick sanity check - generate one sample ──
print("\nSanity check: generating a test sample...")
tokenizer = VitsTokenizer.from_pretrained("mutisya/vits_luo_26_05_f_v4")
model.eval()
model = model.to("cuda")

test_text = "Ber mondo. Nyinga en ngʼa?"
inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model(**inputs)

waveform = output.waveform[0].cpu().numpy()
silence_threshold = np.percentile(np.abs(waveform), 10)
noise_floor_db = 20 * np.log10(silence_threshold + 1e-10)
duration = len(waveform) / model.config.sampling_rate

print(f"  Duration: {duration:.2f}s | Noise Floor: {noise_floor_db:.1f} dB")
assert noise_floor_db < -50, f"Audio quality check failed: NF={noise_floor_db:.1f} dB (expected < -50)"
print("  ✓ Audio quality check passed!")

# ── 7. Save locally ──
print(f"\nSaving model to: {SAVE_DIR}")
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("  ✓ Saved locally!")

# ── 8. Push to Hub ──
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print(f"\nPushing to HuggingFace Hub: {HUB_MODEL_ID}")
    model.push_to_hub(HUB_MODEL_ID, private=True, token=hf_token)
    tokenizer.push_to_hub(HUB_MODEL_ID, private=True, token=hf_token)
    print(f"  ✓ Pushed to {HUB_MODEL_ID}!")
else:
    print("\n  WARNING: HF_TOKEN not set, skipping push to hub")

print("\n" + "=" * 60)
print("DONE! Model saved and pushed.")
print(f"  Local:  {SAVE_DIR}")
print(f"  Hub:    https://huggingface.co/{HUB_MODEL_ID}")
print(f"  Source: checkpoint-153500 (~167 epochs, step 153,500/183,600)")
print("=" * 60)
