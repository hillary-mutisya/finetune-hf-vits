#!/usr/bin/env python3
"""
Generate comparison samples from V4 200ep checkpoint-153500 (~167 epochs)
vs v4 50ep, base, production, v3
"""

import torch
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
from transformers import VitsModel, VitsTokenizer, VitsConfig
from safetensors.torch import load_file as load_safetensors
import json

# Luo test sentences
TEXTS = [
    "Ber mondo. Nyinga en ngʼa?",
    "Agoyo ni erokamano kuom kony mari.",
    "Wan ji ma ohero loso kode kuom weche mag kwan.",
    "Japuonj nopuonjo nyithindo e skul ka.",
    "Chiemo ne nyalo morowa ahinya kawuono.",
    "Jatelo nowacho ni wan duto nyaka wati matek.",
    "Nyathi matin noyudo thuolo mar tugo e pap.",
    "Koth ne ochue matek e otuoma duto.",
]

# Local checkpoint path
V4_200EP_CHECKPOINT = "./outputs/vits_finetuned_luo_sharonm_v4/checkpoint-153500"
V4_200EP_CONFIG = "./outputs/vits_finetuned_luo_sharonm_v4"  # config.json is in output root

MODELS = {
    "v4_200ep_ckpt153500": None,  # Special: load from local checkpoint
    "v4_50ep": "mutisya/vits_luo_26_05_f_v4",
    "base_pre": "mutisya/vits_luo_drL_24_5-v24_27_1_pre",
    "production": "mutisya/vits_luo_drL_24_5-v24_27_1_f",
    "v3_buggy": "mutisya/vits_luo_26_05_f_v3",
}

def generate(model, tokenizer, text, device, noise_scale=0.1, noise_scale_duration=0.0):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs, noise_scale=noise_scale, noise_scale_duration=noise_scale_duration)
        waveform = output.waveform[0].cpu().numpy()
    return waveform, model.config.sampling_rate

def save_audio(waveform, sample_rate, path):
    waveform = waveform / np.max(np.abs(waveform)) * 0.95
    waveform_int16 = (waveform * 32767).astype(np.int16)
    wav.write(str(path), sample_rate, waveform_int16)

def compute_noise_floor(waveform, sr):
    frame_size = int(0.025 * sr)
    hop = frame_size // 2
    energies = []
    for i in range(0, len(waveform) - frame_size, hop):
        frame = waveform[i:i+frame_size]
        rms = np.sqrt(np.mean(frame**2))
        if rms > 0:
            energies.append(20 * np.log10(rms))
    energies.sort()
    n = max(1, len(energies) // 10)
    return np.mean(energies[:n])

def load_from_checkpoint(checkpoint_dir, config_dir, device):
    """Load VitsModel from a training checkpoint, converting weight_norm tensors."""
    config = VitsConfig.from_pretrained(config_dir)
    model = VitsModel(config)
    
    ckpt_path = Path(checkpoint_dir) / "model.safetensors"
    print(f"  Loading weights from {ckpt_path}")
    state_dict = load_safetensors(str(ckpt_path))
    
    # Training checkpoints use weight normalization (weight_g + weight_v)
    # VitsModel expects merged weight tensors. Convert them:
    #   weight = weight_g * weight_v / ||weight_v||
    converted_sd = {}
    weight_v_keys = {k for k in state_dict if k.endswith('.weight_v')}
    
    for key, tensor in state_dict.items():
        if key.endswith('.weight_g'):
            # Will be handled together with weight_v
            continue
        elif key.endswith('.weight_v'):
            base_key = key[:-2]  # remove '_v' to get '.weight'
            g_key = base_key + '_g'
            if g_key in state_dict:
                weight_v = tensor
                weight_g = state_dict[g_key]
                # Reconstruct: weight = g * v / ||v||
                norm = torch.norm(weight_v.reshape(weight_v.shape[0], -1), dim=1)
                # Reshape norm and weight_g to broadcast correctly
                for _ in range(weight_v.dim() - 1):
                    norm = norm.unsqueeze(-1)
                merged_weight = weight_g * weight_v / norm
                converted_sd[base_key] = merged_weight
                continue
        converted_sd[key] = tensor
    
    result = model.load_state_dict(converted_sd, strict=False)
    matched = len(converted_sd) - len(result.unexpected_keys)
    print(f"  Loaded {matched}/{len(model.state_dict())} weights (converted {len(weight_v_keys)} weight_norm pairs)")
    if result.missing_keys:
        print(f"  Missing keys: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")
    
    return model.to(device).eval()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("luo_v4_200ep_samples")
    output_dir.mkdir(exist_ok=True)

    results = []

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Loading {model_name}: {model_id or V4_200EP_CHECKPOINT}")
        print(f"{'='*70}")
        
        try:
            if model_name == "v4_200ep_ckpt153500":
                # Load from local checkpoint
                model = load_from_checkpoint(V4_200EP_CHECKPOINT, V4_200EP_CONFIG, device)
                tokenizer = VitsTokenizer.from_pretrained(V4_200EP_CONFIG)
            else:
                model = VitsModel.from_pretrained(model_id).to(device).eval()
                tokenizer = VitsTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            import traceback; traceback.print_exc()
            continue

        ns, nsd = 0.1, 0.0
        print(f"\n  Config: optimized (ns={ns}, nsd={nsd})")
        print(f"  {'-'*60}")

        noise_floors = []

        for i, text in enumerate(TEXTS):
            fname = f"{model_name}_optimized_{i+1:02d}.wav"
            fpath = output_dir / fname

            waveform, sr = generate(model, tokenizer, text, device, noise_scale=ns, noise_scale_duration=nsd)
            save_audio(waveform, sr, fpath)
            duration = len(waveform) / sr
            nf = compute_noise_floor(waveform, sr)
            noise_floors.append(nf)

            print(f"    [{i+1}] {text[:50]:<50} {duration:.2f}s NF={nf:.1f}dB")

        mean_nf = np.mean(noise_floors)
        print(f"  => Mean Noise Floor: {mean_nf:.1f} dB")
        
        results.append({
            "model": model_name,
            "model_id": model_id or V4_200EP_CHECKPOINT,
            "mean_noise_floor_db": round(float(mean_nf), 1),
        })

        del model, tokenizer
        torch.cuda.empty_cache()

    # Final comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Noise Floor':>12}")
    print(f"{'-'*25} {'-'*12}")
    for r in results:
        print(f"{r['model']:<25} {r['mean_noise_floor_db']:>10.1f} dB")

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    total = len(TEXTS) * len(MODELS)
    print(f"\n{total} audio files saved to {output_dir}/")
    print(f"Results saved to {output_dir}/comparison_results.json")

if __name__ == "__main__":
    main()
