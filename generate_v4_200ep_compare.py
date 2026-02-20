#!/usr/bin/env python3
"""
Compare inference from V4 200ep checkpoint using:
1. VitsModelForPreTraining (native, no weight conversion — same as training validation)
2. VitsModel with weight_norm -> weight conversion

This tests whether the weight_g/weight_v -> weight merging changes audio output.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
from transformers import VitsTokenizer, VitsConfig
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

V4_200EP_CHECKPOINT = "./outputs/vits_finetuned_luo_sharonm_v4/checkpoint-153500"
V4_200EP_CONFIG_DIR = "./outputs/vits_finetuned_luo_sharonm_v4"

HF_MODELS = {
    "v4_50ep": "mutisya/vits_luo_26_05_f_v4",
    "base_pre": "mutisya/vits_luo_drL_24_5-v24_27_1_pre",
    "production": "mutisya/vits_luo_drL_24_5-v24_27_1_f",
}


def generate(model, tokenizer, text, device, noise_scale=0.1, noise_scale_duration=0.0):
    """Generate with explicit noise_scale override."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    # Override the model's noise params directly
    old_ns = model.noise_scale
    old_nsd = model.noise_scale_duration
    model.noise_scale = noise_scale
    model.noise_scale_duration = noise_scale_duration
    with torch.no_grad():
        output = model(**inputs)
        waveform = output.waveform[0].cpu().numpy()
    model.noise_scale = old_ns
    model.noise_scale_duration = old_nsd
    return waveform, model.config.sampling_rate


def save_audio(waveform, sample_rate, path):
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak * 0.95
    waveform_int16 = (waveform * 32767).astype(np.int16)
    wav.write(str(path), sample_rate, waveform_int16)


def compute_noise_floor(waveform, sr):
    frame_size = int(0.025 * sr)
    hop = frame_size // 2
    energies = []
    for i in range(0, len(waveform) - frame_size, hop):
        frame = waveform[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > 0:
            energies.append(20 * np.log10(rms))
    energies.sort()
    n = max(1, len(energies) // 10)
    return np.mean(energies[:n])


def load_training_model(checkpoint_dir, config_dir, device):
    """Load VitsModelForPreTraining directly from training checkpoint — no conversion."""
    from utils.modeling_vits_training import VitsModelForPreTraining

    config = VitsConfig.from_pretrained(config_dir)
    model = VitsModelForPreTraining(config)

    ckpt_path = Path(checkpoint_dir) / "model.safetensors"
    sd = load_safetensors(str(ckpt_path))
    result = model.load_state_dict(sd, strict=False)

    matched = len(sd) - len(result.unexpected_keys)
    total = len(model.state_dict())
    print(f"  [Training model] Loaded {matched}/{total} weights")
    if result.missing_keys:
        # Filter out discriminator keys since we don't need them for inference
        non_disc_missing = [k for k in result.missing_keys if not k.startswith("discriminator.")]
        disc_missing = [k for k in result.missing_keys if k.startswith("discriminator.")]
        if disc_missing:
            print(f"  Discriminator keys missing: {len(disc_missing)} (expected — not in generator checkpoint)")
        if non_disc_missing:
            print(f"  WARNING: Non-discriminator missing keys: {len(non_disc_missing)}")
            for k in non_disc_missing[:5]:
                print(f"    {k}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")
        for k in result.unexpected_keys[:5]:
            print(f"    {k}")

    return model.to(device).eval()


def load_converted_model(checkpoint_dir, config_dir, device):
    """Load VitsModel from training checkpoint — with weight_g/weight_v merging."""
    from transformers import VitsModel

    config = VitsConfig.from_pretrained(config_dir)
    model = VitsModel(config)

    ckpt_path = Path(checkpoint_dir) / "model.safetensors"
    state_dict = load_safetensors(str(ckpt_path))

    # Convert weight_g + weight_v -> weight
    converted_sd = {}
    weight_v_keys = {k for k in state_dict if k.endswith('.weight_v')}

    for key, tensor in state_dict.items():
        if key.endswith('.weight_g'):
            continue  # handled with weight_v
        elif key.endswith('.weight_v'):
            base_key = key[:-2]  # '.weight_v' -> '.weight'
            g_key = base_key + '_g'
            if g_key in state_dict:
                weight_v = tensor
                weight_g = state_dict[g_key]
                # weight_norm formula: weight = g * v / ||v||
                norm = torch.norm(weight_v.reshape(weight_v.shape[0], -1), dim=1)
                for _ in range(weight_v.dim() - 1):
                    norm = norm.unsqueeze(-1)
                merged = weight_g * weight_v / norm
                converted_sd[base_key] = merged
                continue
        converted_sd[key] = tensor

    result = model.load_state_dict(converted_sd, strict=False)
    matched = len(converted_sd) - len(result.unexpected_keys)
    total = len(model.state_dict())
    print(f"  [Converted model] Loaded {matched}/{total} weights (converted {len(weight_v_keys)} weight_norm pairs)")
    if result.missing_keys:
        print(f"  Missing keys: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")

    return model.to(device).eval()


def run_model(model_name, model, tokenizer, output_dir, device):
    """Generate samples for one model and return summary."""
    ns, nsd = 0.1, 0.0
    print(f"\n  Config: optimized (ns={ns}, nsd={nsd})")
    print(f"  {'-' * 60}")

    noise_floors = []
    durations = []

    for i, text in enumerate(TEXTS):
        fname = f"{model_name}_optimized_{i + 1:02d}.wav"
        fpath = output_dir / fname

        waveform, sr = generate(model, tokenizer, text, device, noise_scale=ns, noise_scale_duration=nsd)
        save_audio(waveform, sr, fpath)
        duration = len(waveform) / sr
        nf = compute_noise_floor(waveform, sr)
        noise_floors.append(nf)
        durations.append(duration)

        print(f"    [{i + 1}] {text[:50]:<50} {duration:.2f}s NF={nf:.1f}dB")

    mean_nf = np.mean(noise_floors)
    mean_dur = np.mean(durations)
    print(f"  => Mean Noise Floor: {mean_nf:.1f} dB | Mean Duration: {mean_dur:.2f}s")

    return {
        "model": model_name,
        "mean_noise_floor_db": round(float(mean_nf), 1),
        "mean_duration_s": round(float(mean_dur), 2),
        "noise_floors": [round(float(x), 1) for x in noise_floors],
        "durations": [round(float(x), 2) for x in durations],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("luo_v4_200ep_samples")
    output_dir.mkdir(exist_ok=True)

    # Set seed for reproducibility — same random noise for all models
    torch.manual_seed(555)

    results = []
    tokenizer = VitsTokenizer.from_pretrained(V4_200EP_CONFIG_DIR)

    # ---- 1. VitsModelForPreTraining (native, no conversion) ----
    print(f"\n{'=' * 70}")
    print(f"1. VitsModelForPreTraining (native weights, no conversion)")
    print(f"   Checkpoint: {V4_200EP_CHECKPOINT}")
    print(f"{'=' * 70}")
    torch.manual_seed(555)
    model_native = load_training_model(V4_200EP_CHECKPOINT, V4_200EP_CONFIG_DIR, device)
    r = run_model("v4_200ep_native", model_native, tokenizer, output_dir, device)
    results.append(r)
    del model_native
    torch.cuda.empty_cache()

    # ---- 2. VitsModel (converted weight_norm) ----
    print(f"\n{'=' * 70}")
    print(f"2. VitsModel (weight_g/weight_v -> weight conversion)")
    print(f"   Checkpoint: {V4_200EP_CHECKPOINT}")
    print(f"{'=' * 70}")
    torch.manual_seed(555)
    model_converted = load_converted_model(V4_200EP_CHECKPOINT, V4_200EP_CONFIG_DIR, device)
    r = run_model("v4_200ep_converted", model_converted, tokenizer, output_dir, device)
    results.append(r)
    del model_converted
    torch.cuda.empty_cache()

    # ---- 3-5. HuggingFace models for comparison ----
    from transformers import VitsModel
    for model_name, model_id in HF_MODELS.items():
        print(f"\n{'=' * 70}")
        print(f"Loading {model_name}: {model_id}")
        print(f"{'=' * 70}")
        torch.manual_seed(555)
        try:
            model = VitsModel.from_pretrained(model_id).to(device).eval()
            tok = VitsTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        r = run_model(model_name, model, tok, output_dir, device)
        results.append(r)
        del model, tok
        torch.cuda.empty_cache()

    # ---- Final comparison table ----
    print(f"\n\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Model':<25} {'NF (dB)':>10} {'Avg Dur (s)':>12}")
    print(f"{'-' * 25} {'-' * 10} {'-' * 12}")
    for r in results:
        print(f"{r['model']:<25} {r['mean_noise_floor_db']:>10.1f} {r['mean_duration_s']:>12.2f}")

    # ---- Check if native vs converted differ ----
    if len(results) >= 2:
        native = results[0]
        converted = results[1]
        nf_diff = abs(native["mean_noise_floor_db"] - converted["mean_noise_floor_db"])
        dur_diff = abs(native["mean_duration_s"] - converted["mean_duration_s"])
        print(f"\n  Native vs Converted:")
        print(f"    NF difference:       {nf_diff:.1f} dB")
        print(f"    Duration difference: {dur_diff:.3f}s")
        if nf_diff < 0.5 and dur_diff < 0.01:
            print(f"    => Effectively IDENTICAL (weight_norm merging is mathematically exact)")
        else:
            print(f"    => DIFFERENT — investigate further!")

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    total = len(TEXTS) * (2 + len(HF_MODELS))
    print(f"\n{total} audio files saved to {output_dir}/")


if __name__ == "__main__":
    main()
