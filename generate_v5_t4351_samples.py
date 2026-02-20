#!/usr/bin/env python3
"""
Generate comparison audio samples from the V5 (transformers 4.35.1) model
and compare against V4 200ep, production, and base models.

V5 was trained using transformers 4.35.1 to isolate library version effects.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
from transformers import VitsModel, VitsTokenizer
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

# Models to compare
MODELS = {
    "v5_t4351": {
        "path": "./outputs/vits_finetuned_luo_sharonm_v5_t4351",
        "desc": "V5 - transformers 4.35.1 (200ep, no freezing)",
    },
    "v5_t4351_hub": {
        "path": "mutisya/vits_luo_26_05_f_v5_t4351",
        "desc": "V5 - from HuggingFace Hub",
    },
    "v4_200ep": {
        "path": "mutisya/vits_luo_26_05_f_v4_200ep",
        "desc": "V4 - transformers 5.0 (200ep, frozen text_enc)",
    },
    "v4_50ep": {
        "path": "mutisya/vits_luo_26_05_f_v4",
        "desc": "V4 - transformers 5.0 (50ep, frozen text_enc)",
    },
    "production": {
        "path": "mutisya/vits_luo_drL_24_5-v24_27_1_f",
        "desc": "Production model (transformers 4.35.1, different dataset)",
    },
    "base_pre": {
        "path": "mutisya/vits_luo_drL_24_5-v24_27_1_pre",
        "desc": "Base pretrained (before finetuning)",
    },
}


def generate(model, tokenizer, text, device, noise_scale=0.1, noise_scale_duration=0.0):
    """Generate waveform with explicit noise_scale override."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
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


def run_model(model_name, model, tokenizer, output_dir, device):
    """Generate samples for one model and return summary."""
    ns, nsd = 0.1, 0.0
    print(f"  Config: ns={ns}, nsd={nsd}")
    print(f"  {'-' * 60}")

    noise_floors = []
    durations = []

    for i, text in enumerate(TEXTS):
        fname = f"{model_name}_{i + 1:02d}.wav"
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
    print(f"  => Mean NF: {mean_nf:.1f} dB | Mean Duration: {mean_dur:.2f}s")

    return {
        "model": model_name,
        "mean_noise_floor_db": round(float(mean_nf), 1),
        "mean_duration_s": round(float(mean_dur), 2),
        "noise_floors": [round(float(x), 1) for x in noise_floors],
        "durations": [round(float(x), 2) for x in durations],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("luo_v5_t4351_samples")
    output_dir.mkdir(exist_ok=True)

    results = []

    for model_name, info in MODELS.items():
        print(f"\n{'=' * 70}")
        print(f"  {model_name}: {info['desc']}")
        print(f"  Source: {info['path']}")
        print(f"{'=' * 70}")

        torch.manual_seed(555)
        try:
            model = VitsModel.from_pretrained(info["path"]).to(device).eval()
            tokenizer = VitsTokenizer.from_pretrained(info["path"])
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        r = run_model(model_name, model, tokenizer, output_dir, device)
        results.append(r)
        del model, tokenizer
        torch.cuda.empty_cache()

    # ---- Final comparison table ----
    print(f"\n\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'Description':<50} {'NF (dB)':>8} {'Dur (s)':>8}")
    print(f"{'-' * 20} {'-' * 50} {'-' * 8} {'-' * 8}")
    for r in results:
        desc = MODELS.get(r["model"], {}).get("desc", "")
        print(f"{r['model']:<20} {desc:<50} {r['mean_noise_floor_db']:>8.1f} {r['mean_duration_s']:>8.2f}")

    # ---- V5 vs V4 comparison ----
    v5_result = next((r for r in results if r["model"] == "v5_t4351"), None)
    v4_result = next((r for r in results if r["model"] == "v4_200ep"), None)
    prod_result = next((r for r in results if r["model"] == "production"), None)

    if v5_result and v4_result:
        print(f"\n  V5 (4.35.1) vs V4 (5.0):")
        nf_diff = v5_result["mean_noise_floor_db"] - v4_result["mean_noise_floor_db"]
        dur_diff = v5_result["mean_duration_s"] - v4_result["mean_duration_s"]
        print(f"    NF difference:  {nf_diff:+.1f} dB")
        print(f"    Dur difference: {dur_diff:+.3f}s")

    if v5_result and prod_result:
        print(f"\n  V5 (4.35.1) vs Production:")
        nf_diff = v5_result["mean_noise_floor_db"] - prod_result["mean_noise_floor_db"]
        dur_diff = v5_result["mean_duration_s"] - prod_result["mean_duration_s"]
        print(f"    NF difference:  {nf_diff:+.1f} dB")
        print(f"    Dur difference: {dur_diff:+.3f}s")

    # Save results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    total = len(TEXTS) * len(results)
    print(f"\n{total} audio files saved to {output_dir}/")


if __name__ == "__main__":
    main()
