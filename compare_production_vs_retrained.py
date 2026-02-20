#!/usr/bin/env python3
"""
Compare production Luo VITS model vs our retrained model.

Production: mutisya/vits_luo_drL_24_5-v24_27_1_f (deployed in polyglot-backend)
Retrained:  mutisya/vits_luo_26_05_f_v1 (our new training)

For each model, generates audio with:
  A) Default pipeline params (noise_scale=0.667) — what production actually uses
  B) Optimized params (noise_scale=0.1) — our improved inference
"""

import os
import sys
import torch
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

# Luo test sentences
TEST_TEXTS = [
    "Chike ker ilando kod ker owuon. Nyasaye nochiwo nyathi moro achiel ma ne en gi paro maber.",
    "Joka Nyasaye ne gibet e piny Misri kama ne gin jotich.",
    "Wuod dhano nobiro mondo ores jogo molal.",
    "Ng'ato ka ng'ato nyalo timo gik mabeyo ka ohero Nyasaye.",
    "Koth nochako chue mi pi nopong'o aora.",
]

OUTPUT_DIR = Path("luo_production_comparison")


def generate_audio(model, tokenizer, text, device, noise_scale=0.667, noise_scale_duration=0.8):
    """Generate audio with specified parameters."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs, noise_scale=noise_scale, noise_scale_duration=noise_scale_duration)
    waveform = output.waveform[0].cpu().numpy()
    return waveform, model.config.sampling_rate


def generate_pipeline_style(model_id, text, device_idx=-1):
    """Generate audio the way the production backend does — via HF pipeline."""
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-to-speech", model=model_id, device=device_idx)
    result = pipe(text)
    return result["audio"], result.get("sampling_rate", 16000)


def save_wav(waveform, sr, path, normalize=True):
    """Save waveform to WAV file."""
    if normalize and np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform)) * 0.95
    waveform_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    wav.write(str(path), sr, waveform_int16)


def compute_audio_stats(waveform, sr):
    """Compute audio quality statistics."""
    rms = np.sqrt(np.mean(waveform ** 2))
    duration = len(waveform) / sr

    # Estimate noise floor from quietest 10% of frames
    frame_size = int(0.025 * sr)
    hop = int(0.010 * sr)
    frames = [waveform[i:i + frame_size] for i in range(0, len(waveform) - frame_size, hop)]
    if frames:
        frame_energies = [np.sqrt(np.mean(f ** 2)) for f in frames]
        frame_energies.sort()
        n_quiet = max(1, len(frame_energies) // 10)
        noise_floor = np.mean(frame_energies[:n_quiet])
        noise_floor_db = 20 * np.log10(noise_floor + 1e-10)
    else:
        noise_floor_db = -100

    return {
        "duration": duration,
        "rms": rms,
        "noise_floor_db": noise_floor_db,
        "peak": np.max(np.abs(waveform)),
    }


def compute_utmos(wav_path):
    """Compute UTMOS score for a wav file."""
    try:
        from utmos import Score
        scorer = Score()
        return float(scorer.calculate_wav_file(str(wav_path)))
    except Exception as e:
        print(f"  UTMOS error: {e}")
        return None


def main():
    from transformers import VitsModel, VitsTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Model IDs
    prod_model_id = "mutisya/vits_luo_drL_24_5-v24_27_1_f"
    new_model_id = "mutisya/vits_luo_26_05_f_v1"

    print("=" * 80)
    print("PRODUCTION vs RETRAINED Luo VITS Model Comparison")
    print("=" * 80)

    # Load both models
    print(f"\nLoading production model: {prod_model_id}")
    prod_model = VitsModel.from_pretrained(prod_model_id).to(device).eval()
    prod_tokenizer = VitsTokenizer.from_pretrained(prod_model_id)

    print(f"Loading retrained model:  {new_model_id}")
    new_model = VitsModel.from_pretrained(new_model_id).to(device).eval()
    new_tokenizer = VitsTokenizer.from_pretrained(new_model_id)

    # Configurations to test
    configs = [
        ("pipeline_default", 0.667, 0.8),   # What production backend uses
        ("optimized",        0.1,   0.0),    # Our optimized settings
    ]

    all_results = []

    for i, text in enumerate(TEST_TEXTS):
        print(f"\n{'─' * 70}")
        print(f"Text {i + 1}: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
        print(f"{'─' * 70}")

        for config_name, ns, nsd in configs:
            for model_label, model, tokenizer in [
                ("production", prod_model, prod_tokenizer),
                ("retrained",  new_model,  new_tokenizer),
            ]:
                tag = f"{model_label}_{config_name}"
                filename = f"text{i + 1:02d}_{tag}.wav"
                filepath = OUTPUT_DIR / filename

                waveform, sr = generate_audio(model, tokenizer, text, device,
                                              noise_scale=ns, noise_scale_duration=nsd)
                save_wav(waveform, sr, filepath)

                stats = compute_audio_stats(waveform, sr)
                utmos = compute_utmos(filepath)

                result = {
                    "text_idx": i + 1,
                    "model": model_label,
                    "config": config_name,
                    "noise_scale": ns,
                    "duration": stats["duration"],
                    "noise_floor_db": stats["noise_floor_db"],
                    "rms": stats["rms"],
                    "utmos": utmos,
                    "file": filename,
                }
                all_results.append(result)

                utmos_str = f"{utmos:.3f}" if utmos else "N/A"
                print(f"  {tag:<35} dur={stats['duration']:.2f}s  "
                      f"noise_floor={stats['noise_floor_db']:.1f}dB  "
                      f"UTMOS={utmos_str}  -> {filename}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    # Aggregate by model+config
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        key = f"{r['model']}_{r['config']}"
        groups[key].append(r)

    print(f"\n{'Configuration':<40} {'Mean UTMOS':>12} {'Mean Noise Floor':>18} {'Mean Dur':>10}")
    print("─" * 82)

    for key in ["production_pipeline_default", "production_optimized",
                "retrained_pipeline_default", "retrained_optimized"]:
        results = groups.get(key, [])
        if not results:
            continue

        utmos_scores = [r["utmos"] for r in results if r["utmos"] is not None]
        noise_floors = [r["noise_floor_db"] for r in results]
        durations = [r["duration"] for r in results]

        mean_utmos = np.mean(utmos_scores) if utmos_scores else float("nan")
        mean_nf = np.mean(noise_floors)
        mean_dur = np.mean(durations)

        ns = results[0]["noise_scale"]
        label = f"{key} (ns={ns})"
        print(f"  {label:<38} {mean_utmos:>10.3f}   {mean_nf:>14.1f} dB   {mean_dur:>7.2f}s")

    print(f"\nAll audio saved to: {OUTPUT_DIR}/")
    print(f"Total files: {len(all_results)}")

    # Also save a simple pipeline-generated sample to match exactly what production does
    print(f"\n{'─' * 70}")
    print("Bonus: Generating via HF pipeline (exact production behavior)...")
    pipe_dir = OUTPUT_DIR / "pipeline_exact"
    pipe_dir.mkdir(exist_ok=True)

    for i, text in enumerate(TEST_TEXTS[:3]):
        wf, sr = generate_pipeline_style(prod_model_id, text)
        filepath = pipe_dir / f"text{i + 1:02d}_pipeline_exact.wav"
        save_wav(wf, sr, filepath)
        utmos = compute_utmos(filepath)
        utmos_str = f"{utmos:.3f}" if utmos else "N/A"
        print(f"  pipeline_exact text{i + 1}: UTMOS={utmos_str} -> {filepath.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
