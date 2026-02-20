#!/usr/bin/env python3
"""
Generate comparison samples: v4 (fixed loading + frozen text_enc+dur_pred) vs base vs production vs v3
"""

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

MODELS = {
    "v4_fixed": "mutisya/vits_luo_26_05_f_v4",
    "base_pre": "mutisya/vits_luo_drL_24_5-v24_27_1_pre",
    "production": "mutisya/vits_luo_drL_24_5-v24_27_1_f",
    "v3_buggy": "mutisya/vits_luo_26_05_f_v3",
}

NOISE_CONFIGS = {
    "optimized": {"noise_scale": 0.1, "noise_scale_duration": 0.0},
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("luo_v4_samples")
    output_dir.mkdir(exist_ok=True)

    # Try loading UTMOS
    utmos_scorer = None
    try:
        import utmos
        utmos_scorer = utmos.Score(device=device)
        print("UTMOS scorer loaded")
    except Exception as e:
        print(f"UTMOS not available: {e}")

    results = []

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Loading {model_name}: {model_id}")
        print(f"{'='*70}")
        try:
            model = VitsModel.from_pretrained(model_id).to(device).eval()
            tokenizer = VitsTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        for config_name, config in NOISE_CONFIGS.items():
            print(f"\n  Config: {config_name} (ns={config['noise_scale']}, nsd={config['noise_scale_duration']})")
            print(f"  {'-'*60}")

            scores = []
            noise_floors = []

            for i, text in enumerate(TEXTS):
                fname = f"{model_name}_{config_name}_{i+1:02d}.wav"
                fpath = output_dir / fname

                waveform, sr = generate(
                    model, tokenizer, text, device,
                    noise_scale=config["noise_scale"],
                    noise_scale_duration=config["noise_scale_duration"],
                )
                save_audio(waveform, sr, fpath)
                duration = len(waveform) / sr
                nf = compute_noise_floor(waveform, sr)
                noise_floors.append(nf)

                score = None
                if utmos_scorer:
                    try:
                        score = utmos_scorer.calculate_wav_file(str(fpath))
                        scores.append(score)
                    except:
                        pass

                score_str = f"UTMOS={score:.3f}" if score else ""
                print(f"    [{i+1}] {text[:45]:<45} {duration:.2f}s NF={nf:.1f}dB {score_str}")

            mean_nf = np.mean(noise_floors)
            summary = {
                "model": model_name,
                "model_id": model_id,
                "config": config_name,
                "noise_scale": config["noise_scale"],
                "mean_noise_floor_db": round(float(mean_nf), 1),
            }
            if scores:
                summary["mean_utmos"] = round(float(np.mean(scores)), 3)
                summary["std_utmos"] = round(float(np.std(scores)), 3)
                print(f"\n  => Mean UTMOS: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            print(f"  => Mean Noise Floor: {mean_nf:.1f} dB")
            results.append(summary)

        del model, tokenizer
        torch.cuda.empty_cache()

    # Final comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Config':<12} {'UTMOS':>8} {'Noise Floor':>12}")
    print(f"{'-'*15} {'-'*12} {'-'*8} {'-'*12}")
    for r in results:
        utmos_str = f"{r['mean_utmos']:.3f}" if 'mean_utmos' in r else "N/A"
        print(f"{r['model']:<15} {r['config']:<12} {utmos_str:>8} {r['mean_noise_floor_db']:>10.1f} dB")

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    total = len(TEXTS) * len(MODELS) * len(NOISE_CONFIGS)
    print(f"\n{total} audio files saved to {output_dir}/")
    print(f"Results saved to {output_dir}/comparison_results.json")

if __name__ == "__main__":
    main()
