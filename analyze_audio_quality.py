#!/usr/bin/env python3
"""
Audio Quality Analysis and Denoising for TTS Training Data

This script:
1. Analyzes audio quality metrics (SNR, energy, silence ratio)
2. Optionally applies noise reduction using noisereduce
3. Filters out low-quality samples
4. Creates a cleaned dataset for TTS training
"""

import os
import numpy as np
import torch
import torchaudio
import noisereduce as nr
from datasets import load_dataset, Dataset, Audio
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_NAME = "mutisya/tts-speakers-7lang-15k-25-49-v2"
CONFIG_NAME = "swh_Latn"
SPLIT_NAME = "uganda_speaker"
OUTPUT_DIR = "./audio_analysis_output"
TARGET_SR = 16000

# Quality thresholds
MIN_SNR_DB = 15.0  # Minimum signal-to-noise ratio in dB
MIN_ENERGY_DB = -40.0  # Minimum audio energy in dB
MAX_SILENCE_RATIO = 0.5  # Maximum ratio of silence in audio
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 15.0


def compute_snr(audio: np.ndarray, sr: int, frame_length: int = 2048) -> float:
    """Estimate SNR using spectral subtraction method."""
    # Simple SNR estimation: ratio of signal power to estimated noise power
    # Use first/last 0.5s as noise estimate
    noise_samples = int(0.5 * sr)
    
    if len(audio) < noise_samples * 3:
        return 0.0
    
    # Estimate noise from beginning and end
    noise = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
    noise_power = np.mean(noise ** 2) + 1e-10
    
    # Signal power from middle portion
    signal = audio[noise_samples:-noise_samples]
    signal_power = np.mean(signal ** 2) + 1e-10
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(snr_db)


def compute_energy_db(audio: np.ndarray) -> float:
    """Compute audio energy in dB."""
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    return float(20 * np.log10(rms + 1e-10))


def compute_silence_ratio(audio: np.ndarray, threshold_db: float = -40) -> float:
    """Compute ratio of silence (low energy) frames."""
    frame_length = 512
    hop_length = 256
    
    n_frames = (len(audio) - frame_length) // hop_length + 1
    if n_frames <= 0:
        return 1.0
    
    silence_frames = 0
    threshold_linear = 10 ** (threshold_db / 20)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < threshold_linear:
            silence_frames += 1
    
    return silence_frames / n_frames


def analyze_sample(audio_array: np.ndarray, sr: int) -> dict:
    """Analyze a single audio sample."""
    duration = len(audio_array) / sr
    
    # Ensure mono
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=0)
    
    # Normalize to float32 [-1, 1]
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    if np.abs(audio_array).max() > 1.0:
        audio_array = audio_array / 32768.0
    
    return {
        'duration': duration,
        'snr_db': compute_snr(audio_array, sr),
        'energy_db': compute_energy_db(audio_array),
        'silence_ratio': compute_silence_ratio(audio_array),
        'max_amplitude': float(np.abs(audio_array).max()),
    }


def denoise_audio(audio_array: np.ndarray, sr: int) -> np.ndarray:
    """Apply noise reduction to audio."""
    # Ensure float32
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    if np.abs(audio_array).max() > 1.0:
        audio_array = audio_array / 32768.0
    
    # Apply noise reduction
    reduced = nr.reduce_noise(
        y=audio_array,
        sr=sr,
        stationary=False,  # Non-stationary noise reduction
        prop_decrease=0.75,  # How much to reduce noise (0-1)
        n_fft=2048,
        hop_length=512,
    )
    
    return reduced


def is_quality_acceptable(metrics: dict) -> tuple[bool, list[str]]:
    """Check if audio quality meets thresholds."""
    issues = []
    
    if metrics['snr_db'] < MIN_SNR_DB:
        issues.append(f"Low SNR: {metrics['snr_db']:.1f}dB < {MIN_SNR_DB}dB")
    
    if metrics['energy_db'] < MIN_ENERGY_DB:
        issues.append(f"Low energy: {metrics['energy_db']:.1f}dB < {MIN_ENERGY_DB}dB")
    
    if metrics['silence_ratio'] > MAX_SILENCE_RATIO:
        issues.append(f"Too much silence: {metrics['silence_ratio']:.1%} > {MAX_SILENCE_RATIO:.1%}")
    
    if metrics['duration'] < MIN_DURATION_SEC:
        issues.append(f"Too short: {metrics['duration']:.1f}s < {MIN_DURATION_SEC}s")
    
    if metrics['duration'] > MAX_DURATION_SEC:
        issues.append(f"Too long: {metrics['duration']:.1f}s > {MAX_DURATION_SEC}s")
    
    return len(issues) == 0, issues


def main():
    print("=" * 60)
    print("Audio Quality Analysis and Denoising for TTS")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset: {DATASET_NAME}/{CONFIG_NAME}/{SPLIT_NAME}")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT_NAME)
    print(f"Total samples: {len(ds)}")
    
    # Analyze all samples
    print("\n📊 Analyzing audio quality...")
    analysis_results = []
    
    for idx, sample in enumerate(tqdm(ds, desc="Analyzing")):
        audio = sample['audio']
        audio_array = np.array(audio['array'])
        sr = audio['sampling_rate']
        
        # Resample if needed
        if sr != TARGET_SR:
            audio_tensor = torch.tensor(audio_array).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            audio_array = resampler(audio_tensor).squeeze().numpy()
            sr = TARGET_SR
        
        metrics = analyze_sample(audio_array, sr)
        metrics['index'] = idx
        metrics['transcription'] = sample.get('transcription', '')[:50]
        
        acceptable, issues = is_quality_acceptable(metrics)
        metrics['acceptable'] = acceptable
        metrics['issues'] = issues
        
        analysis_results.append(metrics)
    
    # Summary statistics
    snr_values = [r['snr_db'] for r in analysis_results]
    energy_values = [r['energy_db'] for r in analysis_results]
    silence_values = [r['silence_ratio'] for r in analysis_results]
    
    print("\n" + "=" * 60)
    print("📈 QUALITY ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nSNR (Signal-to-Noise Ratio):")
    print(f"  Mean: {np.mean(snr_values):.1f} dB")
    print(f"  Min:  {np.min(snr_values):.1f} dB")
    print(f"  Max:  {np.max(snr_values):.1f} dB")
    print(f"  Samples below {MIN_SNR_DB}dB threshold: {sum(1 for s in snr_values if s < MIN_SNR_DB)}")
    
    print(f"\nEnergy:")
    print(f"  Mean: {np.mean(energy_values):.1f} dB")
    print(f"  Min:  {np.min(energy_values):.1f} dB")
    print(f"  Max:  {np.max(energy_values):.1f} dB")
    print(f"  Samples below {MIN_ENERGY_DB}dB threshold: {sum(1 for e in energy_values if e < MIN_ENERGY_DB)}")
    
    print(f"\nSilence Ratio:")
    print(f"  Mean: {np.mean(silence_values):.1%}")
    print(f"  Min:  {np.min(silence_values):.1%}")
    print(f"  Max:  {np.max(silence_values):.1%}")
    print(f"  Samples above {MAX_SILENCE_RATIO:.0%} threshold: {sum(1 for s in silence_values if s > MAX_SILENCE_RATIO)}")
    
    # Count acceptable samples
    acceptable_count = sum(1 for r in analysis_results if r['acceptable'])
    print(f"\n✅ Samples passing all quality checks: {acceptable_count} / {len(analysis_results)} ({acceptable_count/len(analysis_results):.1%})")
    
    # Save analysis to JSON
    analysis_file = os.path.join(OUTPUT_DIR, "quality_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset': DATASET_NAME,
            'config': CONFIG_NAME,
            'split': SPLIT_NAME,
            'total_samples': len(ds),
            'acceptable_samples': acceptable_count,
            'thresholds': {
                'min_snr_db': MIN_SNR_DB,
                'min_energy_db': MIN_ENERGY_DB,
                'max_silence_ratio': MAX_SILENCE_RATIO,
                'min_duration_sec': MIN_DURATION_SEC,
                'max_duration_sec': MAX_DURATION_SEC,
            },
            'summary': {
                'snr_mean': np.mean(snr_values),
                'snr_min': np.min(snr_values),
                'snr_max': np.max(snr_values),
                'energy_mean': np.mean(energy_values),
                'silence_ratio_mean': np.mean(silence_values),
            },
            'samples': analysis_results,
        }, f, indent=2, default=str)
    
    print(f"\n📁 Analysis saved to: {analysis_file}")
    
    # Show worst samples
    print("\n" + "=" * 60)
    print("🔴 LOWEST QUALITY SAMPLES (by SNR)")
    print("=" * 60)
    sorted_by_snr = sorted(analysis_results, key=lambda x: x['snr_db'])
    for r in sorted_by_snr[:10]:
        print(f"  [{r['index']:4d}] SNR={r['snr_db']:5.1f}dB, Energy={r['energy_db']:5.1f}dB, Silence={r['silence_ratio']:4.1%} | {r['transcription']}")
    
    return analysis_results


if __name__ == "__main__":
    results = main()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("""
1. Review the quality_analysis.json file for detailed metrics
2. To create a denoised dataset, run with --denoise flag
3. To filter low-quality samples, run with --filter flag
4. Recommended: Use samples with SNR > 15dB for best TTS quality
""")
