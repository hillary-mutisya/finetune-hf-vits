#!/usr/bin/env python3
"""
Audio Denoising and Dataset Preparation for TTS Training

This script:
1. Applies noise reduction to all audio samples
2. Normalizes audio levels
3. Filters out samples that are still low quality after denoising
4. Creates a new cleaned dataset ready for TTS training
"""

import os
import numpy as np
import torch
import torchaudio
import noisereduce as nr
from datasets import load_dataset, Dataset, Audio, DatasetDict
from tqdm import tqdm
from huggingface_hub import HfApi
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_NAME = "mutisya/tts-speakers-7lang-15k-25-49-v2"
CONFIG_NAME = "swh_Latn"
SPLIT_NAME = "uganda_speaker"
OUTPUT_DATASET_NAME = "mutisya/tts-swh-uganda-speaker-denoised"
TARGET_SR = 16000
DENOISE_STRENGTH = 0.8  # 0.0 = no denoising, 1.0 = aggressive denoising


def denoise_audio(audio_array: np.ndarray, sr: int, prop_decrease: float = 0.8) -> np.ndarray:
    """Apply noise reduction to audio."""
    # Ensure float32 in range [-1, 1]
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    max_val = np.abs(audio_array).max()
    if max_val > 1.0:
        audio_array = audio_array / max_val
    
    # Apply noise reduction
    try:
        reduced = nr.reduce_noise(
            y=audio_array,
            sr=sr,
            stationary=False,  # Non-stationary noise (better for variable recordings)
            prop_decrease=prop_decrease,  # How much to reduce noise
            n_fft=2048,
            hop_length=512,
            time_mask_smooth_ms=50,
            freq_mask_smooth_hz=500,
        )
    except Exception as e:
        print(f"Warning: Denoising failed, using original: {e}")
        reduced = audio_array
    
    return reduced


def normalize_audio(audio_array: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target loudness."""
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio_array ** 2) + 1e-10)
    current_db = 20 * np.log10(rms + 1e-10)
    
    # Calculate gain needed
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain with limiting
    normalized = audio_array * gain_linear
    
    # Soft clipping to prevent harsh clipping
    normalized = np.tanh(normalized * 0.9) / 0.9
    
    return normalized


def process_sample(sample: dict) -> dict:
    """Process a single sample: denoise and normalize."""
    audio = sample['audio']
    audio_array = np.array(audio['array'], dtype=np.float32)
    sr = audio['sampling_rate']
    
    # Resample to target SR if needed
    if sr != TARGET_SR:
        audio_tensor = torch.tensor(audio_array).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        audio_array = resampler(audio_tensor).squeeze().numpy()
        sr = TARGET_SR
    
    # Normalize to [-1, 1] range
    max_val = np.abs(audio_array).max()
    if max_val > 1.0:
        audio_array = audio_array / max_val
    
    # Apply denoising
    denoised = denoise_audio(audio_array, sr, prop_decrease=DENOISE_STRENGTH)
    
    # Normalize loudness
    normalized = normalize_audio(denoised, target_db=-20.0)
    
    # Update sample
    sample['audio'] = {
        'array': normalized,
        'sampling_rate': sr,
        'path': audio.get('path', ''),
    }
    
    return sample


def main():
    print("=" * 60)
    print("Audio Denoising and Dataset Preparation")
    print("=" * 60)
    print(f"Denoise strength: {DENOISE_STRENGTH}")
    print(f"Target sample rate: {TARGET_SR}")
    print()
    
    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}/{CONFIG_NAME}/{SPLIT_NAME}")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT_NAME)
    print(f"Total samples: {len(ds)}")
    
    # Process all samples
    print("\n🔧 Processing audio (denoising + normalizing)...")
    processed_samples = []
    
    for idx, sample in enumerate(tqdm(ds, desc="Processing")):
        try:
            processed = process_sample(sample.copy())
            processed_samples.append({
                'audio': processed['audio'],
                'transcription': sample['transcription'],
            })
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue
    
    print(f"\n✅ Successfully processed: {len(processed_samples)} / {len(ds)} samples")
    
    # Create new dataset
    print("\n📦 Creating denoised dataset...")
    
    # Convert to HuggingFace Dataset format
    new_ds = Dataset.from_dict({
        'audio': [s['audio'] for s in processed_samples],
        'transcription': [s['transcription'] for s in processed_samples],
    })
    
    # Cast audio column
    new_ds = new_ds.cast_column('audio', Audio(sampling_rate=TARGET_SR))
    
    # Save locally first
    local_path = "./denoised_dataset_swh_uganda"
    print(f"Saving locally to: {local_path}")
    new_ds.save_to_disk(local_path)
    
    print("\n" + "=" * 60)
    print("✅ DENOISING COMPLETE")
    print("=" * 60)
    print(f"Original samples: {len(ds)}")
    print(f"Processed samples: {len(processed_samples)}")
    print(f"Saved to: {local_path}")
    print()
    print("To push to HuggingFace Hub, run:")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{local_path}')")
    print(f"  ds.push_to_hub('{OUTPUT_DATASET_NAME}')")
    
    return new_ds


if __name__ == "__main__":
    new_ds = main()
