#!/usr/bin/env python3
"""
Full Dataset Denoising with Resemble Enhance

This script:
1. Exports all audio from the HuggingFace dataset to WAV files
2. Runs Resemble Enhance on all files
3. Resamples back to 16kHz
4. Creates a new denoised dataset ready for TTS training
"""

import os
import subprocess
import shutil
from pathlib import Path
import numpy as np
import torch
import torchaudio
from datasets import load_dataset, Dataset, Audio
from tqdm import tqdm
from datetime import datetime
import json

# Configuration
DATASET_NAME = "mutisya/tts-speakers-7lang-15k-25-49-v2"
CONFIG_NAME = "swh_Latn"
SPLIT_NAME = "uganda_speaker"
TARGET_SR = 16000

# Directories
WORK_DIR = Path("./denoise_full_dataset")
RAW_DIR = WORK_DIR / "raw_audio"
ENHANCED_DIR = WORK_DIR / "enhanced_audio"
RESAMPLED_DIR = WORK_DIR / "resampled_16k"
OUTPUT_DATASET_DIR = WORK_DIR / "denoised_dataset"

# Resemble Enhance settings (matching Gradio defaults)
NFE = 64
SOLVER = "midpoint"
TAU = 0.5


def export_dataset_to_wav():
    """Export all audio samples to WAV files."""
    print("=" * 60)
    print("Step 1: Exporting dataset to WAV files")
    print("=" * 60)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT_NAME)
    print(f"Total samples: {len(ds)}")
    
    # Save metadata for later reconstruction
    metadata = []
    
    for idx, sample in enumerate(tqdm(ds, desc="Exporting")):
        audio = sample['audio']
        audio_array = np.array(audio['array'], dtype=np.float32)
        sr = audio['sampling_rate']
        
        # Normalize to prevent clipping
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val * 0.95
        
        # Save as WAV
        output_path = RAW_DIR / f"sample_{idx:05d}.wav"
        audio_tensor = torch.tensor(audio_array).unsqueeze(0)
        torchaudio.save(str(output_path), audio_tensor, sr)
        
        metadata.append({
            'idx': idx,
            'filename': f"sample_{idx:05d}.wav",
            'transcription': sample['transcription'],
            'original_sr': sr,
        })
    
    # Save metadata
    with open(WORK_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Exported {len(metadata)} files to {RAW_DIR}")
    return metadata


def run_resemble_enhance():
    """Run Resemble Enhance on all files."""
    print()
    print("=" * 60)
    print("Step 2: Running Resemble Enhance (this will take a while)")
    print("=" * 60)
    
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "resemble-enhance",
        str(RAW_DIR),
        str(ENHANCED_DIR),
        "--device", "cuda",
        "--nfe", str(NFE),
        "--solver", SOLVER,
        "--tau", str(TAU),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Warning: Resemble Enhance returned code {result.returncode}")
    
    # Count enhanced files
    enhanced_files = list(ENHANCED_DIR.glob("*.wav"))
    print(f"Enhanced {len(enhanced_files)} files")
    
    return len(enhanced_files)


def resample_to_16k():
    """Resample all enhanced files back to 16kHz."""
    print()
    print("=" * 60)
    print("Step 3: Resampling to 16kHz")
    print("=" * 60)
    
    RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    
    enhanced_files = sorted(ENHANCED_DIR.glob("*.wav"))
    print(f"Processing {len(enhanced_files)} files")
    
    for filepath in tqdm(enhanced_files, desc="Resampling"):
        audio, sr = torchaudio.load(str(filepath))
        
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            audio = resampler(audio)
        
        output_path = RESAMPLED_DIR / filepath.name
        torchaudio.save(str(output_path), audio, TARGET_SR)
    
    print(f"Resampled files saved to {RESAMPLED_DIR}")


def create_dataset():
    """Create a HuggingFace dataset from the denoised audio."""
    print()
    print("=" * 60)
    print("Step 4: Creating HuggingFace Dataset")
    print("=" * 60)
    
    # Load metadata
    with open(WORK_DIR / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Build dataset
    audio_files = []
    transcriptions = []
    
    for item in tqdm(metadata, desc="Loading audio"):
        audio_path = RESAMPLED_DIR / item['filename']
        
        if audio_path.exists():
            audio, sr = torchaudio.load(str(audio_path))
            audio_files.append({
                'array': audio.squeeze(0).numpy(),
                'sampling_rate': sr,
                'path': str(audio_path),
            })
            transcriptions.append(item['transcription'])
        else:
            print(f"Warning: Missing file {audio_path}")
    
    print(f"Loaded {len(audio_files)} audio files")
    
    # Create dataset
    ds = Dataset.from_dict({
        'audio': audio_files,
        'transcription': transcriptions,
    })
    
    ds = ds.cast_column('audio', Audio(sampling_rate=TARGET_SR))
    
    # Save to disk
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DATASET_DIR))
    
    print(f"Dataset saved to {OUTPUT_DATASET_DIR}")
    print(f"Total samples: {len(ds)}")
    
    return ds


def main():
    start_time = datetime.now()
    
    print("=" * 60)
    print("FULL DATASET DENOISING PIPELINE")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}/{CONFIG_NAME}/{SPLIT_NAME}")
    print(f"Target sample rate: {TARGET_SR}Hz")
    print(f"Resemble settings: NFE={NFE}, Solver={SOLVER}, Tau={TAU}")
    print(f"Working directory: {WORK_DIR}")
    print()
    
    # Create work directory
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Export to WAV
    metadata = export_dataset_to_wav()
    
    # Step 2: Run Resemble Enhance
    num_enhanced = run_resemble_enhance()
    
    # Step 3: Resample to 16kHz
    resample_to_16k()
    
    # Step 4: Create dataset
    ds = create_dataset()
    
    # Summary
    elapsed = datetime.now() - start_time
    print()
    print("=" * 60)
    print("✅ DENOISING COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed}")
    print(f"Original samples: {len(metadata)}")
    print(f"Denoised samples: {len(ds)}")
    print(f"Output dataset: {OUTPUT_DATASET_DIR}")
    print()
    print("To use this dataset for training, update your config:")
    print(f'  "dataset_name": "{OUTPUT_DATASET_DIR}"')
    print()
    print("Or push to HuggingFace Hub:")
    print("  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{OUTPUT_DATASET_DIR}')")
    print("  ds.push_to_hub('mutisya/tts-swh-uganda-speaker-denoised')")


if __name__ == "__main__":
    main()
