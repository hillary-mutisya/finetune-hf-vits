#!/usr/bin/env python3
"""
Full Dataset Denoising Pipeline using DeepFilterNet
====================================================
Much faster alternative to resemble-enhance (~20x faster than real-time)

Steps:
1. Export audio from HuggingFace dataset to WAV files
2. Run DeepFilterNet on all files (GPU accelerated, batch processing)
3. Resample to target sample rate (16kHz for MMS models)
4. Create new HuggingFace dataset with denoised audio
"""

import os
import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

# Configuration
DATASET_NAME = "mutisya/tts-speakers-7lang-15k-25-49-v2"
CONFIG_NAME = "swh_Latn"
SPLIT_NAME = "uganda_speaker"
TARGET_SAMPLE_RATE = 16000  # MMS model sample rate
WORK_DIR = Path("denoise_deepfilter")

# Directories
RAW_AUDIO_DIR = WORK_DIR / "raw_audio"
ENHANCED_AUDIO_DIR = WORK_DIR / "enhanced_audio"
RESAMPLED_AUDIO_DIR = WORK_DIR / "resampled_audio"
METADATA_FILE = WORK_DIR / "metadata.json"


def export_dataset_to_wav():
    """Step 1: Export dataset audio to WAV files"""
    from datasets import load_dataset
    import soundfile as sf
    from tqdm import tqdm
    
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {DATASET_NAME}/{CONFIG_NAME}/{SPLIT_NAME}")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT_NAME)
    
    print(f"Total samples: {len(ds)}")
    
    metadata = []
    for idx, sample in enumerate(tqdm(ds, desc="Exporting")):
        audio = sample["audio"]
        text = sample["text"]
        
        filename = f"sample_{idx:05d}.wav"
        filepath = RAW_AUDIO_DIR / filename
        
        # Save audio
        sf.write(filepath, audio["array"], audio["sampling_rate"])
        
        metadata.append({
            "idx": idx,
            "filename": filename,
            "text": text,
            "original_sr": audio["sampling_rate"]
        })
    
    # Save metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Exported {len(metadata)} files to {RAW_AUDIO_DIR}")
    return metadata


def run_deepfilter():
    """Step 2: Run DeepFilterNet on all audio files"""
    ENHANCED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get list of input files
    input_files = sorted(RAW_AUDIO_DIR.glob("*.wav"))
    print(f"Processing {len(input_files)} files with DeepFilterNet...")
    
    # DeepFilterNet can process multiple files at once
    # Use batch processing for efficiency
    start_time = time.time()
    
    cmd = [
        "deepFilter",
        str(RAW_AUDIO_DIR),  # Input directory
        "-o", str(ENHANCED_AUDIO_DIR),  # Output directory
        "--pf",  # Enable post-filter for better quality
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"DeepFilterNet failed with return code {result.returncode}")
        return 0
    
    # Count enhanced files
    enhanced_files = list(ENHANCED_AUDIO_DIR.glob("*.wav"))
    print(f"Enhanced {len(enhanced_files)} files in {elapsed:.1f}s ({elapsed/len(enhanced_files):.3f}s per file)")
    
    return len(enhanced_files)


def resample_audio():
    """Step 3: Resample enhanced audio to target sample rate"""
    import torchaudio
    from tqdm import tqdm
    
    RESAMPLED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    enhanced_files = sorted(ENHANCED_AUDIO_DIR.glob("*.wav"))
    print(f"Resampling {len(enhanced_files)} files to {TARGET_SAMPLE_RATE}Hz...")
    
    for filepath in tqdm(enhanced_files, desc="Resampling"):
        waveform, sr = torchaudio.load(filepath)
        
        if sr != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
        
        output_path = RESAMPLED_AUDIO_DIR / filepath.name
        torchaudio.save(output_path, waveform, TARGET_SAMPLE_RATE)
    
    print(f"Resampled {len(enhanced_files)} files to {RESAMPLED_AUDIO_DIR}")
    return len(enhanced_files)


def create_hf_dataset():
    """Step 4: Create HuggingFace dataset from denoised audio"""
    from datasets import Dataset, Audio
    import soundfile as sf
    
    # Load metadata
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    
    print(f"Creating HuggingFace dataset from {len(metadata)} samples...")
    
    # Build dataset records
    records = []
    for item in metadata:
        audio_path = RESAMPLED_AUDIO_DIR / item["filename"]
        if audio_path.exists():
            records.append({
                "audio": str(audio_path),
                "text": item["text"],
                "original_idx": item["idx"]
            })
    
    print(f"Found {len(records)} valid audio files")
    
    # Create dataset
    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    
    # Save dataset
    output_path = WORK_DIR / "dataset"
    ds.save_to_disk(str(output_path))
    print(f"Saved dataset to {output_path}")
    
    return ds


def main():
    print("=" * 60)
    print("FULL DATASET DENOISING PIPELINE (DeepFilterNet)")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}/{CONFIG_NAME}/{SPLIT_NAME}")
    print(f"Target sample rate: {TARGET_SAMPLE_RATE}Hz")
    print(f"Working directory: {WORK_DIR}")
    print()
    
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Export dataset
    print("=" * 60)
    print("Step 1: Exporting dataset to WAV files")
    print("=" * 60)
    if not METADATA_FILE.exists():
        metadata = export_dataset_to_wav()
    else:
        print(f"Using existing exports from {RAW_AUDIO_DIR}")
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        print(f"Found {len(metadata)} samples in metadata")
    print()
    
    # Step 2: Run DeepFilterNet
    print("=" * 60)
    print("Step 2: Running DeepFilterNet")
    print("=" * 60)
    num_enhanced = run_deepfilter()
    if num_enhanced == 0:
        print("ERROR: DeepFilterNet failed!")
        return
    print()
    
    # Step 3: Resample
    print("=" * 60)
    print("Step 3: Resampling to target sample rate")
    print("=" * 60)
    resample_audio()
    print()
    
    # Step 4: Create HF dataset
    print("=" * 60)
    print("Step 4: Creating HuggingFace dataset")
    print("=" * 60)
    ds = create_hf_dataset()
    print()
    
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Denoised dataset saved to: {WORK_DIR / 'dataset'}")
    print(f"Total samples: {len(ds)}")
    print()
    print("To load the dataset:")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{WORK_DIR / 'dataset'}')")


if __name__ == "__main__":
    main()
