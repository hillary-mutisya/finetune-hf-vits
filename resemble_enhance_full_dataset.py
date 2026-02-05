#!/usr/bin/env python3
"""
Full Dataset Enhancement with Resemble Enhance API
===================================================
Processes all audio files and creates a clean HuggingFace dataset.
Uses the exact same API as the Gradio demo with lambd=0.9 for noisy audio.
"""

import torch
import torchaudio
import json
import time
from pathlib import Path
from tqdm import tqdm

# Import the same functions as the Gradio demo
from resemble_enhance.enhancer.inference import denoise, enhance

# Configuration
WORK_DIR = Path("resemble_enhanced_dataset")
RAW_AUDIO_DIR = Path("denoise_deepfilter/raw_audio")
METADATA_FILE = Path("denoise_deepfilter/metadata.json")
TARGET_SAMPLE_RATE = 16000  # MMS model sample rate

# Resemble parameters (same as Gradio demo with denoise enabled)
NFE = 64
SOLVER = "midpoint"
TAU = 0.5
LAMBD = 0.9  # 0.9 for heavy noise (denoise mode)

device = "cuda" if torch.cuda.is_available() else "cpu"


def process_single_file(input_path, output_dir):
    """Process a single audio file"""
    # Load audio
    dwav, sr = torchaudio.load(input_path)
    dwav = dwav.mean(dim=0)  # Convert to mono
    
    # Run enhance (includes denoising with lambd=0.9)
    wav_enhanced, new_sr = enhance(dwav, sr, device, nfe=NFE, solver=SOLVER, lambd=LAMBD, tau=TAU)
    
    # Resample to target sample rate if needed
    if new_sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(new_sr, TARGET_SAMPLE_RATE)
        wav_enhanced = resampler(wav_enhanced.unsqueeze(0)).squeeze(0)
    
    # Save enhanced audio
    basename = Path(input_path).name
    output_path = output_dir / basename
    torchaudio.save(str(output_path), wav_enhanced.unsqueeze(0).cpu(), TARGET_SAMPLE_RATE)
    
    return output_path


def main():
    print("=" * 60)
    print("RESEMBLE ENHANCE - FULL DATASET PROCESSING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Parameters: NFE={NFE}, Solver={SOLVER}, Tau={TAU}, Lambda={LAMBD}")
    print(f"Target sample rate: {TARGET_SAMPLE_RATE}Hz")
    print(f"Input directory: {RAW_AUDIO_DIR}")
    print(f"Output directory: {WORK_DIR}")
    print()
    
    # Create output directory
    enhanced_dir = WORK_DIR / "enhanced_audio"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of input files
    input_files = sorted(RAW_AUDIO_DIR.glob("*.wav"))
    print(f"Total files to process: {len(input_files)}")
    print()
    
    # Check for already processed files (resume support)
    existing_files = set(f.name for f in enhanced_dir.glob("*.wav"))
    files_to_process = [f for f in input_files if f.name not in existing_files]
    
    if existing_files:
        print(f"Already processed: {len(existing_files)} files")
        print(f"Remaining to process: {len(files_to_process)} files")
    print()
    
    if not files_to_process:
        print("All files already processed!")
    else:
        # Process files
        start_time = time.time()
        failed_files = []
        
        for input_path in tqdm(files_to_process, desc="Enhancing"):
            try:
                process_single_file(input_path, enhanced_dir)
            except Exception as e:
                print(f"\nError processing {input_path}: {e}")
                failed_files.append(str(input_path))
        
        elapsed = time.time() - start_time
        print()
        print(f"Processing complete!")
        print(f"Time elapsed: {elapsed:.1f}s ({elapsed/len(files_to_process):.2f}s per file)")
        
        if failed_files:
            print(f"Failed files: {len(failed_files)}")
            with open(WORK_DIR / "failed_files.txt", "w") as f:
                f.write("\n".join(failed_files))
    
    # Copy metadata
    if METADATA_FILE.exists():
        import shutil
        shutil.copy(METADATA_FILE, WORK_DIR / "metadata.json")
        print(f"Copied metadata to {WORK_DIR / 'metadata.json'}")
    
    # Create HuggingFace dataset
    print()
    print("=" * 60)
    print("Creating HuggingFace Dataset")
    print("=" * 60)
    
    try:
        from datasets import Dataset, Audio
        
        # Load metadata
        with open(WORK_DIR / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Build dataset records
        records = []
        for item in metadata:
            audio_path = enhanced_dir / item["filename"]
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
        dataset_path = WORK_DIR / "dataset"
        ds.save_to_disk(str(dataset_path))
        print(f"Saved dataset to {dataset_path}")
        print(f"Total samples: {len(ds)}")
        
    except ImportError:
        print("datasets library not available in this venv")
        print("Run the following in the main venv to create the dataset:")
        print(f"  python -c \"from datasets import Dataset, Audio; ...\"")
    
    print()
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
