#!/usr/bin/env python3
"""
Resemble Enhance - Process first N samples only
"""

import torch
import torchaudio
import json
import time
from pathlib import Path
from tqdm import tqdm

from resemble_enhance.enhancer.inference import enhance

# Configuration
WORK_DIR = Path("resemble_enhanced_dataset")
RAW_AUDIO_DIR = Path("denoise_deepfilter/raw_audio")
METADATA_FILE = Path("denoise_deepfilter/metadata.json")
TARGET_SAMPLE_RATE = 16000
MAX_SAMPLES = 200  # Only process first 200

# Resemble parameters
NFE = 64
SOLVER = "midpoint"
TAU = 0.5
LAMBD = 0.9

device = "cuda" if torch.cuda.is_available() else "cpu"


def process_single_file(input_path, output_dir):
    dwav, sr = torchaudio.load(input_path)
    dwav = dwav.mean(dim=0)
    
    wav_enhanced, new_sr = enhance(dwav, sr, device, nfe=NFE, solver=SOLVER, lambd=LAMBD, tau=TAU)
    
    if new_sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(new_sr, TARGET_SAMPLE_RATE)
        wav_enhanced = resampler(wav_enhanced.unsqueeze(0)).squeeze(0)
    
    basename = Path(input_path).name
    output_path = output_dir / basename
    torchaudio.save(str(output_path), wav_enhanced.unsqueeze(0).cpu(), TARGET_SAMPLE_RATE)
    return output_path


def main():
    print("=" * 60)
    print(f"RESEMBLE ENHANCE - FIRST {MAX_SAMPLES} SAMPLES")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Parameters: NFE={NFE}, Solver={SOLVER}, Tau={TAU}, Lambda={LAMBD}")
    print()
    
    enhanced_dir = WORK_DIR / "enhanced_audio"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    
    # Get first N input files
    input_files = sorted(RAW_AUDIO_DIR.glob("*.wav"))[:MAX_SAMPLES]
    print(f"Target files: {len(input_files)}")
    
    # Check already processed
    existing_files = set(f.name for f in enhanced_dir.glob("*.wav"))
    files_to_process = [f for f in input_files if f.name not in existing_files]
    
    print(f"Already processed: {len(existing_files)}")
    print(f"Remaining: {len(files_to_process)}")
    print()
    
    if not files_to_process:
        print("All 200 files already processed!")
    else:
        start_time = time.time()
        for input_path in tqdm(files_to_process, desc="Enhancing"):
            try:
                process_single_file(input_path, enhanced_dir)
            except Exception as e:
                print(f"\nError processing {input_path}: {e}")
        
        elapsed = time.time() - start_time
        print(f"\nDone! Time: {elapsed:.1f}s ({elapsed/len(files_to_process):.2f}s per file)")
    
    # Copy metadata (only first 200 entries)
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            full_metadata = json.load(f)
        metadata_200 = full_metadata[:MAX_SAMPLES]
        with open(WORK_DIR / "metadata.json", "w") as f:
            json.dump(metadata_200, f, indent=2)
        print(f"Saved metadata for {len(metadata_200)} samples")
    
    print(f"\nEnhanced files: {len(list(enhanced_dir.glob('*.wav')))}")


if __name__ == "__main__":
    main()
