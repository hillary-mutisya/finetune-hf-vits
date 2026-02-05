"""
Save outlier audio samples as WAV files for manual inspection.
"""

import json
import os
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

# Configuration
DATASET_NAME = "mutisya/tts-multi-22lang-15k-26-04-v1"
DATASET_CONFIG = "kik_Latn"
OUTPUT_DIR = "./outlier_samples"
ANALYSIS_FILE = "./speaker_analysis_results/sample_similarities.json"

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load analysis results
    print("Loading analysis results...")
    with open(ANALYSIS_FILE, 'r') as f:
        sample_data = json.load(f)
    
    # Get outlier indices (samples below threshold)
    outliers = [(s['index'], s['similarity'], s['duration'], s['transcription']) 
                for s in sample_data if s['is_outlier']]
    
    # Sort by similarity (lowest first)
    outliers.sort(key=lambda x: x[1])
    
    print(f"Found {len(outliers)} outlier samples to save")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    
    # Save each outlier as WAV
    print(f"Saving outliers to {OUTPUT_DIR}/...")
    for idx, similarity, duration, text_preview in tqdm(outliers, desc="Saving"):
        sample = dataset[idx]
        audio = sample['audio']
        waveform = audio['array']
        sample_rate = audio['sampling_rate']
        
        # Create filename with similarity score for easy sorting
        # Format: sim_0.4430_idx_10241.wav
        filename = f"sim_{similarity:.4f}_idx_{idx:05d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save as WAV
        sf.write(filepath, waveform, sample_rate)
    
    print(f"\nSaved {len(outliers)} outlier samples to {OUTPUT_DIR}/")
    print("\nFiles are named with similarity scores - lower scores = more suspicious")
    print("Listen to them and identify which ones are truly from different speakers.")
    
    # Also save a summary file
    summary_file = os.path.join(OUTPUT_DIR, "outliers_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Outlier Samples Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total outliers: {len(outliers)}\n")
        f.write(f"Threshold used: 0.75\n\n")
        f.write("Files (sorted by similarity, lowest first):\n")
        f.write("-" * 60 + "\n")
        for idx, similarity, duration, text_preview in outliers:
            filename = f"sim_{similarity:.4f}_idx_{idx:05d}.wav"
            f.write(f"\n{filename}\n")
            f.write(f"  Duration: {duration:.1f}s\n")
            f.write(f"  Text: {text_preview}\n")
    
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
