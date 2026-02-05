"""
Add uganda_speaker split to swh_Latn subset of mutisya/tts-speakers-7lang-15k-25-49-v2
Source: mutisya/tts-swh_latn-15k-25-49-v1 (train + test merged)
"""

from datasets import load_dataset, concatenate_datasets, Audio
from huggingface_hub import HfApi
import os

# Configuration
TARGET_DATASET = "mutisya/tts-speakers-7lang-15k-25-49-v2"
SOURCE_DATASET = "mutisya/tts-swh_latn-15k-25-49-v1"
TARGET_CONFIG = "swh_Latn"
NEW_SPLIT_NAME = "uganda_speaker"

def main():
    print("="*60)
    print(f"Adding {NEW_SPLIT_NAME} split to {TARGET_DATASET}/{TARGET_CONFIG}")
    print("="*60)
    
    # Load source dataset (train + test)
    print(f"\nLoading source dataset: {SOURCE_DATASET}")
    source_train = load_dataset(SOURCE_DATASET, split="train")
    source_test = load_dataset(SOURCE_DATASET, split="test")
    
    print(f"  Train: {len(source_train)} samples")
    print(f"  Test: {len(source_test)} samples")
    
    # Merge train and test
    source_merged = concatenate_datasets([source_train, source_test])
    print(f"  Merged: {len(source_merged)} samples")
    
    # Check columns
    print(f"\nSource columns: {source_merged.column_names}")
    
    # Rename 'sentence' to 'transcription' if needed
    if 'sentence' in source_merged.column_names and 'transcription' not in source_merged.column_names:
        print("Renaming 'sentence' -> 'transcription'")
        source_merged = source_merged.rename_column('sentence', 'transcription')
    
    # Keep only audio and transcription columns
    columns_to_keep = ['audio', 'transcription']
    columns_to_remove = [c for c in source_merged.column_names if c not in columns_to_keep]
    if columns_to_remove:
        print(f"Removing extra columns: {columns_to_remove}")
        source_merged = source_merged.remove_columns(columns_to_remove)
    
    print(f"Final columns: {source_merged.column_names}")
    
    # Resample audio to 16kHz (target dataset uses 16kHz)
    print("\nResampling audio to 16kHz...")
    source_merged = source_merged.cast_column("audio", Audio(sampling_rate=16000))
    
    # Verify a sample
    sample = source_merged[0]
    print(f"Sample audio rate: {sample['audio']['sampling_rate']}")
    print(f"Sample transcription: {sample['transcription'][:50]}...")
    
    # Push to hub as new split
    print(f"\nPushing to {TARGET_DATASET} config={TARGET_CONFIG} split={NEW_SPLIT_NAME}...")
    
    # Use push_to_hub with split parameter
    source_merged.push_to_hub(
        TARGET_DATASET,
        config_name=TARGET_CONFIG,
        split=NEW_SPLIT_NAME,
        private=True
    )
    
    print(f"\n✓ Successfully added {NEW_SPLIT_NAME} split with {len(source_merged)} samples")
    print(f"  Dataset: {TARGET_DATASET}")
    print(f"  Config: {TARGET_CONFIG}")
    print(f"  New split: {NEW_SPLIT_NAME}")

if __name__ == "__main__":
    main()
