"""
Filter outliers from specific splits in tts-speakers-7lang-15k-25-49-v2 
and push updated dataset to HuggingFace.

Splits to filter:
- luo_Latn/richardm: 23 outliers
- luo_Latn/sharonm: 175 outliers
- swh_Latn/radio_speaker: 318 outliers
"""

import numpy as np
import torch
import json
import os
from datasets import load_dataset, Dataset
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from tqdm import tqdm

# Configuration
DATASET_NAME = "mutisya/tts-speakers-7lang-15k-25-49-v2"
SIMILARITY_THRESHOLD = 0.75

SPLITS_TO_FILTER = [
    ("luo_Latn", "richardm"),
    ("luo_Latn", "sharonm"),
    ("swh_Latn", "radio_speaker"),
]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load speaker verification model
print("Loading WavLM speaker verification model...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-sv").to(device)
model.eval()

def extract_embedding(audio_array, sampling_rate):
    """Extract speaker embedding from audio."""
    if sampling_rate != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
    
    inputs = feature_extractor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embeddings = model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    
    return embeddings.cpu().numpy()[0]

def analyze_and_filter_split(config, split):
    """Analyze a split and return indices to keep."""
    print(f"\n{'='*60}")
    print(f"Processing {config}/{split}")
    print("="*60)
    
    # Load dataset
    print(f"Loading {config}/{split}...")
    dataset = load_dataset(DATASET_NAME, config, split=split)
    print(f"Loaded {len(dataset)} samples")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = []
    valid_indices = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Extracting")):
        try:
            audio = sample['audio']
            embedding = extract_embedding(audio['array'], audio['sampling_rate'])
            embeddings.append(embedding)
            valid_indices.append(i)
        except Exception as e:
            print(f"Warning: Failed to process sample {i}: {e}")
            continue
    
    embeddings = np.array(embeddings)
    
    # Compute centroid
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    
    # Compute similarities
    similarities = np.dot(embeddings, centroid)
    
    # Find outliers
    outlier_mask = similarities < SIMILARITY_THRESHOLD
    outlier_indices = [valid_indices[i] for i in np.where(outlier_mask)[0]]
    keep_indices = [valid_indices[i] for i in np.where(~outlier_mask)[0]]
    
    print(f"\nResults:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Valid samples: {len(valid_indices)}")
    print(f"  Outliers (< {SIMILARITY_THRESHOLD}): {len(outlier_indices)}")
    print(f"  Samples to keep: {len(keep_indices)}")
    print(f"  Mean similarity: {similarities.mean():.4f}")
    print(f"  Min similarity: {similarities.min():.4f}")
    
    # Filter dataset
    filtered_dataset = dataset.select(keep_indices)
    
    return filtered_dataset, len(outlier_indices)

def main():
    results = []
    
    for config, split in SPLITS_TO_FILTER:
        filtered_ds, outlier_count = analyze_and_filter_split(config, split)
        
        # Push filtered split to hub
        print(f"\nPushing filtered {config}/{split} to HuggingFace...")
        filtered_ds.push_to_hub(
            DATASET_NAME,
            config_name=config,
            split=split,
            private=True
        )
        
        results.append({
            "config": config,
            "split": split,
            "outliers_removed": outlier_count,
            "samples_remaining": len(filtered_ds)
        })
        
        print(f"✓ Updated {config}/{split}: removed {outlier_count} outliers, {len(filtered_ds)} samples remaining")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['config']}/{r['split']}: -{r['outliers_removed']} outliers → {r['samples_remaining']} samples")
    
    print(f"\n✓ All updates pushed to {DATASET_NAME}")

if __name__ == "__main__":
    main()
