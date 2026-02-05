"""
Filter dataset to remove outlier samples and push to HuggingFace.
Also analyzes and filters the test split.
"""

import json
import os
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from tqdm import tqdm
from huggingface_hub import login
import soundfile as sf

# Configuration
DATASET_NAME = "mutisya/tts-multi-22lang-15k-26-04-v1"
DATASET_CONFIG = "kik_Latn"
SIMILARITY_THRESHOLD = 0.75
OUTPUT_DATASET_NAME = "mutisya/tts-kik-estherk-filtered-26-05-v1"
RESULTS_DIR = "./speaker_analysis_results"

# Load the speaker verification model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-sv").to(device)
model.eval()

def extract_embedding(audio_array, sampling_rate):
    """Extract speaker embedding from audio."""
    # Resample to 16kHz if needed
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

def analyze_split(dataset, split_name, centroid=None):
    """Analyze a dataset split and return embeddings and outlier indices."""
    print(f"\nAnalyzing {split_name} split ({len(dataset)} samples)...")
    
    embeddings = []
    valid_indices = []
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
        try:
            audio = sample['audio']
            embedding = extract_embedding(audio['array'], audio['sampling_rate'])
            embeddings.append(embedding)
            valid_indices.append(i)
        except Exception as e:
            print(f"Warning: Failed to process sample {i}: {e}")
            continue
    
    embeddings = np.array(embeddings)
    
    # If no centroid provided, compute from this split
    if centroid is None:
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
    
    # Compute similarities
    similarities = np.dot(embeddings, centroid)
    
    # Find outliers
    outlier_mask = similarities < SIMILARITY_THRESHOLD
    outlier_indices = [valid_indices[i] for i in np.where(outlier_mask)[0]]
    keep_indices = [valid_indices[i] for i in np.where(~outlier_mask)[0]]
    
    print(f"\n{split_name} split results:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Mean similarity: {similarities.mean():.4f}")
    print(f"  Outliers (< {SIMILARITY_THRESHOLD}): {len(outlier_indices)}")
    print(f"  Samples to keep: {len(keep_indices)}")
    
    return embeddings, centroid, outlier_indices, keep_indices, similarities, valid_indices

def save_outlier_wavs(dataset, outlier_indices, similarities, valid_indices, split_name):
    """Save outlier audio files for inspection."""
    output_dir = f"./outlier_samples_{split_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create index to similarity mapping
    idx_to_sim = {valid_indices[i]: similarities[i] for i in range(len(valid_indices))}
    
    print(f"\nSaving {split_name} outliers to {output_dir}/...")
    for idx in tqdm(outlier_indices, desc="Saving"):
        sample = dataset[idx]
        audio = sample['audio']
        sim = idx_to_sim.get(idx, 0)
        
        filename = f"sim_{sim:.4f}_idx_{idx:05d}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio['array'], audio['sampling_rate'])
    
    print(f"Saved {len(outlier_indices)} outliers to {output_dir}/")

def main():
    # Load existing train split analysis
    print("Loading existing train split analysis...")
    with open(os.path.join(RESULTS_DIR, "keep_indices.json"), 'r') as f:
        train_keep_indices = json.load(f)
    
    with open(os.path.join(RESULTS_DIR, "sample_similarities.json"), 'r') as f:
        train_sample_data = json.load(f)
    
    train_outlier_indices = [s['index'] for s in train_sample_data if s['is_outlier']]
    
    print(f"Train split: {len(train_keep_indices)} to keep, {len(train_outlier_indices)} outliers")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    print(f"Train split: {len(dataset['train'])} samples")
    print(f"Test split: {len(dataset['test'])} samples")
    
    # Load train embeddings to get centroid
    print("\nLoading train embeddings for centroid...")
    embeddings_file = os.path.join(RESULTS_DIR, "embeddings.npy")
    if os.path.exists(embeddings_file):
        train_embeddings = np.load(embeddings_file)
        centroid = train_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
    else:
        # Re-compute centroid from train split (using only kept samples)
        print("Re-computing centroid from train samples...")
        train_embeddings = []
        for i in tqdm(train_keep_indices[:500], desc="Computing centroid"):  # Use subset for speed
            sample = dataset['train'][i]
            audio = sample['audio']
            emb = extract_embedding(audio['array'], audio['sampling_rate'])
            train_embeddings.append(emb)
        train_embeddings = np.array(train_embeddings)
        centroid = train_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
    
    # Analyze test split
    test_embeddings, _, test_outlier_indices, test_keep_indices, test_similarities, test_valid_indices = \
        analyze_split(dataset['test'], 'test', centroid=centroid)
    
    # Save test outliers for inspection
    if len(test_outlier_indices) > 0:
        save_outlier_wavs(dataset['test'], test_outlier_indices, test_similarities, test_valid_indices, 'test')
    
    # Filter both splits
    print("\n" + "="*60)
    print("FILTERING DATASET")
    print("="*60)
    
    # Filter train split
    print(f"\nFiltering train split: {len(dataset['train'])} -> {len(train_keep_indices)} samples")
    filtered_train = dataset['train'].select(train_keep_indices)
    
    # Filter test split
    print(f"Filtering test split: {len(dataset['test'])} -> {len(test_keep_indices)} samples")
    filtered_test = dataset['test'].select(test_keep_indices)
    
    # Create new dataset
    filtered_dataset = DatasetDict({
        'train': filtered_train,
        'test': filtered_test
    })
    
    print(f"\nFiltered dataset:")
    print(f"  Train: {len(filtered_dataset['train'])} samples")
    print(f"  Test: {len(filtered_dataset['test'])} samples")
    
    # Push to HuggingFace
    print(f"\nPushing to HuggingFace as {OUTPUT_DATASET_NAME}...")
    filtered_dataset.push_to_hub(
        OUTPUT_DATASET_NAME,
        private=True
    )
    
    print(f"\n✓ Dataset pushed to {OUTPUT_DATASET_NAME}")
    
    # Save filtering summary
    summary = {
        "original_dataset": DATASET_NAME,
        "original_config": DATASET_CONFIG,
        "filtered_dataset": OUTPUT_DATASET_NAME,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "train": {
            "original_samples": len(dataset['train']),
            "filtered_samples": len(filtered_dataset['train']),
            "removed_samples": len(train_outlier_indices),
            "outlier_indices": train_outlier_indices
        },
        "test": {
            "original_samples": len(dataset['test']),
            "filtered_samples": len(filtered_dataset['test']),
            "removed_samples": len(test_outlier_indices),
            "outlier_indices": test_outlier_indices
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "filtering_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {RESULTS_DIR}/filtering_summary.json")

if __name__ == "__main__":
    main()
