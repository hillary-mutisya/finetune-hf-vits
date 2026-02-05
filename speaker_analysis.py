#!/usr/bin/env python3
"""
Analyze speaker consistency in a TTS dataset using speaker embeddings.
This script identifies samples that may be from different speakers.
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
import os

# Configuration
DATASET_NAME = "mutisya/tts-multi-22lang-15k-26-04-v1"
DATASET_CONFIG = "kik_Latn"
OUTPUT_DIR = "./speaker_analysis_results"
BATCH_SIZE = 16
SAMPLE_RATE = 16000

# Thresholds
SIMILARITY_THRESHOLD = 0.75  # Samples below this similarity to centroid may be different speakers


def load_speaker_model():
    """Load WavLM speaker verification model."""
    print("Loading speaker embedding model (WavLM)...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-sv")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    return feature_extractor, model, device


def extract_speaker_embedding(audio_array, feature_extractor, model, device):
    """Extract speaker embedding from audio."""
    # Ensure audio is at correct sample rate and length
    if len(audio_array) < SAMPLE_RATE:
        # Pad short audio
        audio_array = np.pad(audio_array, (0, SAMPLE_RATE - len(audio_array)))
    
    # Process audio
    inputs = feature_extractor(
        audio_array, 
        sampling_rate=SAMPLE_RATE, 
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        embeddings = model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    
    return embeddings.cpu().numpy().squeeze()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {DATASET_NAME} ({DATASET_CONFIG})...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    print(f"Total samples: {len(ds)}")
    
    # Load speaker model
    feature_extractor, model, device = load_speaker_model()
    print(f"Model loaded on {device}")
    
    # Extract embeddings for all samples
    print("\nExtracting speaker embeddings...")
    embeddings = []
    indices = []
    failed_indices = []
    
    for i in tqdm(range(len(ds)), desc="Processing"):
        try:
            sample = ds[i]
            audio = sample["audio"]["array"]
            
            embedding = extract_speaker_embedding(audio, feature_extractor, model, device)
            embeddings.append(embedding)
            indices.append(i)
        except Exception as e:
            failed_indices.append(i)
            print(f"\nFailed on sample {i}: {e}")
    
    embeddings = np.array(embeddings)
    print(f"\nExtracted {len(embeddings)} embeddings")
    
    # Compute centroid (average speaker embedding)
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)  # Normalize
    
    # Compute similarity to centroid for each sample
    similarities = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    
    # Identify outliers (potential different speakers)
    outlier_mask = similarities < SIMILARITY_THRESHOLD
    outlier_indices = [indices[i] for i in np.where(outlier_mask)[0]]
    outlier_similarities = similarities[outlier_mask]
    
    print(f"\n=== Speaker Analysis Results ===")
    print(f"Total samples analyzed: {len(embeddings)}")
    print(f"Mean similarity to centroid: {similarities.mean():.4f}")
    print(f"Min similarity: {similarities.min():.4f}")
    print(f"Max similarity: {similarities.max():.4f}")
    print(f"Std deviation: {similarities.std():.4f}")
    print(f"\nSamples below threshold ({SIMILARITY_THRESHOLD}): {len(outlier_indices)}")
    
    # Cluster to detect if there are multiple speakers
    print("\n=== Clustering Analysis ===")
    for n_clusters in [2, 3]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        print(f"{n_clusters} clusters: sizes = {cluster_sizes}")
    
    # Save results
    results = {
        "total_samples": len(embeddings),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "mean_similarity": float(similarities.mean()),
        "std_similarity": float(similarities.std()),
        "min_similarity": float(similarities.min()),
        "max_similarity": float(similarities.max()),
        "num_outliers": len(outlier_indices),
        "outlier_indices": outlier_indices,
        "failed_indices": failed_indices,
    }
    
    results_path = os.path.join(OUTPUT_DIR, "speaker_analysis.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save per-sample similarities
    sample_data = []
    for i, (idx, sim) in enumerate(zip(indices, similarities)):
        sample = ds[idx]
        sample_data.append({
            "index": idx,
            "similarity": float(sim),
            "is_outlier": bool(sim < SIMILARITY_THRESHOLD),
            "transcription": sample["transcription"][:100],
            "duration": sample["duration"]
        })
    
    # Sort by similarity (lowest first = most likely different speaker)
    sample_data.sort(key=lambda x: x["similarity"])
    
    samples_path = os.path.join(OUTPUT_DIR, "sample_similarities.json")
    with open(samples_path, "w") as f:
        json.dump(sample_data, f, indent=2)
    print(f"Per-sample data saved to {samples_path}")
    
    # Print most suspicious samples
    print("\n=== Top 20 Most Suspicious Samples (lowest similarity) ===")
    for item in sample_data[:20]:
        print(f"  [{item['index']}] sim={item['similarity']:.4f} dur={item['duration']:.1f}s: {item['transcription'][:60]}...")
    
    # Save filtered indices (samples to KEEP)
    keep_indices = [indices[i] for i in np.where(~outlier_mask)[0]]
    keep_path = os.path.join(OUTPUT_DIR, "keep_indices.json")
    with open(keep_path, "w") as f:
        json.dump(keep_indices, f)
    print(f"\nIndices to keep ({len(keep_indices)} samples) saved to {keep_path}")


if __name__ == "__main__":
    main()
