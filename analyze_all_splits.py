"""
Comprehensive speaker analysis for all splits in mutisya/tts-speakers-7lang-15k-25-49-v2
- Analyzes each subset and split for speaker outliers
- Creates a CSV report with outlier counts
- Saves sample audio files (main speaker + outliers) for each split
"""

import os
import json
import csv
import numpy as np
import torch
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from sklearn.cluster import KMeans
from tqdm import tqdm
import soundfile as sf
from collections import defaultdict

# Configuration
DATASET_NAME = "mutisya/tts-speakers-7lang-15k-25-49-v2"
OUTPUT_DIR = "./speaker_analysis_full"
SIMILARITY_THRESHOLD = 0.75
MAX_SAMPLES_PER_CATEGORY = 5  # 5 main speaker, 5 per outlier cluster

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

def analyze_split(dataset, config_name, split_name):
    """Analyze a single split for speaker outliers."""
    print(f"\n  Analyzing {config_name}/{split_name} ({len(dataset)} samples)...")
    
    embeddings = []
    valid_indices = []
    durations = []
    
    for i, sample in enumerate(tqdm(dataset, desc=f"    Extracting embeddings", leave=False)):
        try:
            audio = sample['audio']
            embedding = extract_embedding(audio['array'], audio['sampling_rate'])
            embeddings.append(embedding)
            valid_indices.append(i)
            durations.append(len(audio['array']) / audio['sampling_rate'])
        except Exception as e:
            print(f"    Warning: Failed to process sample {i}: {e}")
            continue
    
    if len(embeddings) < 10:
        print(f"    Too few valid samples ({len(embeddings)}), skipping...")
        return None
    
    embeddings = np.array(embeddings)
    
    # Compute centroid (mean embedding)
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    
    # Compute similarities to centroid
    similarities = np.dot(embeddings, centroid)
    
    # Find outliers
    outlier_mask = similarities < SIMILARITY_THRESHOLD
    outlier_indices = [valid_indices[i] for i in np.where(outlier_mask)[0]]
    main_speaker_indices = [valid_indices[i] for i in np.where(~outlier_mask)[0]]
    
    # Cluster outliers if there are enough
    outlier_clusters = {}
    if len(outlier_indices) >= 3:
        outlier_embeddings = embeddings[outlier_mask]
        n_clusters = min(3, len(outlier_indices) // 2)  # Up to 3 clusters
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            outlier_labels = kmeans.fit_predict(outlier_embeddings)
            for cluster_id in range(n_clusters):
                cluster_mask = outlier_labels == cluster_id
                cluster_indices = [outlier_indices[i] for i in np.where(cluster_mask)[0]]
                if len(cluster_indices) > 0:
                    outlier_clusters[f"cluster_{cluster_id}"] = cluster_indices
        else:
            outlier_clusters["cluster_0"] = outlier_indices
    elif len(outlier_indices) > 0:
        outlier_clusters["cluster_0"] = outlier_indices
    
    results = {
        "config": config_name,
        "split": split_name,
        "total_samples": len(dataset),
        "valid_samples": len(valid_indices),
        "main_speaker_count": len(main_speaker_indices),
        "outlier_count": len(outlier_indices),
        "outlier_percentage": 100 * len(outlier_indices) / len(valid_indices) if valid_indices else 0,
        "mean_similarity": float(similarities.mean()),
        "min_similarity": float(similarities.min()),
        "max_similarity": float(similarities.max()),
        "std_similarity": float(similarities.std()),
        "main_speaker_indices": main_speaker_indices,
        "outlier_indices": outlier_indices,
        "outlier_clusters": outlier_clusters,
        "similarities": {valid_indices[i]: float(similarities[i]) for i in range(len(valid_indices))}
    }
    
    return results

def save_audio_samples(dataset, results, output_dir):
    """Save sample audio files for main speaker and outliers."""
    config = results["config"]
    split = results["split"]
    split_dir = os.path.join(output_dir, f"{config}_{split}")
    os.makedirs(split_dir, exist_ok=True)
    
    # Save main speaker samples (5 highest similarity)
    main_indices = results["main_speaker_indices"]
    if main_indices:
        # Sort by similarity (highest first)
        sorted_main = sorted(main_indices, 
                            key=lambda x: results["similarities"].get(x, 0), 
                            reverse=True)
        for i, idx in enumerate(sorted_main[:MAX_SAMPLES_PER_CATEGORY]):
            sample = dataset[idx]
            audio = sample['audio']
            sim = results["similarities"].get(idx, 0)
            filename = f"main_speaker_{i+1:02d}_sim_{sim:.4f}_idx_{idx:05d}.wav"
            filepath = os.path.join(split_dir, filename)
            sf.write(filepath, audio['array'], audio['sampling_rate'])
    
    # Save outlier samples per cluster
    for cluster_name, cluster_indices in results["outlier_clusters"].items():
        # Sort by similarity (lowest first - most outlier-like)
        sorted_outliers = sorted(cluster_indices, 
                                key=lambda x: results["similarities"].get(x, 1))
        for i, idx in enumerate(sorted_outliers[:MAX_SAMPLES_PER_CATEGORY]):
            sample = dataset[idx]
            audio = sample['audio']
            sim = results["similarities"].get(idx, 0)
            filename = f"outlier_{cluster_name}_{i+1:02d}_sim_{sim:.4f}_idx_{idx:05d}.wav"
            filepath = os.path.join(split_dir, filename)
            sf.write(filepath, audio['array'], audio['sampling_rate'])
    
    # Save a summary for this split
    summary_file = os.path.join(split_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Split: {config}/{split}\n")
        f.write(f"Total samples: {results['total_samples']}\n")
        f.write(f"Main speaker samples: {results['main_speaker_count']}\n")
        f.write(f"Outlier samples: {results['outlier_count']} ({results['outlier_percentage']:.2f}%)\n")
        f.write(f"Mean similarity: {results['mean_similarity']:.4f}\n")
        f.write(f"Min similarity: {results['min_similarity']:.4f}\n")
        f.write(f"\nOutlier clusters:\n")
        for cluster_name, indices in results["outlier_clusters"].items():
            f.write(f"  {cluster_name}: {len(indices)} samples\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print(f"Speaker Analysis for {DATASET_NAME}")
    print("="*70)
    
    # Get all configs
    configs = get_dataset_config_names(DATASET_NAME)
    print(f"\nConfigs (subsets): {configs}")
    
    all_results = []
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Config: {config}")
        print("="*70)
        
        # Get splits for this config
        splits = get_dataset_split_names(DATASET_NAME, config)
        print(f"Splits: {splits}")
        
        for split in splits:
            # Load dataset
            print(f"\n  Loading {config}/{split}...")
            try:
                dataset = load_dataset(DATASET_NAME, config, split=split)
                print(f"  Loaded {len(dataset)} samples")
                
                # Analyze
                results = analyze_split(dataset, config, split)
                
                if results:
                    all_results.append(results)
                    
                    # Save audio samples
                    print(f"    Saving audio samples...")
                    save_audio_samples(dataset, results, OUTPUT_DIR)
                    
                    print(f"    ✓ Outliers: {results['outlier_count']}/{results['valid_samples']} ({results['outlier_percentage']:.2f}%)")
                    
            except Exception as e:
                print(f"  Error loading {config}/{split}: {e}")
                continue
    
    # Create CSV report
    csv_file = os.path.join(OUTPUT_DIR, "speaker_analysis_report.csv")
    print(f"\n{'='*70}")
    print(f"Creating report: {csv_file}")
    print("="*70)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Config", "Split", "Total Samples", "Valid Samples",
            "Main Speaker Count", "Outlier Count", "Outlier %",
            "Mean Similarity", "Min Similarity", "Max Similarity", "Std Similarity"
        ])
        
        for r in all_results:
            writer.writerow([
                r["config"], r["split"], r["total_samples"], r["valid_samples"],
                r["main_speaker_count"], r["outlier_count"], f"{r['outlier_percentage']:.2f}",
                f"{r['mean_similarity']:.4f}", f"{r['min_similarity']:.4f}",
                f"{r['max_similarity']:.4f}", f"{r['std_similarity']:.4f}"
            ])
    
    # Save detailed JSON
    json_file = os.path.join(OUTPUT_DIR, "speaker_analysis_detailed.json")
    # Remove large arrays for JSON
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k not in ['similarities', 'main_speaker_indices', 'outlier_indices']}
        jr['outlier_clusters_sizes'] = {k: len(v) for k, v in r['outlier_clusters'].items()}
        json_results.append(jr)
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*90)
    print("SUMMARY REPORT")
    print("="*90)
    print(f"{'Config':<12} {'Split':<15} {'Total':>8} {'Outliers':>10} {'Outlier %':>10} {'Mean Sim':>10}")
    print("-"*90)
    
    total_samples = 0
    total_outliers = 0
    
    for r in all_results:
        print(f"{r['config']:<12} {r['split']:<15} {r['total_samples']:>8} {r['outlier_count']:>10} {r['outlier_percentage']:>9.2f}% {r['mean_similarity']:>10.4f}")
        total_samples += r['total_samples']
        total_outliers += r['outlier_count']
    
    print("-"*90)
    print(f"{'TOTAL':<28} {total_samples:>8} {total_outliers:>10} {100*total_outliers/total_samples:>9.2f}%")
    print("="*90)
    
    print(f"\n✓ Report saved to: {csv_file}")
    print(f"✓ Audio samples saved to: {OUTPUT_DIR}/<config>_<split>/")
    print(f"✓ Detailed JSON: {json_file}")

if __name__ == "__main__":
    main()
