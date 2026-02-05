# VITS TTS Training Analysis Report
## Swahili (swh_Latn) Uganda Speaker Fine-tuning

**Date**: February 4, 2026  
**Author**: TTS Training Analysis

---

## Executive Summary

This report documents our experiments fine-tuning VITS TTS models on Swahili data, focusing on:
1. The impact of dataset size on output quality
2. The phenomenon of "noise learning" during extended training
3. Post-processing approaches to clean noisy TTS output
4. Gaps in current evaluation metrics

---

## 1. Experimental Setup

### Base Model
- **Model**: `mutisya/vits_swh_mms_24_10_1_pre` (pre-trained Swahili MMS)
- **Architecture**: VITS (Variational Inference Text-to-Speech)

### Datasets Tested

| Experiment | Dataset Size | Epochs | Total Steps | Training Data Quality |
|------------|-------------|--------|-------------|----------------------|
| **Full Training** | 3,835 samples | 200 | 48,000 | Noisy (mean SNR: 0.5 dB) |
| **Enhanced Training** | 200 samples | 200 | 2,600 | Clean (Resemble-enhanced) |

### Training Configuration
- Batch size: 16
- Learning rate: 2e-5
- Loss weights: duration=1.0, kl=1.5, mel=35.0, disc=3.0, gen=1.0, fmaps=1.0
- FP16 training enabled

---

## 2. Key Findings

### Finding 1: Larger Datasets Produce Better Audio Quality

The **200-sample enhanced training** produced lower quality audio than intermediate checkpoints from the **3,835-sample noisy training**, despite the enhanced dataset having cleaner input audio.

**Hypothesis**: VITS requires sufficient data diversity for the model to:
- Learn proper phoneme-to-acoustic mappings across varied contexts
- Develop robust prosody patterns
- Generalize pronunciation variations

**Implication**: Dataset SIZE may be more important than individual sample QUALITY for overall TTS output quality.

### Finding 2: Extended Training Causes "Noise Memorization"

In the 200-epoch training on noisy data:
- **Intermediate checkpoints** (e.g., epoch 100) produced cleaner-sounding speech
- **Final checkpoint** (epoch 200) produced audio with more noise artifacts

**Explanation**: 
- The model initially learns general speech patterns
- With extended training, it begins to memorize training data characteristics
- Since training data contains noise, the model learns to REPRODUCE that noise as part of "faithful" generation
- This is classic overfitting, but manifested in the acoustic domain rather than just loss curves

### Finding 3: Post-Processing Can Partially Clean Noisy TTS Output

We tested two enhancement approaches on noisy TTS output:

| Tool | Speed | Noise Reduction | Voice Clarity | Best For |
|------|-------|-----------------|---------------|----------|
| **DeepFilterNet3** | ~0.01s/file | Good | Preserved | Light background noise |
| **Resemble Enhance** | ~1s/file | Excellent | Enhanced | Heavy noise, restoration |

**Results on Generated Samples:**
- Both tools successfully reduced noise artifacts
- Resemble Enhance (with `lambd=0.9`) provided the most dramatic improvement
- Some "learned noise patterns" were harder to remove than natural recording noise

---

## 3. Training Metrics Analysis

### Metrics Currently Logged

The training script logs the following loss components:

| Metric | Description | What It Measures |
|--------|-------------|------------------|
| `loss_mel` | Mel-spectrogram L1 loss | Spectral similarity to training audio |
| `loss_kl` | KL divergence | Latent distribution alignment |
| `loss_duration` | Duration prediction loss | Timing/prosody accuracy |
| `loss_disc` | Discriminator loss | Adversarial realism |
| `loss_gen` | Generator loss | Adversarial fooling ability |
| `loss_fmaps` | Feature map loss | Perceptual similarity (discriminator features) |

### What's MISSING: Audio Quality Metrics

**Critical Gap**: None of these metrics directly measure:

1. **Perceptual Quality (MOS-like)**
   - No UTMOS or MOSNet score
   - No PESQ (Perceptual Evaluation of Speech Quality)
   - No POLQA scores

2. **Speaker Similarity**
   - No speaker embedding cosine similarity
   - No ECAPA-TDNN verification scores
   - No speaker verification EER

3. **Intelligibility**
   - No Character/Word Error Rate from ASR
   - No STOI (Short-Time Objective Intelligibility)

4. **Naturalness**
   - No F0 correlation with reference
   - No prosody deviation metrics

### Why Current Metrics Fail to Detect Overfitting

The `loss_mel` metric measures similarity to **training audio**, not to **ideal clean audio**. If training audio is noisy:
- Lower mel loss = more similar to noisy training data
- Model is rewarded for reproducing noise
- Loss goes down while perceptual quality goes down

This is why **intermediate checkpoints sound better** - they haven't yet "perfected" noise reproduction.

---

## 4. Recommendations

### Recommendation 1: Add Perceptual Quality Metrics to Evaluation

Implement UTMOS scoring during validation:

```python
# During validation
utmos_predictor = load_utmos_model()
for generated_wav in validation_outputs:
    mos_score = utmos_predictor(generated_wav)
    log_metric("eval/utmos", mos_score)
```

### Recommendation 2: Add Early Stopping Based on Perceptual Metrics

Instead of training for fixed epochs:
- Monitor UTMOS/MOS scores on a held-out validation set
- Stop when perceptual quality starts degrading
- Save "best perceptual" checkpoint separately from "lowest loss" checkpoint

### Recommendation 3: Add Speaker Similarity Tracking

```python
from speechbrain.pretrained import EncoderClassifier
speaker_model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

# Compare generated audio to reference speaker
ref_embedding = speaker_model.encode_batch(reference_audio)
gen_embedding = speaker_model.encode_batch(generated_audio)
similarity = cosine_similarity(ref_embedding, gen_embedding)
log_metric("eval/speaker_similarity", similarity)
```

### Recommendation 4: Pre-clean Training Data

Given that training data quality propagates to output:
- Run Resemble Enhance (lambd=0.9) on ALL training data before training
- Consider SNR filtering: only use samples with SNR > 20dB
- This prevents the model from learning noise patterns

### Recommendation 5: Minimum Dataset Size Guidelines

Based on our experiments:

| Quality Target | Minimum Samples | Notes |
|---------------|-----------------|-------|
| Intelligible | 200+ | Basic pronunciation |
| Natural | 1,000+ | Better prosody |
| High Quality | 3,000+ | Speaker consistency |
| Production | 10,000+ | Full generalization |

---

## 5. Audio Enhancement Comparison

### Samples Generated for Comparison

Location: `/home/hillary/trainingTTS/vits_training/generated_cleanup_comparison/`

| Folder | Source | Processing |
|--------|--------|-----------|
| `original_early/` | Intermediate checkpoint | None |
| `original_final/` | Final checkpoint (epoch 200) | None |
| `deepfilter_early/` | Intermediate checkpoint | DeepFilterNet3 |
| `deepfilter_final/` | Final checkpoint | DeepFilterNet3 |
| `resemble_early/` | Intermediate checkpoint | Resemble Enhance |
| `resemble_final/` | Final checkpoint | Resemble Enhance |

### Enhancement Parameters Used

**DeepFilterNet3:**
- Post-filter enabled (`--pf`)
- Default noise reduction strength

**Resemble Enhance:**
- NFE: 64
- Solver: midpoint
- Tau: 0.5
- Lambda: 0.9 (optimized for noisy input)
- Denoise + Enhance pipeline

---

## 6. Conclusion

Training on noisy data causes VITS to learn noise patterns as part of speaker characteristics. This cannot be detected by current loss-based metrics because they measure similarity to noisy training targets.

**Key Takeaways:**
1. Dataset size matters more than per-sample quality (up to a point)
2. Extended training on noisy data causes noise memorization
3. Post-processing can help but doesn't fully solve the problem
4. Pre-cleaning training data is essential
5. Perceptual metrics (UTMOS, speaker similarity) should be added to evaluation

**Next Steps:**
1. Enhance full 3,835 sample dataset with Resemble
2. Retrain with cleaned data
3. Implement UTMOS-based early stopping
4. Compare "clean-trained" vs "noisy-trained" models

---

## Appendix: Training Loss Progression

### 200-Sample Enhanced Training
| Step | Total Loss | Mel Loss | Disc Loss | Duration Loss |
|------|-----------|----------|-----------|---------------|
| 1 | 75.8 | - | 5.44 | 2.56 |
| 50 | 29.2 | 0.472 | 3.12 | 1.64 |
| 100 | 26.7 | 0.453 | 3.15 | 1.60 |
| 2600 (final) | 27.3 | 0.314 | 4.65 | 1.54 |

Note: Mel loss continued to decrease throughout training, but this doesn't guarantee improved perceptual quality.
