#!/usr/bin/env python3
"""Generate audio samples from the trained VITS model checkpoint."""

import torch
import sys
sys.path.insert(0, '/home/hillary/trainingTTS/vits_training')
from utils import VitsModelForPreTraining, VitsFeatureExtractor
from transformers import VitsTokenizer
import scipy.io.wavfile as wavfile
import os
from datetime import datetime
from safetensors.torch import load_file

# Configuration
CHECKPOINT_PATH = "./outputs/vits_finetuned_swh_uganda_speaker"  # Final model
OUTPUT_DIR = "./generated_samples_swh_uganda_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sample texts in Swahili
SAMPLE_TEXTS = [
    "Habari ya asubuhi. Karibu sana kwenye programu yetu.",
    "Tunataka kukuonyesha jinsi ya kutumia teknolojia hii.",
    "Jambo, jina langu ni Uganda. Ninafuraha kukutana nawe.",
    "Teknolojia ya sauti bandia inaweza kusaidia watu wengi.",
    "Hii ni mfano wa sauti iliyotengenezwa na kompyuta.",
    "Lugha ya Kiswahili ni nzuri sana na inazungumzwa na watu wengi.",
    "Asante sana kwa kusikiliza. Tutaonana tena.",
    "Elimu ni ufunguo wa maisha bora.",
]

def main():
    print(f"Loading model from: {CHECKPOINT_PATH}")
    print(f"Using device: {DEVICE}")
    
    # Load base model from the pretrained hub model
    base_model = "mutisya/vits_swh_mms_24_10_1_pre"
    
    model = VitsModelForPreTraining.from_pretrained(
        base_model,
        ignore_mismatched_sizes=True
    )
    tokenizer = VitsTokenizer.from_pretrained(base_model)
    
    # Apply weight normalization before loading checkpoint weights
    model.decoder.apply_weight_norm()
    for flow in model.flow.flows:
        torch.nn.utils.weight_norm(flow.conv_pre)
        torch.nn.utils.weight_norm(flow.conv_post)
    
    # Load checkpoint weights
    print(f"Loading checkpoint weights from: {CHECKPOINT_PATH}")
    checkpoint_weights = load_file(os.path.join(CHECKPOINT_PATH, "model.safetensors"))
    missing, unexpected = model.load_state_dict(checkpoint_weights, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    model = model.to(DEVICE)
    model.eval()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nGenerating {len(SAMPLE_TEXTS)} audio samples...\n")
    
    for i, text in enumerate(SAMPLE_TEXTS, 1):
        print(f"[{i}/{len(SAMPLE_TEXTS)}] Generating: {text[:50]}...")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        # Generate audio
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get waveform
        waveform = outputs.waveform[0].cpu().numpy()
        
        # Save audio
        sample_rate = model.config.sampling_rate
        filename = f"{OUTPUT_DIR}/sample_{timestamp}_{i:02d}.wav"
        wavfile.write(filename, sample_rate, waveform)
        print(f"   Saved: {filename}")
    
    print(f"\n✅ Generated {len(SAMPLE_TEXTS)} samples in {OUTPUT_DIR}/")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Sample rate: {model.config.sampling_rate} Hz")

if __name__ == "__main__":
    main()
