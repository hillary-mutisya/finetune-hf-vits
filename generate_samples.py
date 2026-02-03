#!/usr/bin/env python3
"""
Generate sample audio from finetuned VITS model for Kikuyu TTS.
"""

import torch
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer
import os

# Model path
MODEL_PATH = "./outputs/vits_finetuned_kik_estherk"
OUTPUT_DIR = "./generated_samples"

# Sample texts in Kikuyu
SAMPLE_TEXTS = [
    "Ngai akauga, Nĩkũgũkorwo na ũtheri. Na ũtheri ũgĩtuĩka o ũguo.",
    "Rĩrĩa Ngai akirora mwĩrĩire ũrĩa aahũmbĩire, nĩonire atĩ nĩ mwega mũno.",
    "Mũthenya mwega. Nĩ ngũkena gũkuona.",
    "Nĩndakũhoya ũndeteere mũhũnja ũcio.",
    "Tũrĩ andũ a Kenya na nĩtũgũthiĩ mbere.",
]

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load model and tokenizer
    model = VitsModel.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    for i, text in enumerate(SAMPLE_TEXTS):
        print(f"\nGenerating sample {i+1}/{len(SAMPLE_TEXTS)}...")
        print(f"  Text: {text}")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Generate audio
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Convert to numpy and save
        waveform = output.squeeze().cpu().numpy()
        sample_rate = model.config.sampling_rate
        
        output_path = os.path.join(OUTPUT_DIR, f"sample_{i+1:02d}.wav")
        wavfile.write(output_path, sample_rate, waveform)
        
        print(f"  Saved to: {output_path}")
        print(f"  Duration: {len(waveform) / sample_rate:.2f}s")
    
    print(f"\n✅ Generated {len(SAMPLE_TEXTS)} samples in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
