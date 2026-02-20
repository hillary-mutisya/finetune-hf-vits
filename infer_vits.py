#!/usr/bin/env python3
"""
VITS TTS Inference Script

Generates audio from text using a trained VITS model.
Optimized settings based on analysis:
- noise_scale=0.1 (reduces vocoder noise, best UTMOS score)
- noise_scale_duration=0.0 (deterministic duration)
"""

import argparse
import torch
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
from transformers import VitsModel, VitsTokenizer


def load_model(model_id: str, device: str = None):
    """Load VITS model and tokenizer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_id}")
    model = VitsModel.from_pretrained(model_id)
    tokenizer = VitsTokenizer.from_pretrained(model_id)
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    return model, tokenizer, device


def generate_audio(
    model,
    tokenizer,
    text: str,
    device: str,
    noise_scale: float = 0.1,  # Optimized default (was 0.667)
    noise_scale_duration: float = 0.0,
    speaking_rate: float = 1.0,
):
    """Generate audio waveform from text.
    
    Args:
        model: VitsModel instance
        tokenizer: VitsTokenizer instance
        text: Input text to synthesize
        device: Device to run inference on
        noise_scale: Controls output variation. Lower = cleaner but less natural.
                     0.1 is optimal for quality (default was 0.667)
        noise_scale_duration: Controls duration variation. 0.0 = deterministic.
        speaking_rate: Speed multiplier (1.0 = normal)
    
    Returns:
        waveform: numpy array of audio samples
        sample_rate: audio sample rate
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model(
            **inputs,
            noise_scale=noise_scale,
            noise_scale_duration=noise_scale_duration,
        )
    
    waveform = output.waveform[0].cpu().numpy()
    sample_rate = model.config.sampling_rate
    
    return waveform, sample_rate


def save_audio(waveform: np.ndarray, sample_rate: int, output_path: str, normalize: bool = True):
    """Save waveform to WAV file."""
    if normalize:
        waveform = waveform / np.max(np.abs(waveform)) * 0.95
    
    waveform_int16 = (waveform * 32767).astype(np.int16)
    wav.write(output_path, sample_rate, waveform_int16)


def main():
    parser = argparse.ArgumentParser(description="VITS TTS Inference")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model ID (e.g., mutisya/vits_luo_26_05_f_v1)")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Text to synthesize")
    parser.add_argument("--text-file", "-f", type=str, default=None,
                        help="File with texts to synthesize (one per line)")
    parser.add_argument("--output-dir", "-o", type=str, default="./generated",
                        help="Output directory for audio files")
    parser.add_argument("--noise-scale", type=float, default=0.1,
                        help="Noise scale for vocoder (default: 0.1, original: 0.667)")
    parser.add_argument("--noise-scale-duration", type=float, default=0.0,
                        help="Noise scale for duration predictor (default: 0.0)")
    parser.add_argument("--speaking-rate", type=float, default=1.0,
                        help="Speaking rate multiplier (default: 1.0)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model, args.device)
    
    # Get texts to synthesize
    texts = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        with open(args.text_file) as f:
            texts.extend([line.strip() for line in f if line.strip()])
    
    if not texts:
        print("No text provided. Use --text or --text-file")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {len(texts)} samples with noise_scale={args.noise_scale}")
    print("-" * 60)
    
    for i, text in enumerate(texts):
        waveform, sample_rate = generate_audio(
            model, tokenizer, text, device,
            noise_scale=args.noise_scale,
            noise_scale_duration=args.noise_scale_duration,
        )
        
        output_path = output_dir / f"sample_{i+1:03d}.wav"
        save_audio(waveform, sample_rate, str(output_path))
        
        duration = len(waveform) / sample_rate
        print(f"  [{i+1}] {text[:50]:<50} -> {output_path.name} ({duration:.2f}s)")
    
    print("-" * 60)
    print(f"Generated {len(texts)} files in {output_dir}/")


if __name__ == "__main__":
    main()
