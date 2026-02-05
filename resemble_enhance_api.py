#!/usr/bin/env python3
"""
Resemble Enhance using the Python API (same as Gradio demo)
============================================================
This uses the exact same API as the HuggingFace Gradio demo.

Key parameters:
- nfe: Number of Function Evaluations (64 = high quality)
- solver: "midpoint" recommended
- tau: Prior temperature (0.5 default)
- lambd: 0.9 for heavy noise, 0.1 for light noise
"""

import torch
import torchaudio
import argparse
from pathlib import Path
import time

# Import the same functions as the Gradio demo
from resemble_enhance.enhancer.inference import denoise, enhance

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def process_audio(input_path, output_dir, nfe=64, solver="midpoint", tau=0.5, denoise_first=True):
    """
    Process audio using the Resemble Enhance API (same as Gradio demo)
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save outputs
        nfe: Number of function evaluations (64 recommended)
        solver: ODE solver ("midpoint", "rk4", "euler")
        tau: Prior temperature (0.5 default)
        denoise_first: If True, use lambd=0.9 (for heavy noise)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    dwav, sr = torchaudio.load(input_path)
    dwav = dwav.mean(dim=0)  # Convert to mono
    
    print(f"Input: {input_path}")
    print(f"  Sample rate: {sr}Hz, Duration: {len(dwav)/sr:.2f}s")
    print(f"  Parameters: nfe={nfe}, solver={solver}, tau={tau}, denoise_first={denoise_first}")
    
    # Set lambd based on denoising setting (exactly like Gradio demo)
    lambd = 0.9 if denoise_first else 0.1
    print(f"  lambd={lambd}")
    
    start_time = time.time()
    
    # Step 1: Denoise (always run this for comparison)
    print("  Running denoise...")
    wav_denoised, new_sr = denoise(dwav, sr, device)
    
    # Step 2: Enhance (with full parameters)
    print("  Running enhance...")
    wav_enhanced, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)
    
    elapsed = time.time() - start_time
    print(f"  Processing time: {elapsed:.2f}s")
    
    # Save outputs
    basename = Path(input_path).stem
    
    # Save denoised
    denoised_path = output_dir / f"{basename}_denoised.wav"
    torchaudio.save(str(denoised_path), wav_denoised.unsqueeze(0).cpu(), new_sr)
    print(f"  Saved: {denoised_path}")
    
    # Save enhanced
    enhanced_path = output_dir / f"{basename}_enhanced.wav"
    torchaudio.save(str(enhanced_path), wav_enhanced.unsqueeze(0).cpu(), new_sr)
    print(f"  Saved: {enhanced_path}")
    
    return denoised_path, enhanced_path


def main():
    parser = argparse.ArgumentParser(description="Resemble Enhance using Python API")
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("-o", "--output", default="resemble_api_output", help="Output directory")
    parser.add_argument("--nfe", type=int, default=64, help="Number of function evaluations (default: 64)")
    parser.add_argument("--solver", choices=["midpoint", "rk4", "euler"], default="midpoint", help="ODE solver")
    parser.add_argument("--tau", type=float, default=0.5, help="Prior temperature (default: 0.5)")
    parser.add_argument("--no-denoise", action="store_true", help="Don't use denoise mode (lambd=0.1 instead of 0.9)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        process_audio(
            input_path, 
            args.output, 
            nfe=args.nfe, 
            solver=args.solver, 
            tau=args.tau,
            denoise_first=not args.no_denoise
        )
    elif input_path.is_dir():
        wav_files = sorted(input_path.glob("*.wav"))
        print(f"Processing {len(wav_files)} files...")
        for wav_file in wav_files:
            process_audio(
                wav_file, 
                args.output, 
                nfe=args.nfe, 
                solver=args.solver, 
                tau=args.tau,
                denoise_first=not args.no_denoise
            )
    else:
        print(f"Error: {input_path} not found")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
