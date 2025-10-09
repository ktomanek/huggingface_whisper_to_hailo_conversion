#!/usr/bin/env python3
"""
Whisper Evaluation on Hailo

Evaluates encoder/decoder combinations on a dataset with ground truth transcriptions.
Audio files are paired with text files: <uid>.wav and <uid>.txt

Usage Example:
   python evaluation.py \
     --audio_folder /path/to/audio_dataset \
     --encoder_hef_file /path/to/encoder.hef \
     --decoder_onnx_dir /path/to/decoder_onnx \
     --variant tiny
"""

import numpy as np
import os
import argparse
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

# For WER calculation
try:
    from jiwer import wer, cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print("Warning: jiwer not installed. Install with: pip install jiwer")


# ============================================================================
# Dataset Loading
# ============================================================================

def load_audio_dataset(audio_folder: str) -> Dict[str, str]:
    """
    Load audio dataset with paired audio (.wav or .mp3) and .txt files.

    Args:
        audio_folder: Path to folder containing <uid>.wav/.mp3 and <uid>.txt files

    Returns:
        Dictionary mapping audio file paths to ground truth transcriptions
    """
    audio_folder = Path(audio_folder)

    if not audio_folder.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_folder}")

    dataset = {}

    # Find all .wav and .mp3 files
    audio_files = list(audio_folder.glob("*.wav")) + list(audio_folder.glob("*.mp3"))

    if not audio_files:
        raise ValueError(f"No .wav or .mp3 files found in {audio_folder}")

    for audio_file in audio_files:
        # Look for corresponding .txt file
        txt_file = audio_file.with_suffix('.txt')

        if not txt_file.exists():
            print(f"Warning: No ground truth found for {audio_file.name}, skipping")
            continue

        # Read ground truth transcription
        with open(txt_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()

        dataset[str(audio_file)] = ground_truth

    print(f"Loaded {len(dataset)} audio files with ground truth")
    return dataset


# ============================================================================
# WER Calculation
# ============================================================================

def calculate_wer(hypothesis: str, reference: str) -> Tuple[float, float]:
    """
    Calculate Word Error Rate (WER) and Character Error Rate (CER).

    Args:
        hypothesis: Predicted transcription
        reference: Ground truth transcription

    Returns:
        (wer_score, cer_score): WER and CER as percentages (0-100)
    """
    if not HAS_JIWER:
        print("Warning: jiwer not available, returning 0.0 for WER/CER")
        return 0.0, 0.0

    # Normalize text (lowercase, strip)
    hypothesis = hypothesis.lower().strip()
    reference = reference.lower().strip()

    wer_score = wer(reference, hypothesis) * 100
    cer_score = cer(reference, hypothesis) * 100

    return wer_score, cer_score


# ============================================================================
# Evaluation
# ============================================================================

def transcribe(audio_file: str) -> str:
    """
    Transcribe audio file using encoder/decoder.

    TODO: Implement this function with encoder/decoder inference

    Args:
        audio_file: Path to audio file

    Returns:
        transcription: Text transcription
    """
    raise NotImplementedError("transcribe() function not yet implemented")


def evaluate_dataset(dataset: Dict[str, str]) -> Dict:
    """
    Evaluate on dataset by transcribing each audio file and calculating WER.

    Args:
        dataset: Dictionary mapping audio files to ground truth

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"Running Evaluation")
    print(f"{'='*70}")
    print(f"Dataset size: {len(dataset)} samples")
    print()

    total_wer = 0.0
    total_cer = 0.0
    num_samples = len(dataset)

    results = []

    for audio_file, ground_truth in tqdm(dataset.items(), desc="Processing"):
        try:
            # Transcribe audio file
            transcription = transcribe(audio_file)

            # Calculate WER/CER
            wer_score, cer_score = calculate_wer(transcription, ground_truth)

            total_wer += wer_score
            total_cer += cer_score

            results.append({
                'audio_file': audio_file,
                'ground_truth': ground_truth,
                'transcription': transcription,
                'wer': wer_score,
                'cer': cer_score
            })

        except Exception as e:
            print(f"\nError processing {audio_file}: {e}")
            num_samples -= 1
            continue

    if num_samples == 0:
        print("Error: No samples successfully processed")
        return None

    # Calculate averages
    avg_wer = total_wer / num_samples
    avg_cer = total_cer / num_samples

    summary = {
        'num_samples': num_samples,
        'avg_wer': avg_wer,
        'avg_cer': avg_cer,
        'results': results
    }

    return summary


def print_evaluation_summary(summary: Dict):
    """Print evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"  Samples:         {summary['num_samples']}")
    print(f"  Average WER:     {summary['avg_wer']:.2f}%")
    print(f"  Average CER:     {summary['avg_cer']:.2f}%")
    print()

    # Show some examples
    print("Sample Results (first 5):")
    for i, result in enumerate(summary['results'][:5]):
        filename = Path(result['audio_file']).name
        print(f"\n  {i+1}. {filename}")
        print(f"     GT:    {result['ground_truth']}")
        print(f"     Pred:  {result['transcription']}")
        print(f"     WER:   {result['wer']:.2f}%")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper Evaluation on Hailo")

    # Dataset
    parser.add_argument("--audio_folder", type=str, required=True,
                        help="Path to folder containing <uid>.wav and <uid>.txt files")

    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (default: all)")

    args = parser.parse_args()

    if not HAS_JIWER:
        print("Error: jiwer is required for WER calculation")
        print("Install with: pip install jiwer")
        return 1

    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_audio_dataset(args.audio_folder)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Limit samples if requested
    if args.max_samples and args.max_samples < len(dataset):
        dataset_items = list(dataset.items())[:args.max_samples]
        dataset = dict(dataset_items)
        print(f"Limited to {len(dataset)} samples")

    # Run evaluation
    try:
        summary = evaluate_dataset(dataset)
        if summary:
            print_evaluation_summary(summary)
        else:
            print("\nNo successful evaluations completed")
            return 1
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print("Please implement the transcribe() function first")
        return 1

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
