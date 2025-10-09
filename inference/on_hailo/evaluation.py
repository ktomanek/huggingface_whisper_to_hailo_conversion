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

