#!/usr/bin/env python3
"""
FasterWhisper Baseline Comparison

Runs FasterWhisper (CPU INT8 baseline) for comparison with Hailo implementations.
Supports single file benchmarking with statistics and folder evaluation with WER.

Usage Examples:

1. Single file with benchmarking:
   python faster_whisper_baseline.py \
     --audio_file test.wav \
     --num_iterations 10

2. Folder evaluation with WER:
   python faster_whisper_baseline.py \
     --audio_folder /path/to/dataset
"""

import numpy as np
import os
import argparse
import time
from pathlib import Path
import librosa

# For evaluation
from evaluation import load_audio_dataset, calculate_wer


def run_faster_whisper_single(audio_file, model, sample_rate=16000):
    """
    Run FasterWhisper on a single audio file.

    Args:
        audio_file: Path to audio file
        model: Loaded FasterWhisper model
        sample_rate: Audio sample rate

    Returns:
        (transcription, total_time_ms): Transcription text and total inference time
    """
    # Load audio
    audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)

    # Measure preprocessing separately
    preprocess_start = time.time()
    features = model.feature_extractor(audio)
    preprocess_time_ms = (time.time() - preprocess_start) * 1000

    # Run full transcription
    inference_start = time.time()
    segments, info = model.transcribe(
        audio,
        beam_size=1,
        language='en',
        task='transcribe',
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=False,
    )

    # Collect all segments
    transcription = ""
    for segment in segments:
        transcription += segment.text.strip() + " "

    total_time_ms = (time.time() - inference_start) * 1000
    transcription = transcription.strip()

    return transcription, total_time_ms, preprocess_time_ms


def main():
    parser = argparse.ArgumentParser(description="FasterWhisper Baseline Comparison")

    # Audio input (mutually exclusive)
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument("--audio_file", type=str,
                        help="Path to audio file to transcribe")
    audio_group.add_argument("--audio_folder", type=str,
                        help="Path to folder containing audio files (.wav/.mp3) and .txt ground truth files for WER evaluation")

    # Options
    parser.add_argument("--model_size", type=str, default="tiny",
                        help="Model size: tiny, base, small, medium, large (default: tiny)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--compute_type", type=str, default="int8",
                        help="Compute type: int8, float16, float32 (default: int8)")
    parser.add_argument("--num_iterations", type=int, default=1,
                        help="Number of iterations for single file benchmarking (default: 1)")

    args = parser.parse_args()

    # Check if audio file/folder exists
    if args.audio_file and not os.path.exists(args.audio_file):
        raise ValueError(f"Error: Audio file not found: {args.audio_file}")
    if args.audio_folder and not os.path.exists(args.audio_folder):
        raise ValueError(f"Error: Audio folder not found: {args.audio_folder}")

    # num_iterations only makes sense for single file
    if args.audio_folder and args.num_iterations > 1:
        print("Warning: --num_iterations is ignored when using --audio_folder")
        args.num_iterations = 1

    print(f"{'='*70}")
    print(f"FASTERWHISPER BASELINE")
    print(f"{'='*70}")
    print(f"Model: {args.model_size}")
    print(f"Device: {args.device}")
    print(f"Compute type: {args.compute_type}")
    if args.audio_file:
        print(f"Audio file: {args.audio_file}")
        if args.num_iterations > 1:
            print(f"Iterations: {args.num_iterations}")
    else:
        print(f"Audio folder: {args.audio_folder}")
    print()

    # Load model
    print("Loading FasterWhisper model...")
    load_start = time.time()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster-whisper not installed")
        print("Install with: pip install faster-whisper")
        return 1

    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)
    load_time_ms = (time.time() - load_start) * 1000
    print(f"✅ Model loaded in {load_time_ms:.1f}ms")
    print()

    # Warmup with dummy audio
    print("Running warmup inference (dummy audio)...")
    warmup_start = time.time()
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    list(model.transcribe(
        dummy_audio,
        beam_size=1,
        language='en',
        task='transcribe',
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=False,
    )[0])
    warmup_time_ms = (time.time() - warmup_start) * 1000
    print(f"✅ Warmup completed in {warmup_time_ms:.1f}ms")
    print()

    # Warmup with actual audio file
    warmup_audio_file = args.audio_file if args.audio_file else list(load_audio_dataset(args.audio_folder).keys())[0]
    print(f"Running warmup inference (actual audio: {Path(warmup_audio_file).name})...")
    warmup_start = time.time()
    _, _, _ = run_faster_whisper_single(warmup_audio_file, model)
    warmup_time_ms = (time.time() - warmup_start) * 1000
    print(f"✅ Warmup completed in {warmup_time_ms:.1f}ms")
    print()

    # Mode 1: Single audio file
    if args.audio_file:
        inference_times = []
        preprocess_times = []
        transcription = None

        for iteration in range(args.num_iterations):
            if args.num_iterations > 1:
                print(f"Iteration {iteration + 1}/{args.num_iterations}...")

            transcription, inference_time_ms, preprocess_time_ms = run_faster_whisper_single(
                args.audio_file, model
            )

            inference_times.append(inference_time_ms)
            preprocess_times.append(preprocess_time_ms)

            if args.num_iterations > 1:
                print(f"  Total time: {inference_time_ms:.1f}ms (preprocessing: {preprocess_time_ms:.1f}ms)")

        # Print benchmarking statistics if multiple iterations
        if args.num_iterations > 1:
            print(f"\n{'='*70}")
            print(f"BENCHMARKING STATISTICS ({args.num_iterations} iterations)")
            print(f"{'='*70}")
            print(f"Inference times (ms):")
            print(f"  Mean: {np.mean(inference_times):.1f} ± {np.std(inference_times):.1f}")
            print(f"  Min/Max: {np.min(inference_times):.1f} / {np.max(inference_times):.1f}")
            print(f"  Median: {np.median(inference_times):.1f}")
            print(f"\nPreprocessing times (ms):")
            print(f"  Mean: {np.mean(preprocess_times):.1f} ± {np.std(preprocess_times):.1f}")
            print(f"  Min/Max: {np.min(preprocess_times):.1f} / {np.max(preprocess_times):.1f}")
            print(f"  Median: {np.median(preprocess_times):.1f}")
            print(f"{'='*70}")
        else:
            print(f"[TIMING] Total time: {inference_times[0]:.1f}ms (preprocessing: {preprocess_times[0]:.1f}ms)")

        # Print transcription result
        print("\n" + "=" * 30 + " TRANSCRIPTION:")
        print(f"{transcription}")
        print("=" * 30)

    # Mode 2: Audio folder with evaluation
    else:
        print("Loading dataset...")
        dataset = load_audio_dataset(args.audio_folder)
        print()

        print(f"{'='*70}")
        print(f"Running Evaluation on {len(dataset)} samples")
        print(f"{'='*70}\n")

        results = []
        total_wer = 0.0
        total_cer = 0.0
        total_inference_time = 0.0
        total_preprocess_time = 0.0

        for idx, (audio_file, ground_truth) in enumerate(dataset.items(), 1):
            print(f"[{idx}/{len(dataset)}] Processing: {Path(audio_file).name}")

            try:
                # Run inference
                transcription, inference_time_ms, preprocess_time_ms = run_faster_whisper_single(
                    audio_file, model
                )

                # Calculate WER/CER
                wer_score, cer_score = calculate_wer(transcription, ground_truth)

                total_wer += wer_score
                total_cer += cer_score
                total_inference_time += inference_time_ms
                total_preprocess_time += preprocess_time_ms

                results.append({
                    'audio_file': audio_file,
                    'ground_truth': ground_truth,
                    'transcription': transcription,
                    'wer': wer_score,
                    'cer': cer_score,
                    'inference_time_ms': inference_time_ms,
                    'preprocess_time_ms': preprocess_time_ms
                })

                print(f"  GT:   {ground_truth}")
                print(f"  Pred: {transcription}")
                print(f"  WER:  {wer_score:.2f}%")
                print()

            except Exception as e:
                print(f"  Error: {e}\n")
                continue

        # Print evaluation summary
        num_successful = len(results)
        if num_successful > 0:
            avg_wer = total_wer / num_successful
            avg_cer = total_cer / num_successful
            avg_inference_time = total_inference_time / num_successful
            avg_preprocess_time = total_preprocess_time / num_successful
            avg_total_time = avg_inference_time + avg_preprocess_time

            print(f"\n{'='*70}")
            print("EVALUATION SUMMARY")
            print(f"{'='*70}")
            print(f"  Samples processed:     {num_successful}/{len(dataset)}")
            print(f"  Average WER:           {avg_wer:.2f}%")
            print(f"  Average CER:           {avg_cer:.2f}%")
            print(f"\n  Timing:")
            print(f"    Avg Preprocessing:   {avg_preprocess_time:.1f}ms")
            print(f"    Avg Inference:       {avg_inference_time:.1f}ms")
            print(f"    Avg Total time:      {avg_total_time:.1f}ms")
            print(f"{'='*70}")
        else:
            print("\nNo samples were successfully processed.")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    exit(main())
