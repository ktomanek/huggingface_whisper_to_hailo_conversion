#!/usr/bin/env python3
"""
Compare the three anti-hallucination configurations on sample audio.

This script tests all three strategies documented in the efficiency notes:
1. Conservative: Full padding (1500), no interventions
2. Balanced: Minimal padding (700), mild penalty + EOS boost
3. Aggressive: No padding (500), strong penalty + aggressive EOS boost
"""

import sys
import os

# Temporarily modify the hailo_inference_example script to test each config
import importlib.util

def run_config(audio_path, config_name, orig_max_len, rep_penalty, eos_threshold):
    """Run inference with a specific configuration."""
    print(f"\n{'='*70}")
    print(f"TESTING: {config_name}")
    print(f"{'='*70}")

    # Load the module and override configs
    spec = importlib.util.spec_from_file_location("hailo_inference", "hailo_inference_example.py")
    module = importlib.util.module_from_spec(spec)

    # Override configuration
    module.ORIG_MAX_LEN = orig_max_len
    module.REPETITION_PENALTY = rep_penalty
    module.EOS_BOOST_THRESHOLD = eos_threshold
    module.DEBUG_OUTPUT = False

    spec.loader.exec_module(module)

    # Run inference
    transcription, timings = module.run_inference(audio_path)

    return transcription, timings


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_configurations.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("CONFIGURATION COMPARISON")
    print("="*70)

    configs = [
        ("Option 1: Conservative (Most Reliable)", 1500, 1.0, 0.0),
        ("Option 2: Balanced (Recommended)", 700, 1.1, 0.2),
        ("Option 3: Aggressive (Maximum Speed)", 500, 1.2, 0.5),
    ]

    results = []

    for config_name, orig_max_len, rep_penalty, eos_threshold in configs:
        trans, timings = run_config(audio_path, config_name, orig_max_len, rep_penalty, eos_threshold)
        results.append((config_name, trans, timings))

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    for config_name, trans, timings in results:
        print(f"\n{config_name}:")
        print(f"  Total time: {timings['total']:.1f}ms")
        print(f"  Decoder time: {timings['decoder_total']:.1f}ms")
        print(f"  Transcription: '{trans}'")
