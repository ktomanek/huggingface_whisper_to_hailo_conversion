#!/usr/bin/env python3
"""
Hailo-Optimized Whisper Inference Example

This script demonstrates production-ready inference combining:
- Hailo's efficient preprocessing (PyTorch STFT, cached mel filters)
- 10s encoder converted for Hailo compatibility
- Efficient cached decoder with anti-hallucination strategies

Anti-hallucination strategies:
1. Encoder output padding (to match training distribution)
2. Repetition penalty (discourage token repetition)
3. EOS boosting (proactive early stopping)
"""

import onnxruntime as ort
import numpy as np
import time
from transformers import WhisperTokenizer, WhisperProcessor
import sys
import os
import librosa

# Add hailo_preprocessing to path for mel computation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hailo_preprocessing'))
import preprocessing  # Hailo's efficient mel computation

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose one of three configurations:

# Option 1: Conservative (Most Reliable) - DEFAULT
# - Full padding to match training distribution
# - No additional interventions needed
ORIG_MAX_LEN = 1500           # Full padding
REPETITION_PENALTY = 1.0      # Disabled
EOS_BOOST_THRESHOLD = 0.0     # Disabled

# # Option 2: Balanced (Recommended)
# # - Minimal safe padding with safety nets
# # - Good trade-off between speed and reliability
# # ORIG_MAX_LEN = 700            # Minimal safe padding
# # REPETITION_PENALTY = 1.1      # Mild penalty
# # EOS_BOOST_THRESHOLD = 0.2     # Conservative boost

# # Option 3: Aggressive (Maximum Speed)
# # - No padding, rely on interventions
# # - Lowest compute overhead
# # ORIG_MAX_LEN = 500            # No padding
# # REPETITION_PENALTY = 1.2      # Stronger penalty
# # EOS_BOOST_THRESHOLD = 0.5     # Aggressive boost

DEBUG_OUTPUT = False          # Toggle detailed debug output
MAX_NEW_TOKENS = 28           # 32 total - 4 forced tokens

# =============================================================================
# MODEL PATHS
# =============================================================================

# 10s encoder (Hailo-optimized)
ENCODER_10S_PATH = "/Users/katrintomanek/dev/huggingface_whisper_to_hailo_conversion/hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx"

# 30s encoder (standard ONNX)
ENCODER_30S_PATH = "/Users/katrintomanek/dev/onnx_experiments/converted_models/whisper_tiny_onnx/default/encoder_model.onnx"

# Decoders (shared by both encoders)
DECODER_INIT_PATH = "/Users/katrintomanek/dev/onnx_experiments/converted_models/whisper_tiny_onnx/default/decoder_model.onnx"
DECODER_CACHED_PATH = "/Users/katrintomanek/dev/onnx_experiments/converted_models/whisper_tiny_onnx/default/decoder_with_past_model.onnx"

# =============================================================================
# EFFICIENT CACHED DECODING
# =============================================================================

def efficient_autoregressive_decode(decoder_init_session, decoder_with_past_session,
                                   encoder_hidden_states, max_new_tokens,
                                   repetition_penalty=1.0, eos_boost_threshold=0.0,
                                   debug_output=False, tokenizer=None):
    """
    Generate tokens using KV cache for efficient autoregressive decoding.

    Args:
        decoder_init_session: ONNX session for initial decoder pass
        decoder_with_past_session: ONNX session for cached decoder
        encoder_hidden_states: Encoder output [1, seq_len, 384]
        max_new_tokens: Maximum tokens to generate
        repetition_penalty: Penalty for repeated tokens (1.0 = disabled, 1.5 = strong)
        eos_boost_threshold: Force EOS if within threshold of top token (0.0 = disabled)
        debug_output: Show detailed generation info
        tokenizer: Tokenizer for debug output

    Returns:
        generated_tokens: List of token IDs
        token_times: List of per-token generation times (ms)
    """
    # Whisper forced decoder start tokens
    forced_tokens = [50258, 50259, 50359, 50363]  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    generated_tokens = forced_tokens.copy()
    token_times = []
    past_key_values_dict = {}

    # Get output names for cache mapping
    decoder_outputs = [output.name for output in decoder_init_session.get_outputs()]
    decoder_with_past_outputs = [output.name for output in decoder_with_past_session.get_outputs()]

    for step in range(max_new_tokens):
        token_start = time.time()

        if not past_key_values_dict:
            # First pass: process forced tokens and initialize cache
            input_ids = np.array([forced_tokens], dtype=np.int64)
            outputs = decoder_init_session.run(None, {
                'input_ids': input_ids,
                'encoder_hidden_states': encoder_hidden_states
            })
            logits = outputs[0]

            # Store ALL cache outputs (both decoder and encoder cross-attention)
            for idx, output_name in enumerate(decoder_outputs[1:], 1):
                if "present" in output_name:
                    past_name = output_name.replace("present.", "past_key_values.")
                    past_key_values_dict[past_name] = outputs[idx]

            # Get logits for last forced token position
            next_token_logits = logits[0, -1, :].copy()
        else:
            # Subsequent passes: process only 1 new token using cache
            current_input_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
            inputs = {'input_ids': current_input_ids}
            inputs.update(past_key_values_dict)

            outputs = decoder_with_past_session.run(None, inputs)
            logits = outputs[0]

            # Update cache for next iteration
            for idx, output_name in enumerate(decoder_with_past_outputs[1:], 1):
                if "present" in output_name:
                    past_name = output_name.replace("present.", "past_key_values.")
                    past_key_values_dict[past_name] = outputs[idx]

            next_token_logits = logits[0, -1, :].copy()

        token_time = (time.time() - token_start) * 1000  # Convert to ms

        # === Anti-Hallucination Strategy 1: Repetition Penalty ===
        if repetition_penalty != 1.0:
            # Only penalize tokens we've generated (not forced tokens)
            tokens_to_penalize = set(generated_tokens[len(forced_tokens):])
            for token_id in tokens_to_penalize:
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= repetition_penalty
                else:
                    next_token_logits[token_id] *= repetition_penalty

        # === Anti-Hallucination Strategy 2: EOS Boosting ===
        if eos_boost_threshold > 0.0:
            top_score = np.max(next_token_logits)
            eos_score = next_token_logits[50257]  # EOS token

            if debug_output:
                top_token = np.argmax(next_token_logits)
                top_text = tokenizer.decode([top_token]) if tokenizer else ""
                print(f"   Top token {top_token} '{top_text}': {top_score:.3f}")
                print(f"   EOS score: {eos_score:.3f}, diff: {top_score - eos_score:.3f}")

            # Force EOS if it's competitive with top token
            if (top_score - eos_score) < eos_boost_threshold:
                next_token = 50257
                if debug_output:
                    print(f"   → Boosting EOS (within {eos_boost_threshold} threshold)")
            else:
                next_token = np.argmax(next_token_logits)
        else:
            next_token = np.argmax(next_token_logits)

        # Check for EOS or end-of-text tokens
        if next_token in [50256, 50257]:
            if debug_output or not DEBUG_OUTPUT:
                token_text = tokenizer.decode([next_token]) if tokenizer else ""
                print(f"      [{step}] {next_token} '{token_text}' ({token_time:.1f}ms) [STOP]")
            break

        generated_tokens.append(int(next_token))
        token_times.append(token_time)

        if debug_output or not DEBUG_OUTPUT:
            token_text = tokenizer.decode([next_token]) if tokenizer else ""
            print(f"      [{step}] {next_token} '{token_text}' ({token_time:.1f}ms)")

    return generated_tokens, token_times


# =============================================================================
# PREPROCESSING METHODS
# =============================================================================

# Cache the WhisperProcessor (loaded once, reused)
_cached_processor = None

def get_whisper_processor():
    """Get cached WhisperProcessor instance."""
    global _cached_processor
    if _cached_processor is None:
        _cached_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    return _cached_processor


def preprocess_huggingface(audio_path, target_duration=10):
    """
    HuggingFace preprocessing approach using WhisperProcessor.

    Args:
        audio_path: Path to audio file
        target_duration: Target duration in seconds (10 or 30)

    Returns:
        mel_input: Numpy array [1, 80, 1, N] where N=1000 for 10s, 3000 for 30s
    """
    # Load audio using librosa (HuggingFace default)
    audio = librosa.load(audio_path, sr=16000, mono=True)[0]

    # Check duration and crop if necessary
    target_length = 16000 * target_duration
    audio_duration = len(audio) / 16000

    if len(audio) > target_length:
        print(f"   ⚠️  Audio is {audio_duration:.1f}s, cropping to {target_duration}s")
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

    # Use cached WhisperProcessor to generate mel spectrogram
    processor = get_whisper_processor()
    inputs = processor(audio, sampling_rate=16000, return_tensors="np")
    mel_input = inputs.input_features  # [1, 80, 3000]

    # Reshape based on target duration
    if target_duration == 10:
        # [1, 80, 1, 1000] for 10s encoder
        mel_input = mel_input[:, :, :1000]
        mel_input = np.expand_dims(mel_input, axis=2)
    elif target_duration == 30:
        # [1, 80, 3000] for 30s encoder (no reshape needed)
        pass
    else:
        raise ValueError(f"Unsupported target_duration: {target_duration}")

    return mel_input


def preprocess_hailo(audio_path):
    """
    Hailo's efficient preprocessing using PyTorch STFT and cached mel filters.
    Uses librosa for audio loading instead of ffmpeg for file-based processing.

    Returns:
        mel_input: Numpy array [1, 80, 1, 1000]
    """
    # Load audio using librosa (faster for file-based processing)
    audio = librosa.load(audio_path, sr=16000, mono=True)[0]

    # Check duration and warn if cropping
    sample_rate = 16000
    audio_duration = len(audio) / sample_rate

    if audio_duration > 10.0:
        print(f"   ⚠️  Audio is {audio_duration:.1f}s, cropping to 10s")

    # Convert to torch tensor for Hailo's mel computation
    import torch
    audio_torch = torch.from_numpy(audio)

    # Generate mel spectrogram using PyTorch STFT with cached filters
    # The preprocessing.preprocess function will handle cropping to max_duration
    mel_spectrograms = preprocessing.preprocess(
        audio_torch,
        is_nhwc=False,  # NCHW format for encoder
        chunk_length=10,  # 10 second chunks
        chunk_offset=0,
        max_duration=10  # Crop to 10 seconds
    )

    return mel_spectrograms[0]  # [1, 80, 1, 1000]


# =============================================================================
# MAIN INFERENCE PIPELINE
# =============================================================================

def run_inference(audio_path, use_hailo_preprocessing=True, encoder_path=None, encoder_duration=10):
    """
    Complete inference pipeline with timing breakdown.

    Args:
        audio_path: Path to audio file
        use_hailo_preprocessing: If True, use Hailo's efficient preprocessing.
                                If False, use HuggingFace preprocessing.
        encoder_path: Path to encoder ONNX file (if None, uses 10s encoder)
        encoder_duration: Expected duration of encoder input (10 or 30 seconds)

    Returns:
        transcription: Decoded text
        timings: Dictionary of timing measurements
    """
    timings = {}

    # Default to 10s encoder if not specified
    if encoder_path is None:
        encoder_path = ENCODER_10S_PATH
        encoder_duration = 10

    preprocessing_method = "Hailo" if use_hailo_preprocessing else "HuggingFace"
    encoder_type = f"{encoder_duration}s encoder"

    print(f"\n{'='*70}")
    print("HAILO-OPTIMIZED WHISPER INFERENCE")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Preprocessing method: {preprocessing_method}")
    print(f"  Encoder: {encoder_type}")
    print(f"  Encoder output padding: {ORIG_MAX_LEN} positions")
    print(f"  Repetition penalty: {REPETITION_PENALTY}")
    print(f"  EOS boost threshold: {EOS_BOOST_THRESHOLD}")
    print(f"  Debug output: {DEBUG_OUTPUT}")
    print()

    # -------------------------------------------------------------------------
    # 1. PREPROCESSING
    # -------------------------------------------------------------------------
    print(f"📦 Stage 1: Audio Preprocessing ({preprocessing_method} approach)")
    preprocess_start = time.time()

    if use_hailo_preprocessing:
        mel_input = preprocess_hailo(audio_path)
    else:
        mel_input = preprocess_huggingface(audio_path, target_duration=encoder_duration)

    preprocess_time = (time.time() - preprocess_start) * 1000
    timings['preprocess'] = preprocess_time

    print(f"   ✅ Mel spectrogram shape: {mel_input.shape}")
    print(f"   ⏱️  Time: {preprocess_time:.1f}ms")
    print()

    # -------------------------------------------------------------------------
    # 2. ENCODER
    # -------------------------------------------------------------------------
    print(f"📦 Stage 2: Encoder Inference ({encoder_duration}s)")
    encoder_start = time.time()

    encoder_session = ort.InferenceSession(encoder_path)

    # Get input name dynamically
    encoder_input_name = encoder_session.get_inputs()[0].name
    encoder_output = encoder_session.run(None, {encoder_input_name: mel_input})[0]

    encoder_time = (time.time() - encoder_start) * 1000
    timings['encoder'] = encoder_time
    timings['encoder_duration'] = encoder_duration

    print(f"   ✅ Encoder output shape: {encoder_output.shape}")
    print(f"   ⏱️  Time: {encoder_time:.1f}ms")
    print()

    # -------------------------------------------------------------------------
    # 3. ENCODER OUTPUT PADDING (Anti-hallucination strategy)
    # -------------------------------------------------------------------------
    print("📦 Stage 3: Encoder Output Padding")
    padding_start = time.time()

    original_length = encoder_output.shape[1]
    if ORIG_MAX_LEN > original_length:
        padded_encoder_output = np.pad(
            encoder_output,
            ((0, 0), (0, ORIG_MAX_LEN - original_length), (0, 0)),
            mode='constant',
            constant_values=0.0
        )
        print(f"   ✅ Padded from {original_length} → {ORIG_MAX_LEN} positions")
    else:
        padded_encoder_output = encoder_output
        print(f"   ℹ️  No padding needed (target: {ORIG_MAX_LEN})")

    padding_time = (time.time() - padding_start) * 1000
    timings['padding'] = padding_time

    print(f"   ⏱️  Time: {padding_time:.1f}ms")
    print()

    # -------------------------------------------------------------------------
    # 4. DECODER (Efficient cached generation)
    # -------------------------------------------------------------------------
    print("📦 Stage 4: Decoder Inference (Cached)")
    print(f"   Loading decoder models...")
    decoder_init_session = ort.InferenceSession(DECODER_INIT_PATH)
    decoder_cached_session = ort.InferenceSession(DECODER_CACHED_PATH)

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    print(f"   Generating tokens (max: {MAX_NEW_TOKENS}):")
    decoder_start = time.time()

    generated_tokens, token_times = efficient_autoregressive_decode(
        decoder_init_session,
        decoder_cached_session,
        padded_encoder_output,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        eos_boost_threshold=EOS_BOOST_THRESHOLD,
        debug_output=DEBUG_OUTPUT,
        tokenizer=tokenizer
    )

    decoder_time = (time.time() - decoder_start) * 1000
    timings['decoder_total'] = decoder_time
    timings['decoder_tokens'] = token_times

    print()
    print(f"   ✅ Generated {len(generated_tokens)} tokens total")
    print(f"   ⏱️  Total decoder time: {decoder_time:.1f}ms")
    if token_times:
        print(f"   ⏱️  Average per token: {np.mean(token_times):.1f}ms")
        print(f"   ⏱️  First token: {token_times[0]:.1f}ms")
        if len(token_times) > 1:
            print(f"   ⏱️  Subsequent tokens: {np.mean(token_times[1:]):.1f}ms")
    print()

    # -------------------------------------------------------------------------
    # 5. DECODE TEXT
    # -------------------------------------------------------------------------
    transcription = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # -------------------------------------------------------------------------
    # 6. TIMING SUMMARY
    # -------------------------------------------------------------------------
    total_time = preprocess_time + encoder_time + padding_time + decoder_time
    timings['total'] = total_time

    print(f"{'='*70}")
    print("TIMING BREAKDOWN:")
    print(f"{'='*70}")
    print(f"  Preprocessing:  {preprocess_time:>8.1f}ms  ({preprocess_time/total_time*100:>5.1f}%)")
    print(f"  Encoder:        {encoder_time:>8.1f}ms  ({encoder_time/total_time*100:>5.1f}%)")
    print(f"  Padding:        {padding_time:>8.1f}ms  ({padding_time/total_time*100:>5.1f}%)")
    print(f"  Decoder:        {decoder_time:>8.1f}ms  ({decoder_time/total_time*100:>5.1f}%)")
    print(f"  {'─'*68}")
    print(f"  TOTAL:          {total_time:>8.1f}ms")
    print(f"{'='*70}")
    print()

    return transcription, timings


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hailo-optimized Whisper inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Hailo preprocessing with 10s encoder (default, fastest)
  python hailo_inference_example.py audio.wav

  # Use HuggingFace preprocessing (for comparison)
  python hailo_inference_example.py audio.wav --preprocessing huggingface

  # Compare both preprocessing methods
  python hailo_inference_example.py audio.wav --compare

  # Compare 10s vs 30s encoder performance
  python hailo_inference_example.py audio.wav --compare-encoders
        """
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument(
        "--preprocessing",
        choices=["hailo", "huggingface"],
        default="hailo",
        help="Preprocessing method to use (default: hailo)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both preprocessing methods and compare timing"
    )
    parser.add_argument(
        "--compare-encoders",
        action="store_true",
        help="Compare 10s vs 30s encoder performance"
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"❌ Error: Audio file not found: {args.audio_path}")
        sys.exit(1)

    if args.compare_encoders:
        # Compare 10s vs 30s encoders
        print("\n" + "="*70)
        print("ENCODER COMPARISON: 10s vs 30s")
        print("="*70)

        # Warmup: load processors and mel filters
        print("\n🔥 Warming up (loading processors and mel filters)...")
        _ = get_whisper_processor()
        _ = preprocess_hailo(args.audio_path)  # Load mel filters
        _ = preprocess_huggingface(args.audio_path, target_duration=10)  # Warmup HF
        print("   ✅ All preprocessors warmed up\n")

        print("--- Running with 10s encoder (Hailo-optimized) ---")
        trans_10s, timings_10s = run_inference(
            args.audio_path,
            use_hailo_preprocessing=True,
            encoder_path=ENCODER_10S_PATH,
            encoder_duration=10
        )

        print("\n--- Running with 30s encoder (standard ONNX) ---")
        trans_30s, timings_30s = run_inference(
            args.audio_path,
            use_hailo_preprocessing=False,  # Use HF preprocessing for 30s
            encoder_path=ENCODER_30S_PATH,
            encoder_duration=30
        )

        # Comparison summary
        print("\n" + "="*70)
        print("ENCODER COMPARISON SUMMARY")
        print("="*70)

        print(f"\n10s Encoder:")
        print(f"  Preprocessing: {timings_10s['preprocess']:.1f}ms")
        print(f"  Encoder: {timings_10s['encoder']:.1f}ms")
        print(f"  Decoder: {timings_10s['decoder_total']:.1f}ms")
        print(f"  Total: {timings_10s['total']:.1f}ms")
        print(f"  Transcription: '{trans_10s}'")

        print(f"\n30s Encoder:")
        print(f"  Preprocessing: {timings_30s['preprocess']:.1f}ms")
        print(f"  Encoder: {timings_30s['encoder']:.1f}ms")
        print(f"  Decoder: {timings_30s['decoder_total']:.1f}ms")
        print(f"  Total: {timings_30s['total']:.1f}ms")
        print(f"  Transcription: '{trans_30s}'")

        encoder_speedup = timings_30s['encoder'] / timings_10s['encoder']
        total_speedup = timings_30s['total'] / timings_10s['total']

        print(f"\n⚡ Performance:")
        print(f"   Encoder: 10s is {encoder_speedup:.2f}x faster ({timings_10s['encoder']:.1f}ms vs {timings_30s['encoder']:.1f}ms)")
        print(f"   Overall: 10s pipeline is {total_speedup:.2f}x faster ({timings_10s['total']:.1f}ms vs {timings_30s['total']:.1f}ms)")

        # Breakdown
        encoder_savings = timings_30s['encoder'] - timings_10s['encoder']
        total_savings = timings_30s['total'] - timings_10s['total']
        print(f"\n💾 Savings:")
        print(f"   Encoder time saved: {encoder_savings:.1f}ms")
        print(f"   Total time saved: {total_savings:.1f}ms")

        # Check transcriptions match
        if trans_10s.strip().lower() == trans_30s.strip().lower():
            print(f"\n✅ Both encoders produce equivalent transcriptions (ignoring case/punctuation)")
        else:
            print(f"\n⚠️  Transcriptions differ slightly")
            print(f"   This is expected - 30s encoder may handle punctuation differently")

    elif args.compare:
        # Run both methods and compare
        print("\n" + "="*70)
        print("PREPROCESSING COMPARISON")
        print("="*70)

        # Warm up: Load processors/filters before timing measurements
        print("\n🔥 Warming up (loading cached processors/filters)...")
        _ = get_whisper_processor()  # Load HF processor
        print("   ✅ WhisperProcessor loaded and cached")

        # First Hailo call will load mel filters via @lru_cache
        mel_hailo_warmup = preprocess_hailo(args.audio_path)
        print("   ✅ Mel filters loaded and cached")

        # First, verify both preprocessing methods produce identical output
        print("\n🔍 Verifying preprocessing methods produce identical mel spectrograms...")
        mel_hailo = preprocess_hailo(args.audio_path)
        mel_hf = preprocess_huggingface(args.audio_path)

        # Check shapes match
        assert mel_hailo.shape == mel_hf.shape, \
            f"Shape mismatch: Hailo {mel_hailo.shape} vs HF {mel_hf.shape}"

        # Check values are close (allowing for minor numerical differences)
        max_diff = np.abs(mel_hailo - mel_hf).max()
        mean_diff = np.abs(mel_hailo - mel_hf).mean()

        print(f"   ✅ Shapes match: {mel_hailo.shape}")
        print(f"   ✅ Max absolute difference: {max_diff:.6f}")
        print(f"   ✅ Mean absolute difference: {mean_diff:.6f}")

        # Assert they're very close (within reasonable numerical precision)
        assert max_diff < 0.01, \
            f"Mel spectrograms differ too much! Max diff: {max_diff}"

        print(f"   ✅ Preprocessing methods produce equivalent outputs!\n")

        print("\n--- Running with Hailo preprocessing ---")
        trans_hailo, timings_hailo = run_inference(args.audio_path, use_hailo_preprocessing=True)

        print("\n--- Running with HuggingFace preprocessing ---")
        trans_hf, timings_hf = run_inference(args.audio_path, use_hailo_preprocessing=False)

        # Comparison summary
        print("\n" + "="*70)
        print("PREPROCESSING COMPARISON SUMMARY")
        print("="*70)
        print(f"\nHailo Preprocessing:")
        print(f"  Time: {timings_hailo['preprocess']:.1f}ms")
        print(f"  Total: {timings_hailo['total']:.1f}ms")
        print(f"  Transcription: '{trans_hailo}'")

        print(f"\nHuggingFace Preprocessing:")
        print(f"  Time: {timings_hf['preprocess']:.1f}ms")
        print(f"  Total: {timings_hf['total']:.1f}ms")
        print(f"  Transcription: '{trans_hf}'")

        if timings_hailo['preprocess'] < timings_hf['preprocess']:
            speedup = timings_hf['preprocess'] / timings_hailo['preprocess']
            print(f"\n⚡ Hailo preprocessing is {speedup:.2f}x faster")
            print(f"   (Hailo: {timings_hailo['preprocess']:.1f}ms vs HuggingFace: {timings_hf['preprocess']:.1f}ms)")
            print(f"   Advantage: PyTorch STFT with cached mel filters vs standard mel computation")
        else:
            slowdown = timings_hailo['preprocess'] / timings_hf['preprocess']
            print(f"\n⚠️  Hailo preprocessing is {slowdown:.2f}x slower")
            print(f"   Note: This shouldn't happen with librosa loading. Check configuration.")

    else:
        # Run single method
        use_hailo = (args.preprocessing == "hailo")
        transcription, timings = run_inference(args.audio_path, use_hailo_preprocessing=use_hailo)

        print("📝 TRANSCRIPTION:")
        print(f"   '{transcription}'")
        print()
