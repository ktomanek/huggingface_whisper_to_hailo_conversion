#!/usr/bin/env python3
"""

Demonstrates 2 decoding technique and compares results:
1. autoregressive decoding without caching to full max length (simulating Hailo NPU behavior)
2. Efficient decoding with KV cache (recommended production approach)


This clearly shows that the hybrid approach is best - encoder on NPU, decoder with KV cache on CPU.

For preprocessing: Use Whisper's official preprocessing
"""

import numpy as np
import onnxruntime as ort
import librosa
import time
from transformers import WhisperProcessor, WhisperTokenizer


# Repetition penalty: 1.0 = no penalty, 1.2 = mild, 1.5 = strong
# Helps prevent hallucination loops when padding is insufficient
REPETITION_PENALTY = 1.0  # Set to 1.2 if you see repetitions

# EOS boosting: If EOS token is within this threshold of top token, choose EOS
# 0.0 = disabled, 0.5 = aggressive, 1.0 = very aggressive
EOS_BOOST_THRESHOLD = 0.0

# ORIG_MAX_LEN = 1500
ORIG_MAX_LEN = 1500

# Debug output: Show detailed penalty/boost decisions
DEBUG_OUTPUT = False  # Set to True to see per-token decisions

def validate_model(new_encoder_onnx_path, reference_encoder_onnx_path, reference_decoder_onnx_path,test_audio="samples/hello_world.wav"):
    """Complete side-by-side inference comparison"""
    print("🔍 COMPREHENSIVE INFERENCE COMPARISON")
    print("=" * 60)
    print("Comparing our custom models vs Hailo reference models")
    print("=" * 60)

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    # Model paths - use provided paths or defaults
    our_encoder = new_encoder_onnx_path
    ref_encoder = reference_encoder_onnx_path
    ref_decoder = reference_decoder_onnx_path

    print(f"Using custom encoder: {our_encoder}")


    our_enc_session = ort.InferenceSession(our_encoder)
    ref_enc_session = ort.InferenceSession(ref_encoder)
    ref_dec_session = ort.InferenceSession(ref_decoder)

    
    print(f"\n🎵 Processing audio: {test_audio}")
    # Load audio
    audio, sr = librosa.load(test_audio, sr=16000, duration=10.0)
    print(f"   📊 Audio: {len(audio)} samples at {sr}Hz")

    # Use Whisper's official preprocessing
    input_features = processor(audio, sampling_rate=16000, return_tensors="np").input_features
    print(f"   🔄 Whisper preprocessing result: {input_features.shape}")

    # Convert from Whisper format [1, 80, 3000] to Hailo format [1, 80, 1, 1000]
    if input_features.shape[-1] >= 1000:
        hailo_input = input_features[:, :, :1000].reshape(1, 80, 1, 1000)
    else:
        pad_width = ((0, 0), (0, 0), (0, 1000 - input_features.shape[-1]))
        padded = np.pad(input_features, pad_width, mode='constant', constant_values=0.0)
        hailo_input = padded.reshape(1, 80, 1, 1000)

    print(f"   ✅ Converted to Hailo format: {hailo_input.shape}")


    # ========================================
    # ENCODER COMPARISON
    # ========================================
    print(f"\n🤖 ENCODER COMPARISON")
    print("=" * 30)

    # Run both encoders with same input

    # Our encoder
    our_enc_input = our_enc_session.get_inputs()[0].name
    our_enc_output = our_enc_session.run(None, {our_enc_input: hailo_input})[0]

    # Reference encoder
    ref_enc_input = ref_enc_session.get_inputs()[0].name
    ref_enc_output = ref_enc_session.run(None, {ref_enc_input: hailo_input})[0]

    print(f"✅ Both encoders completed successfully")
    print(f"   Our encoder output:  {our_enc_output.shape}")
    print(f"   Ref encoder output:  {ref_enc_output.shape}")

    # Compare encoder outputs
    assert our_enc_output.shape == ref_enc_output.shape
    abs_diff = np.abs(our_enc_output - ref_enc_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"\n📊 Encoder Output Comparison:")
    print(f"   Max absolute difference:  {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")
    print(f"   Our encoder stats:  min={our_enc_output.min():.3f}, max={our_enc_output.max():.3f}, mean={our_enc_output.mean():.3f}")
    print(f"   Ref encoder stats:  min={ref_enc_output.min():.3f}, max={ref_enc_output.max():.3f}, mean={ref_enc_output.mean():.3f}")


    # ========================================
    # AUTOREGRESSIVE DECODING TEST (NPU SIMULATION)
    # ========================================
    print(f"\n🔁 AUTOREGRESSIVE DECODING TEST (NPU Simulation)")
    print("=" * 30)
    print("Testing: True token-by-token autoregressive generation")
    print("NOTE: This simulates Hailo NPU decoder behavior:")
    print("  - Processes all 32 positions even when only 1 prediction needed")
    print("  - No KV cache, recomputes attention for all previous tokens")
    print("  - This matches Hailo's fixed-shape ONNX requirements")

    max_new_tokens = 28  # Generate up to 28 new tokens (total 32 with forced)

    def autoregressive_decode(dec_session, encoder_hidden_states, max_new_tokens):
        """Generate tokens one at a time autoregressively"""
        # Start with forced tokens
        generated_tokens = [50258, 50259, 50359, 50363]  # <|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>
        token_times = []  # Track time for each token generation

        for _ in range(max_new_tokens):
            # Pad current tokens to sequence length 32
            current_length = len(generated_tokens)
            padded_tokens = generated_tokens + [50257] * (32 - current_length)
            decoder_input = np.array([padded_tokens], dtype=np.int64)

            # Run decoder
            dec_inputs = {
                "decoder_input_ids": decoder_input,
                "encoder_hidden_states": encoder_hidden_states
            }
            token_start = time.time()
            logits = dec_session.run(None, dec_inputs)[0]
            token_time = time.time() - token_start

            # Get prediction for the next token (at position current_length)
            next_token_logits = logits[0, current_length, :]
            next_token = np.argmax(next_token_logits)

            # Stop if we hit end token or padding token
            if next_token == 50257 or next_token == 50256:
                break

            generated_tokens.append(int(next_token))
            token_times.append(token_time)

        return generated_tokens, token_times

    # Generate with reference decoder
    print(f"\n🤖 Reference decoder autoregressive generation:")
    start_time = time.time()
    ref_autoregressive_tokens, ref_token_times = autoregressive_decode(ref_dec_session, ref_enc_output, max_new_tokens)
    ref_decode_time = time.time() - start_time
    ref_meaningful = ref_autoregressive_tokens[4:]  # Skip forced tokens

    # Print tokens with timing
    print(f"   Generated tokens with timing (ms):")
    for i, (token, token_time) in enumerate(zip(ref_meaningful, ref_token_times)):
        token_text = tokenizer.decode([token], skip_special_tokens=True)
        print(f"      [{i}] {token} '{token_text}' ({token_time*1000:.1f}ms)")

    print(f"   Total time: {ref_decode_time*1000:.1f}ms")
    print(f"   Avg time per token: {np.mean(ref_token_times)*1000:.1f}ms")
    try:
        ref_autoregressive_text = tokenizer.decode(ref_meaningful, skip_special_tokens=True)
        print(f"   Decoded text: '{ref_autoregressive_text}'")
    except Exception as e:
        print(f"   Decode failed: {e}")

    # ========================================
    # EFFICIENT CACHED DECODING TEST (HYBRID ARCHITECTURE)
    # decoder exported and converted with Optimum
    # ========================================
    print(f"\n⚡ EFFICIENT CACHED DECODING TEST (Hybrid: Encoder on NPU, Decoder with KV Cache)")
    print("=" * 30)
    print("Testing: Encoder on NPU (Hailo) + Efficient decoder with KV cache")
    print("  - ✅ Encoder uses NPU (fast, fixed shape - perfect for Hailo)")
    print("  - ✅ Decoder uses KV cache and sees only 1 new token at a time")

    # Load efficient decoder models with KV cache support
    decoder_init_path = "/Users/katrintomanek/dev/onnx_experiments/converted_models/whisper_tiny_onnx/default/decoder_model.onnx"
    decoder_with_past_path = "/Users/katrintomanek/dev/onnx_experiments/converted_models/whisper_tiny_onnx/default/decoder_with_past_model.onnx"

    decoder_init_session = ort.InferenceSession(decoder_init_path)
    decoder_with_past_session = ort.InferenceSession(decoder_with_past_path)

    print(f"\n📦 Loaded efficient decoder models:")
    print(f"   Init decoder: {decoder_init_path.split('/')[-1]}")
    print(f"   Cached decoder: {decoder_with_past_path.split('/')[-1]}")

    def efficient_autoregressive_decode(decoder_init_session, decoder_with_past_session, encoder_hidden_states, max_new_tokens, repetition_penalty=1.0, eos_boost_threshold=0.0, debug_output=False, tokenizer=None):
        """Generate tokens efficiently using KV cache

        Args:
            repetition_penalty: Value > 1.0 penalizes repeated tokens.
                                1.0 = no penalty, 1.2 = mild penalty, 1.5 = strong penalty
            eos_boost_threshold: If EOS score is within this threshold of top score, pick EOS.
                                0.0 = disabled, 0.5 = aggressive, 1.0 = very aggressive
            debug_output: If True, print detailed penalty/boost decisions (slows down timing)
            tokenizer: WhisperTokenizer for decoding token IDs to text (for debug output)
        """
        # Start with forced tokens
        forced_tokens = [50258, 50259, 50359, 50363]
        generated_tokens = forced_tokens.copy()
        token_times = []  # Track time for each token generation

        # Get output names for both sessions
        decoder_outputs = [out.name for out in decoder_init_session.get_outputs()]
        decoder_with_past_outputs = [out.name for out in decoder_with_past_session.get_outputs()]

        past_key_values_dict = {}

        # Account for initial tokens in max_length calculation
        for _ in range(max_new_tokens):
            token_start = time.time()

            if not past_key_values_dict:
                # First pass: process all forced tokens at once and initialize cache
                input_ids = np.array([forced_tokens], dtype=np.int64)

                inputs = {
                    'input_ids': input_ids,
                    'encoder_hidden_states': encoder_hidden_states
                }

                outputs = decoder_init_session.run(None, inputs)
                logits = outputs[0]

                # Store past key values (both decoder and encoder cache)
                for idx, output_name in enumerate(decoder_outputs[1:], 1):
                    if "present" in output_name:
                        past_name = output_name.replace("present.", "past_key_values.")
                        past_key_values_dict[past_name] = outputs[idx]
            else:
                # Subsequent passes: use cached decoder (only process 1 new token at a time)
                # Process only the last token
                current_input_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)

                inputs = {'input_ids': current_input_ids}
                inputs.update(past_key_values_dict)

                outputs = decoder_with_past_session.run(None, inputs)
                logits = outputs[0]

                # Update past key values (only decoder cache changes, encoder cache stays same)
                for idx, output_name in enumerate(decoder_with_past_outputs[1:], 1):
                    if "present" in output_name:
                        past_name = output_name.replace("present.", "past_key_values.")
                        past_key_values_dict[past_name] = outputs[idx]

            token_time = time.time() - token_start

            # Get next token logits
            next_token_logits = logits[0, -1, :].copy()
            original_logits = next_token_logits.copy()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # Get unique tokens from generated sequence (excluding forced tokens)
                tokens_to_penalize = set(generated_tokens[len(forced_tokens):])

                # Debug: show top candidates before penalty
                top_5_indices = np.argsort(original_logits)[-5:][::-1]

                for token_id in tokens_to_penalize:
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty

                if debug_output:
                    # Debug: show top candidates after penalty
                    top_5_indices_after = np.argsort(next_token_logits)[-5:][::-1]

                    # Print comparison for this step
                    print(f"\n      [Debug] Step {len(generated_tokens) - len(forced_tokens)}: Repetition penalty applied")
                    print(f"      Top 5 BEFORE penalty:")
                    for idx in top_5_indices:
                        penalized_mark = "(*)" if idx in tokens_to_penalize else ""
                        token_text = f"'{tokenizer.decode([idx])}'" if tokenizer else ""
                        print(f"        Token {idx} {token_text}: {original_logits[idx]:.3f} {penalized_mark}")
                    print(f"      Top 5 AFTER penalty:")
                    for idx in top_5_indices_after:
                        was_penalized = "(*)" if idx in tokens_to_penalize else ""
                        score_change = f"({original_logits[idx]:.3f} → {next_token_logits[idx]:.3f})" if idx in tokens_to_penalize else ""
                        token_text = f"'{tokenizer.decode([idx])}'" if tokenizer else ""
                        print(f"        Token {idx} {token_text}: {next_token_logits[idx]:.3f} {was_penalized} {score_change}")

            # Greedy decoding with optional EOS boosting
            if eos_boost_threshold > 0.0:
                top_score = np.max(next_token_logits)
                eos_score = next_token_logits[50257]
                score_diff = top_score - eos_score

                if score_diff < eos_boost_threshold:
                    # EOS is competitive, force it to prevent hallucination
                    if debug_output and tokenizer:
                        print(f"      [EOS Boost] Top: {top_score:.3f}, EOS: {eos_score:.3f}, diff: {score_diff:.3f} < {eos_boost_threshold} → Choosing EOS")
                    next_token = 50257
                else:
                    next_token = np.argmax(next_token_logits)
            else:
                next_token = np.argmax(next_token_logits)

            # Stop if we hit end token (50257), padding token (50256), or EOS token (50256/50257)
            # Also check for <|endoftext|> token (50256)
            if next_token in [50256, 50257]:  # EOS or padding
                break

            generated_tokens.append(int(next_token))
            token_times.append(token_time)

        return generated_tokens, token_times

    # Generate with efficient cached decoder using our encoder output
    print(f"\n🚀 Efficient cached generation (our encoder + cached decoder):")

    # EXPERIMENTAL: Pad encoder output from 500 to 1500 to match standard decoder expectations
    print(f"   NOTE: Padding encoder output from {our_enc_output.shape} to [1, 1500, 384]")

    padded_encoder_output = np.pad(
        our_enc_output,
        ((0, 0), (0, ORIG_MAX_LEN - our_enc_output.shape[1]), (0, 0)),
        mode='constant',
        constant_values=0.0
    )
    print(f"   Padded shape: {padded_encoder_output.shape}")


    start_time = time.time()
    efficient_tokens, efficient_token_times = efficient_autoregressive_decode(
        decoder_init_session, decoder_with_past_session, padded_encoder_output, max_new_tokens,
        repetition_penalty=REPETITION_PENALTY,
        eos_boost_threshold=EOS_BOOST_THRESHOLD,
        debug_output=DEBUG_OUTPUT,
        tokenizer=tokenizer
    )
    efficient_decode_time = time.time() - start_time
    efficient_meaningful = efficient_tokens[4:]  # Skip forced tokens

    # Print tokens with timing
    print(f"   Generated tokens with timing (ms):")
    for i, (token, token_time) in enumerate(zip(efficient_meaningful, efficient_token_times)):
        token_text = tokenizer.decode([token], skip_special_tokens=True)
        print(f"      [{i}] {token} '{token_text}' ({token_time*1000:.1f}ms)")

    print(f"   Total time: {efficient_decode_time*1000:.1f}ms")
    print(f"   Avg time per token: {np.mean(efficient_token_times)*1000:.1f}ms")
    try:
        efficient_text = tokenizer.decode(efficient_meaningful, skip_special_tokens=True)
        print(f"   Decoded text: '{efficient_text}'")
    except Exception as e:
        print(f"   Decode failed: {e}")

    # Compare with NPU simulation results
    print(f"\n📊 Efficiency Comparison:")
    print(f"   NPU simulation:")
    print(f"      Generated: {len(ref_autoregressive_tokens)} tokens")
    print(f"      Time: {ref_decode_time*1000:.1f}ms")
    print(f"      Time per token: {ref_decode_time*1000/len(ref_meaningful) if ref_meaningful else 0:.1f}ms")
    print(f"   Efficient cached:")
    print(f"      Generated: {len(efficient_tokens)} tokens")
    print(f"      Time: {efficient_decode_time*1000:.1f}ms")
    print(f"      Time per token: {efficient_decode_time*1000/len(efficient_meaningful) if efficient_meaningful else 0:.1f}ms")

    # Calculate speedup
    if efficient_decode_time > 0:
        speedup = ref_decode_time / efficient_decode_time
        print(f"\n   ⚡ Speedup: {speedup:.2f}x faster with KV cache")
        print(f"      (Efficient cached is {speedup:.2f}x faster than NPU simulation)")

    # Compare token sequences
    min_len = min(len(ref_autoregressive_tokens), len(efficient_tokens))
    token_matches = sum(1 for i in range(min_len) if ref_autoregressive_tokens[i] == efficient_tokens[i])
    print(f"\n   Token agreement: {token_matches}/{min_len} ({100*token_matches/min_len:.1f}%)")


def main():

    new_encoder_onnx_path = "/Users/katrintomanek/dev/huggingface_whisper_to_hailo_conversion/models/hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx"
    default_whisper_decoder = "/Users/katrintomanek/dev/onnx_experiments/converted_models/whisper_tiny_onnx/default/decoder_model.onnx"
    reference_encoder_onnx_path = "/Users/katrintomanek/dev/huggingface_whisper_to_hailo_conversion/models/hailo_reference_models/tiny/tiny-whisper-encoder-10s.onnx"
    reference_decoder_onnx_path = "/Users/katrintomanek/dev/huggingface_whisper_to_hailo_conversion/models/hailo_reference_models/tiny/tiny-whisper-decoder-10s-seq-32.onnx"

    # test_audio="samples/jfk_asknot.wav"
    test_audio="samples/hello_world.wav"
    # test_audio="samples/172.mp3"
    # test_audio="samples/287.mp3"
    validate_model(new_encoder_onnx_path, reference_encoder_onnx_path, reference_decoder_onnx_path,test_audio)


if __name__ == "__main__":
    main()