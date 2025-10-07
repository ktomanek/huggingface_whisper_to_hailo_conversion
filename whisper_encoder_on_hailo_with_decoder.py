#!/usr/bin/env python3
"""
Whisper Encoder+Decoder Benchmarking on Hailo
Compare full transcription outputs across different encoders
"""
import numpy as np
import os
import sys
import argparse
import time

# Add parent directory to path to import from whisper_encoder_on_hailo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything from whisper_encoder_on_hailo
from whisper_encoder_on_hailo import (
    get_audio, get_audio_orig_onnx,
    run_hef_encoder, run_onnx_encoder
)

import onnxruntime as ort


# ============================================================================
# Decoder Inference Functions
# ============================================================================

def run_onnx_decoder(decoder_init_path, decoder_cached_path, encoder_output, max_length=32, sessions=None):
    """
    Run ONNX decoder with KV-cache to generate transcription.

    Args:
        decoder_init_path: Path to decoder_model.onnx (first pass, no cache)
        decoder_cached_path: Path to decoder_with_past_model.onnx (cached passes)
        encoder_output: Encoder hidden states [batch, seq_len, hidden_dim]
        max_length: Maximum tokens to generate
        sessions: Optional tuple of (session_init, session_cached) to reuse loaded models

    Returns:
        (generated_tokens, inference_time_ms): Generated token IDs and total inference time
    """
    # Load decoder models (or reuse existing sessions)
    if sessions:
        session_init, session_cached = sessions
    else:
        print(f"  [DEBUG] Loading decoder models...")
        load_start = time.time()
        session_init = ort.InferenceSession(decoder_init_path)
        session_cached = ort.InferenceSession(decoder_cached_path)
        load_time = (time.time() - load_start) * 1000
        print(f"  [DEBUG] Model loading took {load_time:.1f}ms")

    # Get output names to map "present" to "past_key_values"
    init_output_names = [output.name for output in session_init.get_outputs()]
    cached_output_names = [output.name for output in session_cached.get_outputs()]

    # Use forced tokens like the working implementation
    forced_tokens = [50258, 50259, 50359, 50363]  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    generated_tokens = forced_tokens.copy()
    past_key_values_dict = {}

    # Run decoder loop
    max_new_tokens = max_length - len(forced_tokens)
    step_times = []

    for step in range(max_new_tokens):
        step_start = time.time()

        if not past_key_values_dict:
            # First pass: process forced tokens and initialize cache
            input_ids = np.array([forced_tokens], dtype=np.int64)
            print(f"  [DEBUG] First pass: processing {len(forced_tokens)} forced tokens with decoder_model.onnx")
            outputs = session_init.run(None, {
                'input_ids': input_ids,
                'encoder_hidden_states': encoder_output
            })
            logits = outputs[0]

            # Store ALL cache outputs, converting "present" to "past_key_values"
            for idx, output_name in enumerate(init_output_names[1:], 1):
                if "present" in output_name:
                    past_name = output_name.replace("present.", "past_key_values.")
                    past_key_values_dict[past_name] = outputs[idx]

            print(f"  [DEBUG] Initialized cache with {len(past_key_values_dict)} tensors")

            # Get logits for last forced token position
            next_token_logits = logits[0, -1, :].copy()
        else:
            # Subsequent passes: process only 1 new token using cache
            current_input_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
            inputs = {'input_ids': current_input_ids}
            inputs.update(past_key_values_dict)

            outputs = session_cached.run(None, inputs)
            logits = outputs[0]

            # Update cache for next iteration
            for idx, output_name in enumerate(cached_output_names[1:], 1):
                if "present" in output_name:
                    past_name = output_name.replace("present.", "past_key_values.")
                    past_key_values_dict[past_name] = outputs[idx]

            next_token_logits = logits[0, -1, :].copy()

        # Apply simple repetition penalty
        repetition_penalty = 1.5
        tokens_to_penalize = set(generated_tokens[len(forced_tokens):])
        for token_id in tokens_to_penalize:
            if next_token_logits[token_id] > 0:
                next_token_logits[token_id] /= repetition_penalty
            else:
                next_token_logits[token_id] *= repetition_penalty

        next_token = int(np.argmax(next_token_logits))
        generated_tokens.append(next_token)

        step_time = (time.time() - step_start) * 1000
        step_times.append(step_time)

        decoder_type = "init" if not past_key_values_dict or step == 0 else "cached"
        print(f"  [STEP {step}] {step_time:.1f}ms ({decoder_type}) | token={next_token}")

        # Check for EOS or end-of-text tokens
        if next_token in [50256, 50257]:  # <|endoftext|>
            print(f"  [DEBUG] EOS token reached")
            break

    total_inference_time = sum(step_times)
    print(f"  [DEBUG] Decoder timing: total={total_inference_time:.1f}ms, avg={np.mean(step_times):.1f}ms, first={step_times[0]:.1f}ms, rest={np.mean(step_times[1:]):.1f}ms")

    return generated_tokens, total_inference_time


def decode_tokens(tokens):
    """Convert token IDs to text using transformers tokenizer."""
    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper Encoder+Decoder Benchmarking on Hailo")
    parser.add_argument("--encoder_hef_file", type=str, default=None,
                        help="Path to encoder HEF file (10s, NHWC, Hailo hardware)")
    parser.add_argument("--encoder_onnx_file", type=str, default=None,
                        help="Path to encoder ONNX file (10s, NCHW, INT8, CPU runtime)")
    parser.add_argument("--encoder_orig_onnx_file", type=str, default=None,
                        help="Path to original encoder ONNX file (30s, NCHW, FP32, standard Whisper)")
    parser.add_argument("--audio_file", type=str, required=True,
                        help="Path to audio file to process")
    parser.add_argument("--decoder_onnx_dir", type=str, required=True,
                        help="Path to directory containing decoder_model.onnx and decoder_with_past_model.onnx")
    parser.add_argument("--multi_process_service", action="store_true",
                        help="Enable multi-process service mode (HEF only)")

    args = parser.parse_args()

    # Validate arguments
    if args.encoder_hef_file is None and args.encoder_onnx_file is None and args.encoder_orig_onnx_file is None:
        print("Error: At least one encoder must be provided")
        return 1

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1

    # Validate decoder paths
    decoder_init_path = os.path.join(args.decoder_onnx_dir, "decoder_model.onnx")
    decoder_cached_path = os.path.join(args.decoder_onnx_dir, "decoder_with_past_model.onnx")

    if not os.path.exists(decoder_init_path):
        print(f"Error: Decoder not found: {decoder_init_path}")
        return 1
    if not os.path.exists(decoder_cached_path):
        print(f"Error: Cached decoder not found: {decoder_cached_path}")
        return 1

    print("="*70)
    print("WHISPER ENCODER+DECODER BENCHMARKING ON HAILO")
    print("="*70)
    print(f"Audio file: {args.audio_file}")
    print(f"Decoder: {args.decoder_onnx_dir}")
    print()

    # Load decoder sessions once to avoid reloading overhead
    print("Loading ONNX decoder models...")
    load_start = time.time()
    session_init = ort.InferenceSession(decoder_init_path)
    session_cached = ort.InferenceSession(decoder_cached_path)
    load_time = (time.time() - load_start) * 1000
    print(f"✅ Decoder models loaded in {load_time:.1f}ms")
    print()

    decoder_sessions = (session_init, session_cached)
    transcriptions = {}

    # -------------------------------------------------------------------------
    # 1. HEF ENCODER
    # -------------------------------------------------------------------------
    if args.encoder_hef_file:
        print("\n" + "="*70)
        print("HEF ENCODER + ONNX DECODER")
        print("="*70)

        print("Preprocessing audio (NHWC format for HEF)...")
        mel_input = get_audio(args.audio_file, is_nhwc=True)
        print(f"  Mel spectrogram shape: {mel_input.shape}")

        print("Running HEF encoder...")
        encoder_output, timings = run_hef_encoder(
            args.encoder_hef_file,
            mel_input,
            num_iterations=1,
            multi_process_service=args.multi_process_service
        )
        encoder_time = timings[0]
        print(f"✅ Encoder time: {encoder_time:.2f}ms")
        print(f"✅ Encoder output shape: {encoder_output.shape}")

        print("Running ONNX decoder...")
        tokens, decoder_time = run_onnx_decoder(decoder_init_path, decoder_cached_path, encoder_output, sessions=decoder_sessions)
        transcription = decode_tokens(tokens)

        print(f"✅ Decoder time: {decoder_time:.2f}ms (inference only, excludes model loading)")
        print(f"✅ Total time: {encoder_time + decoder_time:.2f}ms")
        print(f"✅ Generated tokens: {tokens}")
        print(f"✅ Transcription: \"{transcription}\"")

        transcriptions['hef'] = {
            'text': transcription,
            'tokens': tokens,
            'encoder_time': encoder_time,
            'decoder_time': decoder_time,
            'total_time': encoder_time + decoder_time
        }

    # -------------------------------------------------------------------------
    # 2. ONNX INT8 ENCODER (10s)
    # -------------------------------------------------------------------------
    if args.encoder_onnx_file:
        print("\n" + "="*70)
        print("ONNX INT8 ENCODER (10s) + ONNX DECODER")
        print("="*70)

        print("Preprocessing audio (NCHW format for ONNX)...")
        mel_input = get_audio(args.audio_file, is_nhwc=False)
        print(f"  Mel spectrogram shape: {mel_input.shape}")

        print("Running ONNX encoder...")
        encoder_output, timings = run_onnx_encoder(
            args.encoder_onnx_file,
            mel_input,
            num_iterations=1
        )
        encoder_time = timings[0]
        print(f"✅ Encoder time: {encoder_time:.2f}ms")
        print(f"✅ Encoder output shape: {encoder_output.shape}")

        print("Running ONNX decoder...")
        tokens, decoder_time = run_onnx_decoder(decoder_init_path, decoder_cached_path, encoder_output, sessions=decoder_sessions)
        transcription = decode_tokens(tokens)

        print(f"✅ Decoder time: {decoder_time:.2f}ms (inference only, excludes model loading)")
        print(f"✅ Total time: {encoder_time + decoder_time:.2f}ms")
        print(f"✅ Generated tokens: {tokens}")
        print(f"✅ Transcription: \"{transcription}\"")

        transcriptions['onnx'] = {
            'text': transcription,
            'tokens': tokens,
            'encoder_time': encoder_time,
            'decoder_time': decoder_time,
            'total_time': encoder_time + decoder_time
        }

    # -------------------------------------------------------------------------
    # 3. ORIGINAL ONNX ENCODER (30s, FP32)
    # -------------------------------------------------------------------------
    if args.encoder_orig_onnx_file:
        print("\n" + "="*70)
        print("ORIGINAL ONNX ENCODER (30s, FP32) + ONNX DECODER")
        print("="*70)

        print("Preprocessing audio (NCHW format with WhisperProcessor, 30s)...")
        mel_input = get_audio_orig_onnx(args.audio_file, target_duration=30)
        print(f"  Mel spectrogram shape: {mel_input.shape}")

        print("Running original ONNX encoder...")
        encoder_output, timings = run_onnx_encoder(
            args.encoder_orig_onnx_file,
            mel_input,
            num_iterations=1
        )
        encoder_time = timings[0]
        print(f"✅ Encoder time: {encoder_time:.2f}ms")
        print(f"✅ Encoder output shape: {encoder_output.shape}")

        print("Running ONNX decoder...")
        tokens, decoder_time = run_onnx_decoder(decoder_init_path, decoder_cached_path, encoder_output, sessions=decoder_sessions)
        transcription = decode_tokens(tokens)

        print(f"✅ Decoder time: {decoder_time:.2f}ms (inference only, excludes model loading)")
        print(f"✅ Total time: {encoder_time + decoder_time:.2f}ms")
        print(f"✅ Generated tokens: {tokens}")
        print(f"✅ Transcription: \"{transcription}\"")

        transcriptions['onnx_orig'] = {
            'text': transcription,
            'tokens': tokens,
            'encoder_time': encoder_time,
            'decoder_time': decoder_time,
            'total_time': encoder_time + decoder_time
        }

    # -------------------------------------------------------------------------
    # 4. COMPARISON
    # -------------------------------------------------------------------------
    if len(transcriptions) >= 2:
        print("\n" + "="*70)
        print("TRANSCRIPTION COMPARISON")
        print("="*70)

        labels = {
            'hef': 'HEF (Hailo, 10s, NHWC)',
            'onnx': 'ONNX (CPU, 10s, INT8, NCHW)',
            'onnx_orig': 'ONNX Original (CPU, 30s, FP32, NCHW)'
        }

        print("\nTiming Summary:")
        for key in sorted(transcriptions.keys()):
            t = transcriptions[key]
            print(f"\n{labels[key]}:")
            print(f"  Encoder:  {t['encoder_time']:7.2f}ms")
            print(f"  Decoder:  {t['decoder_time']:7.2f}ms")
            print(f"  Total:    {t['total_time']:7.2f}ms")

        print("\n" + "-"*70)
        print("\nTranscriptions:")
        for key in sorted(transcriptions.keys()):
            t = transcriptions[key]
            print(f"\n{labels[key]}:")
            print(f"  Text: \"{t['text']}\"")
            print(f"  Tokens: {t['tokens'][:10]}{'...' if len(t['tokens']) > 10 else ''}")

        # Check if transcriptions match
        texts = [t['text'] for t in transcriptions.values()]
        tokens_list = [tuple(t['tokens']) for t in transcriptions.values()]

        if len(set(texts)) == 1:
            print(f"\n✅ All transcriptions match: \"{texts[0]}\"")
        else:
            print(f"\n⚠️  Transcriptions differ:")
            for key in sorted(transcriptions.keys()):
                print(f"  {labels[key]:40s} \"{transcriptions[key]['text']}\"")

        if len(set(tokens_list)) == 1:
            print(f"✅ All token sequences match")
        else:
            print(f"⚠️  Token sequences differ")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
