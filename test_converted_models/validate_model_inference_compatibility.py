#!/usr/bin/env python3
"""
Run inference and compare results of both models (our converted and Hailo reference).

1. Loads both our custom models AND Hailo reference models
2. Processes real audio (samples/jfk_asknot.wav) with official Whisper preprocessing
3. Runs identical input through both model pairs (encoder → decoder pipeline)
4. Shows detailed side-by-side comparison of outputs:
   - Encoder hidden states comparison
   - Decoder logits comparison
   - Token predictions comparison
   - Decoded text comparison
"""

import numpy as np
import onnxruntime as ort
import librosa
import os
from transformers import WhisperProcessor, WhisperTokenizer

def validate_model(new_encoder_onnx_path, new_decoder_onnx_path, reference_encoder_onnx_path, reference_decoder_onnx_path,test_audio="samples/hello_world.wav"):
    """Complete side-by-side inference comparison"""
    print("🔍 COMPREHENSIVE INFERENCE COMPARISON")
    print("=" * 60)
    print("Comparing our custom models vs Hailo reference models")
    print("=" * 60)

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    # Model paths - use provided paths or defaults
    our_encoder = new_encoder_onnx_path
    our_decoder = new_decoder_onnx_path
    ref_encoder = reference_encoder_onnx_path
    ref_decoder = reference_decoder_onnx_path

    print(f"Using custom encoder: {our_encoder}")
    print(f"Using custom decoder: {our_decoder}")


    our_enc_session = ort.InferenceSession(our_encoder)
    our_dec_session = ort.InferenceSession(our_decoder)
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
    # DECODER COMPARISON
    # ========================================
    print(f"\n🗣️  DECODER COMPARISON")
    print("=" * 30)

    # Standard Whisper task tokens for English transcription
    forced_decoder_ids = [
        50258,  # <|startoftranscript|>
        50259,  # <|en|>
        50359,  # <|transcribe|>
        50363   # <|notimestamps|>
    ]

    decoder_input_ids = np.array([forced_decoder_ids + [50257] * (32 - len(forced_decoder_ids))], dtype=np.int64)

    # Prepare inputs for our decoder
    our_dec_inputs = {
        "decoder_input_ids": decoder_input_ids,
        "encoder_hidden_states": our_enc_output
    }

    # Prepare inputs for reference decoder
    ref_dec_inputs = {
        "decoder_input_ids": decoder_input_ids,
        "encoder_hidden_states": ref_enc_output
    }

    # Run both decoders
    our_dec_output = our_dec_session.run(None, our_dec_inputs)[0]
    ref_dec_output = ref_dec_session.run(None, ref_dec_inputs)[0]

    print(f"✅ Both decoders completed successfully")
    print(f"   Our decoder output: {our_dec_output.shape}")
    print(f"   Ref decoder output: {ref_dec_output.shape}")

    # Compare decoder outputs
    assert our_dec_output.shape == ref_dec_output.shape
    abs_diff = np.abs(our_dec_output - ref_dec_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"\n📊 Decoder Logits Comparison:")
    print(f"   Max absolute difference:  {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")
    print(f"   Our decoder stats:  min={our_dec_output.min():.3f}, max={our_dec_output.max():.3f}, mean={our_dec_output.mean():.3f}")
    print(f"   Ref decoder stats:  min={ref_dec_output.min():.3f}, max={ref_dec_output.max():.3f}, mean={ref_dec_output.mean():.3f}")

    # Get predicted tokens
    our_predicted = np.argmax(our_dec_output[0], axis=-1)
    ref_predicted = np.argmax(ref_dec_output[0], axis=-1)

    print(f"\n🔤 Token Predictions Comparison:")
    print(f"   Our model tokens: {our_predicted[:12].tolist()}")
    print(f"   Ref model tokens: {ref_predicted[:12].tolist()}")

    # Check token agreement
    token_matches = np.sum(our_predicted == ref_predicted)
    total_tokens = len(our_predicted)
    match_percentage = 100 * token_matches / total_tokens
    print(f"   Token agreement: {token_matches}/{total_tokens} ({match_percentage:.1f}%)")

    # Decode both predictions to text
    print(f"\n📝 Decoded Text Comparison:")

    for model_name, predicted_ids in [("Our Model", our_predicted), ("Ref Model", ref_predicted)]:
        meaningful_ids = []
        for i, token_id in enumerate(predicted_ids):
            if i < 4:  # Skip forced tokens
                continue
            if token_id == 50257 or token_id == 50256:  # Skip padding/end tokens
                break
            meaningful_ids.append(int(token_id))

        if meaningful_ids:
            try:
                decoded = tokenizer.decode(meaningful_ids, skip_special_tokens=True)
                print(f"   {model_name}: '{decoded}'")
            except Exception as decode_e:
                print(f"   {model_name}: Raw tokens {meaningful_ids} (decode failed: {decode_e})")
        else:
            print(f"   {model_name}: No meaningful tokens to decode")



    # ========================================
    # CROSS-COMPARISON TEST
    # ========================================
    print(f"\n🔄 CROSS-COMPARISON TEST")
    print("=" * 30)
    print("Testing: Reference encoder output → Our decoder vs Ref decoder (to understand whether decoder leads to same results)")

    # Use reference encoder output with our decoder
    cross_dec_inputs = {
        "decoder_input_ids": decoder_input_ids,
        "encoder_hidden_states": ref_enc_output  # Reference encoder, our decoder
    }

    cross_dec_output = our_dec_session.run(None, cross_dec_inputs)[0]
    cross_predicted = np.argmax(cross_dec_output[0], axis=-1)

    print(f"   Cross-test tokens:     {cross_predicted[:12].tolist()}")
    print(f"   Original ref tokens:   {ref_predicted[:12].tolist()}")

    # Check if using ref encoder output gives same result as ref decoder
    cross_ref_matches = np.sum(cross_predicted == ref_predicted)
    cross_match_percentage = 100 * cross_ref_matches / total_tokens
    print(f"   Agreement with ref decoder: {cross_ref_matches}/{total_tokens} ({cross_match_percentage:.1f}%)")

    # Compare with our original decoder output
    cross_our_matches = np.sum(cross_predicted == our_predicted)
    cross_our_percentage = 100 * cross_our_matches / total_tokens
    print(f"   Agreement with our decoder: {cross_our_matches}/{total_tokens} ({cross_our_percentage:.1f}%)")

    # Decode cross-test result
    cross_meaningful_ids = []
    for i, token_id in enumerate(cross_predicted):
        if i < 4:  # Skip forced tokens
            continue
        if token_id == 50257 or token_id == 50256:  # Skip padding/end tokens
            break
        cross_meaningful_ids.append(int(token_id))

    if cross_meaningful_ids:
        try:
            cross_decoded = tokenizer.decode(cross_meaningful_ids, skip_special_tokens=True)
            print(f"   Cross-test decoded: '{cross_decoded}'")
        except Exception as decode_e:
            print(f"   Cross-test: Raw tokens {cross_meaningful_ids} (decode failed: {decode_e})")
    else:
        print(f"   Cross-test: No meaningful tokens to decode")

    # Analysis
    print(f"\n📊 Cross-Test Analysis:")
    if cross_match_percentage > 90:
        print("   🎯 HIGH AGREEMENT: Decoders behave very similarly")
    else:
        print("   ❌ LOW AGREEMENT: Significant decoder differences")


def main():

    new_encoder_onnx_path = "hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx"
    new_decoder_onnx_path = "hailo_compatible_models/hf_whisper_tiny/whisper_tiny_decoder_10s_hailo_final.onnx"
    reference_encoder_onnx_path = "hailo_reference_models/tiny/tiny-whisper-encoder-10s.onnx"
    reference_decoder_onnx_path = "hailo_reference_models/tiny/tiny-whisper-decoder-10s-seq-32.onnx"

    test_audio="samples/jfk_asknot.wav"
    validate_model(new_encoder_onnx_path, new_decoder_onnx_path, reference_encoder_onnx_path, reference_decoder_onnx_path,test_audio)


if __name__ == "__main__":
    main()