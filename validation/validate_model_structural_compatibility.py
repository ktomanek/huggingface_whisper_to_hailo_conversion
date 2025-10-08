#!/usr/bin/env python3

import numpy as np
np.random.seed(42)

import onnxruntime as ort
import onnx
import os
import sys

def validate_model(new_encoder_onnx_path, new_decoder_onnx_path, reference_encoder_onnx_path, reference_decoder_onnx_path):
    """Validate our final custom models against Hailo baseline"""

    print("📂 Final Model Files:")
    for name, path in [("New Encoder", new_encoder_onnx_path), ("New Decoder", new_decoder_onnx_path)]:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f" * {name}: {size_mb:.1f} MB")
    for name, path in [("Hailo Reference Encoder", reference_encoder_onnx_path), ("Hailo Reference Decoder", reference_decoder_onnx_path)]:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f" * {name}: {size_mb:.1f} MB")


    print(f"\n🤖 ENCODER VALIDATION")
    print("-" * 25)

    our_enc_session = ort.InferenceSession(new_encoder_onnx_path)
    baseline_enc_session = ort.InferenceSession(reference_encoder_onnx_path)

    # Check specifications
    our_enc_input = our_enc_session.get_inputs()[0]
    our_enc_output = our_enc_session.get_outputs()[0]
    baseline_enc_input = baseline_enc_session.get_inputs()[0]
    baseline_enc_output = baseline_enc_session.get_outputs()[0]

    print(f"\nInput compatibility:")
    print(f"   Our model:  {our_enc_input.name} {our_enc_input.shape}")
    print(f"   Baseline:   {baseline_enc_input.name} {baseline_enc_input.shape}")
    input_match = (our_enc_input.name == baseline_enc_input.name and
                    our_enc_input.shape == baseline_enc_input.shape)
    print(f" --> Result: {'✅ MATCH' if input_match else '❌ MISMATCH'}")

    print(f"\nOutput compatibility:")
    print(f"   Our model:  {our_enc_output.name} {our_enc_output.shape}")
    print(f"   Baseline:   {baseline_enc_output.name} {baseline_enc_output.shape}")
    output_match = (our_enc_output.name == baseline_enc_output.name and
                    our_enc_output.shape == baseline_enc_output.shape)
    print(f" --> Result: {'✅ MATCH' if output_match else '❌ MISMATCH'}")

    # Run some random input through both models and compare outputs
    print(f"\n Inference comparison:")
    test_input = np.random.randn(1, 80, 1, 1000).astype(np.float32)
    our_enc_result = our_enc_session.run(None, {our_enc_input.name: test_input})[0]
    baseline_enc_result = baseline_enc_session.run(None, {baseline_enc_input.name: test_input})[0]
    print(f"   Our model output shape: {our_enc_result.shape}")
    print(f"   Baseline output shape:  {baseline_enc_result.shape}")
    # Compare numerical outputs
    if our_enc_result.shape != baseline_enc_result.shape:
        raise ValueError("Output shapes of our and reference encoder incompatible!")
    abs_diff = np.abs(our_enc_result - baseline_enc_result)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"   Max absolute difference:  {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")
    print(f"   Our model stats:  min={our_enc_result.min():.3f}, max={our_enc_result.max():.3f}, mean={our_enc_result.mean():.3f}")
    print(f"   Baseline stats:   min={baseline_enc_result.min():.3f}, max={baseline_enc_result.max():.3f}, mean={baseline_enc_result.mean():.3f}")

    significant_inference_diff = max_diff > 2
    if significant_inference_diff:
        print(f" --> ❌ Significant numerical differences detected in encoder outputs!")
    else:
        print(f" --> ✅ Numerical outputs of both encoders closely match")
            
    encoder_compatible = input_match and output_match


    # Load and validate decoder
    print(f"\n🗣️  DECODER VALIDATION")
    print("-" * 25)
    our_dec_session = ort.InferenceSession(new_decoder_onnx_path)
    baseline_dec_session = ort.InferenceSession(reference_decoder_onnx_path)

    # Check specifications
    our_dec_inputs = {inp.name: inp for inp in our_dec_session.get_inputs()}
    our_dec_output = our_dec_session.get_outputs()[0]
    baseline_dec_inputs = {inp.name: inp for inp in baseline_dec_session.get_inputs()}
    baseline_dec_output = baseline_dec_session.get_outputs()[0]

    print(f"\n Input compatibility:")
    inputs_match = True
    for name in our_dec_inputs:
        if name in baseline_dec_inputs:
            our_shape = our_dec_inputs[name].shape
            baseline_shape = baseline_dec_inputs[name].shape
            match = (our_shape == baseline_shape)
            print(f" --> {name}: {our_shape} vs {baseline_shape} {'✅' if match else '❌'}")
            inputs_match &= match
        else:
            print(f" --> {name}: Missing in baseline ❌")
            inputs_match = False

    print(f"\n Output compatibility:")
    print(f"   Our model:  {our_dec_output.name} {our_dec_output.shape}")
    print(f"   Baseline:   {baseline_dec_output.name} {baseline_dec_output.shape}")
    output_match = (our_dec_output.name == baseline_dec_output.name and
                    our_dec_output.shape == baseline_dec_output.shape)
    print(f" --> Result: {'✅ MATCH' if output_match else '❌ MISMATCH'}")

    # Test inference with same inputs on both models
    print(f"\n Inference comparison:")
    decoder_input_ids = np.array([[50258] + [50259] * 31], dtype=np.int64)
    encoder_hidden_states = np.random.randn(1, 500, 384).astype(np.float32)

    # Prepare inputs for our model
    our_inputs = {}
    for name in our_dec_inputs.items():
        if "decoder_input_ids" in name[0].lower():
            our_inputs[name[0]] = decoder_input_ids
        elif "encoder_hidden_states" in name[0].lower():
            our_inputs[name[0]] = encoder_hidden_states

    # Prepare inputs for baseline model
    baseline_inputs = {}
    for name in baseline_dec_inputs.items():
        if "decoder_input_ids" in name[0].lower() or name[0] == "input_ids":
            baseline_inputs[name[0]] = decoder_input_ids
        elif "encoder_hidden_states" in name[0].lower():
            baseline_inputs[name[0]] = encoder_hidden_states

    our_dec_result = our_dec_session.run(None, our_inputs)[0]
    baseline_dec_result = baseline_dec_session.run(None, baseline_inputs)[0]

    print(f"   Our model output shape: {our_dec_result.shape}")
    print(f"   Baseline output shape:  {baseline_dec_result.shape}")

    # Compare numerical outputs
    if our_dec_result.shape != baseline_dec_result.shape:
        raise ValueError("Output shapes of our and reference decoder incompatible!")            
    abs_diff = np.abs(our_dec_result - baseline_dec_result)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"   Max absolute difference:  {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")
    print(f"   Our model stats:  min={our_dec_result.min():.3f}, max={our_dec_result.max():.3f}, mean={our_dec_result.mean():.3f}")
    print(f"   Baseline stats:   min={baseline_dec_result.min():.3f}, max={baseline_dec_result.max():.3f}, mean={baseline_dec_result.mean():.3f}")

    # Show predicted tokens
    our_predicted_ids = np.argmax(our_dec_result[0], axis=-1)
    baseline_predicted_ids = np.argmax(baseline_dec_result[0], axis=-1)

    print(f"   🔤 Predicted tokens comparison (first 10 positions):")
    print(f"      Our model:  {our_predicted_ids[:10].tolist()}")
    print(f"      Baseline:   {baseline_predicted_ids[:10].tolist()}")

    # Check if predictions match
    token_matches = np.sum(our_predicted_ids == baseline_predicted_ids)
    total_tokens = len(our_predicted_ids)
    match_percentage = 100 * token_matches / total_tokens
    print(f"   📊 Token prediction agreement: {token_matches}/{total_tokens} ({match_percentage:.1f}%)")

    significant_inference_diff = match_percentage < 98
    if significant_inference_diff:
        print(f" --> ❌ Significant differences detected in decoder outputs!")
    else:
        print(f" --> ✅ Outputs of both decoders closely match")


    decoder_compatible = inputs_match and output_match

    # Final summary
    print(f"\n" + "="*50)
    print("🎉 FINAL COMPATIBILITY SUMMARY")
    print("="*50)
    print(f"🤖 Encoder:  {'✅ COMPATIBLE' if encoder_compatible else '❌ INCOMPATIBLE'}")
    print(f"🗣️  Decoder:  {'✅ COMPATIBLE' if decoder_compatible else '❌ INCOMPATIBLE'}")


def main():

    new_encoder_onnx_path = "hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx"
    new_decoder_onnx_path = "hailo_compatible_models/hf_whisper_tiny/whisper_tiny_decoder_10s_hailo_final.onnx"
    reference_encoder_onnx_path = "hailo_reference_models/tiny/tiny-whisper-encoder-10s.onnx"
    reference_decoder_onnx_path = "hailo_reference_models/tiny/tiny-whisper-decoder-10s-seq-32.onnx"

    validate_model(new_encoder_onnx_path, new_decoder_onnx_path, reference_encoder_onnx_path, reference_decoder_onnx_path)

if __name__ == "__main__":
    main()

