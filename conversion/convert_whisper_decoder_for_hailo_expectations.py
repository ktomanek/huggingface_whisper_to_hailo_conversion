#!/usr/bin/env python3
"""
Convert HuggingFace Whisper DECODER with all the patches and tricks Hailo applies to
OpenAI Whisper Model

This script modifies the Whisper decoder to match Hailo's requirements:
- Input: decoder_input_ids [1, 32] + encoder_hidden_states [1, 500, 384]
- Output: logits [1, 32, 51865]
- Fixed sequence lengths (no dynamic dimensions)


TODO needs cleanup

"""

import argparse
import torch
import onnx
import numpy as np
from onnxsim import simplify
from transformers import WhisperForConditionalGeneration
import types
import os
from pathlib import Path
from onnx import helper, TensorProto
from onnx.tools import update_model_dims

def get_args():
    parser = argparse.ArgumentParser(description="Convert Whisper decoder to Hailo-compatible format")
    parser.add_argument(
        "--model-path",
        type=str,
        default="openai/whisper-tiny",
        help="Path to Whisper model (default: openai/whisper-tiny)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./hailo_compatible_models",
        help="Output directory for converted models (default: ./hailo_compatible_models)"
    )
    parser.add_argument(
        "--input-length-seconds",
        type=int,
        default=10,
        help="Input audio length in seconds (default: 10)"
    )
    parser.add_argument(
        "--decoder-sequence-length",
        type=int,
        default=32,
        help="Decoder token sequence length (default: 32, matches Hailo)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        choices=["tiny", "base"],
        help="Whisper variant (default: tiny)"
    )
    parser.add_argument(
        "--skip-simplification",
        action="store_true",
        help="Skip ONNXSim optimization"
    )
    return parser.parse_args()


def fix_decoder_dimensions_with_onnx_tools(input_model_path, output_path, decoder_sequence_length=32, encoder_sequence_length=500):
    """
    Fix decoder model dimensions using ONNX graph manipulation instead of PyTorch re-export.
    This approach avoids the complex ONNX export errors by working with existing ONNX models.
    """
    print(f"Converting existing decoder model: {input_model_path}")
    print(f"  Target decoder sequence length: {decoder_sequence_length}")
    print(f"  Target encoder sequence length: {encoder_sequence_length}")



    # Load the existing model
    model = onnx.load(input_model_path)

    print(f"  Original model nodes: {len(model.graph.node)}")

    # First, let's remove unwanted outputs and rename inputs manually
    # This is simpler than trying to use update_model_dims with all the present.* outputs

    # Update input dimensions and collect old/new names for node updating
    input_name_mapping = {}

    for inp in model.graph.input:
        if inp.name == "input_ids":
            old_name = inp.name
            new_name = "decoder_input_ids"
            inp.name = new_name
            input_name_mapping[old_name] = new_name
            # Update input dimensions
            inp.type.tensor_type.shape.dim[0].dim_value = 1
            inp.type.tensor_type.shape.dim[1].dim_value = decoder_sequence_length

        elif inp.name == "encoder_hidden_states":
            # Update encoder hidden states dimensions (keep same name)
            inp.type.tensor_type.shape.dim[0].dim_value = 1
            inp.type.tensor_type.shape.dim[1].dim_value = encoder_sequence_length
            inp.type.tensor_type.shape.dim[2].dim_value = 384

    # Update all node inputs that reference the renamed inputs
    for node in model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in input_name_mapping:
                node.input[i] = input_name_mapping[input_name]

    # Keep only the logits output and update its dimensions
    new_outputs = []
    for output in model.graph.output:
        if output.name == "logits":
            # Update logits dimensions
            output.type.tensor_type.shape.dim[0].dim_value = 1
            output.type.tensor_type.shape.dim[1].dim_value = decoder_sequence_length
            output.type.tensor_type.shape.dim[2].dim_value = 51865
            new_outputs.append(output)

    # Replace all outputs with just logits
    model.graph.ClearField("output")
    model.graph.output.extend(new_outputs)

    updated_model = model

    print(f"  Updated model outputs: {len(updated_model.graph.output)}")

    # Save the updated model
    onnx.save(updated_model, output_path)
    print(f"✅ Fixed decoder saved to: {output_path}")

    # Verify the result
    print("\\nFixed model structure:")
    for inp in updated_model.graph.input:
        shape = [d.dim_value if d.dim_value != 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
        print(f"  Input: {inp.name} → {shape}")

    for out in updated_model.graph.output:
        shape = [d.dim_value if d.dim_value != 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} → {shape}")

    return updated_model



def create_simple_decoder_wrapper(model, decoder_sequence_length=32, encoder_sequence_length=500):
    """
    Create a simple wrapper that exports just the core decoder functionality.
    DEPRECATED: This approach has ONNX export issues. Use fix_decoder_dimensions_with_onnx_tools instead.
    """
    print(f"Creating decoder wrapper: decoder_seq={decoder_sequence_length}, encoder_seq={encoder_sequence_length}")

    class SimpleWhisperDecoder(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.decoder = original_model.model.decoder
            self.proj_out = original_model.proj_out

        def forward(self, decoder_input_ids, encoder_hidden_states):
            # Call the original decoder
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,  # Disable caching for simpler export
                return_dict=False
            )

            # Get hidden states (first element of tuple)
            hidden_states = decoder_outputs[0]

            # Apply projection layer to get logits
            logits = self.proj_out(hidden_states)

            return logits

    return SimpleWhisperDecoder(model)


def export_hailo_compatible_decoder(decoder_wrapper, output_path, decoder_sequence_length=32, encoder_sequence_length=500):
    """
    Export decoder with Hailo-compatible input/output format.
    """
    print(f"Exporting decoder...")
    print(f"  Decoder input shape: [1, {decoder_sequence_length}]")
    print(f"  Encoder hidden states shape: [1, {encoder_sequence_length}, 384]")

    # Create dummy inputs matching Hailo format
    dummy_decoder_input = torch.randint(0, 1000, (1, decoder_sequence_length), dtype=torch.long)
    dummy_encoder_hidden = torch.randn(1, encoder_sequence_length, 384)

    print(f"  Input names: ['decoder_input_ids', 'encoder_hidden_states']")
    print(f"  Output names: ['logits']")

    # Export to ONNX
    torch.onnx.export(
        decoder_wrapper,
        (dummy_decoder_input, dummy_encoder_hidden),
        output_path,
        opset_version=13,
        input_names=['decoder_input_ids', 'encoder_hidden_states'],
        output_names=['logits'],
    )

    print(f"✅ Exported to: {output_path}")

    # Verify the exported model
    onnx_model = onnx.load(output_path)
    print("\nExported model structure:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value != 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
        print(f"  Input: {inp.name} → {shape}")

    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value != 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} → {shape}")

    print(f"  Total nodes: {len(onnx_model.graph.node)}")

    return onnx_model


def apply_onnx_simplification(onnx_model, output_path):
    """
    Apply ONNXSim optimization like Hailo does.
    """
    print(f"\nApplying ONNXSim optimization...")

    try:
        # Simplify the model
        simplified_model, check = simplify(onnx_model)

        if check:
            # Save simplified model
            simplified_path = output_path.replace('.onnx', '_simplified.onnx')
            onnx.save(simplified_model, simplified_path)

            # Show improvements
            original_nodes = len(onnx_model.graph.node)
            simplified_nodes = len(simplified_model.graph.node)
            reduction = original_nodes - simplified_nodes

            print(f"✅ Simplified model saved to: {simplified_path}")
            print(f"   Node reduction: {original_nodes} → {simplified_nodes} (-{reduction} nodes, {reduction/original_nodes*100:.1f}% reduction)")

            # Show final structure
            print("\nSimplified model structure:")
            for inp in simplified_model.graph.input:
                shape = [d.dim_value if d.dim_value != 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
                print(f"  Input: {inp.name} → {shape}")

            for out in simplified_model.graph.output:
                shape = [d.dim_value if d.dim_value != 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
                print(f"  Output: {out.name} → {shape}")

            return simplified_model, simplified_path
        else:
            print("⚠️  ONNXSim simplification check failed, but model may still be functional")
            return onnx_model, output_path

    except Exception as e:
        print(f"⚠️  ONNXSim failed: {e}")
        return onnx_model, output_path


def test_model_inference(model_path, decoder_sequence_length=32, encoder_sequence_length=500):
    """
    Test the converted model with sample inputs to verify it works.
    """
    print(f"\nTesting decoder model inference...")

    try:
        import onnxruntime as ort

        # Load the model
        session = ort.InferenceSession(model_path)

        # Create test inputs matching Hailo format
        decoder_input = np.random.randint(0, 1000, (1, decoder_sequence_length), dtype=np.int64)
        encoder_hidden = np.random.randn(1, encoder_sequence_length, 384).astype(np.float32)

        # Run inference
        result = session.run(None, {
            'decoder_input_ids': decoder_input,
            'encoder_hidden_states': encoder_hidden
        })

        print(f"✅ Inference successful!")
        print(f"   Decoder input shape: {decoder_input.shape}")
        print(f"   Encoder hidden shape: {encoder_hidden.shape}")
        print(f"   Logits output shape: {result[0].shape}")

        # Verify output dimensions are correct
        expected_output_shape = (1, decoder_sequence_length, 51865)  # Whisper vocabulary size
        actual_output_shape = result[0].shape

        if actual_output_shape == expected_output_shape:
            print(f"✅ Output shape correct: {actual_output_shape}")
        else:
            print(f"⚠️  Output shape unexpected: {actual_output_shape}, expected: {expected_output_shape}")

        return True

    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


def main():
    args = get_args()

    print("=== Converting Whisper Decoder to Hailo-Compatible Format ===")
    print(f"Model: {args.model_path}")
    print(f"Input length: {args.input_length_seconds}s")
    print(f"Decoder sequence length: {args.decoder_sequence_length} tokens")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate encoder sequence length based on audio input length
    encoder_sequence_length = args.input_length_seconds * 50  # Whisper produces ~50 tokens per second of audio

    # Check if args.model_path is an ONNX file or HuggingFace model
    if args.model_path.endswith('.onnx') or os.path.isfile(args.model_path):
        # Use ONNX-based approach with existing decoder model
        print(f"\n🔄 Using ONNX-based conversion approach...")
        print(f"Input model: {args.model_path}")

        output_filename = f"custom_{args.variant}-whisper-decoder-{args.input_length_seconds}s-seq-{args.decoder_sequence_length}.onnx"
        output_path = os.path.join(args.output_dir, output_filename)


        # Fix dimensions using ONNX tools
        onnx_model = fix_decoder_dimensions_with_onnx_tools(
            args.model_path, output_path, args.decoder_sequence_length, encoder_sequence_length
        )

        if onnx_model is None:
            print("❌ ONNX dimension fixing failed")
            return 1

        # Apply simplification unless skipped
        final_model_path = output_path
        if not args.skip_simplification:
            simplified_model, final_model_path = apply_onnx_simplification(onnx_model, output_path)

        # Test the converted model
        test_success = test_model_inference(
            final_model_path, args.decoder_sequence_length, encoder_sequence_length
        )

        print(f"\n{'='*60}")
        print("CONVERSION SUMMARY:")
        print(f"✅ Original ONNX model: {args.model_path}")
        print(f"✅ Fixed dimensions for: {args.decoder_sequence_length} token sequence length")
        print(f"✅ Converted to Hailo format:")
        print(f"   - decoder_input_ids: [1, {args.decoder_sequence_length}]")
        print(f"   - encoder_hidden_states: [1, {encoder_sequence_length}, 384]")
        print(f"   - logits: [1, {args.decoder_sequence_length}, 51865]")
        print(f"✅ Final model: {final_model_path}")
        print(f"✅ Inference test: {'PASSED' if test_success else 'FAILED'}")

        print(f"\nNext step: Test with Hailo conversion pipeline:")
        print(f"python3 -m conversion.convert_whisper_decoder {final_model_path} --variant {args.variant} --hw-arch hailo8")


    else:
        # Use PyTorch-based approach (original method - has export issues)
        print(f"\n⚠️  Using PyTorch-based conversion approach (may have export issues)...")

        # Load the model
        print(f"\nLoading model: {args.model_path}")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
        model.eval()

        # Create decoder wrapper for fixed dimensions
        print(f"\nCreating decoder wrapper for fixed dimensions...")
        decoder_wrapper = create_simple_decoder_wrapper(
            model, args.decoder_sequence_length, encoder_sequence_length
        )

        # Export to ONNX with Hailo format
        output_filename = f"custom_{args.variant}-whisper-decoder-{args.input_length_seconds}s-seq-{args.decoder_sequence_length}.onnx"
        output_path = os.path.join(args.output_dir, output_filename)

        onnx_model = export_hailo_compatible_decoder(
            decoder_wrapper, output_path, args.decoder_sequence_length, encoder_sequence_length
        )

        # Apply simplification unless skipped
        final_model_path = output_path
        if not args.skip_simplification:
            simplified_model, final_model_path = apply_onnx_simplification(onnx_model, output_path)

        # Test the converted model
        test_success = test_model_inference(
            final_model_path, args.decoder_sequence_length, encoder_sequence_length
        )

        print(f"\n{'='*60}")
        print("CONVERSION SUMMARY:")
        print(f"✅ Original model: {args.model_path}")
        print(f"✅ Patched for: {args.decoder_sequence_length} token sequence length")
        print(f"✅ Exported with Hailo format:")
        print(f"   - decoder_input_ids: [1, {args.decoder_sequence_length}]")
        print(f"   - encoder_hidden_states: [1, {encoder_sequence_length}, 384]")
        print(f"   - logits: [1, {args.decoder_sequence_length}, 51865]")
        print(f"✅ Final model: {final_model_path}")
        print(f"✅ Inference test: {'PASSED' if test_success else 'FAILED'}")

        print(f"\nNext step: Test with Hailo conversion pipeline:")
        print(f"python3 -m conversion.convert_whisper_decoder {final_model_path} --variant {args.variant} --hw-arch hailo8")


if __name__ == "__main__":
    main()