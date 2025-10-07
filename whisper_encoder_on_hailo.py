#!/usr/bin/env python3
"""
Whisper Encoder Benchmarking on Hailo
Compare HEF vs ONNX encoder performance on Hailo hardware
"""
import numpy as np
import os
import argparse
import time
from hailo_platform import (VDevice, HailoSchedulingAlgorithm, FormatType)
import onnxruntime as ort

# For efficient audio preprocessing
import librosa
import torch

# Runs
# python whisper_encoder_on_hailo.py --encoder_orig_onnx_file  /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/  --audio_file ~/dev/audio_samples/hello_world.wav
# python whisper_encoder_on_hailo.py --encoder_onnx_file  /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/converted_for_hailo/whisper_tiny_encoder_10s_hailo_final.onnx   --audio_file ~/dev/audio_samples/hello_world.wav

# Usage Examples:
#
# 1. HEF encoder (Hailo hardware):
#    python whisper_encoder_on_hailo.py \
#      --encoder_hef_file /path/to/encoder.hef \
#      --audio_file audio.wav
#
# 2. ONNX encoder INT8 (10s, CPU runtime):
#    python whisper_encoder_on_hailo.py \
#      --encoder_onnx_file /path/to/encoder_10s.onnx \
#      --audio_file audio.wav
#
# 3. Original ONNX encoder FP32 (30s, standard Whisper):
#    python whisper_encoder_on_hailo.py \
#      --encoder_orig_onnx_file /path/to/encoder_model.onnx \
#      --audio_file audio.wav
#
# 4. All three (comparison):
#    python whisper_encoder_on_hailo.py \
#      --encoder_hef_file /path/to/encoder.hef \
#      --encoder_onnx_file /path/to/encoder_10s.onnx \
#      --encoder_orig_onnx_file /path/to/encoder_model.onnx \
#      --audio_file audio.wav


# ============================================================================
# Audio Preprocessing (same as whisper_on_hailo.py)
# ============================================================================

# Audio hyperparameters (from Whisper)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_SAMPLES_10S = 10 * SAMPLE_RATE  # 160000 samples in 10 seconds


def pad_or_trim(array, length: int = N_SAMPLES_10S, *, axis: int = -1):
    """Pad or trim the audio array to specified length."""
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = torch.nn.functional.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
    return array


def log_mel_spectrogram(audio: torch.Tensor, n_mels: int = 80, padding: int = 0):
    """
    Compute log-Mel spectrogram using PyTorch STFT (Hailo-efficient approach).

    Args:
        audio: Audio waveform tensor at 16kHz
        n_mels: Number of mel frequency bands (default: 80)
        padding: Zero samples to pad

    Returns:
        log_spec: Log-mel spectrogram [n_mels, n_frames]
    """
    if padding > 0:
        audio = torch.nn.functional.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    # Compute mel filters on-the-fly using librosa (cached by lru_cache)
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def get_mel_filters(device_str, n_mels):
        """Cached mel filter computation."""
        import librosa
        mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)
        return torch.from_numpy(mel_basis).to(device_str)

    filters = get_mel_filters(str(audio.device), n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def get_audio(audio_file, chunk_length=10, is_nhwc=True):
    """
    Load audio file and convert to mel spectrogram.

    Args:
        audio_file: Path to audio file
        chunk_length: Duration in seconds (default: 10 for 10s encoder)
        is_nhwc: Whether to transpose to NHWC format (default: True for Hailo HEF)

    Returns:
        mel_spectrogram: Numpy array [1, 80, 1, 1000] (NCHW) or [1, 1, 1000, 80] (NHWC)
    """
    # Load audio using librosa
    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

    # Check duration
    audio_duration = len(audio) / SAMPLE_RATE
    if audio_duration > chunk_length:
        print(f"   ⚠️  Audio is {audio_duration:.1f}s, cropping to {chunk_length}s")

    # Convert to torch tensor
    audio_torch = torch.from_numpy(audio)

    # Pad or trim to target length
    segment_samples = chunk_length * SAMPLE_RATE
    audio_torch = pad_or_trim(audio_torch, int(segment_samples))

    # Compute mel spectrogram using efficient PyTorch STFT
    mel = log_mel_spectrogram(audio_torch).to("cpu")

    # Convert to numpy and reshape to [1, 80, 1, 1000]
    mel = mel.numpy()
    mel = np.expand_dims(mel, axis=0)  # Add batch dimension
    mel = np.expand_dims(mel, axis=2)  # Add spatial dimension for conv

    # Transpose to NHWC if needed (Hailo HEF models expect this)
    if is_nhwc:
        mel = np.transpose(mel, [0, 2, 3, 1])  # [1, 80, 1, 1000] -> [1, 1, 1000, 80]

    return mel.astype(np.float32)


def get_audio_orig_onnx(audio_file, target_duration=30):
    """
    Load audio file and preprocess using WhisperProcessor (for original ONNX encoder).
    This matches the preprocessing used in ONNXWhisperTranscriber.

    Args:
        audio_file: Path to audio file
        target_duration: Duration in seconds (default: 30 for original Whisper)

    Returns:
        mel_spectrogram: Numpy array [1, 80, 3000] for 30s audio (NCHW)
    """
    from transformers import WhisperProcessor

    # Load audio using librosa
    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

    # Check duration
    audio_duration = len(audio) / SAMPLE_RATE
    target_length = target_duration * SAMPLE_RATE

    if audio_duration > target_duration:
        print(f"   ⚠️  Audio is {audio_duration:.1f}s, cropping to {target_duration}s")
        audio = audio[:target_length]
    elif len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

    # Use WhisperProcessor to create mel spectrogram (standard Whisper preprocessing)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")

    return inputs.input_features  # [1, 80, 3000]


# ============================================================================
# Encoder Inference Functions
# ============================================================================

def run_hef_encoder(encoder_hef_file, mel_input, num_iterations=10, multi_process_service=False):
    """
    Run encoder using Hailo HEF (hardware accelerated).

    Args:
        encoder_hef_file: Path to encoder HEF file
        mel_input: Preprocessed mel spectrogram [1, 1, 1000, 80] (NHWC)
        num_iterations: Number of times to run inference
        multi_process_service: Enable multi-process service mode

    Returns:
        (encoder_output, timings): Encoder output and list of timing measurements
    """
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    if multi_process_service:
        params.multi_process_service = True
        params.group_id = "SHARED"

    timings = []
    encoder_output = None

    with VDevice(params) as vdevice:
        encoder_infer_model = vdevice.create_infer_model(encoder_hef_file)
        encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
        encoder_infer_model.output().set_format_type(FormatType.FLOAT32)

        with encoder_infer_model.configure() as encoder_configured_infer_model:
            encoder_bindings = encoder_configured_infer_model.create_bindings()

            # Prepare input
            input_mel = np.ascontiguousarray(mel_input)
            encoder_bindings.input().set_buffer(input_mel)
            buffer = np.zeros(encoder_infer_model.output().shape).astype(np.float32)
            encoder_bindings.output().set_buffer(buffer)

            # Run multiple iterations
            for i in range(num_iterations):
                start = time.time()
                encoder_configured_infer_model.run([encoder_bindings], 100000000)  # timeout_ms
                elapsed_ms = (time.time() - start) * 1000
                timings.append(elapsed_ms)

                if i == 0:
                    encoder_output = encoder_bindings.output().get_buffer().copy()

    return encoder_output, timings


def run_onnx_encoder(encoder_onnx_file, mel_input, num_iterations=10):
    """
    Run encoder using ONNX Runtime (CPU).
    Works with both 10s ([1, 80, 1, 1000]) and 30s ([1, 80, 3000]) encoders.

    Args:
        encoder_onnx_file: Path to encoder ONNX file
        mel_input: Preprocessed mel spectrogram (NCHW format)
                   - 10s encoder: [1, 80, 1, 1000]
                   - 30s encoder: [1, 80, 3000]
        num_iterations: Number of times to run inference

    Returns:
        (encoder_output, timings): Encoder output and list of timing measurements
    """
    session = ort.InferenceSession(encoder_onnx_file)

    # Get input info
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    expected_shape = input_info.shape

    # Verify input shape matches (some dimensions might be dynamic, marked as -1 or string)
    print(f"  ONNX model expects input: {input_name} with shape {expected_shape}")
    print(f"  Providing input shape: {mel_input.shape}")

    timings = []
    encoder_output = None

    for i in range(num_iterations):
        start = time.time()
        output = session.run(None, {input_name: mel_input})[0]
        elapsed_ms = (time.time() - start) * 1000
        timings.append(elapsed_ms)

        if i == 0:
            encoder_output = output

    return encoder_output, timings


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper Encoder Benchmarking on Hailo")
    parser.add_argument("--encoder_hef_file", type=str, default=None,
                        help="Path to encoder HEF file (10s, NHWC, Hailo hardware)")
    parser.add_argument("--encoder_onnx_file", type=str, default=None,
                        help="Path to encoder ONNX file (10s, NCHW, INT8, CPU runtime)")
    parser.add_argument("--encoder_orig_onnx_file", type=str, default=None,
                        help="Path to original encoder ONNX file (30s, NCHW, FP32, standard Whisper)")
    parser.add_argument("--audio_file", type=str, required=True,
                        help="Path to audio file to process")
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Number of iterations to run (default: 10)")
    parser.add_argument("--multi_process_service", action="store_true",
                        help="Enable multi-process service mode (HEF only)")

    args = parser.parse_args()

    # Validate arguments
    if args.encoder_hef_file is None and args.encoder_onnx_file is None and args.encoder_orig_onnx_file is None:
        print("Error: At least one encoder must be provided (--encoder_hef_file, --encoder_onnx_file, or --encoder_orig_onnx_file)")
        return 1

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1

    print("="*70)
    print("WHISPER ENCODER BENCHMARKING ON HAILO")
    print("="*70)
    print(f"Audio file: {args.audio_file}")
    print(f"Iterations: {args.num_iterations}")
    print()

    # -------------------------------------------------------------------------
    # 1. PREPROCESSING
    # -------------------------------------------------------------------------
    print("📦 Stage 1: Audio Preprocessing")
    preprocess_start = time.time()

    # Preprocess for HEF (NHWC) if testing HEF
    # Preprocess for ONNX (NCHW) if testing ONNX
    # If testing both, we'll do both preprocessings

    results = {}

    # -------------------------------------------------------------------------
    # 2. HEF ENCODER
    # -------------------------------------------------------------------------
    if args.encoder_hef_file:
        print("\n" + "="*70)
        print("HEF ENCODER (Hailo Hardware)")
        print("="*70)
        print(f"Encoder HEF: {args.encoder_hef_file}")

        print("Preprocessing audio (NHWC format for HEF)...")
        mel_input_hef = get_audio(args.audio_file, is_nhwc=True)
        print(f"  Mel spectrogram shape: {mel_input_hef.shape}")
        print()

        print(f"Running HEF encoder {args.num_iterations} times...")
        encoder_output_hef, timings_hef = run_hef_encoder(
            args.encoder_hef_file,
            mel_input_hef,
            args.num_iterations,
            args.multi_process_service
        )

        results['hef'] = {
            'output': encoder_output_hef,
            'timings': timings_hef,
            'shape': encoder_output_hef.shape
        }

        print(f"✅ Encoder output shape: {encoder_output_hef.shape}")
        print(f"✅ Output range: [{encoder_output_hef.min():.3f}, {encoder_output_hef.max():.3f}]")
        print(f"✅ Output mean: {encoder_output_hef.mean():.3f}, std: {encoder_output_hef.std():.3f}")
        print()
        print("[TIMING] HEF Encoder Results:")
        print(f"  Min:     {min(timings_hef):.2f}ms")
        print(f"  Max:     {max(timings_hef):.2f}ms")
        print(f"  Mean:    {np.mean(timings_hef):.2f}ms")
        print(f"  Median:  {np.median(timings_hef):.2f}ms")
        print(f"  Std:     {np.std(timings_hef):.2f}ms")

    # -------------------------------------------------------------------------
    # 3. ONNX ENCODER
    # -------------------------------------------------------------------------
    if args.encoder_onnx_file:
        print("\n" + "="*70)
        print("ONNX ENCODER (CPU Runtime)")
        print("="*70)
        print(f"Encoder ONNX: {args.encoder_onnx_file}")

        print("Preprocessing audio (NCHW format for ONNX)...")
        mel_input_onnx = get_audio(args.audio_file, is_nhwc=False)
        print(f"  Mel spectrogram shape: {mel_input_onnx.shape}")
        print()

        print(f"Running ONNX encoder {args.num_iterations} times...")
        encoder_output_onnx, timings_onnx = run_onnx_encoder(
            args.encoder_onnx_file,
            mel_input_onnx,
            args.num_iterations
        )

        results['onnx'] = {
            'output': encoder_output_onnx,
            'timings': timings_onnx,
            'shape': encoder_output_onnx.shape
        }

        print(f"✅ Encoder output shape: {encoder_output_onnx.shape}")
        print(f"✅ Output range: [{encoder_output_onnx.min():.3f}, {encoder_output_onnx.max():.3f}]")
        print(f"✅ Output mean: {encoder_output_onnx.mean():.3f}, std: {encoder_output_onnx.std():.3f}")
        print()
        print("[TIMING] ONNX Encoder Results:")
        print(f"  Min:     {min(timings_onnx):.2f}ms")
        print(f"  Max:     {max(timings_onnx):.2f}ms")
        print(f"  Mean:    {np.mean(timings_onnx):.2f}ms")
        print(f"  Median:  {np.median(timings_onnx):.2f}ms")
        print(f"  Std:     {np.std(timings_onnx):.2f}ms")

    # -------------------------------------------------------------------------
    # 4. ORIGINAL ONNX ENCODER (FP32, 30s)
    # -------------------------------------------------------------------------
    if args.encoder_orig_onnx_file:
        print("\n" + "="*70)
        print("ORIGINAL ONNX ENCODER (FP32, 30s, Standard Whisper)")
        print("="*70)
        print(f"Encoder ONNX: {args.encoder_orig_onnx_file}")

        print("Preprocessing audio (NCHW format with WhisperProcessor, 30s)...")
        mel_input_orig = get_audio_orig_onnx(args.audio_file, target_duration=30)
        print(f"  Mel spectrogram shape: {mel_input_orig.shape}")
        print()

        print(f"Running original ONNX encoder {args.num_iterations} times...")
        encoder_output_orig, timings_orig = run_onnx_encoder(
            args.encoder_orig_onnx_file,
            mel_input_orig,
            args.num_iterations
        )

        results['onnx_orig'] = {
            'output': encoder_output_orig,
            'timings': timings_orig,
            'shape': encoder_output_orig.shape
        }

        print(f"✅ Encoder output shape: {encoder_output_orig.shape}")
        print(f"✅ Output range: [{encoder_output_orig.min():.3f}, {encoder_output_orig.max():.3f}]")
        print(f"✅ Output mean: {encoder_output_orig.mean():.3f}, std: {encoder_output_orig.std():.3f}")
        print()
        print("[TIMING] Original ONNX Encoder Results:")
        print(f"  Min:     {min(timings_orig):.2f}ms")
        print(f"  Max:     {max(timings_orig):.2f}ms")
        print(f"  Mean:    {np.mean(timings_orig):.2f}ms")
        print(f"  Median:  {np.median(timings_orig):.2f}ms")
        print(f"  Std:     {np.std(timings_orig):.2f}ms")

    # -------------------------------------------------------------------------
    # 5. COMPARISON
    # -------------------------------------------------------------------------
    if len(results) >= 2:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)

        # Calculate mean times
        means = {}
        for key, data in results.items():
            means[key] = np.mean(data['timings'])

        # Performance comparison
        print("\nPerformance:")
        labels = {
            'hef': 'HEF (Hailo, 10s, NHWC)',
            'onnx': 'ONNX (CPU, 10s, INT8, NCHW)',
            'onnx_orig': 'ONNX Original (CPU, 30s, FP32, NCHW)'
        }

        for key in sorted(results.keys()):
            print(f"  {labels.get(key, key):40s}  {means[key]:7.2f}ms")

        # Speedup comparisons
        if 'hef' in means:
            print(f"\nSpeedup relative to HEF (Hailo):")
            for key in sorted(results.keys()):
                if key != 'hef':
                    speedup = means[key] / means['hef']
                    print(f"  {labels.get(key, key):40s}  {speedup:5.2f}x {'(HEF faster)' if speedup > 1 else '(slower)'}")

        if 'onnx_orig' in means:
            print(f"\nSpeedup relative to Original ONNX (FP32, 30s):")
            for key in sorted(results.keys()):
                if key != 'onnx_orig':
                    speedup = means['onnx_orig'] / means[key]
                    print(f"  {labels.get(key, key):40s}  {speedup:5.2f}x faster")

        # Output comparison (only for encoders with matching sequence lengths)
        if 'hef' in results and 'onnx' in results:
            output_diff = np.abs(results['hef']['output'] - results['onnx']['output'])
            print(f"\nOutput Comparison (HEF vs ONNX 10s):")
            print(f"  Max difference:  {output_diff.max():.6f}")
            print(f"  Mean difference: {output_diff.mean():.6f}")
            print(f"  Outputs match:   {'✅ Yes' if output_diff.max() < 0.01 else '⚠️  No (difference > 0.01)'}")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
