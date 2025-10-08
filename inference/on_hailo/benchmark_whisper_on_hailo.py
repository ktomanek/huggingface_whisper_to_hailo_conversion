#!/usr/bin/env python3
"""
Comprehensive Whisper Benchmarking on Hailo

Combines encoder-only benchmarking with optional end-to-end transcription testing.

Features:
- Encoder benchmarking: HEF (Hailo NPU), ONNX INT8 (10s), ONNX FP32 (30s)
- Statistical performance metrics (min/max/mean/median/std)
- Numerical output comparison between encoders
- Optional decoder integration for end-to-end transcription
- Optional FasterWhisper baseline comparison
- Transcription quality validation

Usage Examples:

1. Encoder-only benchmarking (multiple iterations):
   python benchmark_whisper_on_hailo.py \
     --encoder_hef_file /path/to/encoder.hef \
     --encoder_onnx_file /path/to/encoder_10s.onnx \
     --audio_file audio.wav \
     --num_iterations 10

2. End-to-end transcription comparison (automatically uses decoder if provided):
   python benchmark_whisper_on_hailo.py \
     --encoder_hef_file /path/to/encoder.hef \
     --encoder_onnx_file /path/to/encoder_10s.onnx \
     --audio_file audio.wav \
     --decoder_onnx_dir /path/to/decoder_onnx

3. Full comparison including FasterWhisper:
   python benchmark_whisper_on_hailo.py \
     --encoder_hef_file /path/to/encoder.hef \
     --encoder_onnx_file /path/to/encoder_10s.onnx \
     --encoder_orig_onnx_file /path/to/encoder_30s.onnx \
     --audio_file audio.wav \
     --decoder_onnx_dir /path/to/decoder_onnx \
     --include_faster_whisper
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


# ============================================================================
# Audio Preprocessing
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
    Compute log-Mel spectrogram using PyTorch STFT.

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

def run_hef_encoder(encoder_hef_file, mel_input, num_iterations=1, multi_process_service=False):
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


def run_onnx_encoder(encoder_onnx_file, mel_input, num_iterations=1):
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
        session_init = ort.InferenceSession(decoder_init_path)
        session_cached = ort.InferenceSession(decoder_cached_path)

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

        # Check for EOS or end-of-text tokens
        if next_token in [50256, 50257]:  # <|endoftext|>
            break

    total_inference_time = sum(step_times)

    return generated_tokens, total_inference_time


def decode_tokens(tokens):
    """Convert token IDs to text using transformers tokenizer."""
    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


def run_faster_whisper(audio_file, sample_rate=16000, warmup=True):
    """
    Run FasterWhisper for comparison.

    Args:
        audio_file: Path to audio file
        sample_rate: Audio sample rate
        warmup: Whether to run warmup inference

    Returns:
        (transcription, inference_time_ms): Transcription text and inference time
    """
    from faster_whisper import WhisperModel

    # Load model
    print("  Loading FasterWhisper model...")
    load_start = time.time()
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    load_time = (time.time() - load_start) * 1000
    print(f"  Model loaded in {load_time:.1f}ms")

    # Load audio
    audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)

    # Warmup run
    if warmup:
        print("  Running warmup inference...")
        dummy_audio = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence
        warmup_start = time.time()
        list(model.transcribe(
            dummy_audio,
            beam_size=1,
            language='en',
            task='transcribe',
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )[0])
        warmup_time = (time.time() - warmup_start) * 1000
        print(f"  Warmup completed in {warmup_time:.1f}ms")

    # Measure preprocessing separately
    preprocess_start = time.time()
    features = model.feature_extractor(audio)
    preprocess_time = (time.time() - preprocess_start) * 1000

    # Run full transcription
    print("  Running full transcription...")
    start = time.time()
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

    total_time = (time.time() - start) * 1000
    transcription = transcription.strip()

    # Estimate inference time by subtracting preprocessing
    inference_time = total_time - preprocess_time

    print(f"  Total time: {total_time:.2f}ms (preprocessing: {preprocess_time:.2f}ms, inference: {inference_time:.2f}ms)")

    return transcription, total_time


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Whisper Benchmarking on Hailo")

    # Encoder options
    parser.add_argument("--encoder_hef_file", type=str, default=None,
                        help="Path to encoder HEF file (10s, NHWC, Hailo hardware)")
    parser.add_argument("--encoder_onnx_file", type=str, default=None,
                        help="Path to encoder ONNX file (10s, NCHW, INT8, CPU runtime)")
    parser.add_argument("--encoder_orig_onnx_file", type=str, default=None,
                        help="Path to original encoder ONNX file (30s, NCHW, FP32, standard Whisper)")

    # Audio file
    parser.add_argument("--audio_file", type=str, required=True,
                        help="Path to audio file to process")

    # Decoder options (optional)
    parser.add_argument("--decoder_onnx_dir", type=str, default=None,
                        help="Path to directory containing decoder_model.onnx and decoder_with_past_model.onnx (if provided, decoder will be used automatically)")

    # Benchmarking options
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Number of iterations for encoder benchmarking (default: 10, automatically set to 1 if decoder is provided)")
    parser.add_argument("--multi_process_service", action="store_true",
                        help="Enable multi-process service mode (HEF only)")

    # Comparison options
    parser.add_argument("--include_faster_whisper", action="store_true",
                        help="Include FasterWhisper (CPU INT8) for comparison")

    args = parser.parse_args()

    # Validate arguments
    if args.encoder_hef_file is None and args.encoder_onnx_file is None and args.encoder_orig_onnx_file is None:
        print("Error: At least one encoder must be provided")
        return 1

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1

    # Determine if decoder will be used
    use_decoder = args.decoder_onnx_dir is not None

    if use_decoder:
        decoder_init_path = os.path.join(args.decoder_onnx_dir, "decoder_model.onnx")
        decoder_cached_path = os.path.join(args.decoder_onnx_dir, "decoder_with_past_model.onnx")
        if not os.path.exists(decoder_init_path):
            print(f"Error: Decoder not found: {decoder_init_path}")
            return 1
        if not os.path.exists(decoder_cached_path):
            print(f"Error: Cached decoder not found: {decoder_cached_path}")
            return 1

    # Force num_iterations to 1 if using decoder
    if use_decoder and args.num_iterations != 1:
        print(f"Note: Setting num_iterations to 1 (decoder mode)")
        args.num_iterations = 1

    print("="*70)
    print("COMPREHENSIVE WHISPER BENCHMARKING ON HAILO")
    print("="*70)
    print(f"Audio file: {args.audio_file}")
    print(f"Mode: {'End-to-end transcription' if use_decoder else 'Encoder-only benchmarking'}")
    if not use_decoder:
        print(f"Iterations: {args.num_iterations}")
    print()

    # Load decoder sessions if needed
    decoder_sessions = None
    if use_decoder:
        print("Loading ONNX decoder models...")
        load_start = time.time()
        session_init = ort.InferenceSession(decoder_init_path)
        session_cached = ort.InferenceSession(decoder_cached_path)
        load_time = (time.time() - load_start) * 1000
        print(f"✅ Decoder models loaded in {load_time:.1f}ms")

        # Warmup decoder
        print("Running decoder warmup...")
        warmup_start = time.time()
        dummy_encoder_output = np.random.randn(1, 500, 384).astype(np.float32)
        _, _ = run_onnx_decoder(decoder_init_path, decoder_cached_path, dummy_encoder_output,
                               max_length=10, sessions=(session_init, session_cached))
        warmup_time = (time.time() - warmup_start) * 1000
        print(f"✅ Decoder warmup completed in {warmup_time:.1f}ms")
        print()

        decoder_sessions = (session_init, session_cached)

    results = {}
    transcriptions = {}

    # -------------------------------------------------------------------------
    # 1. HEF ENCODER
    # -------------------------------------------------------------------------
    if args.encoder_hef_file:
        print("\n" + "="*70)
        print("HEF ENCODER (Hailo Hardware)")
        print("="*70)
        print(f"Encoder HEF: {args.encoder_hef_file}")

        print("Preprocessing audio (NHWC format for HEF)...")
        mel_input = get_audio(args.audio_file, is_nhwc=True)
        print(f"  Mel spectrogram shape: {mel_input.shape}")
        print()

        # Warmup if doing multiple iterations
        if args.num_iterations > 1:
            print("Running HEF encoder warmup...")
            dummy_mel = np.random.randn(*mel_input.shape).astype(np.float32)
            _, warmup_times = run_hef_encoder(
                args.encoder_hef_file,
                dummy_mel,
                num_iterations=1,
                multi_process_service=args.multi_process_service
            )
            print(f"  Warmup completed in {warmup_times[0]:.1f}ms")

        print(f"Running HEF encoder ({args.num_iterations} iteration{'s' if args.num_iterations > 1 else ''})...")
        encoder_output, timings = run_hef_encoder(
            args.encoder_hef_file,
            mel_input,
            args.num_iterations,
            args.multi_process_service
        )

        results['hef'] = {
            'output': encoder_output,
            'timings': timings,
            'shape': encoder_output.shape
        }

        print(f"✅ Encoder output shape: {encoder_output.shape}")
        print(f"✅ Output range: [{encoder_output.min():.3f}, {encoder_output.max():.3f}]")
        print(f"✅ Output mean: {encoder_output.mean():.3f}, std: {encoder_output.std():.3f}")
        print()
        print("[TIMING] HEF Encoder Results:")
        if len(timings) > 1:
            print(f"  Min:     {min(timings):.2f}ms")
            print(f"  Max:     {max(timings):.2f}ms")
            print(f"  Mean:    {np.mean(timings):.2f}ms")
            print(f"  Median:  {np.median(timings):.2f}ms")
            print(f"  Std:     {np.std(timings):.2f}ms")
        else:
            print(f"  Time:    {timings[0]:.2f}ms")

        # Run decoder if requested
        if use_decoder:
            print()
            print("Running ONNX decoder...")
            tokens, decoder_time = run_onnx_decoder(
                decoder_init_path, decoder_cached_path, encoder_output,
                sessions=decoder_sessions
            )
            transcription = decode_tokens(tokens)

            print(f"✅ Decoder time: {decoder_time:.2f}ms")
            print(f"✅ Total time: {timings[0] + decoder_time:.2f}ms")
            print(f"✅ Transcription: \"{transcription}\"")

            transcriptions['hef'] = {
                'text': transcription,
                'tokens': tokens,
                'encoder_time': timings[0],
                'decoder_time': decoder_time,
                'total_time': timings[0] + decoder_time
            }

    # -------------------------------------------------------------------------
    # 2. ONNX INT8 ENCODER (10s)
    # -------------------------------------------------------------------------
    if args.encoder_onnx_file:
        print("\n" + "="*70)
        print("ONNX INT8 ENCODER (10s, CPU Runtime)")
        print("="*70)
        print(f"Encoder ONNX: {args.encoder_onnx_file}")

        print("Preprocessing audio (NCHW format for ONNX)...")
        mel_input = get_audio(args.audio_file, is_nhwc=False)
        print(f"  Mel spectrogram shape: {mel_input.shape}")
        print()

        # Warmup if doing multiple iterations
        if args.num_iterations > 1:
            print("Running ONNX encoder warmup...")
            dummy_mel = np.random.randn(*mel_input.shape).astype(np.float32)
            _, warmup_times = run_onnx_encoder(
                args.encoder_onnx_file,
                dummy_mel,
                num_iterations=1
            )
            print(f"  Warmup completed in {warmup_times[0]:.1f}ms")

        print(f"Running ONNX encoder ({args.num_iterations} iteration{'s' if args.num_iterations > 1 else ''})...")
        encoder_output, timings = run_onnx_encoder(
            args.encoder_onnx_file,
            mel_input,
            args.num_iterations
        )

        results['onnx'] = {
            'output': encoder_output,
            'timings': timings,
            'shape': encoder_output.shape
        }

        print(f"✅ Encoder output shape: {encoder_output.shape}")
        print(f"✅ Output range: [{encoder_output.min():.3f}, {encoder_output.max():.3f}]")
        print(f"✅ Output mean: {encoder_output.mean():.3f}, std: {encoder_output.std():.3f}")
        print()
        print("[TIMING] ONNX Encoder Results:")
        if len(timings) > 1:
            print(f"  Min:     {min(timings):.2f}ms")
            print(f"  Max:     {max(timings):.2f}ms")
            print(f"  Mean:    {np.mean(timings):.2f}ms")
            print(f"  Median:  {np.median(timings):.2f}ms")
            print(f"  Std:     {np.std(timings):.2f}ms")
        else:
            print(f"  Time:    {timings[0]:.2f}ms")

        # Run decoder if requested
        if use_decoder:
            print()
            print("Running ONNX decoder...")
            tokens, decoder_time = run_onnx_decoder(
                decoder_init_path, decoder_cached_path, encoder_output,
                sessions=decoder_sessions
            )
            transcription = decode_tokens(tokens)

            print(f"✅ Decoder time: {decoder_time:.2f}ms")
            print(f"✅ Total time: {timings[0] + decoder_time:.2f}ms")
            print(f"✅ Transcription: \"{transcription}\"")

            transcriptions['onnx'] = {
                'text': transcription,
                'tokens': tokens,
                'encoder_time': timings[0],
                'decoder_time': decoder_time,
                'total_time': timings[0] + decoder_time
            }

    # -------------------------------------------------------------------------
    # 3. ORIGINAL ONNX ENCODER (30s, FP32)
    # -------------------------------------------------------------------------
    if args.encoder_orig_onnx_file:
        print("\n" + "="*70)
        print("ORIGINAL ONNX ENCODER (30s, FP32, Standard Whisper)")
        print("="*70)
        print(f"Encoder ONNX: {args.encoder_orig_onnx_file}")

        print("Preprocessing audio (NCHW format with WhisperProcessor, 30s)...")
        mel_input = get_audio_orig_onnx(args.audio_file, target_duration=30)
        print(f"  Mel spectrogram shape: {mel_input.shape}")
        print()

        # Warmup if doing multiple iterations
        if args.num_iterations > 1:
            print("Running original ONNX encoder warmup...")
            dummy_mel = np.random.randn(*mel_input.shape).astype(np.float32)
            _, warmup_times = run_onnx_encoder(
                args.encoder_orig_onnx_file,
                dummy_mel,
                num_iterations=1
            )
            print(f"  Warmup completed in {warmup_times[0]:.1f}ms")

        print(f"Running original ONNX encoder ({args.num_iterations} iteration{'s' if args.num_iterations > 1 else ''})...")
        encoder_output, timings = run_onnx_encoder(
            args.encoder_orig_onnx_file,
            mel_input,
            args.num_iterations
        )

        results['onnx_orig'] = {
            'output': encoder_output,
            'timings': timings,
            'shape': encoder_output.shape
        }

        print(f"✅ Encoder output shape: {encoder_output.shape}")
        print(f"✅ Output range: [{encoder_output.min():.3f}, {encoder_output.max():.3f}]")
        print(f"✅ Output mean: {encoder_output.mean():.3f}, std: {encoder_output.std():.3f}")
        print()
        print("[TIMING] Original ONNX Encoder Results:")
        if len(timings) > 1:
            print(f"  Min:     {min(timings):.2f}ms")
            print(f"  Max:     {max(timings):.2f}ms")
            print(f"  Mean:    {np.mean(timings):.2f}ms")
            print(f"  Median:  {np.median(timings):.2f}ms")
            print(f"  Std:     {np.std(timings):.2f}ms")
        else:
            print(f"  Time:    {timings[0]:.2f}ms")

        # Run decoder if requested
        if use_decoder:
            print()
            print("Running ONNX decoder...")
            tokens, decoder_time = run_onnx_decoder(
                decoder_init_path, decoder_cached_path, encoder_output,
                sessions=decoder_sessions
            )
            transcription = decode_tokens(tokens)

            print(f"✅ Decoder time: {decoder_time:.2f}ms")
            print(f"✅ Total time: {timings[0] + decoder_time:.2f}ms")
            print(f"✅ Transcription: \"{transcription}\"")

            transcriptions['onnx_orig'] = {
                'text': transcription,
                'tokens': tokens,
                'encoder_time': timings[0],
                'decoder_time': decoder_time,
                'total_time': timings[0] + decoder_time
            }

    # -------------------------------------------------------------------------
    # 4. FASTERWHISPER (OPTIONAL)
    # -------------------------------------------------------------------------
    if args.include_faster_whisper:
        print("\n" + "="*70)
        print("FASTERWHISPER (CPU INT8, Baseline)")
        print("="*70)

        print("Running FasterWhisper transcription...")
        transcription, total_time = run_faster_whisper(args.audio_file)

        print(f"✅ Total time: {total_time:.2f}ms")
        print(f"✅ Transcription: \"{transcription}\"")

        if use_decoder:
            transcriptions['faster_whisper'] = {
                'text': transcription,
                'tokens': None,
                'encoder_time': None,
                'decoder_time': None,
                'total_time': total_time
            }

    # -------------------------------------------------------------------------
    # 5. COMPARISON
    # -------------------------------------------------------------------------
    if len(results) >= 2:
        print("\n" + "="*70)
        print("ENCODER COMPARISON")
        print("="*70)

        # Calculate mean times
        means = {}
        for key, data in results.items():
            means[key] = np.mean(data['timings'])

        # Performance comparison
        print("\nEncoder Performance:")
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

    # -------------------------------------------------------------------------
    # 6. TRANSCRIPTION COMPARISON (if decoder was used)
    # -------------------------------------------------------------------------
    if len(transcriptions) >= 2:
        print("\n" + "="*70)
        print("END-TO-END TRANSCRIPTION COMPARISON")
        print("="*70)

        trans_labels = {
            'hef': 'HEF (Hailo, 10s, NHWC)',
            'onnx': 'ONNX (CPU, 10s, INT8, NCHW)',
            'onnx_orig': 'ONNX Original (CPU, 30s, FP32, NCHW)',
            'faster_whisper': 'FasterWhisper (CPU, INT8, Baseline)'
        }

        print("\nTiming Summary:")
        for key in sorted(transcriptions.keys()):
            t = transcriptions[key]
            print(f"\n{trans_labels[key]}:")
            if t['encoder_time'] is not None:
                print(f"  Encoder:  {t['encoder_time']:7.2f}ms")
                print(f"  Decoder:  {t['decoder_time']:7.2f}ms")
            print(f"  Total:    {t['total_time']:7.2f}ms")

        print("\n" + "-"*70)
        print("\nTranscriptions:")
        for key in sorted(transcriptions.keys()):
            t = transcriptions[key]
            print(f"\n{trans_labels[key]}:")
            print(f"  \"{t['text']}\"")

        # Check if transcriptions match
        texts = [t['text'] for t in transcriptions.values()]

        if len(set(texts)) == 1:
            print(f"\n✅ All transcriptions match: \"{texts[0]}\"")
        else:
            print(f"\n⚠️  Transcriptions differ:")
            for key in sorted(transcriptions.keys()):
                print(f"  {trans_labels[key]:40s} \"{transcriptions[key]['text']}\"")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
