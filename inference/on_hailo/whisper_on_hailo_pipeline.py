#!/usr/bin/env python3
"""

This to be run on HAILO directly

Standalone Whisper on Hailo Pipeline (no external dependencies except transformers, numpy, onnxruntime, librosa, torch, hailo_platform)
Based on HailoWhisperPipeline from Hailo Application Code Examples

Can run decoder on HEF (no KV-cache, ~70ms/token) or ONNX (with KV-cache, ~6-12ms/token)

Includes time measurements.


"""
import numpy as np
import os
import argparse
import time
from pathlib import Path
from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)
from transformers import AutoTokenizer
import onnxruntime as ort

# For efficient audio preprocessing
import librosa
import torch

# For evaluation
from evaluation import load_audio_dataset, calculate_wer


# Usage Examples:
#
# 1. HEF mode (Hailo encoder + HEF decoder without KV-cache, ~70ms/token):
#    python whisper_on_hailo.py \
#      --encoder_hef_file /path/to/encoder.hef \
#      --decoder_hef_file /path/to/decoder.hef \
#      --decoder_assets_path /path/to/decoder_assets \
#      --audio_file audio.wav
#
# 2. Hybrid mode (Hailo encoder + ONNX decoder with KV-cache, ~6-12ms/token):
#    python whisper_on_hailo.py \
#      --encoder_hef_file /path/to/encoder.hef \
#      --decoder_onnx_dir /path/to/onnx_decoders \
#      --audio_file audio.wav
#
# Example with full paths:
# HEF mode:
# python whisper_on_hailo.py --encoder_hef_file /home/katrintomanek/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef --decoder_hef_file /home/katrintomanek/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef --decoder_assets_path /home/katrintomanek/dev/hailo-rpi5-examples/whisper/assets/decoder_assets --audio_file ~/dev/audio_samples/hello_world.wav
# /home/katrintomanek/dev/hailo_whisper_convert/my_converted/whisper_tiny_encoder_10s_hailo_final_optimized.hef
#
# Hybrid mode (recommended for speed):
# python whisper_on_hailo.py --encoder_hef_file /home/katrintomanek/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef --decoder_onnx_dir /home/katrintomanek/dev/hailo_whisper_convert/my_converted --audio_file ~/dev/audio_samples/hello_world.wav
# before - download assets
# 
# wget -P decoder_assets/tiny/decoder_tokenization \
#     "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets/tiny/decoder_tokenization/onnx_add_input_tiny.npy"

#   wget -P decoder_assets/tiny/decoder_tokenization \
#     "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets/tiny/decoder_tokenization/token_embedding_weight_tiny.npy"

# ============================================================================
# Postprocessing Functions (embedded from common.postprocessing)
# ============================================================================

excluded_tokens = [11, 13]  # Punctuation tokens to exclude from repetition penalty

REPETITION_PENALTY = 1.5


# ============================================================================
# Audio Preprocessing (Hailo-efficient implementation)
# ============================================================================

# TODO factor out into separate file

MAX_AUDIO_LENGTH_S = 10  # seconds
# Audio hyperparameters (from Whisper)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_SAMPLES_10S = MAX_AUDIO_LENGTH_S * SAMPLE_RATE  # 160000 samples in 10 seconds


def _pad_or_trim(array, length: int = N_SAMPLES_10S, *, axis: int = -1):
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


def _log_mel_spectrogram(audio: torch.Tensor, n_mels: int = 80, padding: int = 0, sample_rate: int = None):
    """
    Compute log-Mel spectrogram using PyTorch STFT (Hailo-efficient approach).

    Args:
        audio: Audio waveform tensor at 16kHz
        n_mels: Number of mel frequency bands (default: 80)
        padding: Zero samples to pad
        sample_rate: Sample rate in Hz (default: SAMPLE_RATE constant)

    Returns:
        log_spec: Log-mel spectrogram [n_mels, n_frames]
    """
    if sample_rate is None:
        sample_rate = SAMPLE_RATE

    if padding > 0:
        audio = torch.nn.functional.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    # Compute mel filters on-the-fly using librosa (cached by lru_cache)
    from functools import lru_cache

    @lru_cache(maxsize=4)
    def get_mel_filters(device_str, n_mels, sr):
        """Cached mel filter computation."""
        import librosa
        mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=n_mels)
        return torch.from_numpy(mel_basis).to(device_str)

    filters = get_mel_filters(str(audio.device), n_mels, sample_rate)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def get_mel_spectrogram(audio_file, target_duration=10, padding_cutoff_delta=1.0, format_4d=True, is_nhwc=True, sample_rate=None):
    """
    Load audio file and convert to mel spectrogram (configurable for different encoder types).

    Args:
        audio_file: Path to audio file
        target_duration: Target duration in seconds for encoder (10 for 10s encoders, 30 for 30s encoders)
        padding_cutoff_delta: Time (in seconds) to cut off before target_duration to avoid boundary hallucinations
                             (default: 1.0, e.g., crop to 9s for 10s target to leave natural padding)
        format_4d: If True, output 4D tensor [1, 80, 1, N]. If False, output 3D tensor [1, 80, N]
        is_nhwc: Whether to transpose to NHWC format (only applies if format_4d=True)
        sample_rate: Target sample rate in Hz (default: SAMPLE_RATE constant = 16000)

    Returns:
        mel_spectrogram: Numpy array
            - format_4d=True, is_nhwc=True:  [1, 1, N, 80] (HEF 10s)
            - format_4d=True, is_nhwc=False: [1, 80, 1, N] (ONNX 10s)
            - format_4d=False: [1, 80, N] (ONNX 30s)
    """
    if sample_rate is None:
        sample_rate = SAMPLE_RATE

    # Load audio using librosa
    audio, sr = librosa.load(audio_file, sr=sample_rate, mono=True)

    # Check duration
    audio_duration = len(audio) / sample_rate
    crop_duration = target_duration - padding_cutoff_delta

    if audio_duration > crop_duration:
        print(f"   ⚠️  Audio is {audio_duration:.1f}s, cropping to {crop_duration:.1f}s")

    # Convert to torch tensor
    audio_torch = torch.from_numpy(audio)

    # Crop audio first (e.g., 9s for 10s target), then pad to target_duration
    # This avoids hallucinations from hard cutoffs at exactly the boundary
    if padding_cutoff_delta > 0:
        crop_samples = int(crop_duration * sample_rate)
        audio_torch = _pad_or_trim(audio_torch, crop_samples)

    # Pad to target duration
    target_samples = int(target_duration * sample_rate)
    audio_torch = _pad_or_trim(audio_torch, target_samples)

    # Compute mel spectrogram using efficient PyTorch STFT
    mel = _log_mel_spectrogram(audio_torch, sample_rate=sample_rate).to("cpu")

    # Convert to numpy and add batch dimension
    mel = mel.numpy()
    mel = np.expand_dims(mel, axis=0)  # [80, N] -> [1, 80, N]

    if format_4d:
        # Add spatial dimension: [1, 80, N] -> [1, 80, 1, N]
        mel = np.expand_dims(mel, axis=2)

        # Transpose to NHWC if needed (HEF models expect this)
        if is_nhwc:
            mel = np.transpose(mel, [0, 2, 3, 1])  # [1, 80, 1, N] -> [1, 1, N, 80]
    # else: 3D format [1, 80, N] for 30s ONNX encoder

    return mel.astype(np.float32)


# ============================================================================
# Main Pipeline Class
# ============================================================================

class HailoWhisperPipeline:
    """
    A pipeline for running inference using Hailo's Whisper models.
    """

    def __init__(self, encoder_hef_path: str = None, decoder_hef_path: str = None, variant="tiny",
                 decoder_assets_path=None, multi_process_service=False, decoder_onnx_dir=None,
                 encoder_onnx_path: str = None, encoder_is_orig_onnx: bool = False):
        """
        Initialize the pipeline.

        :param encoder_hef_path: Path to the encoder HEF file (Hailo hardware, 10s).
        :param encoder_onnx_path: Path to encoder ONNX file (10s or 30s, CPU). If provided, uses ONNX encoder instead of HEF.
        :param encoder_is_orig_onnx: If True, treats encoder as original 30s ONNX (3D format [1, 80, 3000]). Otherwise 10s (4D format).
        :param decoder_hef_path: Path to the decoder HEF file (not used if decoder_onnx_dir is provided).
        :param variant: Model variant (e.g., "tiny", "base").
        :param decoder_assets_path: Path to decoder assets directory (only for HEF decoder).
        :param multi_process_service: Enable multi-process service mode (HEF only).
        :param decoder_onnx_dir: Path to directory containing ONNX decoder files (decoder_model.onnx and decoder_with_past_model.onnx).
                                 If provided, uses ONNX decoder instead of HEF decoder.
        """
        self.encoder_hef_path = encoder_hef_path
        self.encoder_onnx_path = encoder_onnx_path
        self.encoder_is_orig_onnx = encoder_is_orig_onnx
        self.decoder_hef_path = decoder_hef_path
        self.decoder_onnx_dir = decoder_onnx_dir
        self.timeout_ms = 100000000
        self.variant = variant

        self.decoding_sequence_length = 32 if ("tiny" in self.variant) else 24
        self.multi_process_service = multi_process_service

        # load tokenizer
        self._load_tokenizer()

        # Determine encoder and decoder modes
        self.use_onnx_encoder = encoder_onnx_path is not None
        self.use_onnx_decoder = decoder_onnx_dir is not None

        # Set preprocessing parameters based on encoder type
        if self.encoder_is_orig_onnx:
            self.encoder_target_duration = 30
            self.encoder_format_4d = False  # Original 30s ONNX uses [1, 80, 3000]
            self.encoder_is_nhwc = False
        else:
            self.encoder_target_duration = 10
            self.encoder_format_4d = True   # 10s encoders use [1, 80, 1, 1000] or [1, 1, 1000, 80]
            self.encoder_is_nhwc = (encoder_hef_path is not None)  # HEF uses NHWC, ONNX uses NCHW

        if self.use_onnx_encoder:
            # ONNX encoder mode - load ONNX session
            encoder_type = "30s original" if self.encoder_is_orig_onnx else "10s"
            print(f"[INFO] Using ONNX encoder ({encoder_type}) from: {encoder_onnx_path}")
            self.encoder_session = ort.InferenceSession(encoder_onnx_path)

            # Warmup ONNX encoder
            print(f"[INFO] Warming up ONNX encoder...")
            if self.encoder_is_orig_onnx:
                dummy_input = np.random.randn(1, 80, 3000).astype(np.float32)
            else:
                dummy_input = np.random.randn(1, 80, 1, 1000).astype(np.float32)
            input_name = self.encoder_session.get_inputs()[0].name
            _ = self.encoder_session.run(None, {input_name: dummy_input})
            print(f"[INFO] ONNX encoder warmup complete")
            # No VDevice needed for ONNX encoder
        else:
            # HEF encoder mode - initialize VDevice and Hailo resources
            self.params = VDevice.create_params()
            self.params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

            if self.multi_process_service:
                self.params.multi_process_service = True
                self.params.group_id = "SHARED"

            # Create persistent VDevice and load HEF models
            self.vdevice = VDevice(self.params)
            self.encoder_infer_model = self.vdevice.create_infer_model(self.encoder_hef_path)
            self.encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
            self.encoder_infer_model.output().set_format_type(FormatType.FLOAT32)

            # Configure encoder once and create bindings (reduce overhead)
            self.encoder_configured_infer_model = self.encoder_infer_model.configure()
            self.encoder_bindings = self.encoder_configured_infer_model.create_bindings()

            # Warmup HEF encoder
            print(f"[INFO] Warming up HEF encoder...")
            dummy_input = np.random.randn(1, 1, 1000, 80).astype(np.float32)
            dummy_input_contiguous = np.ascontiguousarray(dummy_input)
            self.encoder_bindings.input().set_buffer(dummy_input_contiguous)
            buffer = np.zeros(self.encoder_infer_model.output().shape).astype(np.float32)
            self.encoder_bindings.output().set_buffer(buffer)
            self.encoder_configured_infer_model.run([self.encoder_bindings], self.timeout_ms)
            print(f"[INFO] HEF encoder warmup complete")

        if self.use_onnx_decoder:
            # ONNX decoder mode - no token embeddings needed
            print(f"[INFO] Using ONNX decoder from: {decoder_onnx_dir}")
            self.decoder_init_path = os.path.join(decoder_onnx_dir, "decoder_model.onnx")
            self.decoder_cached_path = os.path.join(decoder_onnx_dir, "decoder_with_past_model.onnx")

            if not os.path.exists(self.decoder_init_path):
                raise FileNotFoundError(f"ONNX decoder not found: {self.decoder_init_path}")
            if not os.path.exists(self.decoder_cached_path):
                raise FileNotFoundError(f"ONNX cached decoder not found: {self.decoder_cached_path}")

            # Load ONNX sessions once
            self.decoder_init_session = ort.InferenceSession(self.decoder_init_path)
            self.decoder_cached_session = ort.InferenceSession(self.decoder_cached_path)

            # Warmup ONNX decoder
            print(f"[INFO] Warming up ONNX decoder...")
            dummy_encoder_output = np.random.randn(1, 500, 384).astype(np.float32)
            dummy_input_ids = np.array([[50258, 50259, 50359, 50363]], dtype=np.int64)
            _ = self.decoder_init_session.run(None, {
                'input_ids': dummy_input_ids,
                'encoder_hidden_states': dummy_encoder_output
            })
            print(f"[INFO] ONNX decoder warmup complete")
        else:
            # HEF decoder mode - need token embeddings
            print(f"[INFO] Using HEF decoder")
            # Set decoder assets path
            if decoder_assets_path is None:
                # Default to looking in the same directory as this script
                base_path = os.path.dirname(os.path.abspath(__file__))
                self.decoder_assets_path = os.path.join(base_path, "decoder_assets")
            else:
                self.decoder_assets_path = decoder_assets_path

            # Token embedding
            self.token_embedding_weight = self._load_token_embedding_weight()
            self.onnx_add_input = self._load_onnx_add_input()
            self.constant_output_0 = np.array([1])  # Unsqueeze axis

            # load HEF decoder
            decoder_hef = HEF(self.decoder_hef_path)
            self.decoder_sorted_output_names = decoder_hef.get_sorted_output_names()
            self.decoder_model_name = decoder_hef.get_network_group_names()[0]

            self.decoder_infer_model = self.vdevice.create_infer_model(self.decoder_hef_path)
            self.decoder_infer_model.input(f"{self.decoder_model_name}/input_layer1").set_format_type(FormatType.FLOAT32)
            self.decoder_infer_model.input(f"{self.decoder_model_name}/input_layer2").set_format_type(FormatType.FLOAT32)
            for output_name in self.decoder_sorted_output_names:
                self.decoder_infer_model.output(output_name).set_format_type(FormatType.FLOAT32)

            # Configure decoder once and create bindings (reduce overhead)
            self.decoder_configured_infer_model = self.decoder_infer_model.configure()
            self.decoder_bindings = self.decoder_configured_infer_model.create_bindings()

            # Warmup HEF decoder
            print(f"[INFO] Warming up HEF decoder...")
            dummy_encoder_output = np.random.randn(1, 500, 384).astype(np.float32)
            dummy_decoder_input_ids = np.array([[50258, 50259, 50359, 50363]], dtype=np.int64)
            dummy_decoder_input_ids = np.concatenate(
                [dummy_decoder_input_ids, np.zeros((1, self.decoding_sequence_length - 4), dtype=np.int64)], axis=1
            )
            tokenized_ids = self._tokenization(dummy_decoder_input_ids)

            self.decoder_bindings.input(f"{self.decoder_model_name}/input_layer1").set_buffer(dummy_encoder_output)
            self.decoder_bindings.input(f"{self.decoder_model_name}/input_layer2").set_buffer(tokenized_ids)
            buffers = [
                np.zeros(self.decoder_infer_model.output(name).shape).astype(np.float32)
                for name in self.decoder_sorted_output_names
            ]
            for name, buffer in zip(self.decoder_sorted_output_names, buffers):
                self.decoder_bindings.output(name).set_buffer(buffer)
            self.decoder_configured_infer_model.run([self.decoder_bindings], self.timeout_ms)
            print(f"[INFO] HEF decoder warmup complete")

    def _load_token_embedding_weight(self):
        """
        Load token embedding weights.
        """
        file_path = os.path.join(
            self.decoder_assets_path,
            f"{self.variant}/decoder_tokenization/token_embedding_weight_{self.variant}.npy"
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Token embedding weight file not found at: {file_path}\n"
                f"Please ensure decoder assets are downloaded and placed in: {self.decoder_assets_path}"
            )
        return np.load(file_path)

    def _load_onnx_add_input(self):
        """
        Load ONNX add input.
        """
        file_path = os.path.join(
            self.decoder_assets_path,
            f"{self.variant}/decoder_tokenization/onnx_add_input_{self.variant}.npy"
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"ONNX add input file not found at: {file_path}\n"
                f"Please ensure decoder assets are downloaded and placed in: {self.decoder_assets_path}"
            )
        return np.load(file_path)

    def _load_tokenizer(self):
        """
        Load the tokenizer for the specified variant.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(f"openai/whisper-{self.variant}")

    def _tokenization(self, decoder_input_ids):
        """
        Perform tokenization operations.

        :param decoder_input_ids: Input token IDs for the decoder.
        :return: Transposed tokenized output.
        """
        # embedding lookup
        gather_output = self.token_embedding_weight[decoder_input_ids]  # Shape: (len(decoder_input_ids), 384)
        # Add bias
        add_output = gather_output + self.onnx_add_input  # Broadcasting with shape (32, 384)
        # insert dimension at axis=1
        unsqueeze_output = np.expand_dims(add_output, axis=int(self.constant_output_0[0]))  # Shape: (32, 1, 384)
        # Transpose (0, 3, 2, 1) + turn into NHWC (0, 2, 3, 1)
        transpose_output = np.transpose(unsqueeze_output, (0, 2, 1, 3))

        return transpose_output

    def cleanup(self):
        """Clean up resources properly to avoid bus errors."""
        try:
            # Release ONNX sessions
            if hasattr(self, 'encoder_session'):
                del self.encoder_session
            if hasattr(self, 'decoder_init_session'):
                del self.decoder_init_session
            if hasattr(self, 'decoder_cached_session'):
                del self.decoder_cached_session

            # Release configured models
            if hasattr(self, 'encoder_configured_infer_model'):
                del self.encoder_configured_infer_model
            if hasattr(self, 'decoder_configured_infer_model'):
                del self.decoder_configured_infer_model

            # Release bindings
            if hasattr(self, 'encoder_bindings'):
                del self.encoder_bindings
            if hasattr(self, 'decoder_bindings'):
                del self.decoder_bindings

            # Release infer models
            if hasattr(self, 'encoder_infer_model'):
                del self.encoder_infer_model
            if hasattr(self, 'decoder_infer_model'):
                del self.decoder_infer_model

            # Release VDevice last
            if hasattr(self, 'vdevice'):
                del self.vdevice
        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def get_transcript(self, input_mel_spec, debug=False, return_timing=False):
        """
        Main inference loop for processing input data and generating transcriptions.

        Args:
            input_mel_spec: Mel spectrogram input
            debug: Enable debug output
            return_timing: If True, returns (transcription, encoder_time_ms, decoder_time_ms)

        Returns:
            transcription (str) or (transcription, encoder_time_ms, decoder_time_ms) if return_timing=True
        """
        if self.use_onnx_decoder:
            return self._inference_onnx(input_mel_spec, debug=debug, return_timing=return_timing)
        else:
            return self._inference_hef(input_mel_spec, debug=debug, return_timing=return_timing)



    def _run_encoder_hef(self, input_mel_spec, debug=False):
        """Run encoder inference using pre-configured bindings."""
        encoder_start_total = time.time()
        input_mel = np.ascontiguousarray(input_mel_spec)

        # Reuse pre-configured bindings
        self.encoder_bindings.input().set_buffer(input_mel)
        buffer = np.zeros(self.encoder_infer_model.output().shape).astype(np.float32)
        self.encoder_bindings.output().set_buffer(buffer)

        # Measure actual inference time
        inference_start = time.time()
        self.encoder_configured_infer_model.run([self.encoder_bindings], self.timeout_ms)
        inference_time_ms = (time.time() - inference_start) * 1000

        encoded_features = self.encoder_bindings.output().get_buffer().copy()

        encoder_time_ms = (time.time() - encoder_start_total) * 1000

        # Print encoder timing info
        print(f"[TIMING] Encoder total: {encoder_time_ms:.1f}ms")
        print(f"[TIMING] Encoder inference only: {inference_time_ms:.1f}ms")
        print(f"[TIMING] Encoder overhead: {encoder_time_ms - inference_time_ms:.1f}ms")
        if debug:
            print(f"[DEBUG] Encoder output shape: {encoded_features.shape}")
            print(f"[DEBUG] Encoder output range: [{encoded_features.min():.3f}, {encoded_features.max():.3f}]")
            print(f"[DEBUG] Encoder output mean: {encoded_features.mean():.3f}, std: {encoded_features.std():.3f}")

        return encoded_features, encoder_time_ms

    def _run_encoder_onnx(self, input_mel_spec, debug=False):
        """Run encoder inference using ONNX Runtime (CPU)."""
        encoder_start_total = time.time()

        # ONNX encoders expect NCHW format: [1, 80, 1, 1000] for 10s or [1, 80, 3000] for 30s
        # Input should already be in correct format from get_mel_spectrogram
        input_name = self.encoder_session.get_inputs()[0].name

        # Measure actual inference time
        inference_start = time.time()
        encoder_output = self.encoder_session.run(None, {input_name: input_mel_spec})[0]
        inference_time_ms = (time.time() - inference_start) * 1000

        encoder_time_ms = (time.time() - encoder_start_total) * 1000

        # Print encoder timing info
        print(f"[TIMING] Encoder total: {encoder_time_ms:.1f}ms")
        print(f"[TIMING] Encoder inference only: {inference_time_ms:.1f}ms")
        print(f"[TIMING] Encoder overhead: {encoder_time_ms - inference_time_ms:.1f}ms")
        if debug:
            print(f"[DEBUG] Encoder output shape: {encoder_output.shape}")
            print(f"[DEBUG] Encoder output range: [{encoder_output.min():.3f}, {encoder_output.max():.3f}]")
            print(f"[DEBUG] Encoder output mean: {encoder_output.mean():.3f}, std: {encoder_output.std():.3f}")

        return encoder_output, encoder_time_ms

    def _run_decoder_hef(self, encoded_features, debug=False):
        """Run decoder inference using pre-loaded HEF model."""
        decoder_start_total = time.time()

        # NOTE: This implementation uses forced decoder tokens, which differs from Hailo's reference implementation.
        # Hailo's reference starts with only [50258] (<|startoftranscript|>) and generates all tokens including
        # language, task, and timestamp tokens. We instead force the standard Whisper control tokens:
        #   50258: <|startoftranscript|>
        #   50259: <|en|> (English language)
        #   50359: <|transcribe|> (transcription task)
        #   50363: <|notimestamps|> (no timestamp prediction)
        # This matches the ONNX decoder behavior and reduces token generation overhead by ~3 tokens per inference.
        forced_tokens = [50258, 50259, 50359, 50363]
        generated_tokens = []
        token_times = []

        # Initialize decoder_input_ids with forced tokens
        decoder_input_ids = np.array([forced_tokens], dtype=np.int64)
        decoder_input_ids = np.concatenate(
            [decoder_input_ids, np.zeros((1, self.decoding_sequence_length - len(forced_tokens)), dtype=np.int64)], axis=1
        )

        if debug:
            print(f"[DEBUG] Decoder input IDs initialized with forced tokens: {decoder_input_ids[0][:8]}")
            print(f"[DEBUG] Forced tokens: {forced_tokens}")
            print(f"[DEBUG] Starting decoder with {self.decoding_sequence_length} max tokens")

        # Reuse pre-configured bindings
        # Run Decoder Iteratively
        inference_times = []  # Track pure inference times
        num_forced = len(forced_tokens)

        for i in range(num_forced - 1, self.decoding_sequence_length - 1):
            step_start = time.time()

            tokenized_ids = self._tokenization(decoder_input_ids)

            self.decoder_bindings.input(f"{self.decoder_model_name}/input_layer1").set_buffer(encoded_features)
            self.decoder_bindings.input(f"{self.decoder_model_name}/input_layer2").set_buffer(tokenized_ids)

            buffers = [
                np.zeros(self.decoder_infer_model.output(name).shape).astype(np.float32)
                for name in self.decoder_sorted_output_names
            ]

            for name, buffer in zip(self.decoder_sorted_output_names, buffers):
                self.decoder_bindings.output(name).set_buffer(buffer)

            # Measure actual inference time
            inference_start = time.time()
            self.decoder_configured_infer_model.run([self.decoder_bindings], self.timeout_ms)
            inference_time_ms = (time.time() - inference_start) * 1000
            inference_times.append(inference_time_ms)

            decoder_outputs = np.concatenate(
                [self.decoder_bindings.output(name).get_buffer() for name in self.decoder_sorted_output_names], axis=2
            )

            # Decoder post-processing (apply penalty only to generated tokens, not forced ones)
            # Match ONNX decoder approach: penalize ALL generated tokens with proper positive/negative handling
            next_token_logits = decoder_outputs[:, i].flatten().copy()
            tokens_to_penalize = set(generated_tokens)
            for token_id in tokens_to_penalize:
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= REPETITION_PENALTY
                else:
                    next_token_logits[token_id] *= REPETITION_PENALTY
            next_token = np.argmax(next_token_logits)

            step_time_ms = (time.time() - step_start) * 1000
            token_times.append(step_time_ms)

            generated_tokens.append(next_token)
            decoder_input_ids[0][i + 1] = next_token

            # Debug: Show token generation with timing
            if debug:
                token_text = self.tokenizer.decode([next_token]) if next_token < len(self.tokenizer) else f"<{next_token}>"
                top_3_tokens = np.argsort(decoder_outputs[:, i].flatten())[-3:][::-1]
                print(f"[TIMING] Step {i}: {step_time_ms:.1f}ms | token={next_token} '{token_text}' | top-3: {top_3_tokens}")

            if next_token == self.tokenizer.eos_token_id:
                if debug:
                    print(f"[DEBUG] EOS token {self.tokenizer.eos_token_id} reached at step {i}")
                break

        # Convert token IDs to text
        transcription = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        total_decoder_time = (time.time() - decoder_start_total) * 1000

        # Print decoder timing summary
        if token_times:
            sum_token_times = sum(token_times)
            sum_inference_times = sum(inference_times)
            avg_token_time = np.mean(token_times)
            avg_inference_time = np.mean(inference_times)
            print(f"\n[TIMING] Decoder Summary:")
            print(f"  Total tokens: {len(token_times)}")
            print(f"  Total time: {total_decoder_time:.1f}ms")
            print(f"  Avg per token (inference only): {avg_inference_time:.1f}ms")
            print(f"  Avg per token (total): {avg_token_time:.1f}ms")
            print(f"  Min/Max: {min(token_times):.1f}ms / {max(token_times):.1f}ms")
            if debug:
                print(f"  Sum of token times: {sum_token_times:.1f}ms")
                print(f"  Sum of inference times: {sum_inference_times:.1f}ms")
                print(f"  Avg per token (inference only): {avg_inference_time:.1f}ms")
                print(f"  Per-token overhead: {avg_token_time - avg_inference_time:.1f}ms")
                print(f"  Total overhead: {total_decoder_time - sum_token_times:.1f}ms")

        return transcription, total_decoder_time

    def _run_decoder_onnx(self, encoded_features, debug=False):
        """Run ONNX decoder with KV-cache."""
        decoder_start_total = time.time()

        # Get output names for cache mapping (from pre-loaded sessions)
        decoder_outputs = [output.name for output in self.decoder_init_session.get_outputs()]
        decoder_with_past_outputs = [output.name for output in self.decoder_cached_session.get_outputs()]

        forced_tokens = [50258, 50259, 50359, 50363]  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        generated_tokens = forced_tokens.copy()
        token_times = []
        inference_times = []  # Track pure inference times
        past_key_values_dict = {}

        if debug:
            print(f"[DEBUG] Starting ONNX decoder with forced tokens: {forced_tokens}")

        # Run Decoder with KV-cache
        max_new_tokens = self.decoding_sequence_length - len(forced_tokens)
        for step in range(max_new_tokens):
            step_start = time.time()

            if not past_key_values_dict:
                # First pass: process forced tokens and initialize cache
                input_ids = np.array([forced_tokens], dtype=np.int64)

                inference_start = time.time()
                outputs = self.decoder_init_session.run(None, {
                    'input_ids': input_ids,
                    'encoder_hidden_states': encoded_features
                })
                inference_time_ms = (time.time() - inference_start) * 1000
                inference_times.append(inference_time_ms)

                logits = outputs[0]

                # Store ALL cache outputs
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

                inference_start = time.time()
                outputs = self.decoder_cached_session.run(None, inputs)
                inference_time_ms = (time.time() - inference_start) * 1000
                inference_times.append(inference_time_ms)

                logits = outputs[0]

                # Update cache for next iteration
                for idx, output_name in enumerate(decoder_with_past_outputs[1:], 1):
                    if "present" in output_name:
                        past_name = output_name.replace("present.", "past_key_values.")
                        past_key_values_dict[past_name] = outputs[idx]

                next_token_logits = logits[0, -1, :].copy()

            # Apply repetition penalty
            tokens_to_penalize = set(generated_tokens[len(forced_tokens):])
            for token_id in tokens_to_penalize:
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= REPETITION_PENALTY
                else:
                    next_token_logits[token_id] *= REPETITION_PENALTY

            next_token = np.argmax(next_token_logits)

            step_time_ms = (time.time() - step_start) * 1000
            token_times.append(step_time_ms)

            # Debug: Show token generation with timing
            if debug:
                token_text = self.tokenizer.decode([next_token]) if next_token < len(self.tokenizer) else f"<{next_token}>"
                top_3_tokens = np.argsort(next_token_logits)[-3:][::-1]
                print(f"[TIMING] Step {step}: {step_time_ms:.1f}ms | token={next_token} '{token_text}' | top-3: {top_3_tokens}")

            # Check for EOS or end-of-text tokens
            if next_token in [50256, 50257]:
                if debug:
                    print(f"[DEBUG] EOS token {next_token} reached at step {step}")
                break

            generated_tokens.append(int(next_token))

        # Convert token IDs to text
        transcription = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        total_decoder_time = (time.time() - decoder_start_total) * 1000

        # Print decoder timing summary
        sum_token_times = sum(token_times)
        sum_inference_times = sum(inference_times)
        avg_token_time = np.mean(token_times)
        avg_inference_time = np.mean(inference_times)
        print(f"\n[TIMING] Decoder Summary (ONNX with KV-cache):")
        print(f"  Total tokens: {len(token_times)}")
        print(f"  Total time: {total_decoder_time:.1f}ms")
        print(f"  Avg per token (total): {avg_token_time:.1f}ms")
        print(f"  Min/Max: {min(token_times):.1f}ms / {max(token_times):.1f}ms")
        if len(token_times) > 1:
            print(f"  First token: {token_times[0]:.1f}ms")
            print(f"  Subsequent tokens: {np.mean(token_times[1:]):.1f}ms")
        if debug:
            print(f"  Sum of token times: {sum_token_times:.1f}ms")
            print(f"  Sum of inference times: {sum_inference_times:.1f}ms")
            print(f"  Avg per token (inference only): {avg_inference_time:.1f}ms")
            print(f"  Per-token overhead: {avg_token_time - avg_inference_time:.1f}ms")
            print(f"  Total overhead: {total_decoder_time - sum_token_times:.1f}ms")

        return transcription, total_decoder_time

    def _inference_hef(self, input_mel_spec, debug=False, return_timing=False):
        """Run full HEF decoder inference (encoder can be HEF or ONNX)."""
        # Use appropriate encoder
        if self.use_onnx_encoder:
            encoded_features, encoder_time_ms = self._run_encoder_onnx(input_mel_spec, debug=debug)
        else:
            encoded_features, encoder_time_ms = self._run_encoder_hef(input_mel_spec, debug=debug)

        transcription, decoder_time_ms = self._run_decoder_hef(encoded_features, debug=debug)

        print(f"\n[TIMING] Total (Encoder + Decoder): {encoder_time_ms + decoder_time_ms:.1f}ms")

        if return_timing:
            return transcription, encoder_time_ms, decoder_time_ms
        return transcription

    def _inference_onnx(self, input_mel_spec, debug=False, return_timing=False):
        """
        Inference using ONNX decoder with KV-cache (encoder can be HEF or ONNX).
        """
        # Use appropriate encoder
        if self.use_onnx_encoder:
            encoded_features, encoder_time_ms = self._run_encoder_onnx(input_mel_spec, debug=debug)
        else:
            encoded_features, encoder_time_ms = self._run_encoder_hef(input_mel_spec, debug=debug)

        transcription, decoder_time_ms = self._run_decoder_onnx(encoded_features, debug=debug)

        print(f"\n[TIMING] Total (Encoder + Decoder): {encoder_time_ms + decoder_time_ms:.1f}ms")

        if return_timing:
            return transcription, encoder_time_ms, decoder_time_ms
        return transcription


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper on Hailo Inference Pipeline (Supports HEF/ONNX encoders and decoders)")

    # Encoder options (mutually exclusive)
    encoder_group = parser.add_mutually_exclusive_group(required=True)
    encoder_group.add_argument("--encoder_hef_file", type=str,
                        help="Path to encoder HEF file (Hailo hardware, 10s)")
    encoder_group.add_argument("--encoder_onnx_file", type=str,
                        help="Path to encoder ONNX file (10s, CPU)")
    encoder_group.add_argument("--encoder_orig_onnx_file", type=str,
                        help="Path to original encoder ONNX file (30s, FP32, CPU)")

    # Decoder options
    parser.add_argument("--decoder_hef_file", type=str, default=None,
                        help="Path to decoder HEF file (not used if --decoder_onnx_dir is provided)")
    parser.add_argument("--decoder_onnx_dir", type=str, default=None,
                        help="Path to directory containing ONNX decoder files (decoder_model.onnx and decoder_with_past_model.onnx)")

    # Audio input (mutually exclusive)
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument("--audio_file", type=str,
                        help="Path to audio file to transcribe")
    audio_group.add_argument("--audio_folder", type=str,
                        help="Path to folder containing audio files (.wav/.mp3) and .txt ground truth files for WER evaluation")

    # Other options
    parser.add_argument("--variant", type=str, default="tiny",
                        help="Model variant (e.g., 'tiny', 'base')")
    parser.add_argument("--decoder_assets_path", type=str, default=None,
                        help="Path to decoder assets directory (only for HEF decoder, default: ./decoder_assets)")
    parser.add_argument("--multi_process_service", action="store_true",
                        help="Enable multi-process service mode (HEF only)")
    parser.add_argument("--num_iterations", type=int, default=1,
                        help="Number of iterations for single file benchmarking (default: 1)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output (default: False)")

    args = parser.parse_args()

    # Check if audio file/folder exists
    if args.audio_file and not os.path.exists(args.audio_file):
        raise ValueError(f"Error: Audio file not found: {args.audio_file}")
    if args.audio_folder and not os.path.exists(args.audio_folder):
        raise ValueError(f"Error: Audio folder not found: {args.audio_folder}")

    # Validate decoder arguments
    if args.decoder_onnx_dir is None and args.decoder_hef_file is None:
        raise ValueError("Error: Either --decoder_hef_file or --decoder_onnx_dir must be provided")

    # Determine encoder path and type
    encoder_path = args.encoder_hef_file or args.encoder_onnx_file or args.encoder_orig_onnx_file
    encoder_onnx_path = args.encoder_onnx_file or args.encoder_orig_onnx_file
    encoder_is_orig = args.encoder_orig_onnx_file is not None

    # Validate encoder/decoder compatibility (only HEF encoder works with HEF decoder)
    if args.decoder_hef_file and (args.encoder_onnx_file or args.encoder_orig_onnx_file):
        print("Warning: HEF decoder requires VDevice. Using ONNX encoder with HEF decoder may not work correctly.")

    # num_iterations only makes sense for single file
    if args.audio_folder and args.num_iterations > 1:
        print("Warning: --num_iterations is ignored when using --audio_folder")
        args.num_iterations = 1


    print(f"Initializing Whisper on Hailo Pipeline...")

    # Print encoder info
    if args.encoder_hef_file:
        print(f"  Encoder: HEF (Hailo hardware, 10s)")
        print(f"    Path: {args.encoder_hef_file}")
    elif args.encoder_orig_onnx_file:
        print(f"  Encoder: ONNX Original (CPU, 30s)")
        print(f"    Path: {args.encoder_orig_onnx_file}")
    else:
        print(f"  Encoder: ONNX (CPU, 10s)")
        print(f"    Path: {args.encoder_onnx_file}")

    # Print decoder info
    if args.decoder_onnx_dir:
        print(f"  Decoder: ONNX (KV-cache)")
        print(f"    Path: {args.decoder_onnx_dir}")
    else:
        print(f"  Decoder: HEF")
        print(f"    Path: {args.decoder_hef_file}")

    print(f"  Variant: {args.variant}")

    # Print audio input info
    if args.audio_file:
        print(f"  Audio file: {args.audio_file}")
        if args.num_iterations > 1:
            print(f"  Iterations: {args.num_iterations}")
    else:
        print(f"  Audio folder: {args.audio_folder}")

    print()

    # Initialize the pipeline
    pipeline = HailoWhisperPipeline(
        encoder_hef_path=args.encoder_hef_file,
        encoder_onnx_path=encoder_onnx_path,
        encoder_is_orig_onnx=encoder_is_orig,
        decoder_hef_path=args.decoder_hef_file,
        variant=args.variant,
        decoder_assets_path=args.decoder_assets_path,
        multi_process_service=args.multi_process_service,
        decoder_onnx_dir=args.decoder_onnx_dir
    )

    print("Pipeline initialized successfully!")
    print()

    # Mode 1: Single audio file
    if args.audio_file:
        # Preprocess the audio file
        print("Preprocessing audio...")
        preprocess_start = time.time()
        mel_spectrogram = get_mel_spectrogram(
            args.audio_file,
            target_duration=pipeline.encoder_target_duration,
            format_4d=pipeline.encoder_format_4d,
            is_nhwc=pipeline.encoder_is_nhwc
        )
        preprocess_time_ms = (time.time() - preprocess_start) * 1000
        print(f"  Mel spectrogram shape: {mel_spectrogram.shape}")
        print(f"[TIMING] Preprocessing: {preprocess_time_ms:.1f}ms")
        print()

        # Run inference (potentially multiple iterations for benchmarking)
        encoder_times = []
        decoder_times = []
        transcription = None

        for iteration in range(args.num_iterations):
            if args.num_iterations > 1:
                print(f"Iteration {iteration + 1}/{args.num_iterations}...")

            t1 = time.time()
            result = pipeline.get_transcript(mel_spectrogram, debug=args.debug, return_timing=True)
            total_time_ms = (time.time() - t1) * 1000

            if args.num_iterations > 1:
                # Multiple iterations: collect timing stats
                transcription, encoder_time, decoder_time = result
                encoder_times.append(encoder_time)
                decoder_times.append(decoder_time)
                print(f"[TIMING] Total get_transcript() time: {total_time_ms:.1f}ms\n")
            else:
                # Single iteration: standard output (transcription only or with timing)
                if isinstance(result, tuple):
                    transcription = result[0]
                else:
                    transcription = result
                print(f"[TIMING] Total get_transcript() time: {total_time_ms:.1f}ms")

        # Print benchmarking statistics if multiple iterations
        if args.num_iterations > 1:
            print(f"\n{'='*70}")
            print(f"BENCHMARKING STATISTICS ({args.num_iterations} iterations)")
            print(f"{'='*70}")
            print(f"Encoder times (ms):")
            print(f"  Mean: {np.mean(encoder_times):.1f} ± {np.std(encoder_times):.1f}")
            print(f"  Min/Max: {np.min(encoder_times):.1f} / {np.max(encoder_times):.1f}")
            print(f"  Median: {np.median(encoder_times):.1f}")
            print(f"\nDecoder times (ms):")
            print(f"  Mean: {np.mean(decoder_times):.1f} ± {np.std(decoder_times):.1f}")
            print(f"  Min/Max: {np.min(decoder_times):.1f} / {np.max(decoder_times):.1f}")
            print(f"  Median: {np.median(decoder_times):.1f}")
            total_times = np.array(encoder_times) + np.array(decoder_times)
            print(f"\nTotal times (ms):")
            print(f"  Mean: {np.mean(total_times):.1f} ± {np.std(total_times):.1f}")
            print(f"  Min/Max: {np.min(total_times):.1f} / {np.max(total_times):.1f}")
            print(f"  Median: {np.median(total_times):.1f}")
            print(f"{'='*70}")

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
        total_encoder_time = 0.0
        total_decoder_time = 0.0

        for idx, (audio_file, ground_truth) in enumerate(dataset.items(), 1):
            print(f"[{idx}/{len(dataset)}] Processing: {Path(audio_file).name}")

            try:
                # Preprocess audio
                mel_spectrogram = get_mel_spectrogram(
                    audio_file,
                    target_duration=pipeline.encoder_target_duration,
                    format_4d=pipeline.encoder_format_4d,
                    is_nhwc=pipeline.encoder_is_nhwc
                )

                # Run inference with timing
                transcription, encoder_time_ms, decoder_time_ms = pipeline.get_transcript(
                    mel_spectrogram, debug=args.debug, return_timing=True
                )

                # Calculate WER/CER
                wer_score, cer_score = calculate_wer(transcription, ground_truth)

                total_wer += wer_score
                total_cer += cer_score
                total_encoder_time += encoder_time_ms
                total_decoder_time += decoder_time_ms

                results.append({
                    'audio_file': audio_file,
                    'ground_truth': ground_truth,
                    'transcription': transcription,
                    'wer': wer_score,
                    'cer': cer_score,
                    'encoder_time_ms': encoder_time_ms,
                    'decoder_time_ms': decoder_time_ms
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
            avg_encoder_time = total_encoder_time / num_successful
            avg_decoder_time = total_decoder_time / num_successful
            avg_total_time = avg_encoder_time + avg_decoder_time

            print(f"\n{'='*70}")
            print("EVALUATION SUMMARY")
            print(f"{'='*70}")
            print(f"  Samples processed:     {num_successful}/{len(dataset)}")
            print(f"  Average WER:           {avg_wer:.2f}%")
            print(f"  Average CER:           {avg_cer:.2f}%")
            print(f"\n  Timing:")
            print(f"    Avg Encoder time:    {avg_encoder_time:.1f}ms")
            print(f"    Avg Decoder time:    {avg_decoder_time:.1f}ms")
            print(f"    Avg Total time:      {avg_total_time:.1f}ms")
            print(f"{'='*70}")
        else:
            print("\nNo samples were successfully processed.")

    # Clean up resources to avoid bus errors
    pipeline.cleanup()


if __name__ == "__main__":
    exit(main())
