#!/usr/bin/env python3
"""

This to be run on HAILO directly

Standalone Whisper on Hailo Pipeline
Based on HailoWhisperPipeline from Hailo Application Code Examples
"""
import numpy as np
import os
import argparse
import time
from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)
from transformers import AutoTokenizer
from queue import Queue, Empty
from threading import Thread
import onnxruntime as ort

# For efficient audio preprocessing
import librosa
import torch


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


def apply_repetition_penalty(logits, generated_tokens, penalty=1.5, last_window=8):
    """
    Apply repetition penalty to the logits.
    Args:
        logits: The logits from the model (shape: (vocab_size,)).
        generated_tokens: List of previously generated tokens.
        penalty: The penalty factor (higher values discourage repetitions).
    Returns:
        logits: The logits with repetition penalty applied.
    """
    logits = np.squeeze(logits, axis=0)
    recent_tokens = generated_tokens[-last_window:] if len(generated_tokens) >= last_window else generated_tokens

    # Combine the tokens from both windows
    recent_tokens = set(recent_tokens)

    for token in recent_tokens:
        if token not in excluded_tokens:
            logits[token] /= penalty
    return logits


# ============================================================================
# Audio Preprocessing (Hailo-efficient implementation)
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
    Load audio file and convert to mel spectrogram (Hailo-efficient implementation).

    Args:
        audio_file: Path to audio file
        chunk_length: Duration in seconds (default: 10 for 10s encoder)
        is_nhwc: Whether to transpose to NHWC format (default: True for Hailo models)

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

    # Transpose to NHWC if needed (Hailo models expect this)
    if is_nhwc:
        mel = np.transpose(mel, [0, 2, 3, 1])  # [1, 80, 1, 1000] -> [1, 1, 1000, 80]

    return mel.astype(np.float32)


# ============================================================================
# Main Pipeline Class
# ============================================================================

class HailoWhisperPipeline:
    """
    A pipeline for running inference using Hailo's Whisper models.
    """

    def __init__(self, encoder_model_path: str, decoder_model_path: str = None, variant="tiny",
                 decoder_assets_path=None, multi_process_service=False, decoder_onnx_dir=None):
        """
        Initialize the pipeline.

        :param encoder_model_path: Path to the encoder HEF file.
        :param decoder_model_path: Path to the decoder HEF file (not used if decoder_onnx_dir is provided).
        :param variant: Model variant (e.g., "tiny", "base").
        :param decoder_assets_path: Path to decoder assets directory (only for HEF decoder).
        :param multi_process_service: Enable multi-process service mode.
        :param decoder_onnx_dir: Path to directory containing ONNX decoder files (decoder_model.onnx and decoder_with_past_model.onnx).
                                 If provided, uses ONNX decoder instead of HEF decoder.
        """
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        self.decoder_onnx_dir = decoder_onnx_dir
        self.timeout_ms = 100000000
        self.variant = variant

        self.decoding_sequence_length = 32 if ("tiny" in self.variant) else 24
        self.multi_process_service = multi_process_service

        # Determine decoder mode
        self.use_onnx_decoder = decoder_onnx_dir is not None

        if self.use_onnx_decoder:
            # ONNX decoder mode - no token embeddings needed
            print(f"[INFO] Using ONNX decoder from: {decoder_onnx_dir}")
            self.decoder_init_path = os.path.join(decoder_onnx_dir, "decoder_model.onnx")
            self.decoder_cached_path = os.path.join(decoder_onnx_dir, "decoder_with_past_model.onnx")

            if not os.path.exists(self.decoder_init_path):
                raise FileNotFoundError(f"ONNX decoder not found: {self.decoder_init_path}")
            if not os.path.exists(self.decoder_cached_path):
                raise FileNotFoundError(f"ONNX cached decoder not found: {self.decoder_cached_path}")
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

        self._load_tokenizer()

        self.data_queue = Queue()
        self.results_queue = Queue()
        self.running = True
        self.thread = Thread(target=self._inference_loop)
        self.thread.start()

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

    def _inference_loop(self):
        """
        Main inference loop for processing input data and generating transcriptions.
        """
        if self.use_onnx_decoder:
            self._inference_loop_onnx()
        else:
            self._inference_loop_hef()

    def _inference_loop_hef(self):
        """
        Inference loop using HEF decoder (without KV-cache).
        """
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        if self.multi_process_service:
            params.multi_process_service = True
            params.group_id = "SHARED"

        # get output info
        decoder_hef = HEF(self.decoder_model_path)
        sorted_output_names = decoder_hef.get_sorted_output_names()
        decoder_model_name = decoder_hef.get_network_group_names()[0]

        with VDevice(params) as vdevice:
            encoder_infer_model = vdevice.create_infer_model(self.encoder_model_path)
            decoder_infer_model = vdevice.create_infer_model(self.decoder_model_path)
            encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
            encoder_infer_model.output().set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input(f"{decoder_model_name}/input_layer1").set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input(f"{decoder_model_name}/input_layer2").set_format_type(FormatType.FLOAT32)

            # model's outputs will be concatenated on the host
            for output_name in sorted_output_names:
                decoder_infer_model.output(output_name).set_format_type(FormatType.FLOAT32)


            with encoder_infer_model.configure() as encoder_configured_infer_model:
                with decoder_infer_model.configure() as decoder_configured_infer_model:
                    encoder_bindings = encoder_configured_infer_model.create_bindings()
                    decoder_bindings = decoder_configured_infer_model.create_bindings()

                    while self.running:
                        try:
                            # Wait for new data with a timeout to allow clean exit
                            input_mel = self.data_queue.get(timeout=1)

                            transcriptions = []
                            input_mel = np.ascontiguousarray(input_mel)
                            encoder_bindings.input().set_buffer(input_mel)
                            buffer = np.zeros(encoder_infer_model.output().shape).astype(np.float32)
                            encoder_bindings.output().set_buffer(buffer)

                            # Time encoder inference
                            encoder_start = time.time()
                            encoder_configured_infer_model.run([encoder_bindings], self.timeout_ms)
                            encoder_time_ms = (time.time() - encoder_start) * 1000
                            encoded_features = encoder_bindings.output().get_buffer()

                            # Debug: Print encoder output info
                            print(f"[TIMING] Encoder: {encoder_time_ms:.1f}ms")
                            print(f"[DEBUG] Encoder output shape: {encoded_features.shape}")
                            print(f"[DEBUG] Encoder output range: [{encoded_features.min():.3f}, {encoded_features.max():.3f}]")
                            print(f"[DEBUG] Encoder output mean: {encoded_features.mean():.3f}, std: {encoded_features.std():.3f}")

                            # Decoder - Hailo approach: start with just <|startoftranscript|> and generate everything
                            start_token_id = [50258]
                            decoder_input_ids = np.array(
                                [[start_token_id[0]]], dtype=np.int64
                            )  # Shape (1,1)
                            decoder_input_ids = np.concatenate(
                                [decoder_input_ids, np.zeros((1, self.decoding_sequence_length - 1), dtype=np.int64)], axis=1
                            )

                            print(f"[DEBUG] Decoder input IDs initialized: {decoder_input_ids[0][:8]}")

                            generated_tokens = []
                            decoder_outputs = None
                            token_times = []

                            print(f"[DEBUG] Starting decoder with {self.decoding_sequence_length} max tokens")

                            # Run Decoder Iteratively
                            for i in range(self.decoding_sequence_length - 1):
                                step_start = time.time()

                                tokenized_ids = self._tokenization(decoder_input_ids)

                                decoder_bindings.input(f"{decoder_model_name}/input_layer1").set_buffer(encoded_features)
                                decoder_bindings.input(f"{decoder_model_name}/input_layer2").set_buffer(tokenized_ids)

                                buffers = [
                                    np.zeros(decoder_infer_model.output(name).shape).astype(np.float32) for name in sorted_output_names
                                ]

                                for name, buffer in zip(sorted_output_names, buffers):
                                    decoder_bindings.output(name).set_buffer(buffer)

                                decoder_configured_infer_model.run([decoder_bindings], self.timeout_ms)  # run decoder

                                decoder_outputs = np.concatenate(
                                    [decoder_bindings.output(name).get_buffer() for name in sorted_output_names], axis=2
                                )


                                # Decoder post-processing
                                repetition_penalty = 1.5
                                logits = apply_repetition_penalty(decoder_outputs[:, i], generated_tokens, penalty=repetition_penalty)
                                next_token = np.argmax(logits)

                                step_time_ms = (time.time() - step_start) * 1000
                                token_times.append(step_time_ms)

                                # Debug: Show token generation with timing
                                token_text = self.tokenizer.decode([next_token]) if next_token < len(self.tokenizer) else f"<{next_token}>"
                                top_3_tokens = np.argsort(decoder_outputs[:, i].flatten())[-3:][::-1]
                                print(f"[TIMING] Step {i}: {step_time_ms:.1f}ms | token={next_token} '{token_text}' | top-3: {top_3_tokens}")

                                generated_tokens.append(next_token)
                                decoder_input_ids[0][i + 1] = np.array([[next_token]], dtype=np.int64)

                                if next_token == self.tokenizer.eos_token_id:
                                    print(f"[DEBUG] EOS token {self.tokenizer.eos_token_id} reached at step {i}")
                                    break

                            # Convert token IDs to text
                            transcription = self.tokenizer.decode(
                                generated_tokens, skip_special_tokens=True
                            )

                            # Print decoder timing summary
                            if token_times:
                                total_decoder_time = sum(token_times)
                                avg_token_time = np.mean(token_times)
                                print(f"\n[TIMING] Decoder Summary:")
                                print(f"  Total tokens: {len(token_times)}")
                                print(f"  Total time: {total_decoder_time:.1f}ms")
                                print(f"  Avg per token: {avg_token_time:.1f}ms")
                                print(f"  Min/Max: {min(token_times):.1f}ms / {max(token_times):.1f}ms")
                                print(f"\n[TIMING] Total (Encoder + Decoder): {encoder_time_ms + total_decoder_time:.1f}ms")

                            self.results_queue.put(transcription)
                            transcriptions.append(transcription)
                        except Empty:
                            pass  # No data yet, continue looping

    def _inference_loop_onnx(self):
        """
        Inference loop using ONNX decoder with KV-cache (hybrid mode: Hailo encoder + ONNX decoder).
        """
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        if self.multi_process_service:
            params.multi_process_service = True
            params.group_id = "SHARED"

        # Load ONNX decoder sessions
        decoder_init_session = ort.InferenceSession(self.decoder_init_path)
        decoder_cached_session = ort.InferenceSession(self.decoder_cached_path)

        # Get output names for cache mapping
        decoder_outputs = [output.name for output in decoder_init_session.get_outputs()]
        decoder_with_past_outputs = [output.name for output in decoder_cached_session.get_outputs()]

        with VDevice(params) as vdevice:
            encoder_infer_model = vdevice.create_infer_model(self.encoder_model_path)
            encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
            encoder_infer_model.output().set_format_type(FormatType.FLOAT32)

            with encoder_infer_model.configure() as encoder_configured_infer_model:
                encoder_bindings = encoder_configured_infer_model.create_bindings()

                while self.running:
                    try:
                        # Wait for new data with a timeout to allow clean exit
                        input_mel = self.data_queue.get(timeout=1)

                        transcriptions = []
                        input_mel = np.ascontiguousarray(input_mel)
                        encoder_bindings.input().set_buffer(input_mel)
                        buffer = np.zeros(encoder_infer_model.output().shape).astype(np.float32)
                        encoder_bindings.output().set_buffer(buffer)

                        # Time encoder inference
                        encoder_start = time.time()
                        encoder_configured_infer_model.run([encoder_bindings], self.timeout_ms)
                        encoder_time_ms = (time.time() - encoder_start) * 1000
                        encoded_features = encoder_bindings.output().get_buffer()

                        # Debug: Print encoder output info
                        print(f"[TIMING] Encoder: {encoder_time_ms:.1f}ms")
                        print(f"[DEBUG] Encoder output shape: {encoded_features.shape}")
                        print(f"[DEBUG] Encoder output range: [{encoded_features.min():.3f}, {encoded_features.max():.3f}]")
                        print(f"[DEBUG] Encoder output mean: {encoded_features.mean():.3f}, std: {encoded_features.std():.3f}")

                        # Decoder - Use Whisper's forced decoder start tokens for ONNX
                        forced_tokens = [50258, 50259, 50359, 50363]  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
                        generated_tokens = forced_tokens.copy()
                        token_times = []
                        past_key_values_dict = {}

                        print(f"[DEBUG] Starting ONNX decoder with forced tokens: {forced_tokens}")

                        # Run Decoder with KV-cache
                        max_new_tokens = self.decoding_sequence_length - len(forced_tokens)
                        for step in range(max_new_tokens):
                            step_start = time.time()

                            if not past_key_values_dict:
                                # First pass: process forced tokens and initialize cache
                                input_ids = np.array([forced_tokens], dtype=np.int64)
                                outputs = decoder_init_session.run(None, {
                                    'input_ids': input_ids,
                                    'encoder_hidden_states': encoded_features
                                })
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

                                outputs = decoder_cached_session.run(None, inputs)
                                logits = outputs[0]

                                # Update cache for next iteration
                                for idx, output_name in enumerate(decoder_with_past_outputs[1:], 1):
                                    if "present" in output_name:
                                        past_name = output_name.replace("present.", "past_key_values.")
                                        past_key_values_dict[past_name] = outputs[idx]

                                next_token_logits = logits[0, -1, :].copy()

                            # Apply repetition penalty
                            repetition_penalty = 1.5
                            tokens_to_penalize = set(generated_tokens[len(forced_tokens):])
                            for token_id in tokens_to_penalize:
                                if next_token_logits[token_id] > 0:
                                    next_token_logits[token_id] /= repetition_penalty
                                else:
                                    next_token_logits[token_id] *= repetition_penalty

                            next_token = np.argmax(next_token_logits)

                            step_time_ms = (time.time() - step_start) * 1000
                            token_times.append(step_time_ms)

                            # Debug: Show token generation with timing
                            token_text = self.tokenizer.decode([next_token]) if next_token < len(self.tokenizer) else f"<{next_token}>"
                            top_3_tokens = np.argsort(next_token_logits)[-3:][::-1]
                            print(f"[TIMING] Step {step}: {step_time_ms:.1f}ms | token={next_token} '{token_text}' | top-3: {top_3_tokens}")

                            # Check for EOS or end-of-text tokens
                            if next_token in [50256, 50257]:
                                print(f"[DEBUG] EOS token {next_token} reached at step {step}")
                                break

                            generated_tokens.append(int(next_token))

                        # Convert token IDs to text (skip forced tokens)
                        transcription = self.tokenizer.decode(
                            generated_tokens, skip_special_tokens=True
                        )

                        # Print decoder timing summary
                        if token_times:
                            total_decoder_time = sum(token_times)
                            avg_token_time = np.mean(token_times)
                            print(f"\n[TIMING] Decoder Summary (ONNX with KV-cache):")
                            print(f"  Total tokens: {len(token_times)}")
                            print(f"  Total time: {total_decoder_time:.1f}ms")
                            print(f"  Avg per token: {avg_token_time:.1f}ms")
                            print(f"  Min/Max: {min(token_times):.1f}ms / {max(token_times):.1f}ms")
                            if len(token_times) > 1:
                                print(f"  First token: {token_times[0]:.1f}ms")
                                print(f"  Subsequent tokens: {np.mean(token_times[1:]):.1f}ms")
                            print(f"\n[TIMING] Total (Encoder + Decoder): {encoder_time_ms + total_decoder_time:.1f}ms")

                        self.results_queue.put(transcription)
                        transcriptions.append(transcription)
                    except Empty:
                        pass  # No data yet, continue looping

    def send_data(self, data):
        """
        Send new data to the queue.

        :param data: Input data to process.
        """
        self.data_queue.put(data)

    def get_transcription(self):
        """
        Retrieve the next transcription result.

        :return: Transcription result.
        """
        return self.results_queue.get()

    def stop(self):
        """
        Stop the processing loop.
        """
        self.running = False
        self.thread.join()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper on Hailo Inference Pipeline (Hybrid Mode Supported)")
    parser.add_argument("--encoder_hef_file", type=str, required=True,
                        help="Path to encoder HEF file")
    parser.add_argument("--decoder_hef_file", type=str, default=None,
                        help="Path to decoder HEF file (not used if --decoder_onnx_dir is provided)")
    parser.add_argument("--audio_file", type=str, required=True,
                        help="Path to audio file to transcribe")
    parser.add_argument("--variant", type=str, default="tiny",
                        help="Model variant (e.g., 'tiny', 'base')")
    parser.add_argument("--decoder_assets_path", type=str, default=None,
                        help="Path to decoder assets directory (only for HEF decoder, default: ./decoder_assets)")
    parser.add_argument("--decoder_onnx_dir", type=str, default=None,
                        help="Path to directory containing ONNX decoder files (enables hybrid mode: Hailo encoder + ONNX decoder with KV-cache)")
    parser.add_argument("--multi_process_service", action="store_true",
                        help="Enable multi-process service mode")

    args = parser.parse_args()

    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1

    # Validate decoder arguments
    if args.decoder_onnx_dir is None and args.decoder_hef_file is None:
        print("Error: Either --decoder_hef_file or --decoder_onnx_dir must be provided")
        return 1

    print(f"Initializing Whisper on Hailo Pipeline...")
    print(f"  Encoder HEF: {args.encoder_hef_file}")
    if args.decoder_onnx_dir:
        print(f"  Decoder: ONNX (hybrid mode)")
        print(f"  ONNX decoder dir: {args.decoder_onnx_dir}")
    else:
        print(f"  Decoder: HEF")
        print(f"  Decoder HEF: {args.decoder_hef_file}")
    print(f"  Variant: {args.variant}")
    print(f"  Audio file: {args.audio_file}")
    print()

    try:
        # Initialize the pipeline
        pipeline = HailoWhisperPipeline(
            encoder_model_path=args.encoder_hef_file,
            decoder_model_path=args.decoder_hef_file,
            variant=args.variant,
            decoder_assets_path=args.decoder_assets_path,
            multi_process_service=args.multi_process_service,
            decoder_onnx_dir=args.decoder_onnx_dir
        )

        print("Pipeline initialized successfully!")
        print()

        # Preprocess the audio file
        print("Preprocessing audio...")
        preprocess_start = time.time()
        mel_spectrogram = get_audio(args.audio_file)
        preprocess_time_ms = (time.time() - preprocess_start) * 1000
        print(f"  Mel spectrogram shape: {mel_spectrogram.shape}")
        print(f"[TIMING] Preprocessing: {preprocess_time_ms:.1f}ms")
        print()

        # Send data to pipeline
        print("Running inference...")
        pipeline.send_data(mel_spectrogram)

        # Wait for and get transcription
        transcription = pipeline.get_transcription()

        # Stop the pipeline
        pipeline.stop()

        # Print result
        print()
        print("=" * 70)
        print("TRANSCRIPTION:")
        print("=" * 70)
        print(f"{transcription}")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
