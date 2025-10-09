#!/usr/bin/env python3
"""
Hailo Whisper Captioning App

A self-contained real-time speech captioning application using Hailo HEF encoder + ONNX decoder.
Optimized for best speed/accuracy tradeoff on Hailo hardware.

Usage:
    python hailo_captioning_app.py \
        --encoder_hef_file /path/to/encoder.hef \
        --decoder_onnx_dir /path/to/decoder_onnx \
        --language en \
        --min_partial_duration 0.2
"""

import argparse
import logging
import numpy as np
import os
import psutil
import pyaudio
import queue
import time
import threading

# Hailo and ONNX imports
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
import onnxruntime as ort
from transformers import AutoTokenizer
import torch
import librosa

# Embedded dependencies
from silero_vad import VADIterator, load_silero_vad

# Import shared helper functions from whisper_on_hailo_pipeline
from whisper_on_hailo_pipeline import (
    _pad_or_trim,
    _log_mel_spectrogram,
    N_FFT,
    HOP_LENGTH,
    SAMPLE_RATE as SAMPLING_RATE,
    REPETITION_PENALTY
)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_LANGUAGE = 'en'
FORMAT = pyaudio.paInt16
CHANNELS = 1
AUDIO_FRAMES_TO_CAPTURE = 512  # VAD strictly needs this number
VAD_THRESHOLD = 0.5
EOS_MIN_SILENCE = 100
MAXIMUM_SEGMENT_DURATION = 9.0 #because of 10sec encoder max (where 1sec padding helps to avoid hallucinations)


def get_mel_spectrogram(audio_data, target_duration=10, padding_cutoff_delta=1.0, sample_rate=None):
    """
    Convert audio data (numpy array) to mel spectrogram for Hailo HEF encoder (NHWC format).
    Uses shared helper functions from whisper_on_hailo_pipeline.

    Args:
        audio_data: Audio waveform (numpy array)
        target_duration: Target duration in seconds (default: 10)
        padding_cutoff_delta: Time to cut off before target to avoid hallucinations (default: 1.0)
        sample_rate: Sample rate in Hz (default: SAMPLING_RATE constant = 16000)

    Returns:
        mel_spectrogram: [1, 1, N, 80] (NHWC format for HEF)
    """
    if sample_rate is None:
        sample_rate = SAMPLING_RATE

    # Convert to torch tensor
    audio_torch = torch.from_numpy(audio_data)

    # Crop audio first, then pad to target duration
    crop_duration = target_duration - padding_cutoff_delta
    if padding_cutoff_delta > 0:
        crop_samples = int(crop_duration * sample_rate)
        audio_torch = _pad_or_trim(audio_torch, crop_samples)

    # Pad to target duration
    target_samples = int(target_duration * sample_rate)
    audio_torch = _pad_or_trim(audio_torch, target_samples)

    # Compute mel spectrogram using shared function
    mel = _log_mel_spectrogram(audio_torch, sample_rate=sample_rate).to("cpu")

    # Convert to numpy and reshape to NHWC [1, 1, N, 80]
    mel = mel.numpy()
    mel = np.expand_dims(mel, axis=0)  # [80, N] -> [1, 80, N]
    mel = np.expand_dims(mel, axis=2)  # [1, 80, N] -> [1, 80, 1, N]
    mel = np.transpose(mel, [0, 2, 3, 1])  # [1, 80, 1, N] -> [1, 1, N, 80] (NHWC)

    return mel.astype(np.float32)


class Transcriber:
    """Base transcriber class"""

    def __init__(self, model_name_or_path, sampling_rate, show_word_confidence_scores=False, language=DEFAULT_LANGUAGE, output_streaming=True):
        self.number_of_partials_transcribed = 0
        self.speech_segments_transcribed = 0
        self.speech_frames_transcribed = 0
        self.compute_time = 0.0
        self.memory_used = 0

        self.sampling_rate = sampling_rate
        self.model_name = model_name_or_path
        self.show_word_confidence_scores = show_word_confidence_scores
        self.output_streaming = output_streaming

        self.language = language
        print(f"Setting model language to: {self.language}")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        self._load_model(model_name_or_path)
        mem_after = process.memory_info().rss
        self.memory_used = mem_after - mem_before

        self._warmup_model()

    def _load_model(self, model_name_or_path):
        raise NotImplementedError("Subclasses should implement this method to load the model.")

    def _warmup_model(self):
        """Warm up the model by running a dummy transcription."""
        data = np.zeros((self.sampling_rate,), dtype=np.float32)  # 1 second of silence
        _ = self._transcribe(data, segment_end=True)
        logging.debug('Model warmed up...')

    def _transcribe(self, audio_data, segment_end):
        raise NotImplementedError("Subclasses should implement this method to transcribe.")

    def transcribe(self, audio_data, segment_end):
        """Generator that yields transcription results as they become available."""
        t1 = time.time()

        yielded_any = False
        for result in self._transcribe(audio_data, segment_end):
            yielded_any = True
            yield result

        # Only update stats after all results have been yielded
        self.compute_time += (time.time() - t1)
        self.speech_frames_transcribed += len(audio_data)
        if segment_end:
            self.speech_segments_transcribed += 1
        else:
            self.number_of_partials_transcribed += 1

        # Handle case where _transcribe yields nothing
        if not yielded_any:
            yield ""

    def get_stats(self):
        speech_time_transcribes = self.speech_frames_transcribed / self.sampling_rate
        rtfx = speech_time_transcribes / self.compute_time if self.compute_time > 0 else 0.0
        print(f"Model uses {self.memory_used / (1024 * 1024):.2f} MB of RAM")
        print(f"Number of inference calls total: {self.speech_segments_transcribed + self.number_of_partials_transcribed}")
        print(f"Number of partial segments transcribed: {self.number_of_partials_transcribed}")
        print(f"Number of full segments transcribed: {self.speech_segments_transcribed}")
        print(f"Number of frames transcribed: {self.speech_frames_transcribed}")
        print(f"Total speech time transcribed: {speech_time_transcribes:.2f} sec")
        print(f"Total inference time: {self.compute_time:.2f} sec")
        print(f"Inverse real-time factor (RTFx): {rtfx:.2f}")


class HailoWhisperTranscriber(Transcriber):
    """Hailo HEF encoder + ONNX decoder transcriber"""

    MAX_OUTPUT_LEN = 447  # Whisper upper bound (0-indexed)

    def __init__(self, model_name_or_path, sampling_rate, show_word_confidence_scores=False, language=DEFAULT_LANGUAGE,
                 encoder_hef_path=None, decoder_onnx_dir=None, output_streaming=True, variant="tiny"):
        self.encoder_hef_path = encoder_hef_path
        self.decoder_onnx_dir = decoder_onnx_dir
        self.variant = variant

        if not encoder_hef_path:
            raise ValueError("encoder_hef_path is required for HailoWhisperTranscriber")
        if not decoder_onnx_dir:
            raise ValueError("decoder_onnx_dir is required for HailoWhisperTranscriber")

        super().__init__(model_name_or_path, sampling_rate, show_word_confidence_scores, language, output_streaming)

    def _load_model(self, model_name_or_path):
        """Load Hailo HEF encoder and ONNX decoder"""
        print(f"[INFO] Loading Hailo HEF encoder from: {self.encoder_hef_path}")
        print(f"[INFO] Loading ONNX decoder from: {self.decoder_onnx_dir}")

        # Initialize Hailo HEF encoder
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.vdevice = VDevice(params)
        self.encoder_infer_model = self.vdevice.create_infer_model(self.encoder_hef_path)
        self.encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
        self.encoder_infer_model.output().set_format_type(FormatType.FLOAT32)

        # Configure encoder and create bindings
        self.encoder_configured_infer_model = self.encoder_infer_model.configure()
        self.encoder_bindings = self.encoder_configured_infer_model.create_bindings()

        print(f"[INFO] HEF encoder loaded successfully")

        # Load ONNX decoder
        decoder_init_path = os.path.join(self.decoder_onnx_dir, "decoder_model.onnx")
        decoder_cached_path = os.path.join(self.decoder_onnx_dir, "decoder_with_past_model.onnx")

        if not os.path.exists(decoder_init_path):
            raise FileNotFoundError(f"ONNX decoder not found: {decoder_init_path}")
        if not os.path.exists(decoder_cached_path):
            raise FileNotFoundError(f"ONNX cached decoder not found: {decoder_cached_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.decoder_init_session = ort.InferenceSession(decoder_init_path, sess_options=sess_options)
        self.decoder_cached_session = ort.InferenceSession(decoder_cached_path, sess_options=sess_options)

        print(f"[INFO] ONNX decoder loaded successfully")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"openai/whisper-{self.variant}")

        # Special tokens
        self.sot_token = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.eot_token = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.no_timestamps_token = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.transcribe_token = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")

        # Dynamic language token
        self.language_token = self._get_language_token(self.language)

        # Get output names for cache mapping
        self.decoder_outputs = [out.name for out in self.decoder_init_session.get_outputs()]
        self.decoder_with_past_outputs = [out.name for out in self.decoder_cached_session.get_outputs()]

        print(f"[INFO] Using language: {self.language}")

    def _get_language_token(self, language):
        """Get the language token ID for the specified language code."""
        language_token_map = {
            'en': '<|en|>', 'es': '<|es|>', 'fr': '<|fr|>', 'de': '<|de|>', 'it': '<|it|>',
            'pt': '<|pt|>', 'ru': '<|ru|>', 'ja': '<|ja|>', 'ko': '<|ko|>', 'zh': '<|zh|>',
        }

        language_token = language_token_map.get(language)
        if not language_token:
            logging.warning(f"Language '{language}' not found in supported languages, falling back to English")
            language_token = '<|en|>'

        token_id = self.tokenizer.convert_tokens_to_ids(language_token)
        if token_id is None:
            raise ValueError(f"Language token '{language_token}' not found in tokenizer vocabulary")

        return token_id

    def _encode_audio(self, input_mel_spec):
        """Encode audio using Hailo HEF encoder"""
        input_mel = np.ascontiguousarray(input_mel_spec)

        # Reuse pre-configured bindings
        self.encoder_bindings.input().set_buffer(input_mel)
        buffer = np.zeros(self.encoder_infer_model.output().shape).astype(np.float32)
        self.encoder_bindings.output().set_buffer(buffer)

        # Run inference
        self.encoder_configured_infer_model.run([self.encoder_bindings], 100000000)  # timeout_ms
        encoded_features = self.encoder_bindings.output().get_buffer().copy()

        return encoded_features

    def _transcribe(self, audio_data, segment_end):
        """Perform transcription using Hailo encoder + ONNX decoder"""
        try:
            # Preprocess audio to mel spectrogram
            input_mel_spec = get_mel_spectrogram(audio_data, target_duration=10, padding_cutoff_delta=1.0)

            # Encode audio with Hailo HEF
            encoder_hidden_states = self._encode_audio(input_mel_spec)

            if self.output_streaming:
                # Stream decode tokens one by one
                yield from self._decode_streaming(encoder_hidden_states)
            else:
                # Accumulate all tokens and yield complete result
                complete_text = ""
                for token in self._decode_streaming(encoder_hidden_states):
                    if token:
                        complete_text += token
                if complete_text.strip():
                    yield complete_text.strip()

        except Exception as e:
            logging.error(f"Hailo transcription error: {e}")
            yield f"[Error: {str(e)}]"

    def _decode_streaming(self, encoder_hidden_states, max_length=None):
        """Streaming decoding with ONNX decoder + KV-cache"""
        if max_length is None:
            max_length = self.MAX_OUTPUT_LEN

        # Initialize decoder input with start tokens
        forced_tokens = [self.sot_token, self.language_token, self.transcribe_token, self.no_timestamps_token]
        generated_tokens = forced_tokens.copy()
        past_key_values_dict = {}

        # Account for initial tokens in max_length calculation
        max_new_tokens = max_length - len(forced_tokens)

        for step in range(max_new_tokens):
            if not past_key_values_dict:
                # First pass: process forced tokens and initialize cache
                input_ids = np.array([forced_tokens], dtype=np.int64)

                outputs = self.decoder_init_session.run(None, {
                    'input_ids': input_ids,
                    'encoder_hidden_states': encoder_hidden_states
                })
                logits = outputs[0]

                # Store ALL cache outputs
                for idx, output_name in enumerate(self.decoder_outputs[1:], 1):
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

                outputs = self.decoder_cached_session.run(None, inputs)
                logits = outputs[0]

                # Update cache for next iteration
                for idx, output_name in enumerate(self.decoder_with_past_outputs[1:], 1):
                    if "present" in output_name:
                        past_name = output_name.replace("present.", "past_key_values.")
                        past_key_values_dict[past_name] = outputs[idx]

                next_token_logits = logits[0, -1, :].copy()

            # Apply repetition penalty (penalize ALL generated tokens, not just last 8)
            tokens_to_penalize = set(generated_tokens[len(forced_tokens):])
            for token_id in tokens_to_penalize:
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= REPETITION_PENALTY
                else:
                    next_token_logits[token_id] *= REPETITION_PENALTY

            # Get next token (greedy decoding)
            next_token_id = np.argmax(next_token_logits)

            # Check for end of transcript
            if next_token_id in [self.eot_token, 50256, 50257]:
                break

            generated_tokens.append(int(next_token_id))

            # Decode token to text for streaming output
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            if token_text.strip():  # Only yield non-empty tokens
                if self.show_word_confidence_scores:
                    # Calculate confidence if requested
                    exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
                    token_probs = exp_logits / np.sum(exp_logits)
                    confidence = token_probs[next_token_id]
                    yield f"{token_text}/{confidence:.2f}"
                else:
                    yield token_text


class PlainCaptionPrinter:
    """Simple caption printer"""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def start(self):
        print("---------------------------  Transcribing speech ----------------------------")

    def stop(self):
        print("-----------------------------------------------------------------------------")

    def print(self, transcript, duration=None, partial=False, is_recent_chunk_mode=False, recent_chunk_duration=None):
        """Update the caption display with the latest transcription"""
        if partial:
            if self.verbose and is_recent_chunk_mode and recent_chunk_duration:
                print(f"\r\033[2K\rPARTIAL (recent-chunk, {recent_chunk_duration:.1f}s chunk/{duration:.1f}s total): {transcript}", flush=True, end='')
            elif self.verbose and duration:
                print(f"\r\033[2K\rPARTIAL (retranscribe, {duration:.1f}s total): {transcript}", flush=True, end='')
            else:
                print(f"\r\033[2K\rPARTIAL: {transcript}", flush=True, end='')
        else:
            if self.verbose and duration:
                print(f"\rSEGMENT ({duration:.1f}s total): {transcript}")
            else:
                print(f"\rSEGMENT: {transcript}")


class TranscriptionWorker:
    """Transcription worker"""

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.is_speech_recording = False
        self.had_speech = False
        self.frames_since_last_speech = 0
        self.transcribed_segments = []
        self.last_partial_transcribed_length = 0
        self.accumulated_partial_text = ""

    def reset(self):
        self.is_speech_recording = False
        self.had_speech = False
        self.frames_since_last_speech = 0
        self.transcribed_segments = []
        self.last_partial_transcribed_length = 0
        self.accumulated_partial_text = ""

    def transcription_worker(
            self,
            asr,
            audio_queue,
            caption_printer,
            vad,
            stop_threads,
            min_partial_duration=0.1,
            max_segment_duration=MAXIMUM_SEGMENT_DURATION):
        """Worker thread that processes audio chunks for transcription"""

        speech_buffer = np.empty(0, dtype=np.float32)
        self.is_speech_recording = False
        time_since_last_transcription = time.time()

        while not stop_threads.is_set():
            try:
                # Read new chunk from queue and add to buffer
                chunk = audio_queue.get(timeout=0.05)
                chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                speech_buffer = np.concatenate((speech_buffer, chunk_np))
                speech_buffer_duration = len(speech_buffer) / self.sampling_rate
                current_recording_duration = time.time() - time_since_last_transcription

                # Process speech in buffer depending on VAD event
                vad_event = vad(chunk_np)
                if vad_event:
                    logging.debug(f"VAD event detected: {vad_event}")
                    if "start" in vad_event:
                        self.is_speech_recording = True
                        self.had_speech = True
                        self.frames_since_last_speech = 0
                        self.last_partial_transcribed_length = 0
                        self.accumulated_partial_text = ""
                        time_since_last_transcription = time.time()
                    elif "end" in vad_event:
                        # Finish the segment by processing all so far and then flushing buffer
                        self.is_speech_recording = False
                        self.frames_since_last_speech += len(chunk_np)

                        complete_text = ""
                        for text_chunk in asr.transcribe(speech_buffer, segment_end=True):
                            if text_chunk:
                                complete_text += text_chunk
                        complete_text = complete_text.strip()
                        if complete_text:
                            caption_printer.print(complete_text, duration=speech_buffer_duration, partial=False)
                            self.transcribed_segments.append(complete_text)
                        speech_buffer = np.empty(0, dtype=np.float32)
                        self.last_partial_transcribed_length = 0
                        self.accumulated_partial_text = ""
                        time_since_last_transcription = time.time()
                else:
                    # No VAD event means recording state hasn't changed
                    if self.is_speech_recording:
                        # Force end a segment if it is getting too long
                        if speech_buffer_duration > max_segment_duration:
                            logging.debug(f"Max segment duration reached, ending segment: {speech_buffer_duration:.2f} sec")

                            complete_text = ""
                            for text_chunk in asr.transcribe(speech_buffer, segment_end=True):
                                if text_chunk:
                                    complete_text += text_chunk

                            complete_text = complete_text.strip()
                            if complete_text:
                                caption_printer.print(complete_text, duration=speech_buffer_duration, partial=False)
                                self.transcribed_segments.append(complete_text)
                            speech_buffer = np.empty(0, dtype=np.float32)
                            self.last_partial_transcribed_length = 0
                            self.accumulated_partial_text = ""
                            time_since_last_transcription = time.time()

                        # If we have enough data in the buffer, transcribe a partial
                        elif current_recording_duration > min_partial_duration:
                            logging.debug(f"Transcribing partial segment: {current_recording_duration:.2f} sec")

                            # Retranscribe all accumulated audio (more accurate)
                            self.accumulated_partial_text = ""
                            for text_chunk in asr.transcribe(speech_buffer, segment_end=False):
                                if text_chunk:
                                    self.accumulated_partial_text += text_chunk
                                    d = len(speech_buffer) / self.sampling_rate
                                    caption_printer.print(self.accumulated_partial_text, duration=d, partial=True,
                                                         is_recent_chunk_mode=False, recent_chunk_duration=None)

                            time_since_last_transcription = time.time()
                    else:
                        empty_frames_to_keep = int(0.1 * self.sampling_rate)
                        speech_buffer = speech_buffer[-empty_frames_to_keep:]
                        self.frames_since_last_speech += len(chunk_np)

            except queue.Empty:
                if stop_threads.is_set():
                    break
                continue
            except Exception as e:
                if stop_threads.is_set():
                    break
                print(f"\nTranscription error: {e}")
                continue

        # Flush remaining speech buffer
        if len(speech_buffer) > 0:
            logging.debug("Flushing remaining speech buffer...")

            complete_text = ""
            for text_chunk in asr.transcribe(speech_buffer, segment_end=True):
                if text_chunk:
                    complete_text += text_chunk

            complete_text = complete_text.strip()
            if complete_text:
                caption_printer.print(complete_text, duration=len(speech_buffer) / self.sampling_rate, partial=False)
                self.transcribed_segments.append(complete_text)


def get_vad(eos_min_silence=EOS_MIN_SILENCE, vad_threshold=VAD_THRESHOLD, sampling_rate=SAMPLING_RATE):
    """Create and initialize VAD"""
    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=sampling_rate,
        threshold=vad_threshold,
        min_silence_duration_ms=eos_min_silence,
    )
    print('VAD loaded.')
    return vad_iterator


def get_audio_stream(audio, input_device_index=1):
    """Create audio input stream"""
    print('Using audio input device:', audio.get_device_info_by_index(input_device_index).get('name'))
    audio_stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLING_RATE,
                        input=True,
                        frames_per_buffer=AUDIO_FRAMES_TO_CAPTURE,
                        input_device_index=input_device_index)
    return audio_stream


def find_default_input_device():
    """Find the default microphone device using PyAudio"""
    p = pyaudio.PyAudio()
    try:
        default_info = p.get_default_input_device_info()
        if default_info:
            print(f"Default input device: {default_info['name']} (index: {default_info['index']})")
            return {
                'name': default_info['name'],
                'index': default_info['index']
            }
    except:
        print("No default input device found")
        return None
    finally:
        p.terminate()


def list_audio_devices():
    """List all available audio devices"""
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"* Device [{i}]: {device_info['name']} \t input channels: {device_info['maxInputChannels']}, output channels: {device_info['maxOutputChannels']}")
    p.terminate()


def capture_audio_from_stream(audio_stream, audio_queue, stop_threads, caption_printer):
    """Capture audio from microphone stream"""
    print("Recording started. Press Ctrl+C to stop.")
    caption_printer.start()

    try:
        while True:
            data = audio_stream.read(AUDIO_FRAMES_TO_CAPTURE)
            audio_queue.put(data, timeout=0.1)
    except queue.Full:
        logging.warning("Audio queue is full, skipping this chunk.")
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(0.2)
        stop_threads.set()

        # Empty queue
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
                audio_queue.task_done()
            except queue.Empty:
                break

        # Clean up audio resources
        audio_stream.stop_stream()
        audio_stream.close()


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Hailo Whisper Real-time Captioning (HEF encoder + ONNX decoder)")
    parser.add_argument(
        "--encoder_hef_file",
        type=str,
        default="models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef",
        # required=True,
        help="Path to Hailo HEF encoder file",
    )
    parser.add_argument(
        "--decoder_onnx_dir",
        type=str,
        default="models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8",
        #required=True,
        help="Path to directory containing ONNX decoder files (decoder_model.onnx, decoder_with_past_model.onnx)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        help="Whisper model variant (e.g., tiny, base, small) for tokenizer (default: tiny)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help=f"Language to transcribe in (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--min_partial_duration",
        type=float,
        default=0.2,
        help="Minimum duration in seconds for partial transcriptions to be displayed (default: 0.2)",
    )
    parser.add_argument(
        "--max_segment_duration",
        type=float,
        default=MAXIMUM_SEGMENT_DURATION,
        help=f"Maximum duration in seconds for a segment before it is transcribed (default: {MAXIMUM_SEGMENT_DURATION})",
    )
    parser.add_argument(
        "--eos_min_silence",
        type=int,
        default=EOS_MIN_SILENCE,
        help=f"Minimum silence duration in milliseconds to consider the end of a segment (default: {EOS_MIN_SILENCE})",
    )
    parser.add_argument(
        "--audio_input_device_index",
        type=int,
        help="Index of the audio input device to use (default is system default)",
    )
    parser.add_argument(
        "--show_audio_devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--show_word_confidence_scores",
        action="store_true",
        default=False,
        help="Calculate and show per-word confidence scores",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed transcription information including timing",
    )

    return parser.parse_args()


def main():
    """Main function for Hailo captioning"""
    args = get_args()

    if args.show_audio_devices:
        list_audio_devices()
        return

    # Create caption printer
    caption_printer = PlainCaptionPrinter(verbose=args.verbose)

    # Initialize VAD
    vad = get_vad(eos_min_silence=args.eos_min_silence)

    # Load Hailo Whisper model
    print(f"Loading Hailo Whisper model (HEF encoder + ONNX decoder)")
    asr_model = HailoWhisperTranscriber(
        model_name_or_path='hailo-whisper',
        sampling_rate=SAMPLING_RATE,
        show_word_confidence_scores=args.show_word_confidence_scores,
        language=args.language,
        encoder_hef_path=args.encoder_hef_file,
        decoder_onnx_dir=args.decoder_onnx_dir,
        output_streaming=False,  # Use retranscribe mode (default)
        variant=args.variant
    )

    # Print configuration info
    print(f"Transcription mode: Retranscribe mode (token streaming disabled)")
    print(f"Partial duration: {args.min_partial_duration}s")

    # Initialize audio queue and threading
    audio_queue = queue.Queue(maxsize=1000)
    stop_threads = threading.Event()

    # Start transcription worker thread
    transcription_handler = TranscriptionWorker(sampling_rate=SAMPLING_RATE)
    transcriber = threading.Thread(target=transcription_handler.transcription_worker,
                                   kwargs={'vad': vad,
                                           'asr': asr_model,
                                           'audio_queue': audio_queue,
                                           'caption_printer': caption_printer,
                                           'stop_threads': stop_threads,
                                           'min_partial_duration': args.min_partial_duration,
                                           'max_segment_duration': args.max_segment_duration,
                                           })
    transcriber.daemon = True
    transcriber.start()

    # Initialize audio system
    audio = pyaudio.PyAudio()
    device_index = args.audio_input_device_index
    if device_index:
        print(f"Using user specified audio input device index: {device_index}")
    else:
        # Find default device index
        input_device = find_default_input_device()
        if input_device:
            print(f"Using default audio input device: {input_device}")
            device_index = input_device['index']
        else:
            print("No default input device found, using device index 1")
            device_index = 1

    # Start audio capture
    audio_stream = get_audio_stream(audio, input_device_index=device_index)
    capture_audio_from_stream(audio_stream, audio_queue, stop_threads, caption_printer)

    # Cleanup
    audio.terminate()
    caption_printer.stop()
    print("\nRecording stopped.")

    # Print model statistics
    print("\n>>> Model stats:")
    asr_model.get_stats()


if __name__ == "__main__":
    main()
