python whisper_encoder_on_hailo.py --audio_file ~/dev/audio_samples/hello_world.wav --encoder_hef_file /home/katrintomanek/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef   --encoder_orig_onnx_file  /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/default_int8/encoder_model.onnx --encoder_onnx_file  /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/converted_for_hailo/whisper_tiny_encoder_10s_hailo_final.onnx
2025-10-07 16:13:49.998662435 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
======================================================================
WHISPER ENCODER BENCHMARKING ON HAILO
======================================================================
Audio file: /home/katrintomanek/dev/audio_samples/hello_world.wav
Iterations: 10

📦 Stage 1: Audio Preprocessing

======================================================================
HEF ENCODER (Hailo Hardware)
======================================================================
Encoder HEF: /home/katrintomanek/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef
Preprocessing audio (NHWC format for HEF)...
  Mel spectrogram shape: (1, 1, 1000, 80)

Running HEF encoder 10 times...
✅ Encoder output shape: (1, 500, 384)
✅ Output range: [-16.975, 18.281]
✅ Output mean: 0.035, std: 1.354

[TIMING] HEF Encoder Results:
  Min:     30.53ms
  Max:     31.96ms
  Mean:    30.69ms
  Median:  30.54ms
  Std:     0.42ms

======================================================================
ONNX ENCODER (CPU Runtime)
======================================================================
Encoder ONNX: /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/converted_for_hailo/whisper_tiny_encoder_10s_hailo_final.onnx
Preprocessing audio (NCHW format for ONNX)...
  Mel spectrogram shape: (1, 80, 1, 1000)

Running ONNX encoder 10 times...
  ONNX model expects input: x.1 with shape [1, 80, 1, 1000]
  Providing input shape: (1, 80, 1, 1000)
✅ Encoder output shape: (1, 500, 384)
✅ Output range: [-17.419, 18.559]
✅ Output mean: 0.037, std: 1.365

[TIMING] ONNX Encoder Results:
  Min:     116.73ms
  Max:     147.26ms
  Mean:    120.00ms
  Median:  116.82ms
  Std:     9.10ms

======================================================================
ORIGINAL ONNX ENCODER (FP32, 30s, Standard Whisper)
======================================================================
Encoder ONNX: /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/default_int8/encoder_model.onnx
Preprocessing audio (NCHW format with WhisperProcessor, 30s)...
  Mel spectrogram shape: (1, 80, 3000)

Running original ONNX encoder 10 times...
  ONNX model expects input: input_features with shape ['batch_size', 80, 3000]
  Providing input shape: (1, 80, 3000)
✅ Encoder output shape: (1, 1500, 384)
✅ Output range: [-17.303, 15.469]
✅ Output mean: 0.032, std: 1.225

[TIMING] Original ONNX Encoder Results:
  Min:     429.34ms
  Max:     437.08ms
  Mean:    431.08ms
  Median:  430.38ms
  Std:     2.18ms

======================================================================
COMPARISON
======================================================================

Performance:
  HEF (Hailo, 10s, NHWC)                      30.69ms
  ONNX (CPU, 10s, INT8, NCHW)                120.00ms
  ONNX Original (CPU, 30s, FP32, NCHW)       431.08ms

Speedup relative to HEF (Hailo):
  ONNX (CPU, 10s, INT8, NCHW)                3.91x (HEF faster)
  ONNX Original (CPU, 30s, FP32, NCHW)      14.05x (HEF faster)

Speedup relative to Original ONNX (FP32, 30s):
  HEF (Hailo, 10s, NHWC)                    14.05x faster
  ONNX (CPU, 10s, INT8, NCHW)                3.59x faster

Output Comparison (HEF vs ONNX 10s):
  Max difference:  13.033251
  Mean difference: 0.188791
  Outputs match:   ⚠️  No (difference > 0.01)

======================================================================
BENCHMARK COMPLETE
======================================================================

======================================================================
TRANSCRIPTION COMPARISON
======================================================================

Timing Summary:

HEF (Hailo, 10s, NHWC):
  Encoder:    31.97ms
  Decoder:    66.97ms
  Total:      98.95ms

ONNX (CPU, 10s, INT8, NCHW):
  Encoder:   118.33ms
  Decoder:    68.64ms
  Total:     186.97ms

ONNX Original (CPU, 30s, FP32, NCHW):
  Encoder:   437.91ms
  Decoder:   109.27ms
  Total:     547.17ms



#####

python whisper_encoder_on_hailo_with_decoder.py --audio_file ~/dev/audio_samples/hello_world.wav --encoder_hef_file /home/katrintomanek/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef   --encoder_orig_onnx_file  /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/default_int8/encoder_model.onnx --encoder_onnx_file  /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/converted_for_hailo/whisper_tiny_encoder_10s_hailo_final.onnx --decoder_onnx_dir ~/dev/hailo_whisper_convert/my_converted
2025-10-07 16:22:28.187906685 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
======================================================================
WHISPER ENCODER+DECODER BENCHMARKING ON HAILO
======================================================================
Audio file: /home/katrintomanek/dev/audio_samples/hello_world.wav
Decoder: /home/katrintomanek/dev/hailo_whisper_convert/my_converted

Loading ONNX decoder models...
✅ Decoder models loaded in 759.4ms


======================================================================
HEF ENCODER + ONNX DECODER
======================================================================
Preprocessing audio (NHWC format for HEF)...
  Mel spectrogram shape: (1, 1, 1000, 80)
Running HEF encoder...
✅ Encoder time: 31.97ms
✅ Encoder output shape: (1, 500, 384)
Running ONNX decoder...
  [DEBUG] First pass: processing 4 forced tokens with decoder_model.onnx
  [DEBUG] Initialized cache with 16 tensors
  [STEP 0] 28.2ms (init) | token=2425
  [STEP 1] 11.6ms (cached) | token=1002
  [STEP 2] 15.0ms (cached) | token=13
  [STEP 3] 12.2ms (cached) | token=50257
  [DEBUG] EOS token reached
  [DEBUG] Decoder timing: total=67.0ms, avg=16.7ms, first=28.2ms, rest=12.9ms
✅ Decoder time: 66.97ms (inference only, excludes model loading)
✅ Total time: 98.95ms
✅ Generated tokens: [50258, 50259, 50359, 50363, 2425, 1002, 13, 50257]
✅ Transcription: " Hello world."

======================================================================
ONNX INT8 ENCODER (10s) + ONNX DECODER
======================================================================
Preprocessing audio (NCHW format for ONNX)...
  Mel spectrogram shape: (1, 80, 1, 1000)
Running ONNX encoder...
  ONNX model expects input: x.1 with shape [1, 80, 1, 1000]
  Providing input shape: (1, 80, 1, 1000)
✅ Encoder time: 118.33ms
✅ Encoder output shape: (1, 500, 384)
Running ONNX decoder...
  [DEBUG] First pass: processing 4 forced tokens with decoder_model.onnx
  [DEBUG] Initialized cache with 16 tensors
  [STEP 0] 28.2ms (init) | token=2425
  [STEP 1] 16.5ms (cached) | token=1002
  [STEP 2] 11.8ms (cached) | token=13
  [STEP 3] 12.2ms (cached) | token=50257
  [DEBUG] EOS token reached
  [DEBUG] Decoder timing: total=68.6ms, avg=17.2ms, first=28.2ms, rest=13.5ms
✅ Decoder time: 68.64ms (inference only, excludes model loading)
✅ Total time: 186.97ms
✅ Generated tokens: [50258, 50259, 50359, 50363, 2425, 1002, 13, 50257]
✅ Transcription: " Hello world."

======================================================================
ORIGINAL ONNX ENCODER (30s, FP32) + ONNX DECODER
======================================================================
Preprocessing audio (NCHW format with WhisperProcessor, 30s)...
  Mel spectrogram shape: (1, 80, 3000)
Running original ONNX encoder...
  ONNX model expects input: input_features with shape ['batch_size', 80, 3000]
  Providing input shape: (1, 80, 3000)
✅ Encoder time: 437.91ms
✅ Encoder output shape: (1, 1500, 384)
Running ONNX decoder...
  [DEBUG] First pass: processing 4 forced tokens with decoder_model.onnx
  [DEBUG] Initialized cache with 16 tensors
  [STEP 0] 60.8ms (init) | token=2425
  [STEP 1] 21.1ms (cached) | token=1002
  [STEP 2] 13.6ms (cached) | token=13
  [STEP 3] 13.7ms (cached) | token=50257
  [DEBUG] EOS token reached
  [DEBUG] Decoder timing: total=109.3ms, avg=27.3ms, first=60.8ms, rest=16.1ms
✅ Decoder time: 109.27ms (inference only, excludes model loading)
✅ Total time: 547.17ms
✅ Generated tokens: [50258, 50259, 50359, 50363, 2425, 1002, 13, 50257]
✅ Transcription: " Hello world."

======================================================================
TRANSCRIPTION COMPARISON
======================================================================

Timing Summary:

HEF (Hailo, 10s, NHWC):
  Encoder:    31.97ms
  Decoder:    66.97ms
  Total:      98.95ms

ONNX (CPU, 10s, INT8, NCHW):
  Encoder:   118.33ms
  Decoder:    68.64ms
  Total:     186.97ms

ONNX Original (CPU, 30s, FP32, NCHW):
  Encoder:   437.91ms
  Decoder:   109.27ms
  Total:     547.17ms

----------------------------------------------------------------------

Transcriptions:

HEF (Hailo, 10s, NHWC):
  Text: " Hello world."
  Tokens: [50258, 50259, 50359, 50363, 2425, 1002, 13, 50257]

ONNX (CPU, 10s, INT8, NCHW):
  Text: " Hello world."
  Tokens: [50258, 50259, 50359, 50363, 2425, 1002, 13, 50257]

ONNX Original (CPU, 30s, FP32, NCHW):
  Text: " Hello world."
  Tokens: [50258, 50259, 50359, 50363, 2425, 1002, 13, 50257]

✅ All transcriptions match: " Hello world."
✅ All token sequences match

======================================================================
BENCHMARK COMPLETE
======================================================================
