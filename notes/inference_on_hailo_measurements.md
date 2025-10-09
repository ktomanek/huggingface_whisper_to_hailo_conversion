

## Refactoreed Pipeline - Final Runs

# Original 30s ONNX encoder + ONNX decoder
  python whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file encoder_30s.onnx \
    --decoder_onnx_dir decoder_onnx \
    --audio_file test.wav

  # Original 30s ONNX encoder + ONNX decoder, folder evaluation
  python whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file encoder_30s.onnx \
    --decoder_onnx_dir decoder_onnx \
    --audio_folder /path/to/dataset

# HEF Only
python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef \
    --audio_file samples/singles/hello_world.wav --num_iterations 10

Encoder times (ms):
  Mean: 33.0 ± 0.0
  Min/Max: 32.9 / 33.0
  Median: 33.0

Decoder times (ms):
  Mean: 296.8 ± 0.5
  Min/Max: 296.2 / 298.1
  Median: 296.7

Total times (ms):
  Mean: 329.8 ± 0.5
  Min/Max: 329.2 / 331.0
  Median: 329.6


python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef \
    --audio_file samples/singles/jfk_asknot.wav --num_iterations 10

Encoder times (ms):
  Mean: 33.2 ± 0.2
  Min/Max: 32.9 / 33.3
  Median: 33.2

Decoder times (ms):
  Mean: 1327.9 ± 5.1
  Min/Max: 1319.2 / 1332.4
  Median: 1330.5

Total times (ms):
  Mean: 1361.0 ± 5.3
  Min/Max: 1352.2 / 1365.7
  Median: 1363.8

python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef \
    --audio_folder samples/audio_folder  

  Samples processed:     11/11
  Average WER:           57.92%
  Average CER:           38.69%

  Timing:
    Avg Encoder time:    33.1ms
    Avg Decoder time:    651.2ms
    Avg Total time:      684.3ms

python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef \
    --audio_folder samples/librispeech_devclean_small/prepared     

  Samples processed:     24/24
  Average WER:           34.53%
  Average CER:           15.02%

  Timing:
    Avg Encoder time:    33.3ms
    Avg Decoder time:    1257.4ms
    Avg Total time:      1290.7ms  

# HEF encoder, ONNX decoder
python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_file samples/singles/hello_world.wav --num_iterations 10

Encoder times (ms):
  Mean: 31.5 ± 0.5
  Min/Max: 31.2 / 32.9
  Median: 31.4

Decoder times (ms):
  Mean: 32.4 ± 2.0
  Min/Max: 30.7 / 37.8
  Median: 31.6

Total times (ms):
  Mean: 64.0 ± 2.2
  Min/Max: 62.1 / 69.2
  Median: 63.0

python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_file samples/singles/jfk_asknot.wav --num_iterations 10

Encoder times (ms):
  Mean: 31.7 ± 0.7
  Min/Max: 31.2 / 33.2
  Median: 31.4

Decoder times (ms):
  Mean: 105.4 ± 3.2
  Min/Max: 101.6 / 109.6
  Median: 103.3

Total times (ms):
  Mean: 137.1 ± 3.3
  Min/Max: 134.2 / 142.3
  Median: 134.7    

python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_folder samples/audio_folder

  Samples processed:     11/11
  Average WER:           54.33%
  Average CER:           40.94%

  Timing:
    Avg Encoder time:    31.6ms
    Avg Decoder time:    60.0ms
    Avg Total time:      91.5ms


python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_folder samples/librispeech_devclean_small/prepared
  Samples processed:     24/24
  Average WER:           34.71%
  Average CER:           14.02%

  Timing:
    Avg Encoder time:    31.4ms
    Avg Decoder time:    101.1ms
    Avg Total time:      132.5ms    

# ONNX Encoder (10 sec), ONNX decoder
python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_onnx_file models/onnx/hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_file samples/singles/hello_world.wav --num_iterations 10

Encoder times (ms):
  Mean: 125.3 ± 2.8
  Min/Max: 117.3 / 127.7
  Median: 125.8

Decoder times (ms):
  Mean: 45.4 ± 3.5
  Min/Max: 39.1 / 49.9
  Median: 46.6

Total times (ms):
  Mean: 170.7 ± 4.5
  Min/Max: 164.1 / 177.6
  Median: 170.7

python inference/on_hailo/whisper_on_hailo_pipeline.py \
--encoder_onnx_file models/onnx/hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx \
--decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
--audio_file samples/singles/jfk_asknot.wav --num_iterations 10

Encoder times (ms):
  Mean: 126.1 ± 3.1
  Min/Max: 118.1 / 130.8
  Median: 126.9

Decoder times (ms):
  Mean: 114.2 ± 3.6
  Min/Max: 109.8 / 120.7
  Median: 113.1

Total times (ms):
  Mean: 240.4 ± 2.7
  Min/Max: 236.8 / 245.7
  Median: 239.4

python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_onnx_file models/onnx/hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_folder samples/audio_folder

  Samples processed:     11/11
  Average WER:           43.16%
  Average CER:           25.75%

  Timing:
    Avg Encoder time:    118.3ms
    Avg Decoder time:    63.9ms
    Avg Total time:      182.1ms
  
python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_onnx_file models/onnx/hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_folder samples/librispeech_devclean_small/prepared  

  Samples processed:     24/24
  Average WER:           33.65%
  Average CER:           13.62%

  Timing:
    Avg Encoder time:    117.3ms
    Avg Decoder time:    108.5ms
    Avg Total time:      225.8ms

# Old ONNX decoder, ONNX decoder
python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8/encoder_model.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_file samples/singles/hello_world.wav --num_iterations 10

Encoder times (ms):
  Mean: 438.7 ± 1.0
  Min/Max: 437.1 / 440.7
  Median: 438.8

Decoder times (ms):
  Mean: 64.9 ± 3.0
  Min/Max: 59.4 / 68.5
  Median: 65.1

Total times (ms):
  Mean: 503.6 ± 3.3
  Min/Max: 496.7 / 507.3
  Median: 503.8

python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8/encoder_model.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_file samples/singles/jfk_asknot.wav --num_iterations 10

Encoder times (ms):
  Mean: 439.1 ± 1.3
  Min/Max: 436.8 / 440.9
  Median: 439.2

Decoder times (ms):
  Mean: 226.6 ± 2.2
  Min/Max: 223.7 / 230.8
  Median: 226.0

Total times (ms):
  Mean: 665.6 ± 2.0
  Min/Max: 663.0 / 670.2
  Median: 665.3

python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8/encoder_model.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_folder samples/audio_folder

  Samples processed:     11/11
  Average WER:           58.59%
  Average CER:           33.92%

  Timing:
    Avg Encoder time:    435.0ms
    Avg Decoder time:    90.7ms
    Avg Total time:      525.7ms

python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default/encoder_model.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default \
    --audio_folder samples/audio_folder

  Samples processed:     11/11
  Average WER:           30.82%
  Average CER:           15.29%

  Timing:
    Avg Encoder time:    584.9ms
    Avg Decoder time:    168.4ms
    Avg Total time:      753.3ms


python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8/encoder_model.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default \
    --audio_folder samples/audio_folder
  Samples processed:     11/11
  Average WER:           53.74%
  Average CER:           32.52%

  Timing:
    Avg Encoder time:    435.9ms
    Avg Decoder time:    164.3ms
    Avg Total time:      600.2ms

python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default/encoder_model.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_folder samples/audio_folder
  Samples processed:     11/11
  Average WER:           27.71%
  Average CER:           14.25%

  Timing:
    Avg Encoder time:    583.5ms
    Avg Decoder time:    91.3ms
    Avg Total time:      674.7ms

--> Encoder int8 hurts most!


python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default/encoder_model.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_folder samples/librispeech_devclean_small/prepared

  Samples processed:     24/24
  Average WER:           32.77%
  Average CER:           12.45%

  Timing:
    Avg Encoder time:    584.5ms
    Avg Decoder time:    148.8ms
    Avg Total time:      733.2ms


python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_orig_onnx_file models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8/encoder_model.onnx \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_folder samples/librispeech_devclean_small/prepared

  Samples processed:     24/24
  Average WER:           42.55%
  Average CER:           19.05%

  Timing:
    Avg Encoder time:    434.9ms
    Avg Decoder time:    149.9ms
    Avg Total time:      584.9ms

# Faster Whisper
int8

python inference/on_hailo/faster_whisper_baseline.py --audio_file samples/singles/hello_world.wav --num_iterations 10

Inference times (ms):
  Mean: 814.2 ± 48.8
  Min/Max: 777.4 / 899.0
  Median: 786.3


python inference/on_hailo/faster_whisper_baseline.py --audio_folder samples/audio_folder
  Samples processed:     11/11
  Average WER:           27.71%
  Average CER:           13.69%

  Timing:
    Avg Preprocessing:   9.0ms
    Avg Inference:       1007.1ms
    Avg Total time:      1016.1ms

python inference/on_hailo/faster_whisper_baseline.py --audio_folder samples/librispeech_devclean_small/prepared
  Samples processed:     24/24
  Average WER:           30.68%
  Average CER:           10.88%

  Timing:
    Avg Preprocessing:   9.2ms
    Avg Inference:       1067.4ms
    Avg Total time:      1076.6ms


## Results Summary Tables

### LibriSpeech Dataset (24 samples, clean speech)

| Configuration | Avg WER | Avg Encoder (ms) | Avg Decoder (ms) | Avg Total (ms) |
|--------------|---------|------------------|------------------|----------------|
| **HEF Encoder + HEF Decoder** | 34.53% | 33.3 | 1257.4 | 1290.7 |
| **HEF Encoder + ONNX Decoder (INT8)** | 34.71% | 31.4 | 101.1 | 132.5 |
| **ONNX Encoder (10s) + ONNX Decoder (INT8)** | 33.65% | 117.3 | 108.5 | 225.8 |
| **ONNX Encoder (30s, FP32) + ONNX Decoder (INT8)** | 32.77% | 584.5 | 148.8 | 733.2 |
| **FasterWhisper (INT8, CPU baseline)** | 30.68% | - | - | 1076.6 |

**Key Findings:**
- **Best accuracy**: FasterWhisper baseline (30.68% WER) and ONNX 30s FP32 encoder (32.77% WER)
- **Fastest inference**: HEF Encoder + ONNX Decoder INT8 (132.5ms total) - **8.1x faster than baseline**
- **Best speed/accuracy tradeoff**: HEF Encoder + ONNX Decoder INT8 (34.71% WER, 132.5ms)
- **INT8 encoder quantization impact**: Increases WER by ~10% (32.77% → 42.55%)

---

### Audio Folder Dataset (11 samples, Katrin accent)

| Configuration | Avg WER | Avg Encoder (ms) | Avg Decoder (ms) | Avg Total (ms) |
|--------------|---------|------------------|------------------|----------------|
| **HEF Encoder + HEF Decoder** | 57.92% | 33.1 | 651.2 | 684.3 |
| **HEF Encoder + ONNX Decoder (INT8)** | 54.33% | 31.6 | 60.0 | 91.5 |
| **ONNX Encoder (10s) + ONNX Decoder (INT8)** | 43.16% | 118.3 | 63.9 | 182.1 |
| **ONNX Encoder (30s, FP32) + ONNX Decoder (FP32)** | 30.82% | 584.9 | 168.4 | 753.3 |
| **ONNX Encoder (30s, FP32) + ONNX Decoder (INT8)** | 27.71% | 583.5 | 91.3 | 674.7 |
| **ONNX Encoder (30s, INT8) + ONNX Decoder (FP32)** | 53.74% | 435.9 | 164.3 | 600.2 |
| **ONNX Encoder (30s, INT8) + ONNX Decoder (INT8)** | 58.59% | 435.0 | 90.7 | 525.7 |
| **FasterWhisper (INT8, CPU baseline)** | 27.71% | - | - | 1016.1 |

**Key Findings:**
- **Best accuracy**: FasterWhisper baseline (27.71% WER) = ONNX 30s FP32 + ONNX INT8 decoder (27.71% WER)
- **Fastest inference**: HEF Encoder + ONNX Decoder INT8 (91.5ms total) - **11.1x faster than baseline**
- **HEF decoder**: Comparable to ONNX decoder (57.92% vs 54.33% WER) after repetition penalty fix
- **Decoder quantization**: INT8 decoder improves WER vs FP32 when paired with FP32 encoder (30.82% → 27.71%)
- **Encoder quantization is critical**: INT8 encoder causes major degradation (27.71% → 58.59% WER)