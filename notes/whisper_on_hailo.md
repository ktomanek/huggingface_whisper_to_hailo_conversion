# Whisper on Hailo - Implementation Guide

## Overview

This document explains the `whisper_on_hailo.py` implementation, which provides a standalone pipeline for running OpenAI's Whisper speech recognition model on Hailo hardware accelerators (Hailo-8, Hailo-8L, Hailo-10H).

## Architecture

The pipeline consists of three main components:

1. **Audio Preprocessing** - Converts audio files to mel spectrograms
2. **HailoWhisperPipeline** - Threaded inference pipeline running encoder and decoder
3. **Token Generation** - Autoregressive decoding with optional repetition penalty

## Key Implementation Details

### 1. Data Layout: NCHW vs NHWC

One of the most critical aspects of this implementation is understanding the data layout difference between ONNX models and Hailo HEF models.

#### Format Comparison

**NCHW (Channels First)**
- Layout: (Batch, Channels, Height, Width) = `[1, 80, 1, 1000]`
- Used by: PyTorch, ONNX Runtime
- Memory layout: All values for channel 0, then channel 1, etc.

**NHWC (Channels Last)**
- Layout: (Batch, Height, Width, Channels) = `[1, 1, 1000, 80]`
- Used by: TensorFlow, Hailo Hardware
- Memory layout: All channels for position 0, then position 1, etc.

#### Why Hailo Uses NHWC

Hardware accelerators like Hailo prefer NHWC format for several architectural reasons:

1. **Memory Access Patterns**
   - Convolution operations process spatial locations (H×W) and compute across all channels
   - NHWC keeps all channel data for a spatial position contiguous in memory
   - This reduces memory bandwidth and improves cache utilization

2. **Parallel Processing**
   - Hailo's architecture processes multiple channels simultaneously
   - NHWC layout allows efficient SIMD (Single Instruction, Multiple Data) operations
   - All channels for a position can be loaded in a single burst

3. **Reduced Data Movement**
   - Convolution kernels naturally iterate over spatial dimensions
   - NHWC minimizes data reshuffling between operations
   - Fewer transpose operations needed on the accelerator

4. **Hardware Pipeline Efficiency**
   - Modern AI accelerators have specialized memory hierarchies
   - NHWC aligns with how data flows through convolution units
   - Better utilization of on-chip buffers and registers

#### Practical Impact

**In our code:**
```python
def get_audio(audio_file, chunk_length=10, is_nhwc=True):
    # ... compute mel spectrogram ...
    mel = np.expand_dims(mel, axis=0)  # [80, 1000] -> [1, 80, 1000]
    mel = np.expand_dims(mel, axis=2)  # [1, 80, 1000] -> [1, 80, 1, 1000] (NCHW)

    # Transpose to NHWC for Hailo hardware
    if is_nhwc:
        mel = np.transpose(mel, [0, 2, 3, 1])  # -> [1, 1, 1000, 80] (NHWC)
```

**What happens without transpose:**
- Model interprets 80 mel bins as "height" and 1000 time steps as "channels"
- Completely scrambles the input data
- Results in garbage predictions (e.g., `<|nocaptions|>` token immediately)

#### ONNX vs HEF Models

**ONNX Models** (used in `hailo_inference_example.py`):
- Expect NCHW format: `[1, 80, 1, 1000]`
- Run on CPU/GPU with ONNX Runtime
- Use `is_nhwc=False` in preprocessing

**HEF Models** (used in `whisper_on_hailo.py`):
- Expect NHWC format: `[1, 1, 1000, 80]`
- Run on Hailo hardware accelerator
- Use `is_nhwc=True` in preprocessing
- Format conversion happens during ONNX→HEF compilation

### 2. Decoder Token Embeddings

The Hailo decoder HEF has the token embedding layer **stripped out** for efficiency:

**Why embeddings are separate:**
- Token embeddings are simple lookup tables (not compute-intensive)
- Running them on host CPU is more efficient
- Reduces model size on the Hailo accelerator
- Allows more complex layers to run on hardware

**Assets required:**
```
decoder_assets/
└── tiny/
    └── decoder_tokenization/
        ├── token_embedding_weight_tiny.npy  # Token lookup table
        └── onnx_add_input_tiny.npy          # Bias/offset
```

Download from:
```bash
wget -P decoder_assets/tiny/decoder_tokenization \
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets/tiny/decoder_tokenization/onnx_add_input_tiny.npy"

wget -P decoder_assets/tiny/decoder_tokenization \
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets/tiny/decoder_tokenization/token_embedding_weight_tiny.npy"
```

### 3. Audio Preprocessing

The `get_audio()` function implements Hailo's efficient preprocessing:

**Key features:**
- Uses PyTorch STFT (faster than librosa's implementation)
- Caches mel filter banks with `@lru_cache`
- Matches OpenAI Whisper's preprocessing exactly
- Automatically handles audio padding/cropping to 10 seconds

**Preprocessing pipeline:**
1. Load audio with librosa at 16kHz
2. Convert to PyTorch tensor
3. Compute STFT with Hann window
4. Apply mel filterbank (80 bands)
5. Convert to log scale and normalize
6. Transpose to NHWC format

### 4. Decoder Start Tokens

The Hailo implementation uses a simplified start sequence:

**Hailo approach:**
```python
start_token_id = [50258]  # Just <|startoftranscript|>
```

**Standard Whisper approach:**
```python
forced_tokens = [50258, 50259, 50359, 50363]  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
```

The Hailo decoder **generates all tokens** including language and task tokens, rather than forcing them. This provides flexibility but means the first few generated tokens are typically:
1. `50259` - `<|en|>` (language)
2. `50359` - `<|transcribe|>` (task)
3. `50363` - `<|notimestamps|>` (format)
4. Then actual transcription tokens

### 5. Repetition Penalty

The `apply_repetition_penalty()` function prevents the model from repeating tokens:

**Implementation:**
```python
def apply_repetition_penalty(logits, generated_tokens, penalty=1.5, last_window=8):
    # Penalize tokens that appeared in the last 8 generated tokens
    # Excludes punctuation tokens (11, 13) from penalty
```

**Why it's needed:**
- Autoregressive models can get stuck in repetition loops
- Without penalty: "Hello world. Hello world. Hello world..."
- With penalty (1.5): "Hello world." (stops after first occurrence)

**Typical values:**
- `1.0` - No penalty (disabled)
- `1.1-1.3` - Mild penalty
- `1.5` - Strong penalty (default)

### 6. Multi-Process Service

The `multi_process_service` parameter enables sharing the Hailo device:

**When enabled:**
```python
params.multi_process_service = True
params.group_id = "SHARED"
```

**Use case:**
- Running multiple inference pipelines on the same Hailo chip
- Other applications need concurrent access
- Shared device management across processes

## Usage

### Basic Usage

```bash
python whisper_on_hailo.py \
  --encoder_hef_file /path/to/encoder.hef \
  --decoder_hef_file /path/to/decoder.hef \
  --audio_file /path/to/audio.wav
```

### Full Example

```bash
python whisper_on_hailo.py \
  --encoder_hef_file tiny-whisper-encoder-10s_15dB_h8l.hef \
  --decoder_hef_file tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef \
  --audio_file hello_world.wav \
  --variant tiny \
  --decoder_assets_path ./decoder_assets \
  --multi_process_service
```

### Command-Line Arguments

- `--encoder_hef_file` (required) - Path to encoder HEF file
- `--decoder_hef_file` (required) - Path to decoder HEF file
- `--audio_file` (required) - Path to audio file to transcribe
- `--variant` - Model variant: "tiny" or "base" (default: "tiny")
- `--decoder_assets_path` - Path to decoder assets (default: ./decoder_assets)
- `--multi_process_service` - Enable multi-process service mode

## Pipeline Flow

```
Audio File
    ↓
[get_audio()] → Mel Spectrogram [1, 1, 1000, 80] (NHWC)
    ↓
[Encoder HEF] → Encoded Features [1, 500, 384]
    ↓
[_tokenization()] → Token Embeddings (CPU)
    ↓
[Decoder HEF] → Logits [1, seq_len, vocab_size]
    ↓
[apply_repetition_penalty()] → Modified Logits
    ↓
[argmax] → Next Token
    ↓
[Repeat until EOS]
    ↓
Transcription Text
```

## Performance Considerations

1. **First inference is slower** - Mel filter computation is cached after first run
2. **Background thread** - Pipeline runs in separate thread for async processing
3. **Memory layout** - NHWC transpose is crucial for correct inference
4. **Token generation** - Each token requires one decoder forward pass (autoregressive)

## Troubleshooting

### Issue: Empty transcription or `<|nocaptions|>`

**Cause:** Incorrect input format (NCHW instead of NHWC)

**Solution:** Ensure `is_nhwc=True` in `get_audio()`

### Issue: Repeated transcription

**Cause:** Repetition penalty disabled or too low

**Solution:** Enable repetition penalty with `penalty=1.5`

### Issue: Missing decoder assets

**Cause:** Token embedding files not downloaded

**Solution:** Download .npy files from Hailo's S3 bucket (see section 2)

### Issue: Audio duration warning

**Cause:** Audio longer than 10 seconds

**Solution:** Audio is automatically cropped to 10s for tiny model

## Comparison with ONNX Implementation

| Aspect | ONNX (`hailo_inference_example.py`) | HEF (`whisper_on_hailo.py`) |
|--------|-------------------------------------|------------------------------|
| Runtime | ONNX Runtime (CPU/GPU) | Hailo Hardware Accelerator |
| Input Format | NCHW `[1, 80, 1, 1000]` | NHWC `[1, 1, 1000, 80]` |
| Decoder Embeddings | Included in ONNX model | Separate .npy files (CPU) |
| Speed | Slower (software) | Faster (hardware acceleration) |
| Setup | Just ONNX files needed | HEF + decoder assets needed |

## Dependencies

```
numpy
torch
librosa
hailo_platform
transformers
```

## References

- Original Hailo implementation: [Hailo-Application-Code-Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples)
- OpenAI Whisper: [openai/whisper](https://github.com/openai/whisper)
- Hailo Developer Zone: [hailo.ai/developer-zone](https://hailo.ai/developer-zone/)
