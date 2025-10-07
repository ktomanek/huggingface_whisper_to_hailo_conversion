# Hailo-Optimized Whisper Inference

Production-ready inference script combining efficient preprocessing with optimized decoder strategies.

## Features

- **Two preprocessing methods:**
  - **Hailo** (PyTorch STFT, cached mel filters, ffmpeg) - **32x faster**
  - **HuggingFace** (librosa, WhisperProcessor) - standard approach

- **Anti-hallucination strategies:**
  1. Encoder output padding (match training distribution)
  2. Repetition penalty (discourage token repetition)
  3. EOS boosting (proactive early stopping)

- **Efficient cached decoder:**
  - KV cache eliminates O(n²) overhead
  - Per-token timing measurements
  - ~1ms per token after first token

## Usage

### Basic Usage (Hailo preprocessing - default)

```bash
python hailo_inference_example.py audio.wav
```

### Use HuggingFace Preprocessing

```bash
python hailo_inference_example.py audio.wav --preprocessing huggingface
```

### Compare Preprocessing Methods

```bash
python hailo_inference_example.py audio.wav --compare
```

This will:
1. Verify both methods produce identical mel spectrograms
2. Run inference with both methods
3. Show timing comparison

### Compare 10s vs 30s Encoder

```bash
python hailo_inference_example.py audio.wav --compare-encoders
```

This will:
1. Run inference with 10s encoder (Hailo-optimized)
2. Run inference with 30s encoder (standard ONNX)
3. Show detailed performance comparison

## Performance Comparison

### First Run (includes loading processor)

**Example output (beckett.wav - 8 seconds):**

```
Hailo Preprocessing:
  Time: ~110ms (includes mel filters loading)
  Total: ~200ms

HuggingFace Preprocessing:
  Time: ~1500ms (includes WhisperProcessor loading)
  Total: ~1600ms

⚡ Hailo is ~13x faster on first run
```

### Subsequent Runs (cached, both using librosa)

```
Hailo Preprocessing:
  Time: ~2.2ms
  Total: ~85ms

HuggingFace Preprocessing:
  Time: ~5.1ms
  Total: ~86ms

⚡ Hailo is 2.37x faster (mel computation)
```

**Performance Breakdown:**
```
Both methods use librosa for audio loading (~0.4ms)

Hailo mel computation:    ~2.2ms (PyTorch STFT + cached filters)
HuggingFace mel:          ~5.1ms (standard mel computation)

Speedup: 2.37x
```

**Key Insight:**
- Hailo's advantage is PyTorch STFT with pre-computed mel filters (@lru_cache)
- HuggingFace uses standard mel spectrogram computation (slower)
- Both methods produce **identical** outputs (verified via assertion)

### Encoder Comparison: 10s vs 30s

**Example output (beckett.wav - 8 seconds):**

```
10s Encoder:
  Preprocessing: 2.0ms
  Encoder: 40.0ms
  Decoder: 42.0ms
  Total: 84.2ms

30s Encoder:
  Preprocessing: 4.7ms
  Encoder: 97.1ms
  Decoder: 32.6ms
  Total: 134.5ms

⚡ Performance:
   Encoder: 10s is 2.43x faster (40.0ms vs 97.1ms)
   Overall: 10s pipeline is 1.60x faster (84.2ms vs 134.5ms)

💾 Savings:
   Encoder time saved: 57.1ms
   Total time saved: 50.3ms
```

**Key Insights:**
- 10s encoder is **~2.4x faster** than 30s encoder
- Overall pipeline **~1.6x faster** with 10s
- Encoder accounts for **47% of total time** (10s) vs **72%** (30s)
- Decoder padding overhead (+4.8ms) is negligible compared to encoder savings (57ms)

**Recommendation:**
Use 10s encoder with full padding (1500 positions) for optimal balance of speed and reliability.

## Configuration Options

Edit the configuration section in `hailo_inference_example.py`:

### Option 1: Conservative (Most Reliable)
```python
ORIG_MAX_LEN = 1500           # Full padding
REPETITION_PENALTY = 1.0      # Disabled
EOS_BOOST_THRESHOLD = 0.0     # Disabled
```
- Most robust across all inputs
- ~1.9x decoder overhead from padding

### Option 2: Balanced (Recommended)
```python
ORIG_MAX_LEN = 700            # Minimal safe padding
REPETITION_PENALTY = 1.1      # Mild penalty
EOS_BOOST_THRESHOLD = 0.2     # Conservative boost
```
- Good trade-off between speed and reliability
- Multiple safety mechanisms

### Option 3: Aggressive (Maximum Speed)
```python
ORIG_MAX_LEN = 500            # No padding
REPETITION_PENALTY = 1.2      # Stronger penalty
EOS_BOOST_THRESHOLD = 0.5     # Aggressive boost
```
- Lowest compute overhead
- Relies heavily on EOS boosting

## Architecture

**Hybrid approach:**
- **Encoder:** Hailo-compatible 10s model (ready for NPU deployment)
- **Decoder:** Efficient cached decoder on CPU

This is optimal for Raspberry Pi 5 + Hailo-8 NPU:
- Encoder uses fixed shapes (perfect for NPU)
- Decoder benefits from KV cache (not practical on NPU)

## Model Paths

Update these paths in the script if needed:

```python
ENCODER_PATH = "../hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx"
DECODER_INIT_PATH = "/path/to/decoder_model.onnx"
DECODER_CACHED_PATH = "/path/to/decoder_with_past_model.onnx"
```

## Input Constraints

- Audio is automatically cropped to **10 seconds** (with warning)
- Encoder expects 10s input (1000 mel frames)
- Decoder generates up to 32 tokens (4 forced + 28 new)

## Dependencies

```bash
pip install onnxruntime numpy transformers librosa torch
```

For Hailo preprocessing, the following files are required:
- `hailo_preprocessing/audio_utils.py`
- `hailo_preprocessing/preprocessing.py`
- `hailo_preprocessing/assets/mel_filters.npz`

## References

See `../onnx_conversion_practical_efficiency_notes.md` for detailed analysis of:
- Distribution shift problem and hallucination causes
- Performance analysis of padding strategies
- Early stopping benefits
- Hybrid architecture rationale
