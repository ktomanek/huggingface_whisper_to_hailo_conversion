# HuggingFace Whisper Decoder to Hailo ONNX Conversion Notes

## Overview
This document outlines the necessary modifications to convert the HuggingFace Transformers Whisper decoder model to ONNX format compatible with Hailo NPU.

## Sources
- **Patch file**: `/Users/katrintomanek/dev/onnx_experiments/hailo-whisper/hailo-compatibility-changes.patch`
- **Export script**: `/Users/katrintomanek/dev/onnx_experiments/hailo-whisper/export/export_whisper_model.py`
- **Encoder reference**: `/Users/katrintomanek/dev/huggingface_whisper_to_hailo_conversion/convert_encoder.ipynb`

---

## Required Decoder Modifications

### 1. Use Eager Attention (No SDPA)
**Source**: Patch line 43
```python
SDPA_AVAILABLE = False  # forcing it to avoid issues
```

**Implementation for HuggingFace**:
```python
model = WhisperForConditionalGeneration.from_pretrained(
    base_model_name,
    attn_implementation='eager'
)
```

**Reason**: Hailo doesn't support scaled dot product attention optimizations. Must use explicit attention computation.

---

### 2. Token Embedding Reshape Operations
**Source**: Patch lines 144-146
```python
x = x.unsqueeze(1)
x = x.transpose(1, -1)
x = x.flatten(2).permute(0, 2, 1)
```

**Location**: After token embedding + positional embedding in decoder forward pass

**Purpose**: Transform embeddings to match Hailo's expected tensor format. These operations prepare the hidden states for processing through the decoder layers.

**What it does**:

The reshape operations appear to return to the same shape, but they change the internal memory layout and computational graph structure for ONNX export:

```python
# After: hidden_states = inputs_embeds + positions
# Shape: [1, 32, 384]

hidden_states = hidden_states.unsqueeze(1)      # [1, 32, 384] -> [1, 1, 32, 384]
hidden_states = hidden_states.transpose(1, -1)  # [1, 1, 32, 384] -> [1, 384, 32, 1]
hidden_states = hidden_states.flatten(2).permute(0, 2, 1)  # [1, 384, 32, 1] -> [1, 32, 384]
```

**Why this is necessary**:

These operations create a specific computational pattern in the ONNX graph that Hailo's compiler expects. Without them, the compiler may:
- Fail to recognize the pattern
- Generate suboptimal code
- Reject the model entirely

Think of it like reformatting a document - the content looks the same, but the structure is different under the hood.

**Where to apply**: In the custom `decoder_forward()` function, immediately after combining token embeddings with positional embeddings and before processing through decoder layers.

---

### 3. Split Final Matmul into Chunks
**Source**: Patch lines 108-125 (method definition) and line 160 (usage)

**Original Operation** (patch lines 156-158):
```python
# Too large for Hailo-8
logits = (
    x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
).float()
```

**Modified Operation** (patch lines 108-125):
```python
def split_conv2d_method(self, x):
    vocab_size = self.token_embedding.weight.shape[0]
    chunk_size = vocab_size // 4
    logit_chunks = []

    W = self.token_embedding.weight.to(x.dtype)

    for i in range(4):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < 3 else vocab_size
        W_chunk = W[start:end]
        logits_chunk = torch.matmul(x, W_chunk.T)
        logit_chunks.append(logits_chunk)

    logits = torch.cat(logit_chunks, dim=-1)
    return logits
```

**Reason**: The vocabulary size for Whisper is 51865, making the final matmul too large for Hailo-8 hardware constraints. Splitting into 4 chunks (~12966 each) fits within hardware limits.

**How it works**:

The final decoder layer computes logits by multiplying hidden states with the embedding weight matrix:

**Original operation (too large for Hailo-8)**:
```python
logits = x @ W.T  # Shape: (1, 32, 384) @ (384, 51865) = (1, 32, 51865)
```

**Split approach**:
Instead of one large matmul, split the vocabulary dimension into 4 chunks:

- **x**: (1, 32, 384) - batch 1, 32 tokens, 384 hidden dims
- **W**: (51865, 384) - vocab size × hidden dims

Process each chunk:
- Chunk 0: (1, 32, 384) @ (384, 12966) → (1, 32, 12966)
- Chunk 1: (1, 32, 384) @ (384, 12966) → (1, 32, 12966)
- Chunk 2: (1, 32, 384) @ (384, 12966) → (1, 32, 12966)
- Chunk 3: (1, 32, 384) @ (384, 12967) → (1, 32, 12967) [gets remainder]
- **Concatenate** along last dimension → (1, 32, 51865)

This produces mathematically identical results while fitting within Hailo-8's MAC (multiply-accumulate) operation size limits.

**Usage**: Applied after layer normalization in decoder forward pass (patch line 160)

---

### 4. Fixed Decoder Sequence Length
**Source**: Export script lines 60-64, 84, 108

**Configuration for tiny model**:
```python
INPUT_LENGTH_SECONDS = 10       # Scaled from 30s
DECODER_SEQUENCE_LENGTH = 32    # Maximum number of tokens
ENCODER_SEQ_LEN = 500          # 1500 / 3 for 10s input
HIDDEN_STATES_CHANNELS = 384   # for tiny model
```

**Source** (export script lines 60-61):
```python
default_input_length = 10
default_decoder_sequence_length = 32  # Maximum number of tokens
```

**Reason**: Hailo requires fixed input shapes. Dynamic shapes are not supported.

---

## ONNX Export Configuration

### Export Parameters
**Source**: Export script lines 119-126

```python
torch.onnx.export(
    decoder,
    (decoder_input_ids, encoder_hidden_states),
    decoder_onnx_path,
    opset_version=13,
    input_names=["decoder_input_ids", "encoder_hidden_states"],
    output_names=["logits"],
)
```

**Key settings**:
- `opset_version=13`: Required for Hailo compatibility
- Fixed input shapes (no dynamic_axes)
- Two inputs: decoder token IDs and encoder hidden states

### Input Shapes
**Source**: Export script lines 130-133

```python
input_shapes = {
    "decoder_input_ids": [1, decoder_sequence_length],
    "encoder_hidden_states": [1, encoder_seq_len, hidden_states_channels]
}
```

**For tiny model with 10s input**:
- `decoder_input_ids`: [1, 32]
- `encoder_hidden_states`: [1, 500, 384]

### ONNX Simplification
**Source**: Export script lines 134-142

```python
model_onnx = onnx.load(decoder_onnx_path)
model_simp, check = simplify(model_onnx, overwrite_input_shapes=input_shapes)
onnx.save(model_simp, decoder_onnx_path)
```

**Purpose**: Simplify the ONNX graph for better Hailo compilation and performance.

---

## Decoder Input Creation
**Source**: Export script line 108

```python
decoder_input_ids = torch.cat([
    torch.tensor([[50258]], dtype=torch.int64),  # Start token
    torch.zeros((1, decoder_sequence_length - 1), dtype=torch.int64)
], dim=1)
```

**Token 50258**: Special start-of-transcript token for Whisper decoder

---

## Model-Specific Parameters

### Tiny Model
**Source**: Export script lines 58-61
- Input length: 10 seconds
- Decoder sequence: 32 tokens
- Hidden size: 384
- Encoder output length: 500

### Base Model
**Source**: Export script lines 62-65
- Input length: 5 seconds
- Decoder sequence: 24 tokens
- Hidden size: 512
- Encoder output length: 333 (1500 / 4.5 scaling)

---

## Detailed Function Explanations

### `decoder_forward()` Function

This is a modified version of the HuggingFace Whisper decoder's forward pass. It processes token IDs and produces hidden states.

**Function signature**:
```python
def decoder_forward(
    self,
    input_ids=None,              # Token IDs to decode [batch, seq_len]
    attention_mask=None,          # Mask for padding tokens
    encoder_hidden_states=None,   # Output from encoder [batch, enc_seq, hidden]
    head_mask=None,               # Mask specific attention heads
    cross_attn_head_mask=None,    # Mask for cross-attention heads
    past_key_values=None,         # Cached keys/values for generation
    inputs_embeds=None,           # Pre-computed embeddings (alternative to input_ids)
    use_cache=None,               # Whether to return key/value cache
    output_attentions=None,       # Return attention weights
    output_hidden_states=None,    # Return all layer hidden states
    return_dict=None,             # Return as dict vs tuple
)
```

**Processing steps**:

1. **Configuration and Validation**
   - Set defaults from model config
   - Validate that either `input_ids` or `inputs_embeds` is provided (but not both)

2. **Get Token Embeddings**
   ```python
   # Calculate past sequence length for positional embeddings
   past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

   # Convert token IDs to embeddings
   if inputs_embeds is None:
       inputs_embeds = self.embed_tokens(input_ids)  # [1, 32] -> [1, 32, 384]

   # Get positional embeddings (learned, not sinusoidal like encoder)
   positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)

   # Combine token + positional embeddings
   hidden_states = inputs_embeds + positions  # [1, 32, 384]
   hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
   ```

3. **Apply Hailo Reshape Operations** (see section 2 above)
   ```python
   hidden_states = hidden_states.unsqueeze(1).transpose(1, -1).flatten(2).permute(0, 2, 1)
   ```

4. **Process Through Decoder Layers**
   ```python
   for idx, decoder_layer in enumerate(self.layers):
       layer_outputs = decoder_layer(
           hidden_states,
           attention_mask=attention_mask,
           encoder_hidden_states=encoder_hidden_states,  # For cross-attention
           past_key_value=past_key_value,
           output_attentions=output_attentions,
           use_cache=use_cache,
       )
       hidden_states = layer_outputs[0]  # Update hidden states
   ```

   Each decoder layer contains:
   - **Self-attention**: Token attends to previous tokens (with causal masking)
   - **Cross-attention**: Token attends to encoder outputs
   - **Feed-forward network**: MLP transformation

5. **Final Layer Normalization**
   ```python
   hidden_states = self.layer_norm(hidden_states)  # [1, 32, 384]
   ```

6. **Return Results**
   ```python
   return BaseModelOutputWithPastAndCrossAttentions(
       last_hidden_state=hidden_states,      # [1, 32, 384] - main output
       past_key_values=next_cache,           # Cached k/v for next generation step
       hidden_states=all_hidden_states,      # All intermediate layer outputs
       attentions=all_self_attns,            # Self-attention weights
       cross_attentions=all_cross_attentions, # Cross-attention weights
   )
   ```

**Output**: The `hidden_states` `[1, 32, 384]` is then passed to `split_matmul_method()` to compute logits `[1, 32, 51865]`.

**Key modification**: Only the reshape operations (step 3) are added for Hailo compatibility. The rest is standard Whisper decoder logic.

---

## Key Differences from Encoder Conversion

1. **No Conv1D to Conv2D conversion**: Decoder uses embeddings, not convolutional layers
2. **No positional embedding rescaling**: Decoder uses learnable positional embeddings (no sinusoidal)
3. **Split matmul required**: Large vocabulary size requires chunking the final projection
4. **Cross-attention**: Decoder attends to encoder outputs (not present in encoder)
5. **Reshape operations**: Specific to decoder embedding processing

---

## Summary Checklist

- [ ] ✓ Use eager attention (no SDPA)
- [ ] ✓ Add token embedding reshape operations (unsqueeze, transpose, flatten)
- [ ] ✓ Implement split matmul for final logits (4 chunks)
- [ ] ✓ Fix decoder sequence length (32 for tiny)
- [ ] ✓ Export with opset_version=13
- [ ] ✓ Apply ONNX simplification with fixed input shapes
- [ ] ✓ Verify output shape: [1, 32, 51865] for tiny model

---

## Implementation Notes for HuggingFace Transformers

### Method Overriding

The notebook correctly overrides two methods:

```python
# Override decoder forward pass (adds reshape operations)
model.model.decoder.forward = types.MethodType(decoder_forward, model.model.decoder)

# Override model forward pass (uses split matmul for logits)
model.forward = types.MethodType(model_forward, model)
```

### Weight Tying in HuggingFace

HuggingFace Whisper has a `proj_out` layer that computes final logits:
```python
model.proj_out = Linear(in_features=384, out_features=51865, bias=False)
```

**Important**: The `proj_out.weight` and `model.decoder.embed_tokens.weight` are **tied** (share the same memory). This means:
- Using `embed_tokens.weight` in `split_matmul_method()` is equivalent to using `proj_out.weight`
- By overriding `model.forward()`, we bypass `proj_out` and use our split matmul instead
- This is correct and produces identical results (aside from the chunking optimization)

### Naming Differences

| OpenAI Whisper | HuggingFace Transformers |
|----------------|-------------------------|
| `token_embedding` | `embed_tokens` |
| `positional_embedding` | `embed_positions` |
| Direct access | Wrapped in `proj_out` layer |

### Critical Points

- The patch modifies OpenAI's original Whisper, but the concepts apply to HuggingFace Transformers with adaptation
- HuggingFace uses slightly different naming: `embed_tokens` instead of `token_embedding`
- The reshape operations (lines 144-146) are critical and easy to miss
- Without split matmul, Hailo compilation will fail due to operation size limits
- All notebook cells must be executed in order for the method overrides to work

---

## KV Cache and Autoregressive Decoding - IMPORTANT EFFICIENCY CONSIDERATION

### What's Missing: No KV Cache in ONNX Export

**Verification**: The reference export script (lines 119-126) exports the decoder with only **2 inputs**:
```python
torch.onnx.export(
    decoder,
    (decoder_input_ids, encoder_hidden_states),  # NO past_key_values!
    decoder_onnx_path,
    ...
)
```

This means the Hailo ONNX model does **NOT** use KV caching during autoregressive decoding.

### Two Distinct Sources of Inefficiency

The Hailo approach suffers from **two independent inefficiencies** that multiply together:

#### Inefficiency #1: Computing Unused Positions (32x overhead)

**Problem**: Fixed shape `[1, 32]` forces decoder to process all 32 positions in every forward pass.

**What happens**: At step 5 (generating the 6th token):
```python
decoder_input_ids = [50258, 1234, 5678, 9012, 3456, 0, 0, ..., 0]  # [1, 32]
#                    tok0   tok1  tok2  tok3  tok4  PAD PAD ... PAD
#                    ^^^^^ 5 real tokens ^^^^^  ^^^ 27 padding ^^^

logits = decoder(decoder_input_ids, encoder_hidden_states)  # [1, 32, 51865]

# Only use position 5 to generate next token
next_token = logits[0, 5, :].argmax()

# Discard:
# - logits[0, 0:5, :] -> Already generated in previous steps
# - logits[0, 6:32, :] -> Future positions, not ready yet
```

**Overhead**: Process 32 positions, use 1 → **32x waste per step**

**Total across 32 steps**: 32 steps × 32 positions = 1,024 logit computations vs. needed 32 = **32x overhead**

**Root cause**: ONNX fixed shape requirement for decoder inputs

---

#### Inefficiency #2: No KV Cache (16.5x overhead)

**Problem**: Without cache, each position recomputes attention over ALL previous tokens.

**What happens**: At step 31 (generating the 32nd token):
- Position 31 computes self-attention over positions 0-31 (all 32 tokens)
- This attention was partially computed in previous steps but thrown away
- Must recompute from scratch every time

**With cache**: Each new token only computes attention for itself, reusing cached keys/values from previous tokens (O(n) complexity)

**Without cache**: Each token recomputes attention over its entire history (O(n²) complexity)

**Overhead**: For 32-token sequence:
- With cache: 32 attention operations (one per step)
- Without cache: 1 + 2 + 3 + ... + 32 = 528 attention operations
- **16.5x overhead**

**Root cause**: ONNX fixed shape requirement for KV cache inputs (dynamic cache needs variable shapes)

---

#### Combined Impact: 528x Total Overhead

The two inefficiencies **multiply together**:

| Scenario | Positions/step | Cache | Attention ops | Overhead |
|----------|---------------|-------|---------------|----------|
| **Ideal (CPU)** | 1 (dynamic) | Yes | ~32 | **1x** |
| Fixed shape only | 32 (fixed) | Yes | ~1,024 | **32x** |
| No cache only | 1 (dynamic) | No | ~528 | **16.5x** |
| **Hailo (both)** | 32 (fixed) | No | ~16,896 | **~528x** |

**Result**: The Hailo decoder performs **~528x more attention operations** than an efficient cached implementation!

### Why Does Hailo Do This?

1. **Fixed shape requirement**: Hailo NPU requires static tensor shapes at compile time. Supporting dynamic cache would require variable-length inputs.

2. **Hardware speed**: Even with 16x more operations, Hailo's NPU is fast enough for short sequences (24-32 tokens). The NPU acceleration may compensate for the inefficiency.

3. **Simplification**: Avoiding cache simplifies the ONNX graph and deployment. No need to manage stateful inference.

4. **Short sequences**: Whisper decoder typically generates 24-32 tokens, not hundreds. The O(n²) overhead is acceptable for small n.

### Real-World Impact: Raspberry Pi 5 + Hailo-8 Analysis

**Hardware specs**:
- **Raspberry Pi 5 CPU**: 4x ARM Cortex-A76 @ 2.4 GHz (~40-80 GFLOPS total)
- **Hailo-8 NPU**: 13 TOPS = 13,000 GFLOPS (**160-325x faster raw compute**)

**Can Hailo overcome the 528x overhead with 160-325x speed advantage?**

**Math**:
- Best case: 528x operations ÷ 325x speed = **~1.6x slower** than CPU
- Worst case: 528x operations ÷ 160x speed = **~3.3x slower** than CPU

**BUT** this assumes ideal conditions. Reality is worse because:

**Where Hailo NPU shines** (Encoder workload):
- Large matrix multiplications
- High parallelism (entire sequences processed at once)
- Large hidden dimensions
- Batch processing

**Where Hailo NPU struggles** (Decoder workload):
- Small matrices (32 × 384 for tiny model)
- Sequential processing (32 autoregressive steps, cannot parallelize across time)
- Memory-bound operations (loading weights 32× per inference)
- Low utilization of 13 TOPS capacity

**The decoder is the wrong workload for an NPU!**

**Predicted performance on Raspberry Pi 5 + Hailo-8**:

| Component | CPU (with cache) | Hailo NPU (no cache) | Winner |
|-----------|------------------|----------------------|--------|
| **Encoder** | 200-500ms | 10-50ms | **Hailo: 4-50x faster** ✓ |
| **Decoder** | 50-100ms | 500-2000ms | **CPU: 5-20x faster** ✓ |
| **Total (Hybrid)** | **60-150ms** | - | **Optimal** |
| **Total (Full NPU)** | - | **510-2050ms** | **3-13x slower** |

**Conclusion**: On Raspberry Pi 5 + Hailo-8, hybrid architecture (encoder on NPU, decoder on CPU) is **significantly faster** than full NPU deployment.

### Alternative Architectures to Consider

For production deployment, consider these alternatives:

1. **Hybrid CPU-NPU approach**: Run encoder on Hailo NPU (where it shines for parallel processing), run decoder on CPU with KV cache (efficient autoregressive generation)

2. **Batch processing**: If transcribing multiple audio files, batch them to amortize setup costs

3. **Non-autoregressive decoder**: Research models like Whisper variants that decode all tokens in parallel (no KV cache needed)

4. **Hailo-specific optimization**: Work with Hailo to support dynamic cache shapes (if hardware permits)

### Fixed Shapes: Encoder vs Decoder

**Encoder**: Processes entire audio in one shot (parallel). Fixed shape = fixed audio duration (e.g., 10 seconds). Shorter audio gets padded to 10s.

**Decoder**: Generates tokens one-by-one (autoregressive). Fixed shape decision has two options:
- **Option A** (current): Process all 32 token positions at once, no cache (O(n²) but parallel on NPU)
- **Option B** (alternative): Process tokens sequentially with cache, but requires dynamic shapes OR separate compiled models for each step

The Hailo implementation chose **Option A** - sacrifice computational efficiency for deployment simplicity and fixed shapes.

### ⚠️ IMPORTANT TODO

**For production deployment, evaluate**:
1. Actual inference latency on Hailo hardware (may be acceptable despite 16x overhead)
2. Compare hybrid CPU-NPU architecture (encoder on Hailo, decoder on CPU with cache)
3. Profile memory bandwidth vs computation trade-off
4. Consider whether real-time performance is achieved for target use case

**Question to revisit**: Is the 16.5x computational overhead acceptable, or should we pursue hybrid architecture?

---

## Recommended Architecture: Hybrid CPU-NPU Deployment

### Why Hybrid Makes Sense

The optimal deployment strategy may be to **split the workload**:

**Encoder on Hailo NPU**:
- Highly parallel computation (convolutional layers + self-attention on entire mel spectrogram)
- This is where NPU acceleration provides maximum benefit
- Encoder is the heavier part of the model (more parameters, more compute)
- Processes fixed-size input naturally (e.g., 500 frames for 10s audio)
- Single forward pass per audio clip

**Decoder on CPU with KV cache**:
- Sequential autoregressive generation (token-by-token)
- CPU can efficiently handle cached computation
- Avoids 16.5x computational overhead from recomputing attention
- Decoder is lighter (fewer parameters than encoder)
- Benefits from O(n) complexity vs O(n²)

### Performance Analysis

**Data Transfer Cost**:
- Encoder output: [1, 500, 384] float32 = 500 × 384 × 4 bytes = **~768 KB**
- Transfer from NPU to CPU memory: negligible overhead (~microseconds)
- This is a one-time cost per audio clip

**Computational Savings**:
- Decoder with cache: 32 × 4 layers = **128 attention operations**
- Decoder without cache: 528 × 4 layers = **2,112 attention operations**
- **Savings: 1,984 operations** (94% reduction)

**Trade-off**:
- **Hybrid**: Fast encoder (NPU) + efficient decoder (CPU with cache)
- **Full NPU**: Fast encoder (NPU) + inefficient decoder (NPU, no cache, 16x overhead)

**Likely outcome**: Unless Hailo NPU is >16x faster than your CPU for attention operations, the hybrid approach will be more efficient overall.

### Implementation Considerations

**For hybrid architecture**:
1. Export encoder to Hailo ONNX (already done in `convert_encoder.ipynb`)
2. Keep decoder as PyTorch model on CPU with standard HuggingFace generation (includes KV cache)
3. Pipeline: Audio → Hailo encoder → CPU decoder (with cache) → Transcription

**Code example**:
```python
# Encoder on Hailo NPU
encoder_output = hailo_encoder_inference(mel_spectrogram)  # [1, 500, 384]

# Decoder on CPU with cache (standard HuggingFace)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
decoder_input_ids = torch.tensor([[50258]])  # Start token

generated_ids = model.generate(
    decoder_input_ids=decoder_input_ids,
    encoder_outputs=(encoder_output,),  # Use Hailo encoder output
    max_length=32,
    use_cache=True  # Enable KV cache for efficiency
)
```

**No ONNX export needed for decoder** - just use standard HuggingFace model on CPU.

### Recommendation for Raspberry Pi 5 + Hailo-8

**Strongly recommended: Hybrid architecture**
- ✓ Encoder on Hailo NPU (4-50x faster than CPU, 80%+ of compute)
- ✓ Decoder on CPU with KV cache (5-20x faster than NPU, avoids 528x overhead)
- ✓ 768 KB data transfer between NPU and CPU (negligible)
- ✓ Expected total speedup: 3-13x faster than full NPU

**When to consider full NPU** (unlikely for your hardware):
- Different hardware with better decoder characteristics
- Hailo provides dynamic shape or cache support (future optimization)
- Benchmarking proves otherwise (always measure!)

**The decoder ONNX export work is still valuable for**:
- Understanding model architecture
- Educational purposes
- Benchmarking to validate hybrid is better
- Future hardware compatibility

---


# Final notes

* ONNX export is taking extremely long and not finishing
    * we are constantly stuck at the same position: Still stuck at the exact same place - layer 1, self-attention, line 238.
    
* but overall we likely don't want the decoder anyways
