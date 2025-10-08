# compare reference Hailo onnx encoder and our converted encoder
import onnxruntime as ort
import numpy as np

our_encoder_path = "models/hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx"
ref_encoder_path = "models/hailo_reference_models/tiny/tiny-whisper-encoder-10s.onnx"


your_session = ort.InferenceSession(our_encoder_path)
ref_session = ort.InferenceSession(ref_encoder_path)
print("instantiated sessions")

input_data = np.random.randn(1, 80, 1, 1000).astype(np.float32)

your_outputs = your_session.run(None, {"x.1": input_data})
ref_outputs = ref_session.run(None, {"x.1": input_data})

print(f"Encoder Output Comparison:")
print(f"  Your mean: {your_outputs[0].mean():.6f}, std: {your_outputs[0].std():.6f}")
print(f"  Ref mean: {ref_outputs[0].mean():.6f}, std: {ref_outputs[0].std():.6f}")
print(f"  Max abs difference: {np.abs(your_outputs[0] - ref_outputs[0]).max():.6f}")
print(f"  Mean abs difference: {np.abs(your_outputs[0] - ref_outputs[0]).mean():.6f}")