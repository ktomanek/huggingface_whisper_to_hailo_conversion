
import onnx
from onnx import helper
import onnxruntime as ort
import sys

import numpy as np
np.random.seed(42)


our_encoder_path = "hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx"
ref_encoder_path = "hailo_reference_models/tiny/tiny-whisper-encoder-10s.onnx"

input_data = np.random.randn(1, 80, 1, 1000).astype(np.float32)    

your_model = onnx.load(our_encoder_path)
ref_model = onnx.load(ref_encoder_path)


##############
def compare_node_outputs(node_idx, our_encoder_path, ref_encoder_path, input_data):    
   
    your_model = onnx.load(our_encoder_path)        
    your_model.graph.output.append(
        helper.make_tensor_value_info(
            your_model.graph.node[node_idx].output[0],
            onnx.TensorProto.FLOAT,
            None
        )
    )
    onnx.save(your_model, f"/tmp/your_debug_{node_idx}.onnx")
    
    ref_model = onnx.load(ref_encoder_path)
    ref_model.graph.output.append(
        helper.make_tensor_value_info(
            ref_model.graph.node[node_idx].output[0],
            onnx.TensorProto.FLOAT,
            None
        )
    )
    onnx.save(ref_model, f"/tmp/ref_debug_{node_idx}.onnx")
    
    your_session = ort.InferenceSession(f"/tmp/your_debug_{node_idx}.onnx")
    ref_session = ort.InferenceSession(f"/tmp/ref_debug_{node_idx}.onnx")
    
    your_outputs = your_session.run(None, {"x.1": input_data})
    ref_outputs = ref_session.run(None, {"x.1": input_data})
    
    print(f"\tNode {node_idx} ({your_model.graph.node[node_idx].op_type}):")
    print(f"\tYour shape: {your_outputs[1].shape}, Ref shape: {ref_outputs[1].shape}")
    if {your_outputs[1].shape} == {ref_outputs[1].shape}:
        difference = np.abs(your_outputs[1] - ref_outputs[1]).max()
        print(f"\t\tDiff: {difference:.6f}")
        print(f"\t\tYour mean: {your_outputs[1].mean():.6f}, Ref mean: {ref_outputs[1].mean():.6f}")        
        if difference > 1e-4:
            print("\t\t  ❌ Significant difference!")
    else:
        print("  Shapes differ, cannot compare values.")



print("\n\n>>>> Comparing all nodes by type and mean/std diff")
for node_idx in range(1, len(your_model.graph.node)):
# for node_idx in range(20,23):
    our_node = your_model.graph.node[node_idx]
    ref_node = ref_model.graph.node[node_idx]
    print(f"Node {node_idx}")
    print(f"\tours:{our_node.op_type}\t{our_node.name if our_node.name else ''}")
    print(f"\t ref:{ref_node.op_type}\t{ref_node.name if ref_node.name else ''}")

    # compare
    compare_node_outputs(node_idx, our_encoder_path, ref_encoder_path, input_data)
