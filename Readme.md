# Convert HuggingFace Transformers Whisper model to ONNX in format that is compatible for Hailo conversion (Hailo 8/10)

 
# Install locally

* make a new venv
* install `requirements.txt`

# Install on Hailo

* make a new venv
* then run `./hailo_python_installation.sh`
* then run `pip install -r hailo_requirements.txt`


# new

# Run full pipeline

# HEF only 
python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef \
    --audio_file samples/hello_world.wav

# Hybrid HEF and onnx
python inference/on_hailo/whisper_on_hailo_pipeline.py \
    --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 \
    --audio_file samples/hello_world.wav

# Run benchmarking on encoder and decoder

python inference/on_hailo/benchmark_whisper_on_hailo.py \
    --encoder_hef_file models/hef/my_model/whisper_tiny_encoder_10s_hailo_final_optimized.hef \
    --encoder_onnx_file models/onnx/hailo_compatible_models/hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx \
    --encoder_orig_onnx_file models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default/encoder_model.onnx \
    --audio_file samples/hello_world.wav \
    --include_faster_whisper \
    --decoder_onnx_dir models/onnx/whisper_optimum_onnx_conversions/whisper_tiny_onnx/default_int8 


or use original HEF from Hailo: --encoder_hef_file models/hef/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef
