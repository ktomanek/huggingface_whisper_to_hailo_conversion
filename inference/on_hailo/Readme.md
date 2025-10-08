This code needs to be run on Hailo directly, ie on Raspberry Pi or elsewhere where the hailo chip is attached



# new

# Run full pipeline

# HEF only 
python whisper_on_hailo_pipeline.py --encoder_hef_file ~/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef --decoder_hef_file ~/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef --audio_file ~/dev/audio_samples/hello_world.wav --decoder_assets_path /home/katrintomanek/dev/hailo-rpi5-examples/whisper/assets/decoder_assets

# Hybrid HEF and onnx
python whisper_on_hailo_pipeline.py --encoder_hef_file ~/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef --decoder_onnx_dir ~/dev/hailo_whisper_convert/my_converted/ --audio_file ~/dev/audio_samples/hello_world.wav --decoder_assets_path /home/katrintomanek/dev/hailo-rpi5-examples/whisper/assets/decoder_assets

# Run benchmarking on encoder and decoder

python benchmark_whisper_on_hailo.py --encoder_hef_file ~/dev/hailo_whisper_convert/HEF_h8l_from_hailo/tiny-whisper-encoder-10s_15dB_h8l.hef --encoder_onnx_file /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/converted_for_hailo/whisper_tiny_encoder_10s_hailo_final.onnx --encoder_orig_onnx_file /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/default/encoder_model.onnx --audio_file ~/dev/audio_samples/hello_world.wav --include_faster_whisper --decoder_onnx_dir /home/katrintomanek/dev/hailo-rpi5-examples/whisper/models/onnx/default_int8 

