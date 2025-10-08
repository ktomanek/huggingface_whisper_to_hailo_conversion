import modal

app = modal.App("ubuntu-console")

volume = modal.Volume.from_name("hailo_dfw_volume", create_if_missing=True)

@app.function(
    image=modal.Image.from_registry(
        #"nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04" # TODO TEST
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
        #"nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", # this works as in it starts but missing devel!
        # "ubuntu:22.04",  # Plain Ubuntu 22.04 - Hailo doesn't need CUDA
        add_python="3.11"
    ).apt_install(
        "wget", "curl", "git", "vim",
        "python3", "python3-pip", "python3-dev",
        "python3.11-dev",
        "build-essential",
        "gcc", "g++",
        "clang",
        "graphviz", "graphviz-dev", "libgraphviz-dev",
        "pkg-config",
        "ffmpeg", "libportaudio2",
        "lsb-release",
        )
        .run_commands(
            "pip3 install --upgrade pip wheel setuptools",
            "pip3 install --no-build-isolation pygraphviz",
        ),
        gpu="A100",
        #gpu="A10G",
        cpu=8.0,              # Request 8 vCPUs (default is 1-2)
        memory=32768,         # Request 32 GB RAM (in MB)
        volumes={"/hailo_data": volume},
        timeout=3600,         # 1 hour timeout (default is 5 minutes)
)
def run_console():
    import subprocess
    subprocess.run(["/bin/bash"])

# Run with: 
# modal shell modal_shell.py::run_console


# uplooad stuff
#  modal volume put hailo_dfw_volume ~/Downloads/hailo_dataflow_compiler-3.32.0-py3-none-linux_x86_64.whl
#  modal volume put hailo_dfw_volume hailo_compatible_models/hf_whisper_tiny

# cd /hailo_data/hailo-whisper/
# source  whisper_env/bin/activate
# cd /hailo_data/hailo-whisper/ & source  whisper_env/bin/activate
# python3 -u -m conversion.convert_whisper_encoder --variant tiny --hw-arch hailo8l --load-calib-set ../hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx 
# python3 -u -m conversion.convert_whisper_encoder --variant tiny --hw-arch hailo8l  ../hf_whisper_tiny/whisper_tiny_encoder_10s_hailo_final.onnx 