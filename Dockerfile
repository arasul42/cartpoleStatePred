# Use Ubuntu 20.04 as base image
FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04


# Set non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install essential dependencies in one step to optimize caching
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    x11-utils mesa-utils \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    stable-baselines3

RUN pip install gymnasium[classic-control] \
    imageio[ffmpeg] \
    opencv-python


# Set default command
CMD ["bash"]
