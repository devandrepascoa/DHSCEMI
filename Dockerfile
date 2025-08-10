# Use NVIDIA CUDA base image for GPU support
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    ca-certificates \
    sudo \
    docker.io \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY . .

# Install UV, sync dependencies, create models directory, and run benchmark
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv sync && \
    mkdir -p models && \
    chmod +x scripts/*.sh && \
    chmod +x run_benchmark.sh

# Run the benchmark script
CMD ["./run_benchmark.sh"]
