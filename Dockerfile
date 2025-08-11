# Use Docker-in-Docker base image for running Docker commands within container
FROM docker:dind

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apk update && apk add --no-cache \
    curl \
    wget \
    git \
    build-base \
    cmake \
    pkgconfig \
    openssl-dev \
    ca-certificates \
    python3 \
    python3-dev \
    py3-pip \
    bash \
    shadow

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

# Create startup script that starts Docker daemon and runs benchmark
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'set -e' >> /start.sh && \
    echo 'echo "Starting Docker daemon..."' >> /start.sh && \
    echo 'dockerd-entrypoint.sh &' >> /start.sh && \
    echo 'sleep 10' >> /start.sh && \
    echo 'echo "Docker daemon started, running benchmark..."' >> /start.sh && \
    echo 'cd /app' >> /start.sh && \
    echo './run_benchmark.sh' >> /start.sh && \
    chmod +x /start.sh

# Run the startup script
CMD ["/start.sh"]
