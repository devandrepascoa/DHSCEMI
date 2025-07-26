#!/bin/bash

# LLM Benchmarking System Setup Script

set -e

echo "🚀 Setting up LLM Benchmarking System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available (modern docker compose)
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose plugin."
    echo "   On Ubuntu/Debian: sudo apt-get install docker-compose-plugin"
    echo "   On other systems: Follow Docker documentation"
    exit 1
fi

# Check NVIDIA Docker support for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    if docker info | grep -q nvidia; then
        echo "✅ NVIDIA Docker runtime available"
    else
        echo "⚠️  NVIDIA Docker runtime not found. CUDA benchmarking may not work."
    fi
else
    echo "⚠️  NVIDIA GPU not detected. CUDA benchmarking will be skipped."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models results scripts docker

# Pull prebuilt Docker images
echo "🐳 Pulling prebuilt Docker images..."

# Pull CPU image (full version includes llama-bench)
echo "  Pulling CPU image..."
docker pull ghcr.io/ggml-org/llama.cpp:full
docker tag ghcr.io/ggml-org/llama.cpp:full llama-bench:cpu

# Pull CUDA image if nvidia-smi is available (full-cuda version includes llama-bench)
if command -v nvidia-smi &> /dev/null; then
    echo "  Pulling CUDA image..."
    docker pull ghcr.io/ggml-org/llama.cpp:full-cuda
    docker tag ghcr.io/ggml-org/llama.cpp:full-cuda llama-bench:cuda
else
    echo "  Skipping CUDA image (no NVIDIA GPU detected)"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.py scripts/*.sh

# Test the system
echo "🧪 Testing system setup..."
python3 scripts/test_setup.py

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add your model files (.gguf or .bin) to the models/ directory"
echo "2. Run: python3 scripts/benchmark.py"
echo ""
echo "📖 For detailed instructions, see README.md"
