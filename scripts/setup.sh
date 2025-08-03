#!/bin/bash

set -e

echo "🚀 Setting up LLM Benchmarking System..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
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

echo "🐳 Pulling prebuilt Docker images..."

echo "  Pulling CPU image..."
docker pull ghcr.io/ggml-org/llama.cpp:full

if command -v nvidia-smi &> /dev/null; then
    echo "  Pulling CUDA image..."
    docker pull ghcr.io/ggml-org/llama.cpp:full-cuda
else
    echo "  Skipping CUDA image (no NVIDIA GPU detected)"
fi

echo "📦 Installing Uv dependencies..."
uv sync

echo "🧪 Testing system setup..."
uv run scripts/test_setup.py

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add your model files (.gguf or .bin) to the models/ directory"
echo "2. Run: uv run main.py"
echo ""
echo "📖 For detailed instructions, see README.md"
