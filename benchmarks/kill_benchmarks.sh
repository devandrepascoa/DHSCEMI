pkill -f "uv run python -u scripts/benchmark"
docker rm -f $(docker ps -a | grep "ghcr.io/ggml-org/llama.cpp" | awk '{print $1}')
