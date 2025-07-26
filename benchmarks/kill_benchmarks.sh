pkill -f "python3 -u scripts/benchmark.py"
docker rm -f $(docker ps -a | grep "ghcr.io/ggml-org/llama.cpp" | awk '{print $1}')