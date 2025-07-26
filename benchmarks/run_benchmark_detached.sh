#!/bin/bash

# Run benchmark with timestamped log
nohup python3 -u scripts/benchmark.py --cpu-only > log_$(date +%Y%m%d_%H%M%S).log &

echo "Benchmark started in background"
echo "Log file: log_$(date +%Y%m%d_%H%M%S).log"
