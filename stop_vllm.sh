#!/bin/bash
# Stop vLLM server

echo "Stopping vLLM server..."

# Find and kill vLLM processes
pkill -f "vllm.entrypoints.openai.api_server"

# Wait for processes to stop
sleep 2

# Verify
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "Warning: Some vLLM processes still running, forcing kill..."
    pkill -9 -f "vllm.entrypoints.openai.api_server"
    sleep 1
fi

# Check GPU memory
echo ""
echo "GPU Memory Status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

echo ""
echo "vLLM server stopped."
