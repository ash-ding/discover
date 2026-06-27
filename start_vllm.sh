#!/bin/bash
# Universal vLLM server startup script
# Default configuration matches paper settings (TP=4)

# ============================================================================
# Configuration (can be overridden by environment variables)
# ============================================================================

MODEL_PATH=${MODEL_PATH:-/workspace/home/asherding/models/Qwen3-8B}
VLLM_PORT=${VLLM_PORT:-8888}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.9}
MAX_LORA_RANK=${MAX_LORA_RANK:-64}

# GPU allocation (default: 5 GPUs for TP=4)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ============================================================================
# Display configuration
# ============================================================================

echo "======================================================================"
echo "  vLLM Server Startup"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  Model Path:           $MODEL_PATH"
echo "  Port:                 $VLLM_PORT"
echo "  Tensor Parallelism:   $TENSOR_PARALLEL"
echo "  Max Model Length:     $MAX_MODEL_LEN"
echo "  GPU Memory Util:      $GPU_MEMORY_UTIL"
echo "  Max LoRA Rank:        $MAX_LORA_RANK"
echo "  CUDA Devices:         $CUDA_VISIBLE_DEVICES"
echo ""
echo "======================================================================"
echo ""

# ============================================================================
# Check for existing vLLM processes and clean up
# ============================================================================

echo "🔍 Checking for existing vLLM processes..."

# Check if vLLM server is already running
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "⚠️  WARNING: Existing vLLM server process detected!"
    echo ""
    ps aux | grep "[v]llm.entrypoints.openai.api_server"
    echo ""
    echo "🧹 Cleaning up existing vLLM server..."

    # Run stop script
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    bash "$SCRIPT_DIR/stop_vllm.sh"

    echo ""
    echo "✅ Cleanup complete. Proceeding with new vLLM server startup..."
    echo ""
elif lsof -i :$VLLM_PORT > /dev/null 2>&1; then
    # Port occupied by non-vLLM process
    echo "⚠️  WARNING: Port $VLLM_PORT is occupied by another process!"
    echo ""
    lsof -i :$VLLM_PORT
    echo ""
    echo "❌ Cannot start vLLM. Please free port $VLLM_PORT first."
    exit 1
else
    echo "✅ No conflicts detected."
    echo ""
fi

# ============================================================================
# Start vLLM server
# ============================================================================

echo "🚀 Starting vLLM server..."
echo ""

# Activate conda environment (vLLM is installed in all task environments)
# Use discover_math as default (can be overridden with VLLM_CONDA_ENV)
VLLM_CONDA_ENV=${VLLM_CONDA_ENV:-discover_math}
source ~/.bashrc
conda activate $VLLM_CONDA_ENV

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port $VLLM_PORT \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --max-model-len $MAX_MODEL_LEN \
    --enable-lora \
    --max-lora-rank $MAX_LORA_RANK \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --disable-custom-all-reduce
