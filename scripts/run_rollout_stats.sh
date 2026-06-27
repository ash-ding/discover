#!/bin/bash
# Wrapper script to run rollout statistics collection with proper conda environments

set -e

OUTPUT_FILE="rollout_length_statistics.json"
LOG_DIR="rollout_stats_logs"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "Starting Rollout Length Statistics Collection"
echo "================================================================================"
echo "Total tasks: 4"
echo "Execution mode: SEQUENTIAL (one task at a time, proper conda env per task)"
echo "================================================================================"
echo ""

# Set required environment variables (same as run.sh scripts)
export VLLM_BASE_URL="http://localhost:8888"
export WANDB_MODE="offline"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

echo "Environment variables set:"
echo "  VLLM_BASE_URL=$VLLM_BASE_URL"
echo "  WANDB_MODE=$WANDB_MODE"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# Task 1: Circle Packing (discover_math environment)
echo "================================================================================"
echo "Task 1/4: CIRCLE_PACKING"
echo "================================================================================"
echo "Environment: discover_math"
echo "Config: examples/circle_packing/config_paper.yaml"
echo "================================================================================"
echo ""

conda run -n discover_math --no-capture-output python scripts/run_single_task.py \
    --task-name circle_packing \
    --config-path examples/circle_packing/config_paper.yaml \
    --output-file "$OUTPUT_FILE" \
    2>&1 | tee "$LOG_DIR/circle_packing.log"

echo ""
echo "✓ Circle Packing completed"
echo ""

# Task 2: AHC (discover_ale environment or discover_math)
echo "================================================================================"
echo "Task 2/4: AHC"
echo "================================================================================"
echo "Environment: discover_ale (or discover_math as fallback)"
echo "Config: examples/ahc/config_paper.yaml"
echo "================================================================================"
echo ""

# Try discover_ale first, fallback to discover_math
if conda env list | grep -q "discover_ale"; then
    ENV_NAME="discover_ale"
else
    ENV_NAME="discover_math"
    echo "Warning: discover_ale not found, using discover_math"
fi

conda run -n "$ENV_NAME" --no-capture-output python scripts/run_single_task.py \
    --task-name ahc \
    --config-path examples/ahc/config_paper.yaml \
    --output-file "$OUTPUT_FILE" \
    2>&1 | tee "$LOG_DIR/ahc.log"

echo ""
echo "✓ AHC completed"
echo ""

# Task 3: GPU Mode (discover_gpumode environment)
echo "================================================================================"
echo "Task 3/4: GPU_MODE"
echo "================================================================================"
echo "Environment: discover_gpumode"
echo "Config: examples/gpu_mode/config_paper.yaml"
echo "================================================================================"
echo ""

if conda env list | grep -q "discover_gpumode"; then
    conda run -n discover_gpumode --no-capture-output python scripts/run_single_task.py \
        --task-name gpu_mode \
        --config-path examples/gpu_mode/config_paper.yaml \
        --output-file "$OUTPUT_FILE" \
        2>&1 | tee "$LOG_DIR/gpu_mode.log"
    echo ""
    echo "✓ GPU Mode completed"
else
    echo "✗ discover_gpumode environment not found, skipping GPU Mode"
    python scripts/run_single_task.py \
        --task-name gpu_mode \
        --config-path examples/gpu_mode/config_paper.yaml \
        --output-file "$OUTPUT_FILE" \
        --skip \
        2>&1 | tee "$LOG_DIR/gpu_mode.log"
fi

echo ""

# Task 4: Denoising (discover_denoising environment)
echo "================================================================================"
echo "Task 4/4: DENOISING"
echo "================================================================================"
echo "Environment: discover_denoising"
echo "Config: examples/denoising/config_paper.yaml"
echo "================================================================================"
echo ""

if conda env list | grep -q "discover_denoising"; then
    conda run -n discover_denoising --no-capture-output python scripts/run_single_task.py \
        --task-name denoising \
        --config-path examples/denoising/config_paper.yaml \
        --output-file "$OUTPUT_FILE" \
        2>&1 | tee "$LOG_DIR/denoising.log"
    echo ""
    echo "✓ Denoising completed"
else
    echo "✗ discover_denoising environment not found, skipping Denoising"
    python scripts/run_single_task.py \
        --task-name denoising \
        --config-path examples/denoising/config_paper.yaml \
        --output-file "$OUTPUT_FILE" \
        --skip \
        2>&1 | tee "$LOG_DIR/denoising.log"
fi

echo ""
echo "================================================================================"
echo "ALL TASKS COMPLETED"
echo "================================================================================"
echo "Results saved to: $OUTPUT_FILE"
echo "Logs saved to: $LOG_DIR/"
echo "================================================================================"
