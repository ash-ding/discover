#!/bin/bash
# Run Denoising experiment
# Usage: bash run.sh <config_file>
# Example: bash run.sh config_paper.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ENV="discover_denoising"

log_info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[1;32m[✓]\033[0m $1"; }
log_error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; }

# Check argument
if [ $# -eq 0 ]; then
    log_error "Usage: bash run.sh <config_file>"
    log_info "Examples:"
    log_info "  bash run.sh config_paper.yaml      # 50 epochs"
    log_info "  bash run.sh config_validate.yaml   # 1 epoch"
    log_info "  bash run.sh /path/to/custom.yaml   # custom config"
    exit 1
fi

CONFIG_FILE="$1"

# Resolve relative path
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
fi

log_info "Denoising Experiment"
log_info "Config: $CONFIG_FILE"
echo ""

# Check prerequisites
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

if ! conda env list | grep -q "^$CONDA_ENV "; then
    log_error "Conda environment not found: $CONDA_ENV"
    log_info "Create it with: conda create -n $CONDA_ENV python=3.11 -y"
    exit 1
fi

if ! curl -s "http://localhost:8888/v1/models" > /dev/null 2>&1; then
    log_error "vLLM server not running"
    log_info "Start it with: bash $PROJECT_ROOT/start_vllm.sh"
    exit 1
fi

log_success "All prerequisites met"

# Set environment
export TTT_CONFIG_PATH="$CONFIG_FILE"
export VLLM_BASE_URL="http://localhost:8888"
export WANDB_MODE="offline"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Activate and run
source ~/.bashrc
conda activate $CONDA_ENV
cd "$PROJECT_ROOT"

log_info "Starting experiment..."
python -m examples.denoising.env --local

log_success "Experiment completed!"
