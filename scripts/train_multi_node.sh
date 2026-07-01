#!/usr/bin/env bash
# Multi-node training (2x 8x H100 = 16 GPUs, colocate mode)
#
# Prerequisites:
#   1. Both nodes have verl_discover conda env and synced code
#   2. Passwordless SSH between nodes
#   3. Ray cluster started: bash scripts/start_ray_cluster.sh
#
# Usage:
#   bash scripts/train_multi_node.sh circle_packing          # 50 epochs
#   bash scripts/train_multi_node.sh circle_packing 10       # 10 epochs
#
# Resume:
#   RESUME_DIR=checkpoints/ttt-discover/old-run INPLACE=true \
#     bash scripts/train_multi_node.sh circle_packing 50
#
# Sync code first if changed:
#   bash scripts/sync_worker.sh

set -euo pipefail

TASK=${1:?Usage: bash scripts/train_multi_node.sh <task> [epochs]}
EPOCHS=${2:-50}

export TOTAL_EPOCHS=${EPOCHS}
export NNODES=2
export CONDA_ENV=${CONDA_ENV:-verl_discover}

# Check Ray cluster
source /workspace/home/asherding/.conda/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}
if ! ray status &>/dev/null; then
    echo "ERROR: Ray cluster not running. Start it first:"
    echo "  bash scripts/start_ray_cluster.sh"
    exit 1
fi

# Count GPUs in cluster
TOTAL_GPUS=$(python3 -c "import ray; ray.init(address='auto'); print(int(ray.cluster_resources().get('GPU', 0))); ray.shutdown()" 2>/dev/null || echo "0")
echo "========================================="
echo "Multi-node training"
echo "  Task:   ${TASK}"
echo "  Epochs: ${EPOCHS}"
echo "  Nodes:  2"
echo "  GPUs:   ${TOTAL_GPUS} (colocate)"
echo "========================================="

if [ "${TOTAL_GPUS}" -lt 16 ]; then
    echo "WARNING: Expected 16 GPUs but found ${TOTAL_GPUS}. Check Ray cluster."
fi

exec bash run_verl.sh ${TASK}
