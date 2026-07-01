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
#   bash scripts/train_multi_node.sh ac1 50                  # AC1, 50 epochs
#
# Resume:
#   RESUME_DIR=checkpoints/ttt-discover/old-run INPLACE=true \
#     bash scripts/train_multi_node.sh circle_packing 50
#
# Sync code to worker node before training:
#   bash scripts/sync_worker.sh
#
# All run_verl.sh env vars are supported.

set -euo pipefail

TASK=${1:?Usage: bash scripts/train_multi_node.sh <task> [epochs]}
EPOCHS=${2:-50}

HEAD_NODE=${HEAD_NODE:-10.241.128.30}
WORKER_NODE=${WORKER_NODE:-10.241.128.16}

export TOTAL_EPOCHS=${EPOCHS}
export NNODES=2
export CONDA_ENV=${CONDA_ENV:-verl_discover}

# Check Ray cluster is running
if ! ray status &>/dev/null; then
    echo "ERROR: Ray cluster not running. Start it first:"
    echo "  bash scripts/start_ray_cluster.sh"
    exit 1
fi

NODE_COUNT=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
if [ "$NODE_COUNT" -lt 2 ]; then
    echo "WARNING: Only ${NODE_COUNT} node(s) in Ray cluster. Expected 2."
    echo "Start the cluster: bash scripts/start_ray_cluster.sh"
fi

echo "========================================="
echo "Multi-node training"
echo "  Task:   ${TASK}"
echo "  Epochs: ${EPOCHS}"
echo "  Nodes:  2 (${HEAD_NODE} + ${WORKER_NODE})"
echo "  GPUs:   16 (colocate)"
echo "========================================="

exec bash run_verl.sh ${TASK}
