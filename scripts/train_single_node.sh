#!/usr/bin/env bash
# Single-node training (8x H100, colocate mode)
#
# Usage:
#   bash scripts/train_single_node.sh circle_packing          # 50 epochs
#   bash scripts/train_single_node.sh circle_packing 10       # 10 epochs
#   bash scripts/train_single_node.sh ac1 50                  # AC1, 50 epochs
#
# Resume:
#   RESUME_DIR=checkpoints/ttt-discover/old-run INPLACE=true \
#     bash scripts/train_single_node.sh circle_packing 50
#
# All run_verl.sh env vars (ROLLOUT_N, SP_SIZE, etc.) are supported.

set -euo pipefail

TASK=${1:?Usage: bash scripts/train_single_node.sh <task> [epochs]}
EPOCHS=${2:-50}

export TOTAL_EPOCHS=${EPOCHS}
export NNODES=1
export CONDA_ENV=${CONDA_ENV:-verl_discover}

echo "========================================="
echo "Single-node training"
echo "  Task:   ${TASK}"
echo "  Epochs: ${EPOCHS}"
echo "  GPUs:   8 (colocate)"
echo "========================================="

exec bash run_verl.sh ${TASK}
