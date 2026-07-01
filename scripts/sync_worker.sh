#!/usr/bin/env bash
# Sync code and VERL fork to worker node.
# Run this after any code changes before multi-node training.
#
# Usage:
#   bash scripts/sync_worker.sh

set -euo pipefail

WORKER_NODE=${WORKER_NODE:-10.241.128.16}
PROJECT_DIR="/workspace/home/asherding/code/discover"

echo "Syncing code to ${WORKER_NODE}..."
rsync -avz --delete \
    --exclude='checkpoints' --exclude='wandb' --exclude='outputs' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='tinker_log' \
    --exclude='.git' \
    ${PROJECT_DIR}/ \
    asherding@${WORKER_NODE}:${PROJECT_DIR}/ \
    2>&1 | tail -5

echo ""
echo "Reinstalling verl on worker..."
ssh asherding@${WORKER_NODE} "
    source /workspace/home/asherding/.conda/etc/profile.d/conda.sh && \
    conda activate verl_discover && \
    pip install -e ${PROJECT_DIR}/verl 2>&1 | tail -3
"

echo ""
echo "Sync complete!"
