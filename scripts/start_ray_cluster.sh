#!/usr/bin/env bash
# Start a Ray cluster across 2 nodes for multi-node VERL training.
#
# Usage:
#   bash scripts/start_ray_cluster.sh          # Start cluster
#   bash scripts/start_ray_cluster.sh stop     # Stop cluster
#
# After starting, run training with:
#   NNODES=2 bash run_verl.sh circle_packing

set -euo pipefail

HEAD_NODE=${HEAD_NODE:-10.241.128.30}
WORKER_NODE=${WORKER_NODE:-10.241.128.16}
RAY_PORT=${RAY_PORT:-6379}
CONDA_ENV=${CONDA_ENV:-verl_discover}
CONDA_SH="/workspace/home/asherding/.conda/etc/profile.d/conda.sh"

# CUDA 13 runtime for vLLM
CUDA13_LIB="/workspace/home/asherding/.conda/envs/${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cu13/lib"

ACTION=${1:-start}

stop_cluster() {
    echo "Stopping Ray cluster..."
    ray stop --force 2>/dev/null || true
    ssh asherding@${WORKER_NODE} "ray stop --force 2>/dev/null || true" 2>/dev/null || true
    echo "Ray cluster stopped."
}

start_cluster() {
    # Stop any existing cluster first
    stop_cluster

    echo "========================================="
    echo "Starting Ray cluster"
    echo "  Head:   ${HEAD_NODE}"
    echo "  Worker: ${WORKER_NODE}"
    echo "========================================="

    # Start head node
    echo "[Head] Starting Ray head..."
    source ${CONDA_SH} && conda activate ${CONDA_ENV}
    export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}"

    ray start --head \
        --port=${RAY_PORT} \
        --num-cpus=160 \
        --num-gpus=8 \
        --block &
    HEAD_PID=$!

    # Wait for head to be ready
    sleep 10
    echo "[Head] Ray head started."

    # Start worker node
    echo "[Worker] Starting Ray worker on ${WORKER_NODE}..."
    ssh asherding@${WORKER_NODE} "
        source ${CONDA_SH} && conda activate ${CONDA_ENV} && \
        export LD_LIBRARY_PATH='${CUDA13_LIB}:\${LD_LIBRARY_PATH:-}' && \
        ray start --address=${HEAD_NODE}:${RAY_PORT} \
            --num-cpus=160 \
            --num-gpus=8 \
            --block &
    " &

    sleep 10
    echo "[Worker] Ray worker started."

    # Verify cluster
    echo ""
    echo "========================================="
    echo "Cluster status:"
    ray status 2>&1 | head -20
    echo "========================================="
    echo ""
    echo "Cluster ready! Run training with:"
    echo "  NNODES=2 bash run_verl.sh circle_packing"

    # Keep head running
    wait $HEAD_PID
}

case "${ACTION}" in
    start)
        start_cluster
        ;;
    stop)
        stop_cluster
        ;;
    *)
        echo "Usage: $0 [start|stop]"
        exit 1
        ;;
esac
