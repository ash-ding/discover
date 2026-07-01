#!/usr/bin/env bash
# Start a Ray cluster across 2 nodes for multi-node VERL training.
#
# Usage:
#   bash scripts/start_ray_cluster.sh          # Start cluster
#   bash scripts/start_ray_cluster.sh stop     # Stop cluster
#   bash scripts/start_ray_cluster.sh status   # Check cluster status
#
# After starting, run training with:
#   NNODES=2 bash run_verl.sh circle_packing

set -euo pipefail

HEAD_NODE=${HEAD_NODE:-10.241.128.30}
WORKER_NODE=${WORKER_NODE:-10.241.128.16}
RAY_PORT=${RAY_PORT:-6379}
CONDA_ENV=${CONDA_ENV:-verl_discover}
CONDA_SH="/workspace/home/asherding/.conda/etc/profile.d/conda.sh"
CUDA13_LIB="/workspace/home/asherding/.conda/envs/${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cu13/lib"

ACTION=${1:-start}

stop_cluster() {
    echo "Stopping Ray cluster..."
    source ${CONDA_SH} && conda activate ${CONDA_ENV}
    ray stop --force 2>/dev/null || true
    ssh asherding@${WORKER_NODE} "
        source ${CONDA_SH} && conda activate ${CONDA_ENV} && \
        ray stop --force
    " 2>/dev/null || true
    echo "Ray cluster stopped."
}

start_cluster() {
    stop_cluster
    sleep 3

    echo "========================================="
    echo "Starting Ray cluster"
    echo "  Head:   ${HEAD_NODE}"
    echo "  Worker: ${WORKER_NODE}"
    echo "========================================="

    # Start head node (no --block, runs as daemon)
    echo "[Head] Starting Ray head..."
    source ${CONDA_SH} && conda activate ${CONDA_ENV}
    export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}"

    ray start --head \
        --port=${RAY_PORT} \
        --num-cpus=160 \
        --num-gpus=8

    echo "[Head] Ray head started."
    sleep 5

    # Start worker node (nohup to survive SSH disconnect)
    echo "[Worker] Starting Ray worker on ${WORKER_NODE}..."
    ssh asherding@${WORKER_NODE} "
        source ${CONDA_SH} && conda activate ${CONDA_ENV} && \
        export LD_LIBRARY_PATH='${CUDA13_LIB}:\${LD_LIBRARY_PATH:-}' && \
        ray start --address=${HEAD_NODE}:${RAY_PORT} \
            --num-cpus=160 \
            --num-gpus=8
    "
    echo "[Worker] Ray worker started."
    sleep 5

    # Verify
    show_status
    echo ""
    echo "Cluster ready! Run training with:"
    echo "  NNODES=2 bash run_verl.sh circle_packing"
}

show_status() {
    echo "========================================="
    echo "Ray cluster status:"
    source ${CONDA_SH} && conda activate ${CONDA_ENV}
    ray status 2>&1 | head -25
    echo "========================================="
}

case "${ACTION}" in
    start)  start_cluster ;;
    stop)   stop_cluster ;;
    status) show_status ;;
    *)
        echo "Usage: $0 [start|stop|status]"
        exit 1
        ;;
esac
