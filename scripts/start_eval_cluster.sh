#!/usr/bin/env bash
# Start a Ray cluster for GPU Mode training with remote evaluation.
#
# Node 1 (worker node) = head, runs VERL training on 8 GPUs
# Node 0 (head node) = eval worker, runs GPU kernel benchmarks on 1-2 GPUs
#
# Usage:
#   bash scripts/start_eval_cluster.sh          # Start cluster
#   bash scripts/start_eval_cluster.sh stop     # Stop cluster
#   bash scripts/start_eval_cluster.sh status   # Check status
#
# After starting, run training with:
#   GPU_EVAL_REMOTE=true bash run_verl.sh gpu_mode

set -euo pipefail

# Node 1 = training head, Node 0 = eval worker
HEAD_NODE=${HEAD_NODE:-10.241.128.16}
EVAL_NODE=${EVAL_NODE:-10.241.128.30}
RAY_PORT=${RAY_PORT:-6379}
CONDA_ENV=${CONDA_ENV:-verl_discover}
CONDA_SH="/workspace/home/asherding/.conda/etc/profile.d/conda.sh"
CUDA13_LIB="/workspace/home/asherding/.conda/envs/${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cu13/lib"
NUM_EVAL_GPUS=${NUM_EVAL_GPUS:-2}
EVAL_GPU_IDS=${EVAL_GPU_IDS:-}

ACTION=${1:-start}

stop_cluster() {
    echo "Stopping eval cluster..."
    source ${CONDA_SH} && conda activate ${CONDA_ENV}

    # Stop on training node (head)
    ray stop --force 2>/dev/null || true

    # Stop on eval node
    ssh asherding@${EVAL_NODE} "
        source ${CONDA_SH} && conda activate ${CONDA_ENV} && \
        ray stop --force
    " 2>/dev/null || true

    echo "Eval cluster stopped."
}

start_cluster() {
    stop_cluster
    sleep 3

    # Compute GPU ID list for display
    if [ -n "${EVAL_GPU_IDS}" ]; then
        DISPLAY_GPU_IDS="${EVAL_GPU_IDS}"
    else
        DISPLAY_GPU_IDS=$(seq -s, 0 $((NUM_EVAL_GPUS - 1)))
    fi

    echo "========================================="
    echo "Starting eval cluster"
    echo "  Head (training): ${HEAD_NODE}"
    echo "  Eval worker:     ${EVAL_NODE}"
    echo "  Eval GPUs:       ${NUM_EVAL_GPUS} (IDs: ${DISPLAY_GPU_IDS})"
    echo "========================================="

    # Start head on training node (Node 1)
    echo "[Head] Starting Ray head on ${HEAD_NODE}..."
    source ${CONDA_SH} && conda activate ${CONDA_ENV}
    export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}"

    ray start --head \
        --port=${RAY_PORT} \
        --num-cpus=160 \
        --num-gpus=8

    echo "[Head] Ray head started."
    sleep 5

    # Start eval worker on Node 0
    # num-gpus=0: prevent VERL from scheduling training workers here
    # eval_gpu=N: custom resource for GPU kernel evaluation
    echo "[Eval] Starting eval worker on ${EVAL_NODE}..."
    ssh asherding@${EVAL_NODE} "
        source ${CONDA_SH} && conda activate ${CONDA_ENV} && \
        export LD_LIBRARY_PATH='${CUDA13_LIB}:\${LD_LIBRARY_PATH:-}' && \
        ray start --address=${HEAD_NODE}:${RAY_PORT} \
            --num-cpus=0 \
            --num-gpus=0 \
            --resources='{\"eval_gpu\": ${NUM_EVAL_GPUS}}'
    "
    echo "[Eval] Eval worker started."
    sleep 5

    show_status
    echo ""
    echo "Cluster ready! For GPU Mode with HTTP eval, start eval_server.py on the eval node:"
    echo "  PYTHONPATH=\${PWD} python examples/gpu_mode/eval_server.py --port 8890 --num-gpus ${NUM_EVAL_GPUS}"
    echo "Then run training with:"
    echo "  GPU_EVAL_SERVER=${EVAL_NODE}:8890 bash run_verl.sh gpu_mode"
}

show_status() {
    echo "========================================="
    echo "Eval cluster status:"
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
