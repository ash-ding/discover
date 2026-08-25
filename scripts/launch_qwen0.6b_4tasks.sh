#!/usr/bin/env bash
# Launch 4 concurrent Qwen3-0.6B experiments on a single 8xH100 node.
#
# Each experiment gets 2 GPUs with TP=1 / DP=2:
#   circle_packing -> GPU 0,1
#   erdos          -> GPU 2,3
#   ac1            -> GPU 4,5
#   ac2            -> GPU 6,7
#
# 50 epochs each (dataset = 8 rows -> 1 epoch = 1 step = 8x64 = 512 rollouts).
# Algorithm hyperparameters left at run_verl.sh reproduction defaults.
#
# Concurrency isolation: per-job Ray temp dir + dashboard disabled + num_cpus=40
# (4 x 40 = 160 physical cores) + per-job Triton cache dir.
set -uo pipefail

cd /workspace/home/asherding/code/discover

export WANDB_MODE=offline
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-0.6B
# verl_discover env does not exist on this machine; lumen has verl 0.9.0.dev +
# vllm 0.23.0+cu129 (FA3/sm_90a) and passes scripts/check_vllm_build.py.
export CONDA_ENV=lumen

LOGDIR="logs/qwen0.6b_4tasks_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"
echo "Log dir: $LOGDIR"

launch () {
    local task="$1" gpus="$2" name="$3"
    CUDA_VISIBLE_DEVICES="$gpus" \
    TRITON_CACHE_DIR="/tmp/triton_${name}" \
    NGPUS_PER_NODE=2 \
    ROLLOUT_TP=1 \
    ROLLOUT_GPU_MEM_UTIL=0.7 \
    TOTAL_EPOCHS=50 \
    SAVE_FREQ=0 \
    nohup bash run_verl.sh "$task" \
        +ray_kwargs.ray_init._temp_dir="/tmp/ray_${name}" \
        +ray_kwargs.ray_init.include_dashboard=False \
        +ray_kwargs.ray_init.num_cpus=40 \
        > "$LOGDIR/${name}.log" 2>&1 &
    echo "launched ${task} on GPU ${gpus} -> ${LOGDIR}/${name}.log (pid $!)"
}

launch circle_packing 0,1 circle_packing
sleep 30
launch erdos          2,3 erdos
sleep 30
launch ac1            4,5 ac1
sleep 30
launch ac2            6,7 ac2

echo "All 4 experiments launched. Tail logs with: tail -f ${LOGDIR}/*.log"
echo "$LOGDIR" > "$LOGDIR/../.last_qwen0.6b_launch_dir"
