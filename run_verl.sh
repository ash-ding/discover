#!/usr/bin/env bash
# TTT-Discover unified VERL launcher for all tasks
# 8xH100 colocate, FSDP, GRPO with entropic adaptive beta, LoRA rank=32
#
# Usage:
#   bash run_verl.sh circle_packing        # Circle Packing 26
#   bash run_verl.sh ac1                   # AC Inequalities AC1
#   bash run_verl.sh ac2                   # AC Inequalities AC2
#   bash run_verl.sh erdos                 # Erdős Min Overlap
#   bash run_verl.sh denoising             # Denoising
#   bash run_verl.sh gpu_mode              # GPU Mode (trimul)
#   bash run_verl.sh ahc039               # AHC 039
#
# Override epochs: TOTAL_EPOCHS=50 bash run_verl.sh circle_packing

set -xeuo pipefail

TASK=${1:?Usage: bash run_verl.sh <task>}
shift  # remaining args passed to python

# Activate correct conda env
CONDA_ENV=${CONDA_ENV:-verl_discover}
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate "$CONDA_ENV"
fi

# CUDA 13 runtime for vLLM
CONDA_PREFIX="${CONDA_PREFIX:-${HOME}/.conda/envs/${CONDA_ENV}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Multi-node communication: force IPv4 on eth0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

########################### Task-specific config ###########################
case "${TASK}" in
    circle_packing|cp|cp26)
        export DISCOVER_ENV_MODULE=examples.circle_packing.env
        export DISCOVER_ENV_CLASS=CirclePackingEnv
        export DISCOVER_PROBLEM_TYPE=26
        export DISCOVER_PHASE1_MAX_TOKENS=26000
        export DISCOVER_EVAL_TIMEOUT=530
        export DISCOVER_NUM_CPUS_PER_TASK=1
        export DISCOVER_DATA_SOURCE=circle_packing_26
        ACTOR_LR=${ACTOR_LR:-4e-5}
        KL_COEF=${KL_COEF:-0.1}
        DATA_FILE=data/circle_packing_train.parquet
        EXPERIMENT_TAG="circle-packing-26"
        ;;
    cp32)
        export DISCOVER_ENV_MODULE=examples.circle_packing.env
        export DISCOVER_ENV_CLASS=CirclePackingEnv
        export DISCOVER_PROBLEM_TYPE=32
        export DISCOVER_PHASE1_MAX_TOKENS=26000
        export DISCOVER_EVAL_TIMEOUT=530
        export DISCOVER_NUM_CPUS_PER_TASK=1
        export DISCOVER_DATA_SOURCE=circle_packing_32
        ACTOR_LR=${ACTOR_LR:-4e-5}
        KL_COEF=${KL_COEF:-0.1}
        DATA_FILE=data/circle_packing_train.parquet
        EXPERIMENT_TAG="circle-packing-32"
        ;;
    ac1|ac_inequalities_ac1)
        export DISCOVER_ENV_MODULE=examples.ac_inequalities.env
        export DISCOVER_ENV_CLASS=AutoCorrInequalityEnv
        export DISCOVER_PROBLEM_TYPE=ac1
        export DISCOVER_PHASE1_MAX_TOKENS=26000
        export DISCOVER_EVAL_TIMEOUT=1100
        export DISCOVER_NUM_CPUS_PER_TASK=2
        export DISCOVER_DATA_SOURCE=ac_inequalities_ac1
        ACTOR_LR=${ACTOR_LR:-4e-5}
        KL_COEF=${KL_COEF:-0.1}
        DATA_FILE=data/ac_inequalities_ac1_train.parquet
        EXPERIMENT_TAG="ac-inequalities-ac1"
        ;;
    ac2|ac_inequalities_ac2)
        export DISCOVER_ENV_MODULE=examples.ac_inequalities.env
        export DISCOVER_ENV_CLASS=AutoCorrInequalityEnv
        export DISCOVER_PROBLEM_TYPE=ac2
        export DISCOVER_PHASE1_MAX_TOKENS=26000
        export DISCOVER_EVAL_TIMEOUT=1100
        export DISCOVER_NUM_CPUS_PER_TASK=2
        export DISCOVER_DATA_SOURCE=ac_inequalities_ac2
        ACTOR_LR=${ACTOR_LR:-4e-5}
        KL_COEF=${KL_COEF:-0.1}
        DATA_FILE=data/ac_inequalities_ac2_train.parquet
        EXPERIMENT_TAG="ac-inequalities-ac2"
        ;;
    erdos|erdos_min_overlap)
        export DISCOVER_ENV_MODULE=examples.erdos_min_overlap.env
        export DISCOVER_ENV_CLASS=ErdosMinOverlapEnv
        export DISCOVER_PROBLEM_TYPE=""
        export DISCOVER_PHASE1_MAX_TOKENS=26000
        export DISCOVER_EVAL_TIMEOUT=1100
        export DISCOVER_NUM_CPUS_PER_TASK=1
        export DISCOVER_DATA_SOURCE=erdos_min_overlap
        ACTOR_LR=${ACTOR_LR:-4e-5}
        KL_COEF=${KL_COEF:-0.1}
        DATA_FILE=data/erdos_min_overlap_train.parquet
        EXPERIMENT_TAG="erdos-min-overlap"
        ;;
    denoising)
        export DISCOVER_ENV_MODULE=examples.denoising.env
        export DISCOVER_ENV_CLASS=DenoisingEnv
        export DISCOVER_PROBLEM_TYPE=""
        export DISCOVER_PHASE1_MAX_TOKENS=26000
        export DISCOVER_EVAL_TIMEOUT=530
        export DISCOVER_NUM_CPUS_PER_TASK=1
        export DISCOVER_DATA_SOURCE=denoising
        ACTOR_LR=${ACTOR_LR:-4e-5}
        KL_COEF=${KL_COEF:-0.1}
        DATA_FILE=data/denoising_train.parquet
        EXPERIMENT_TAG="denoising"
        ;;
    gpu_mode|trimul)
        export DISCOVER_ENV_MODULE=examples.gpu_mode.env
        export DISCOVER_ENV_CLASS=GpuModeEnv
        export DISCOVER_PROBLEM_TYPE=trimul
        export DISCOVER_PHASE1_MAX_TOKENS=26000
        export DISCOVER_EVAL_TIMEOUT=530
        export DISCOVER_NUM_CPUS_PER_TASK=1
        export DISCOVER_DATA_SOURCE=gpu_mode_trimul
        export GPU_EVAL_SERVER=${GPU_EVAL_SERVER:-}
        export KERNEL_EVAL_GPU=${KERNEL_EVAL_GPU:-0}
        export NUM_EVAL_GPUS=${NUM_EVAL_GPUS:-2}
        export EVAL_GPU_IDS=${EVAL_GPU_IDS:-}
        ACTOR_LR=${ACTOR_LR:-4e-5}
        KL_COEF=${KL_COEF:-0.01}
        DATA_FILE=data/gpu_mode_trimul_train.parquet
        EXPERIMENT_TAG="gpu-mode-trimul"
        ;;
    ahc039|ahc)
        export DISCOVER_ENV_MODULE=examples.ahc.env
        export DISCOVER_ENV_CLASS=AhcEnv
        export DISCOVER_PROBLEM_TYPE=ahc039
        export DISCOVER_PHASE1_MAX_TOKENS=22000
        export DISCOVER_EVAL_TIMEOUT=600
        export DISCOVER_NUM_CPUS_PER_TASK=2
        export DISCOVER_DATA_SOURCE=ahc_039
        ACTOR_LR=${ACTOR_LR:-2e-5}
        KL_COEF=${KL_COEF:-0.01}
        DATA_FILE=data/ahc_039_train.parquet
        EXPERIMENT_TAG="ahc-039"
        ;;
    *)
        echo "Unknown task: ${TASK}"
        echo "Available: circle_packing cp26 cp32 ac1 ac2 erdos denoising gpu_mode ahc039"
        exit 1
        ;;
esac

########################### Shared config ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
ROLLOUT_N=${ROLLOUT_N:-64}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
# Set to total samples (train_batch_size * rollout_n) for single optimizer step
# matching original TTT-Discover behavior (1 update per training iteration)
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-28672}
LORA_RANK=${LORA_RANK:-32}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
SP_SIZE=${SP_SIZE:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-${EXPERIMENT_TAG}_verl_$(date +%Y%m%d_%H%M)}

# Resume from checkpoint:
#   RESUME_DIR=checkpoints/ttt-discover/old_run INPLACE=true  bash run_verl.sh cp  # same dir
#   RESUME_DIR=checkpoints/ttt-discover/old_run INPLACE=false bash run_verl.sh cp  # new dir
#   RESUME_DIR=checkpoints/ttt-discover/old_run                bash run_verl.sh cp  # new dir (default)
RESUME_DIR=${RESUME_DIR:-}
INPLACE=${INPLACE:-false}

if [ -n "$RESUME_DIR" ]; then
    RESUME_MODE="resume_path"
    RESUME_PATH="$RESUME_DIR/latest"

    if [ "$INPLACE" = "true" ]; then
        # Use the resume dir's name as experiment name → results stay in same dir
        EXPERIMENT_NAME=$(basename "$RESUME_DIR")
    else
        # Copy PUCT state from source to new experiment dir (preserve step-indexed filename)
        SOURCE_PUCT=$(find "$RESUME_DIR" -name "puct_sampler_step_*.json" | sort | tail -1)
        if [ -n "$SOURCE_PUCT" ]; then
            DEST_DIR="checkpoints/ttt-discover/${EXPERIMENT_NAME}"
            mkdir -p "$DEST_DIR"
            cp "$SOURCE_PUCT" "$DEST_DIR/$(basename "$SOURCE_PUCT")"
            echo "Copied PUCT state from $SOURCE_PUCT to $DEST_DIR/$(basename "$SOURCE_PUCT")"
        fi
    fi
else
    RESUME_MODE="auto"
    RESUME_PATH=""
fi

export DISCOVER_MAX_MODEL_LEN=${DISCOVER_MAX_MODEL_LEN:-32768}
export DISCOVER_LOG_DIR=${DISCOVER_LOG_DIR:-./tinker_log}
export DISCOVER_PUCT_FILE_PATH=${DISCOVER_PUCT_FILE_PATH:-./checkpoints/ttt-discover/${EXPERIMENT_NAME}/puct_sampler.json}
export DISCOVER_PUCT_C=${DISCOVER_PUCT_C:-1.0}
export DISCOVER_TOPK_CHILDREN=${DISCOVER_TOPK_CHILDREN:-2}
export DISCOVER_MAX_BUFFER_SIZE=${DISCOVER_MAX_BUFFER_SIZE:-1000}

########################### Launch ###########################
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=entropic_adaptive_beta \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="${PWD}/${DATA_FILE}" \
    data.val_files="${PWD}/${DATA_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_RANK} \
    actor_rollout_ref.model.target_modules=all-linear \
    "++actor_rollout_ref.model.lora.merge=True" \
    \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.95]" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.clip_ratio=1000.0 \
    actor_rollout_ref.actor.clip_ratio_low=1000.0 \
    actor_rollout_ref.actor.clip_ratio_high=1000.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE} \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.enforce_eager=True \
    "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=ttt_discover.verl_integration.agent_loop.DiscoverAgentLoopManagerTQ" \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE} \
    \
    reward.custom_reward_function.path=${PWD}/ttt_discover/verl_integration/verl_reward.py \
    reward.custom_reward_function.name=compute_score \
    \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=ttt-discover \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ:-0} \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.rollout_data_dir=checkpoints/ttt-discover/${EXPERIMENT_NAME}/rollouts \
    trainer.val_before_train=False \
    trainer.resume_mode=${RESUME_MODE} \
    ${RESUME_PATH:+trainer.resume_from_path=${RESUME_PATH}} \
    ${NNODES:+$([ "$NNODES" -gt 1 ] && echo "+ray_kwargs.ray_init.address=auto" || true)} \
    "$@"
