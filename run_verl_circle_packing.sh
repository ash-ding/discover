#!/usr/bin/env bash
# TTT-Discover Circle Packing via VERL native trainer
# 8xH100 colocate, FSDP, GRPO with entropic adaptive beta, LoRA rank=32
#
# Usage:
#   bash run_verl_circle_packing.sh           # 1 epoch validation
#   TOTAL_EPOCHS=50 bash run_verl_circle_packing.sh  # full run

set -xeuo pipefail

# Activate correct conda env
CONDA_ENV=${CONDA_ENV:-verl_discover_math}
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    echo "Activating conda env: $CONDA_ENV"
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate "$CONDA_ENV"
fi

# CUDA 13 runtime required by vLLM 0.23
CONDA_PREFIX="${CONDA_PREFIX:-${HOME}/.conda/envs/${CONDA_ENV}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

# RL hyperparameters (from paper Table 9)
ROLLOUT_N=${ROLLOUT_N:-64}                    # group_size
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}         # 8 prompts × n=64 completions = 512 samples
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-28672}  # 32768 - 4096
ACTOR_LR=${ACTOR_LR:-4e-5}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.1}
LORA_RANK=${LORA_RANK:-32}

# VERL infrastructure
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
SP_SIZE=${SP_SIZE:-4}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-circle_packing_26_verl_$(date +%Y%m%d_%H%M)}

# TTT-Discover task config (passed to custom AgentLoop via env vars)
export DISCOVER_ENV_MODULE=${DISCOVER_ENV_MODULE:-examples.circle_packing.env}
export DISCOVER_ENV_CLASS=${DISCOVER_ENV_CLASS:-CirclePackingEnv}
export DISCOVER_PROBLEM_TYPE=${DISCOVER_PROBLEM_TYPE:-26}
export DISCOVER_PHASE1_MAX_TOKENS=${DISCOVER_PHASE1_MAX_TOKENS:-26000}
export DISCOVER_MAX_MODEL_LEN=${DISCOVER_MAX_MODEL_LEN:-32768}
export DISCOVER_EVAL_TIMEOUT=${DISCOVER_EVAL_TIMEOUT:-530}
export DISCOVER_NUM_CPUS_PER_TASK=${DISCOVER_NUM_CPUS_PER_TASK:-1}
export DISCOVER_LOG_DIR=${DISCOVER_LOG_DIR:-./tinker_log}
export DISCOVER_PUCT_FILE_PATH=${DISCOVER_PUCT_FILE_PATH:-./tinker_log/puct_sampler.json}
export DISCOVER_PUCT_C=${DISCOVER_PUCT_C:-1.0}
export DISCOVER_TOPK_CHILDREN=${DISCOVER_TOPK_CHILDREN:-2}
export DISCOVER_MAX_BUFFER_SIZE=${DISCOVER_MAX_BUFFER_SIZE:-1000}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=entropic_adaptive_beta \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="$PWD/data/circle_packing_train.parquet" \
    data.val_files="$PWD/data/circle_packing_train.parquet" \
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
    \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.95]" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE} \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=ttt_discover.verl_integration.agent_loop.DiscoverAgentLoopManagerTQ" \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE} \
    \
    reward.custom_reward_function.path=ttt_discover/verl_integration/verl_reward.py \
    reward.custom_reward_function.name=compute_score \
    \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=ttt-discover \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"
