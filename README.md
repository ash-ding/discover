# TTT-Discover: Local Reproduction with VERL

Local reproduction of [TTT-Discover](https://arxiv.org/abs/2601.16175) using **VERL colocate mode** for RL training. All 8 GPUs are shared between inference (vLLM) and training (FSDP) via sleep/wake memory management.

## Quick Start

### Prerequisites

- **Hardware**: 8x NVIDIA H100 80GB
- **CUDA Driver**: 12.9+
- **Software**: Conda, Python 3.11

### Setup

```bash
# 1. Create unified conda environment
conda create -n verl_discover python=3.11 -y
conda activate verl_discover

# 2. Install base dependencies
pip install -r requirements/requirements-base.txt

# 3. Install FlashInfer
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/

# 4. Install Flash Attention (limit jobs to avoid CPU overload)
MAX_JOBS=8 pip install flash-attn --no-build-isolation --no-cache-dir

# 5. Install VERL (local fork with custom extensions)
pip install -e verl

# 6. Task-specific dependencies (install whichever you need)
pip install -r requirements/requirements-gpumode.txt                  # GPU Mode
pip install -r requirements/requirements-ahc.txt                      # AHC
pip install -r requirements/denoising/requirements-denoising.txt      # Denoising
# Math tasks (Circle Packing, AC, Erdős) need no extra deps
```

### Run

```bash
conda activate verl_discover
TOTAL_EPOCHS=50 bash run_verl.sh circle_packing   # Full 50-epoch training
TOTAL_EPOCHS=1  bash run_verl.sh circle_packing   # 1-epoch validation
```

No manual vLLM server management needed — VERL handles everything internally.

## Available Tasks

| Task | Command | LR | KL Coef | Notes |
|------|---------|-----|---------|-------|
| Circle Packing (26) | `bash run_verl.sh circle_packing` | 4e-5 | 0.1 | |
| Circle Packing (32) | `bash run_verl.sh cp32` | 4e-5 | 0.1 | |
| AC Inequalities (AC1) | `bash run_verl.sh ac1` | 4e-5 | 0.1 | Minimize |
| AC Inequalities (AC2) | `bash run_verl.sh ac2` | 4e-5 | 0.1 | Maximize |
| Erdős Min Overlap | `bash run_verl.sh erdos` | 4e-5 | 0.1 | |
| Denoising | `bash run_verl.sh denoising` | 4e-5 | 0.1 | Requires openproblems |
| GPU Mode (trimul) | `bash run_verl.sh gpu_mode` | 4e-5 | 0.01 | |
| AHC (ahc039) | `bash run_verl.sh ahc039` | 2e-5 | 0.01 | SP=2 recommended |

Override defaults via environment variables:

```bash
TOTAL_EPOCHS=50 ROLLOUT_N=64 SP_SIZE=2 bash run_verl.sh ahc039
```

## Architecture

```
8x H100 Colocate Mode:
  Inference: vLLM TP=4 × 2 replicas (all 8 GPUs)
  Training:  FSDP DP=8 (all 8 GPUs)
  Phases alternate via sleep/wake — no idle GPUs
```

Key components:
- **VERL** (`verl/`): Training framework with colocate infrastructure
- **Custom AgentLoop** (`ttt_discover/verl_integration/agent_loop.py`): PUCT state reuse + two-phase completion
- **Entropic Adaptive Beta** (`verl/verl/trainer/ppo/adv_estimators/`): Custom advantage estimator

## Paper Configuration (Table 9)

| Parameter | Default | AHC | GPU Mode |
|-----------|---------|-----|----------|
| model | Qwen/Qwen3-8B | same | same |
| group_size | 64 | 64 | 64 |
| groups_per_batch | 8 | 8 | 8 |
| num_epochs | 50 | 50 | 50 |
| learning_rate | 4e-5 | 2e-5 | 4e-5 |
| kl_penalty_coef | 0.1 | 0.01 | 0.01 |
| phase1_max_tokens | 26000 | 22000 | 26000 |
| lora_rank | 32 | 32 | 32 |
| temperature | 1.0 | 1.0 | 1.0 |

## Output Structure

```
checkpoints/ttt-discover/<experiment_name>/
├── latest/                              # Every step (for crash recovery)
│   ├── actor/                           # LoRA weights + optimizer state
│   └── data.pt                          # Dataloader state
├── global_step_N/                       # Every N steps (for analysis)
│   └── actor/                           # LoRA weights only
├── puct_sampler.json                    # PUCT state (every step)
└── latest_checkpointed_iteration.txt    # Current step number
```

## Key Differences from Original Codebase

| Aspect | Original | This Version |
|--------|----------|-------------|
| Inference | Manual vLLM server (TP=4, GPUs 0-3) | VERL-managed vLLM (TP=4, 2 replicas, all 8 GPUs) |
| Training | Custom PEFT LoRA (4-GPU DP) | VERL FSDP (8-GPU DP) |
| GPU allocation | Fixed 4+4 split | Colocate 8 GPUs with sleep/wake |
| vLLM management | `start_vllm.sh` / `stop_vllm.sh` | Automatic (VERL internal) |
| Environment | 4 separate conda envs | Single `verl_discover` env |
