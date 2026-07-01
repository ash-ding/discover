# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TTT-Discover is a research implementation of test-time training for LLMs using reinforcement learning. This is a **local reproduction fork** using **VERL colocate mode** for RL training: all 8 GPUs are shared between inference (vLLM) and training (FSDP) via sleep/wake memory management.

**Key Architecture Decision**: We use **VERL's colocated infrastructure** where vLLM inference and FSDP training alternate on the same GPUs. No manual vLLM server management needed.

## Quick Start (New Machine)

```bash
git clone --recursive https://github.com/ash-ding/discover.git
cd discover
conda create -n verl_discover python=3.11 -y
conda activate verl_discover
pip install -r requirements/requirements-base.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/
MAX_JOBS=8 pip install flash-attn --no-build-isolation --no-cache-dir
pip install -e verl
```

If you already cloned without `--recursive`, initialize the submodule with:
```bash
git submodule update --init --recursive
```

## Critical Setup Requirements

### Hardware Requirements
- **Required**: 8x NVIDIA H100 80GB (colocate mode — all GPUs shared)
- CUDA Driver: 12.9+

### Environment Setup

All tasks use a single unified conda environment `verl_discover`:

```bash
conda create -n verl_discover python=3.11 -y
conda activate verl_discover
pip install -r requirements/requirements-base.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/
MAX_JOBS=8 pip install flash-attn --no-build-isolation --no-cache-dir
pip install -e verl
```

See `requirements/README.md` for task-specific extra dependencies.

### vLLM Server (Managed by VERL)

vLLM is managed automatically by VERL's colocate infrastructure. **No manual server management needed.** The `run_verl.sh` script handles everything.

## Configuration System

All tasks use YAML configuration files instead of hardcoded parameters.

### Structure
Each task has:
- **config_paper.yaml** — Full 50-epoch run (paper parameters from Table 9)
- **config_validate.yaml** — 1-epoch validation run (same parameters, epochs=1)
- **run.sh** — Launcher script that loads config and runs experiment

### Running Tasks
```bash
cd examples/<task>
bash run.sh full      # Uses config_paper.yaml
bash run.sh validate  # Uses config_validate.yaml
```

### Custom Configs
```bash
export TTT_CONFIG_PATH=/path/to/custom_config.yaml
cd examples/<task>
bash run.sh full
```

The config loader (`ttt_discover/utils/config_loader.py`) parses YAML and passes parameters to DiscoverConfig.

### YAML Configuration Files

Each task has two standard configs:
- **`config_paper.yaml`** - Full 50-epoch training (Table 9 parameters)
- **`config_validate.yaml`** - Quick 1-epoch validation (same params except num_epochs)

### Loading Priority

When a task starts, configuration is loaded in this order:
1. **YAML file** (via `TTT_CONFIG_PATH` environment variable) - highest priority
2. **Hard-coded defaults** in task's `env.py` file
3. **DiscoverConfig defaults** in `discovery.py`

If `TTT_CONFIG_PATH` is set, the YAML config overrides all defaults.

### Environment Variables

Some parameters are controlled externally via environment variables, **not** YAML:

| Variable | Purpose | Default | Set By |
|----------|---------|---------|--------|
| `VLLM_BASE_URL` | vLLM server address | `http://localhost:8888` | `run.sh` or manual |
| `WANDB_MODE` | WandB logging mode | `offline` | `run.sh` or `~/.bashrc` |
| `CUDA_VISIBLE_DEVICES` | GPU visibility | `0,1,2,3,4` | `run.sh` |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | Enable LoRA hot-reload | `true` | `start_vllm.sh` (required) |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA allocator config | `expandable_segments:True` | `run.sh` |

**Critical**: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` must be set **before** starting vLLM. Without it, training will fail with "LoRA adapter not loaded" errors. The `start_vllm.sh` script sets this automatically.

### Configuration Validation

Before starting experiments, validate your config:
```bash
# Quick syntax check
python3 -c "import yaml; yaml.safe_load(open('config_paper.yaml'))"

# Full validation (GPU config, parameter ranges, etc.)
# See .claude/skills/config-validation.md for detailed checks
```

## Mandatory Rules

1. **Always activate `verl_discover` conda environment before running any task.** All tasks share this single environment.

2. **Never use conda `base` environment** to run or debug code.

## Standard Workflow

```bash
conda activate verl_discover

# Run any task (VERL manages vLLM internally, no manual server needed)
TOTAL_EPOCHS=50 bash run_verl.sh circle_packing   # Full training
TOTAL_EPOCHS=1  bash run_verl.sh circle_packing   # Validation (1 epoch)

# Available tasks: circle_packing, cp32, ac1, ac2, erdos, denoising, gpu_mode, ahc039
```

For AHC (long prompts), use SP=2 to avoid OOM:
```bash
TOTAL_EPOCHS=50 SP_SIZE=2 bash run_verl.sh ahc039
```

### Available Tasks

| Task | Environment | Requirements File | Notes |
|------|-------------|-------------------|-------|
| Circle Packing | `discover_math` | `requirements-math.txt` | 26 or 32 circles |
| AC Inequalities | `discover_math` | `requirements-math.txt` | AC1 (minimize) or AC2 (maximize) |
| Erdős Min Overlap | `discover_math` | `requirements-math.txt` | C₅ constant optimization |
| Denoising | `discover_denoising` | `denoising/requirements-denoising.txt` | Requires openproblems patch |
| GPU Mode | `discover_gpumode` | `requirements-gpumode.txt` | Local evaluation only |
| AHC | `discover_ale` | `requirements-ahc.txt` | Must run in container |

Each task has detailed documentation in its respective `examples/<task>/README.md`.


## Code Architecture

### Core Structure

```
ttt_discover/
├── verl_integration/        # VERL training backend
│   ├── agent_loop.py        # Custom AgentLoop: PUCT + two-phase completion
│   ├── verl_reward.py       # Reward function wrapper for sandbox evaluator
│   ├── discover_trainer.py  # Custom trainer (legacy, for mock testing)
│   ├── puct_data_source.py  # Dynamic PUCT-driven data source
│   └── config/              # Task YAML configs
├── compat/
│   └── tinker_types.py      # Type compatibility shim for removed local_backend
├── rl/
│   ├── train.py             # Original RL training loop (reference, not used with VERL)
│   ├── rollouts.py          # PUCT sampling, trajectory generation
│   └── data_processing.py   # Advantage computation (entropic adaptive beta)
├── tinker_utils/
│   ├── completers.py        # Qwen3TwoPhaseTokenCompleter (reference implementation)
│   ├── sampler.py           # PUCT state reuse
│   └── renderers.py         # Qwen3Renderer (prompt formatting)
└── environments/
    └── sandbox_reward_evaluator.py  # Ray-based sandboxed code execution

verl/                        # VERL fork with custom extensions
├── verl/trainer/ppo/adv_estimators/
│   └── entropic_adaptive_beta.py  # Custom advantage estimator
└── verl/workers/utils/losses.py   # Fine-grained mask support
```

### verl_integration/ Design

This directory contains the VERL integration layer (~600 lines):

- **`agent_loop.py`**: Custom VERL AgentLoopManager with PUCT state reuse and two-phase token completion. Replaces VERL's default rollout with TTT-Discover's algorithm.
- **`verl_reward.py`**: Wraps `SandboxRewardEvaluator` in VERL's `compute_score()` interface
- **`puct_data_source.py`**: Dynamic prompt generation from PUCT sampler state
- **`discover_trainer.py`**: Legacy custom trainer for mock/CPU testing

VERL extensions (in `verl/` fork):
- **`entropic_adaptive_beta.py`**: Custom advantage estimator with KL centered adjustment
- **`losses.py`**: Fine-grained mask support for two-phase completion prefill tokens
- **`bucketed_weight_transfer.py`**: IPC handle fix for PyTorch 2.11+

## GPU Allocation Strategy

**VERL Colocate (8 GPUs shared)**: All GPUs alternate between inference and training via sleep/wake.

```
Inference phase: vLLM TP=4 × 2 replicas (all 8 GPUs, gpu_memory_utilization=0.85)
Training phase:  FSDP DP=8 (all 8 GPUs, vLLM sleeps to release memory)
```

For long-sequence tasks (AHC), use sequence parallelism to reduce activation memory:
```bash
SP_SIZE=2 bash run_verl.sh ahc039   # SP=2, DP=4
```

## Common Issues and Fixes

### vLLM OOM during KL penalty computation

**Symptom**: `torch.cuda.OutOfMemoryError` during `incorporate_kl_penalty` step

**Root Cause**: KL penalty uses `echo=True` to compute logprobs for entire sequences (prompt + generation). This creates logits tensors of `[seq_len, vocab_size]`. For 32K sequences: `32768 × 152064 × 2 bytes ≈ 9.3 GiB` per request. Multiple concurrent requests exhaust GPU memory.

**Solutions** (in order of preference):
1. Lower `gpu_memory_utilization` in vLLM startup (e.g., `0.70` or `0.80` instead of `0.90`)
2. Add `--max-num-seqs 2` to vLLM startup to limit concurrent sequences
3. Reduce `max_model_len` in both vLLM and `DiscoverConfig`

### Training OOM

**Symptom**: `torch.cuda.OutOfMemoryError` during training step (after rollouts)

**Fix**: Training already uses gradient checkpointing and flash-attn. If still OOM, reduce sequence length:
```python
# In training_client.py
max_train_seq_len = 8192  # from 32768
```

Note: This truncates prompts but keeps full responses, reducing training quality.

### LoRA adapter not loaded

**Symptom**: vLLM returns `400 Bad Request: LoRA adapter 'lora_v1' not loaded`

**Fix**: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` was not set when starting vLLM. Restart vLLM with the environment variable.

### Ray CUDA_VISIBLE_DEVICES override

**Symptom**: vLLM or training uses wrong GPUs despite `CUDA_VISIBLE_DEVICES`

**Fix**: Ray overrides GPU visibility by default. This is already fixed in `discovery.py` with:
```python
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
```

### AHC Container Issues

For AHC tasks, the job must run inside `yimjk/ale-bench:cpp20-202301` container. Common issues:

- **PID limit**: Add `--pids-limit=-1` to podman run
- **Shared memory**: Add `--shm-size=16g` to podman run
- **GPU access**: Use `--device nvidia.com/gpu=all`

## Key Configuration Parameters

All tasks now use YAML configuration files (`config_paper.yaml`, `config_validate.yaml`). Below are the key parameters and their meanings:

| Parameter | Default | AHC | GPU Mode | Notes |
|-----------|---------|-----|----------|-------|
| `model_name` | `"Qwen/Qwen3-8B"` | same | same | Must match vLLM model |
| `use_local_backend` | `True` | `True` | `True` | Enables local_backend/ adapter |
| `inference_tp_size` | `4` | `4` | `4` | Must match vLLM `--tensor-parallel-size` |
| `max_model_len` | `32768` | `32768` | `32768` | Must match vLLM `--max-model-len` |
| `group_size` | `64` | `64` | `64` | Completions per prompt |
| `groups_per_batch` | `8` | `8` | `8` | Different prompts per step |
| `num_epochs` | `50` | `50` | `50` | Training steps (1 for validate) |
| `phase1_max_tokens` | `26000` | **22000** | `26000` | Prompt + thinking budget |
| `kl_penalty_coef` | `0.1` | **0.01** | **0.01** | KL penalty coefficient |
| `lora_rank` | `32` | `32` | `32` | LoRA rank |
| `learning_rate` | `4e-5` | **2e-5** | `4e-5` | Adam learning rate |
| `training_gpu_ids` | `[4,5,6,7]` | same | same | Multi-GPU parallel training (4-way DP) |
| `training_batch_size` | `1` | `1` | `1` | Per-GPU micro batch size (keep 1 for 32K) |

**Task-specific parameter overrides** (bold = differs from default):
- **AHC**: Lower all three key params (phase1_max_tokens, kl_penalty_coef, learning_rate)
- **GPU Mode**: Lower kl_penalty_coef only
- **All others**: Use default values from Table 9

**Config tiers** (all use same hyperparameters, only scale differs):

| Config | group_size | groups_per_batch | samples/step | epochs | Purpose |
|--------|-----------|-----------------|-------------|--------|---------|
| `config_paper.yaml` | 64 | 8 | 512 | 50 | Full paper reproduction |
| `config_validate.yaml` | 64 | 8 | 512 | 1 | Pre-training verification |
| `config_smoke_test.yaml` | 4 | 2 | 8 | 1 | Quick code validation |

## Testing

No automated test suite. Validation is done via:
1. Quick validation runs (1 epoch, small group_size)
2. Monitoring WandB logs for `env/all/raw_score/max` (maximization tasks) or `env/all/raw_score/min` (minimization tasks)
3. Checking checkpoint saves to `tinker_log/local_checkpoints/<experiment_name>/`

## Performance Notes

- **Denoising, Math tasks**: CPU-intensive evaluation. Needs HPC-grade CPUs or many cores.
- **vLLM TP scaling**: TP=4 is ~2x faster than TP=1 for group_size=64. Minimal difference for small group_size.
- **Estimated runtime** (Circle Packing, 50 epochs, TP=4): 8-12 hours on 5×H100 80GB

## Output Structure

```
checkpoints/ttt-discover/<experiment_name>/
├── latest/                          # Every step (crash recovery)
│   ├── actor/                       # LoRA weights + optimizer state
│   └── data.pt                      # Dataloader state
├── global_step_N/                   # Every N steps (analysis)
│   └── actor/                       # LoRA weights only
├── puct_sampler.json                # PUCT state (every step)
└── latest_checkpointed_iteration.txt # Current step number
```

## Creating Custom Environments

See `examples/circle_packing/env.py` for a complete example. Key steps:

1. Subclass `Environment` and implement `get_question()` to build the prompt
2. Subclass `BaseRewardEvaluator` (or use `SandboxRewardEvaluator` for code execution)
3. Define `DiscoverConfig` with your environment type and call `discover(config)`

The RL algorithm (GRPO, entropic adaptive beta, PUCT state reuse) is unchanged from the original codebase.

## Key Differences from Original Codebase

| Aspect | Original | This Fork |
|--------|----------|-----------|
| Model | gpt-oss-120b | Qwen3-8B |
| Training framework | Tinker remote service | VERL colocate (FSDP + vLLM) |
| GPU allocation | Tinker auto-managed | 8-GPU colocate with sleep/wake |
| Inference | Tinker auto-managed | VERL-managed vLLM (TP=4, 2 replicas) |
| Training | Tinker remote | FSDP DP=8 with LoRA |
| Environment | Multiple conda envs | Single `verl_discover` env |
| Phase 2 marker | Channel tokens | `</think>` tag |
| Max train seq len | Unlimited | 32768 (32K) |

## Important Files

- `README.md` — Local reproduction guide (comprehensive)
- `reproduce.md` — Detailed Chinese reproduction notes, troubleshooting, lessons learned
- `docs/reproducing.md` — Original paper reproduction guide (Tinker-based)
- `docs/api.md` — Public API reference
- `examples/<task>/env.py` — Task-specific entry points
- `requirements/requirements-<task>-local.txt` — Combined task + RL backend dependencies

## Development Workflow

When modifying the VERL integration:

1. Changes to rollout/generation → edit `ttt_discover/verl_integration/agent_loop.py`
2. Changes to advantage computation → edit `verl/verl/trainer/ppo/adv_estimators/entropic_adaptive_beta.py`
3. Changes to reward evaluation → edit `ttt_discover/verl_integration/verl_reward.py`
4. Changes to PUCT logic → edit `ttt_discover/tinker_utils/sampler.py`
5. Adding a new task → create `examples/<task>/env.py`, add case to `run_verl.sh`, create parquet in `data/`

The VERL integration is **task-agnostic**. Custom AgentLoop reads task config from environment variables set by `run_verl.sh`.

## WandB Configuration

Set in `~/.bashrc`:
```bash
export WANDB_API_KEY="..."
export WANDB_ENTITY="..."
```

Or use `WANDB_MODE=offline` to log locally without syncing.
