# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TTT-Discover is a research implementation of test-time training for LLMs using reinforcement learning. The original codebase relied on the Tinker platform (remote LLM training service). This is a **local reproduction fork** that replaces Tinker with a local backend using **Qwen3-8B + vLLM (inference) + PEFT LoRA (training)**.

**Key Architecture Decision**: We run a **standalone vLLM V1 server** that the training loop communicates with via HTTP. The vLLM process is managed separately from the RL training loop.

## Critical Setup Requirements

### Hardware Requirements
- **Minimum**: 2x NVIDIA H100 80GB (1 for inference, 1 for training)
- **Paper configuration**: 5x H100 80GB (4 for inference via TP=4, 1 for training)
- CUDA Driver: 12.9+

### Environment Setup Pattern

Each task has its own conda environment. **Installation order matters**:

1. Install torch 2.11.0+cu129 first (from PyTorch index)
2. Install vllm 0.23.0 second (auto-pulls torch 2.11 from wheels.vllm.ai)
3. Install flashinfer from the specialized index third
4. Install flash-attn last with special flags

```bash
conda create -n <env_name> python=3.11 -y
conda activate <env_name>
pip install -r requirements/requirements-<task>.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/
pip install flash-attn==2.8.3.post1 --no-build-isolation --no-cache-dir
```

**Why this order?** Installing vllm before pinning torch can pull incompatible versions. Installing flashinfer before vllm can cause ABI incompatibility. Flash-attn must be installed last with --no-build-isolation to avoid rebuilding against wrong torch.

### vLLM Server Must Run Separately

**CRITICAL**: Before running any task, start the universal vLLM server in the root directory:

```bash
bash start_vllm.sh
```

This script starts vLLM with TP=4 by default (GPUs 0-3 for inference). Override with environment variables:

```bash
TENSOR_PARALLEL=2 VLLM_PORT=8000 GPU_MEMORY_UTIL=0.8 bash start_vllm.sh
```

**Key flags set by start_vllm.sh**:
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` — enables hot-reload via `/v1/load_lora_adapter`
- `--tensor-parallel-size=4` — uses GPUs 0-3 for inference
- `--enable-lora --max-lora-rank=64` — LoRA support with max rank 64
- `--gpu-memory-utilization=0.9` — configurable via GPU_MEMORY_UTIL env var
- `--disable-custom-all-reduce` — required for compatibility (uses standard NCCL)

Verify startup:
```bash
curl http://localhost:8888/v1/models
```

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

## Standard Workflow

1. Start vLLM server once (shared across all tasks):
   ```bash
   bash start_vllm.sh
   ```

2. Run task with run.sh script:
   ```bash
   cd examples/<task>
   bash run.sh config_paper.yaml      # 50 epochs (paper config)
   bash run.sh config_validate.yaml   # 1 epoch (quick validation)
   bash run.sh /path/to/custom.yaml   # custom configuration
   ```

### Available Tasks

| Task | Environment | Requirements File | Notes |
|------|-------------|-------------------|-------|
| Circle Packing | `discover_math` | `requirements-math.txt` | 26 or 32 circles |
| AC Inequalities | `discover_math` | `requirements-math.txt` | AC1 (minimize) or AC2 (maximize) |
| Erdős Min Overlap | `discover_math` | `requirements-math.txt` | C₅ constant optimization |
| Denoising | `discover_denoising` | `denoising/requirements-denoising.txt` | Requires openproblems patch |
| GPU Mode | `discover_gpumode` | `requirements-gpumode.txt` | Requires Modal account |
| AHC | `discover_ale` | `requirements-ahc.txt` | Must run in container |

Each task has detailed documentation in its respective `examples/<task>/README.md`.


## Code Architecture

### Core Structure

```
ttt_discover/
├── discovery.py              # DiscoverConfig and main entry point
├── rl/
│   ├── train.py             # RL training loop, KL penalty computation
│   ├── rollouts.py          # PUCT sampling, trajectory generation
│   └── data_processing.py   # Advantage computation (entropic adaptive beta)
├── local_backend/           # ⚠️ Custom replacement for Tinker SDK
│   ├── service_client.py    # Orchestrates inference + training clients
│   ├── sampling_client.py   # HTTP client to vLLM OpenAI API
│   ├── training_client.py   # HuggingFace + PEFT LoRA training
│   ├── loss.py             # importance_sampling_loss, ppo_clip_loss
│   └── types.py            # Type definitions matching Tinker SDK
├── tinker_utils/
│   ├── completers.py        # Qwen3TwoPhaseTokenCompleter (Phase 2 parsing)
│   └── sampler.py          # PUCT state reuse
└── environments/
    └── sandbox_reward_evaluator.py  # Ray-based sandboxed code execution
```

### local_backend/ Design

This directory contains ~800 lines of adapter code that makes the original Tinker-dependent codebase work locally:

- **`LocalServiceClient`**: Entry point. Creates vLLM sampling client + local training client, manages GPU allocation
- **`LocalSamplingClient`**: Wraps vLLM's OpenAI-compatible `/v1/completions` API. Handles LoRA hot-reload via `/v1/load_lora_adapter`
- **`LocalTrainingClient`**: Implements HuggingFace Trainer + PEFT LoRA training. Uses gradient checkpointing and flash-attn to fit 32K sequences in 80GB
- **`loss.py`**: Pure PyTorch implementations of importance sampling and PPO losses

The interfaces match Tinker's SDK so `ttt_discover/rl/train.py` works with minimal changes.

## GPU Allocation Strategy

vLLM always uses `cuda:0` through `cuda:tp_size-1`. Training uses the GPU after the inference GPUs.

```bash
# TP=1 (2 GPUs): inference on 0, training on 1
CUDA_VISIBLE_DEVICES=0,1 ... inference_gpu_id=0, training_gpu_id=1, inference_tp_size=1

# TP=2 (3 GPUs): inference on 0-1, training on 2
CUDA_VISIBLE_DEVICES=0,1,2 ... inference_gpu_id=0, training_gpu_id=2, inference_tp_size=2

# TP=4 (5 GPUs): inference on 0-3, training on 4
CUDA_VISIBLE_DEVICES=0,1,2,3,4 ... inference_gpu_id=0, training_gpu_id=4, inference_tp_size=4
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

**Task-specific parameter overrides** (bold = differs from default):
- **AHC**: Lower all three key params (phase1_max_tokens, kl_penalty_coef, learning_rate)
- **GPU Mode**: Lower kl_penalty_coef only
- **All others**: Use default values from Table 9

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
tinker_log/<experiment_name>/
├── metrics.jsonl              # Per-step scores, rewards, advantages
├── checkpoints.jsonl          # Checkpoint path index
├── config.json               # Training config snapshot
└── train.log                 # Python logs

tinker_log/local_checkpoints/<experiment_name>/
├── state_<step>/             # LoRA weights + optimizer (for resume)
├── sampler_<step>/           # LoRA weights only (for evaluation)
└── latest_sampler/           # Current LoRA (vLLM hot-reloads from here)
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
| PyTorch | N/A | 2.11.0+cu129 |
| vLLM | Tinker remote service | vLLM 0.23.0 local server |
| FlashInfer | N/A | 0.6.12 (cu129/torch2.11) |
| Flash-Attn | N/A | 2.8.3.post1 |
| CUDA Driver | N/A | 12.9+ |
| Inference | Tinker auto-managed | Manual vLLM server (TP=4) |
| Training | Tinker remote | Local PEFT LoRA |
| vLLM management | Auto-managed by Tinker | Manual (bash start_vllm.sh) |
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

When modifying the local backend:

1. Changes to sampling/inference → edit `ttt_discover/local_backend/sampling_client.py`
2. Changes to training → edit `ttt_discover/local_backend/training_client.py`
3. Changes to loss functions → edit `ttt_discover/local_backend/loss.py`
4. Changes to RL algorithm → edit `ttt_discover/rl/train.py` or `ttt_discover/rl/rollouts.py`

The local_backend is **task-agnostic**. If you add support for a new RL task, you should not need to modify local_backend/ — only create a new env.py and reward evaluator.

## WandB Configuration

Set in `~/.bashrc`:
```bash
export WANDB_API_KEY="..."
export WANDB_ENTITY="..."
```

Or use `WANDB_MODE=offline` to log locally without syncing.
