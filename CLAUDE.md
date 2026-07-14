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
- **Optional second node**: 8x H100 for multi-node training (16 GPUs total)
- CUDA Driver: 12.9+

### Cluster Nodes

| Node | Hostname | Internal IP | External IP | Role |
|------|----------|-------------|-------------|------|
| Node 0 | ai-innovation-h100-10-preserve | 10.241.128.30 | 169.62.23.172 | Head (default) |
| Node 1 | ai-innovation-h100-11-preserve | 10.241.128.16 | 169.62.18.122 | Worker (optional) |

Scripts default to internal IPs (lower latency for Ray/NCCL). Override via `HEAD_NODE` and `WORKER_NODE` env vars. SSH must be passwordless between nodes.

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

## Pre-Flight Checklist

Before ANY training run, ensure these environment variables are set:

```bash
# 1. Activate the unified conda environment
conda activate verl_discover

# 2. Use local model path (default Qwen/Qwen3-8B downloads from HF Hub — will hang without HF_TOKEN)
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-8B

# 3. WandB offline mode (avoids API key auth errors)
export WANDB_MODE=offline

# 4. Verify parquet data exists for your task
ls data/<task>_train.parquet
```

**Add to `~/.bashrc` for persistence:**
```bash
export WANDB_MODE=offline
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-8B
```

## Mandatory Rules

1. **Always activate `verl_discover` conda environment before running any task.** All tasks share this single environment.

2. **Never use conda `base` environment** to run or debug code.

## Standard Workflow

```bash
conda activate verl_discover
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-8B
export WANDB_MODE=offline

# Run any task (VERL manages vLLM internally, no manual server needed)
TOTAL_EPOCHS=50 bash run_verl.sh circle_packing   # Full training
TOTAL_EPOCHS=1  bash run_verl.sh circle_packing   # Validation (1 epoch)

# Smoke test (fast, minimal samples)
TOTAL_EPOCHS=1 ROLLOUT_N=4 TRAIN_BATCH_SIZE=2 bash run_verl.sh circle_packing
```

For AHC (long prompts), use SP=2 to avoid OOM:
```bash
TOTAL_EPOCHS=50 SP_SIZE=2 bash run_verl.sh ahc039
```

For GPU Mode, set the eval server:
```bash
GPU_EVAL_SERVER=http://10.241.128.30:8890 TOTAL_EPOCHS=50 bash run_verl.sh gpu_mode
```

### Resume Training

```bash
# Same-dir resume (crash recovery, overwrites latest)
TOTAL_EPOCHS=50 RESUME_DIR=checkpoints/ttt-discover/my-run INPLACE=true bash run_verl.sh circle_packing

# New-dir resume (preserves old results)
TOTAL_EPOCHS=50 RESUME_DIR=checkpoints/ttt-discover/my-run bash run_verl.sh circle_packing

# Cross-config resume (different GPU count) — export LoRA first
python scripts/export_lora.py checkpoints/ttt-discover/my-run/latest/actor
TOTAL_EPOCHS=50 RESUME_DIR=checkpoints/ttt-discover/my-run bash run_verl.sh circle_packing
```

### Available Tasks

| Task | Command | Metric | Special | README |
|------|---------|--------|---------|--------|
| Circle Packing 26 | `bash run_verl.sh circle_packing` | `raw_score/max` >= 2.636 | | [README](examples/circle_packing/README.md) |
| Circle Packing 32 | `bash run_verl.sh cp32` | `raw_score/max` >= 2.940 | | [README](examples/circle_packing/README.md) |
| AC Inequalities 1 | `bash run_verl.sh ac1` | `raw_score/max` | timeout=1100s | [README](examples/ac_inequalities/README.md) |
| AC Inequalities 2 | `bash run_verl.sh ac2` | `raw_score/max` | timeout=1100s | [README](examples/ac_inequalities/README.md) |
| Erdos Min Overlap | `bash run_verl.sh erdos` | `raw_score/min` <= 0.380 | timeout=1100s | [README](examples/erdos_min_overlap/README.md) |
| Denoising | `bash run_verl.sh denoising` | `raw_score/max` | openproblems patch | [README](examples/denoising/README.md) |
| GPU Mode (trimul) | `bash run_verl.sh gpu_mode` | `raw_score/min` | eval GPU needed | [README](examples/gpu_mode/README.md) |
| AHC 039 | `SP_SIZE=2 bash run_verl.sh ahc039` | `raw_score/max` | container + SP=2 | [README](examples/ahc/README.md) |

All tasks use the unified `verl_discover` conda environment.


### Multi-Node Training

For 2-node training (16 GPUs total):

```bash
# 1. Start Ray cluster
bash scripts/start_ray_cluster.sh

# 2. Run training with NNODES=2
NNODES=2 TOTAL_EPOCHS=50 bash run_verl.sh circle_packing

# 3. Stop cluster when done
bash scripts/start_ray_cluster.sh stop
```

The Ray head node runs on Node 0 (10.241.128.30), worker on Node 1 (10.241.128.16). SSH must be passwordless between nodes. Override via `HEAD_NODE` and `WORKER_NODE` env vars.

### Overridable Environment Variables

All variables below can be set before `bash run_verl.sh <task>`. Task-specific defaults are in `run_verl.sh`.

**Model & Infrastructure:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_PATH` | `Qwen/Qwen3-8B` | Model path. **Use local path** to avoid HF Hub download |
| `NGPUS_PER_NODE` | `8` | GPUs per node |
| `NNODES` | `1` | Number of nodes (set 2 for multi-node) |
| `SP_SIZE` | `1` | Sequence parallel size (set 2 for AHC) |
| `ROLLOUT_TP` | `4` | vLLM tensor parallel size |
| `ROLLOUT_GPU_MEM_UTIL` | `0.5` | vLLM GPU memory fraction |

**Training:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `TOTAL_EPOCHS` | `1` | Training epochs (50 for full run) |
| `ROLLOUT_N` | `64` | Completions per prompt |
| `TRAIN_BATCH_SIZE` | `8` | Prompts per batch |
| `LORA_RANK` | `32` | LoRA rank and alpha |
| `ACTOR_LR` | Task-dependent | Learning rate |
| `KL_COEF` | Task-dependent | KL penalty coefficient |
| `SAVE_FREQ` | `0` | Checkpoint save frequency (0 = every step) |

**PUCT Sampling:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `DISCOVER_PUCT_C` | `1.0` | PUCT exploration constant |
| `DISCOVER_TOPK_CHILDREN` | `2` | Top-k children in PUCT tree |
| `DISCOVER_MAX_BUFFER_SIZE` | `1000` | Max PUCT buffer size |

**Logging:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `WANDB_MODE` | `online` | **Set to `offline`** to avoid auth errors |
| `WANDB_API_KEY` | (none) | Required only if `WANDB_MODE=online` |
| `WANDB_ENTITY` | (none) | WandB team/org name |
| `VERL_LOGGING_LEVEL` | `INFO` | Agent loop log verbosity |

**GPU Mode Only:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_EVAL_SERVER` | (empty) | HTTP eval server URL for remote evaluation |
| `KERNEL_EVAL_GPU` | `0` | GPU ID for local kernel evaluation |
| `KERNEL_EVAL_TIMEOUT` | `1200` | Per-kernel eval timeout (seconds) |
| `KERNEL_EVAL_RETRIES` | `2` | Eval retry count |
| `KERNEL_EVAL_USE_CONTAINER` | `true` | Use Docker/Podman isolation |

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

## Task-Specific Parameters

Parameters that differ from defaults (bold = non-standard):

| Parameter | Default | AHC | GPU Mode |
|-----------|---------|-----|----------|
| `phase1_max_tokens` | `26000` | **22000** | `26000` |
| `kl_penalty_coef` | `0.1` | **0.01** | **0.01** |
| `learning_rate` | `4e-5` | **2e-5** | `4e-5` |
| `eval_timeout` | `530s` | **600s** | `530s` |
| `num_cpus_per_task` | `1` | **2** | `1` |
| `SP_SIZE` | `1` | **2** | `1` |

These are handled automatically by `run_verl.sh` — no manual override needed unless experimenting.

**Config tiers** (via env var overrides to `run_verl.sh`):

| Tier | ROLLOUT_N | TRAIN_BATCH_SIZE | samples/step | TOTAL_EPOCHS | Purpose |
|------|-----------|-----------------|-------------|--------------|---------|
| Full | 64 | 8 | 512 | 50 | Paper reproduction |
| Validate | 64 | 8 | 512 | 1 | Pre-training check |
| Smoke test | 4 | 2 | 8 | 1 | Quick code validation |

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
│   ├── actor/                       # FSDP sharded weights + optimizer
│   │   ├── model_world_size_8_rank_*.pt
│   │   ├── optim_world_size_8_rank_*.pt
│   │   ├── fsdp_config.json         # Parallel config (world_size)
│   │   ├── lora_train_meta.json     # LoRA rank/alpha
│   │   └── exported_lora/           # (optional) PEFT adapter for cross-config resume
│   └── data.pt                      # Dataloader state
├── rollouts/                        # Per-step rollout logs
│   └── N.jsonl                      # Prompt + response + reward
├── puct_sampler_step_*.json         # PUCT state history
└── latest_checkpointed_iteration.txt # Current step number
```

**LoRA export** (for cross-config resume or analysis):
```bash
python scripts/export_lora.py checkpoints/ttt-discover/my-run/latest/actor
# → creates exported_lora/adapter_model.safetensors (~80MB)
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

Add to `~/.bashrc`:

```bash
export WANDB_MODE=offline
```

For cloud sync, set `WANDB_API_KEY` and `WANDB_ENTITY` instead and remove the `WANDB_MODE=offline` line.
