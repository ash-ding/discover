# TTT-Discover: Local Reproduction

This is a local reproduction of [TTT-Discover](https://arxiv.org/abs/2601.16175) that replaces the Tinker platform with a local backend using **Qwen3-8B + vLLM 0.23.0 + PEFT LoRA**. The RL algorithm (GRPO with entropic adaptive beta, PUCT state reuse) remains unchanged from the original codebase.

## Quick Start

### Prerequisites

- **Hardware**: 2x NVIDIA H100 80GB minimum (5x for paper configuration)
- **CUDA Driver**: 12.9+
- **Software**: Conda, Python 3.11

### Setup

**1. Create conda environment for your task**

```bash
# Choose one based on your task
conda create -n discover_math python=3.11 -y          # Circle Packing, Erdős, AC
conda create -n discover_denoising python=3.11 -y    # Denoising
conda create -n discover_gpumode python=3.11 -y      # GPU Mode
conda create -n discover_ale python=3.11 -y          # AHC

# Activate environment
conda activate discover_<task>
```

**CRITICAL**: Install dependencies in this **exact order** to avoid CUDA ABI conflicts:

```bash
# 1. PyTorch first (from PyTorch index)
pip install torch==2.11.0+cu129 -i https://download.pytorch.org/whl/cu129

# 2. vLLM second (auto-pulls torch 2.11 from wheels.vllm.ai)
pip install vllm==0.23.0

# 3. FlashInfer third (from specialized index)
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/

# 4. Flash-attn last (with special flags)
pip install flash-attn==2.8.3.post1 --no-build-isolation --no-cache-dir

# 5. Other dependencies (if using requirements file)
pip install -r requirements/requirements-<task>.txt
```

**Why this order?** Installing vllm before pinning torch can pull incompatible versions. Installing flashinfer before vllm can cause ABI incompatibility. Flash-attn must be installed last with `--no-build-isolation` to avoid rebuilding against wrong torch version. Installing in wrong order causes version downgrades and runtime errors.

**2. Start vLLM server** (once, shared across all tasks)

```bash
bash start_vllm.sh
```

This starts vLLM with TP=4 (GPUs 0-3 for inference). Override defaults with environment variables:
```bash
TENSOR_PARALLEL=2 VLLM_PORT=8000 GPU_MEMORY_UTIL=0.8 bash start_vllm.sh
```

Verify server is running:
```bash
curl http://localhost:8888/v1/models
```

**3. Run experiment**

```bash
cd examples/<task>
bash run.sh config_smoke_test.yaml   # smoke test (~5-10 min, 16 samples/step)
bash run.sh config_validate.yaml     # validation (~30-60 min, 512 samples/step)
bash run.sh config_paper.yaml        # full training (~40-60 hrs, 512 samples/step)
bash run.sh /path/to/custom.yaml     # custom configuration
```

## Configuration System

Each task uses YAML configuration files instead of hardcoded parameters:

- **`config_paper.yaml`** - Full 50-epoch training with paper parameters (Table 9)
  - group_size=64, groups_per_batch=8 → 512 samples/step
- **`config_validate.yaml`** - Quick 1-epoch validation (same params, epochs=1)
  - group_size=64, groups_per_batch=8 → 512 samples/step
- **`config_smoke_test.yaml`** - Ultra-fast smoke test for pipeline verification
  - group_size=8, groups_per_batch=2 → 16 samples/step (32× faster)

### Environment Variables

Some parameters are controlled via environment variables, **not** YAML:

```bash
# vLLM server address (used by local_backend/service_client.py)
export VLLM_BASE_URL="http://localhost:8888"

# WandB logging mode (used by ml_log.py)
export WANDB_MODE="offline"

# GPU memory utilization (used by start_vllm.sh)
export GPU_MEMORY_UTIL=0.9

# CRITICAL: Enable LoRA hot-reload (REQUIRED for training)
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
```

The `start_vllm.sh` script sets `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` automatically. Without this, training will fail with "LoRA adapter not loaded" errors.

### Custom Configurations

```bash
# Use custom config file
export TTT_CONFIG_PATH=/path/to/custom_config.yaml
cd examples/<task>
bash run.sh full
```

See [CLAUDE.md](CLAUDE.md) for complete parameter reference and configuration details.

## Available Tasks

| Task | Environment | Command | Runtime (50 epochs) |
|------|-------------|---------|---------------------|
| **Circle Packing** | `discover_math` | `cd examples/circle_packing && bash run.sh config_paper.yaml` | ~40-50h |
| **AC Inequalities** | `discover_math` | `cd examples/ac_inequalities && bash run.sh config_paper.yaml` | ~40-60h |
| **Erdős Min Overlap** | `discover_math` | `cd examples/erdos_min_overlap && bash run.sh config_paper.yaml` | ~40-60h |
| **Denoising** | `discover_denoising` | `cd examples/denoising && bash run.sh config_paper.yaml` | ~30-40h |
| **GPU Mode** | `discover_gpumode` | `cd examples/gpu_mode && bash run.sh config_paper.yaml` | ~20-30h |
| **AHC** | `discover_ale` | `cd examples/ahc && bash run.sh config_paper.yaml` | ~50-60h |

**Task-specific notes**:
- **AC Inequalities**: Two variants (AC1 minimize, AC2 maximize), 2 CPUs per task
- **Erdős Min Overlap**: Minimize C₅ constant, target ≤ 0.38080
- **Denoising**: Requires `openproblems_api` patch (see `requirements/denoising/README.md`)
- **GPU Mode**: Requires Modal account and API key
- **AHC**: Must run inside `yimjk/ale-bench:cpp20-202301` container

Each task has detailed documentation in `examples/<task>/README.md`.

## Paper Configuration (Table 9)

All parameters below exactly match the TTT-Discover paper (Table 9).

### Common Parameters (All Tasks)

- **Model**: Qwen/Qwen3-8B
- **group_size**: 64 (completions generated per prompt)
- **groups_per_batch**: 8 (different prompts per training step → 512 samples/step total)
- **num_epochs**: 50
- **Inference**: TP=4 (Tensor Parallelism, uses GPUs 0-3)
- **Training**: Single GPU (GPU 4)
- **LoRA rank**: 32
- **Learning rate**: 4e-5
- **Save frequency**: every 2 epochs

### Task-Specific Parameters

#### Circle Packing (26 or 32 circles)
- **phase1_max_tokens**: 26000 (prompt + thinking token budget)
- **kl_penalty_coef**: 0.1 (KL divergence penalty coefficient)
- **eval_timeout**: 530s
- **Target score**: 2.636 (26 circles) / 2.940 (32 circles)

#### Denoising (single-cell RNA)
- **phase1_max_tokens**: 26000
- **kl_penalty_coef**: 0.1
- **eval_timeout**: 530s

#### GPU Mode (kernel optimization)
- **phase1_max_tokens**: 26000
- **kl_penalty_coef**: 0.1
- **eval_timeout**: 300s

#### AHC (competitive programming)
- **phase1_max_tokens**: 22000 ⚠️ **(shorter than other tasks)**
- **kl_penalty_coef**: 0.01 ⚠️ **(10x smaller than other tasks)**
- **learning_rate**: 2e-5 ⚠️ **(half of other tasks)**
- **eval_timeout**: 600s

⚠️ = Different from common parameters

## Configuration System

Each task has two YAML configuration files:

- **config_paper.yaml** — Full 50-epoch run with paper parameters (Table 9)
- **config_validate.yaml** — 1-epoch validation run (same parameters, epochs=1)

To use a custom configuration:

```bash
export TTT_CONFIG_PATH=/path/to/custom_config.yaml
cd examples/<task>
bash run.sh full
```

Configuration files are loaded by `ttt_discover/utils/config_loader.py` and passed to `DiscoverConfig`.

## Output Structure

```
tinker_log/<experiment_name>/
├── metrics.jsonl       # Per-step metrics (rewards, scores, advantages)
├── config.json         # Training config snapshot
└── train.log           # Python logs

tinker_log/local_checkpoints/<experiment_name>/
├── state_0/, state_2/, state_4/, ...  # LoRA weights + optimizer state
└── latest_sampler/                     # Current LoRA weights (vLLM hot-reloads from here)
```

Checkpoints are saved every 2 epochs by default (configurable via `save_every` in config).

## Architecture Overview

We created `ttt_discover/local_backend/` — a drop-in adapter layer that replaces Tinker's client interfaces:

| Tinker Component | Local Replacement | Implementation |
|---|---|---|
| `tinker.ServiceClient` | `LocalServiceClient` | Manages vLLM server + training client |
| `tinker.TrainingClient` | `LocalTrainingClient` | HuggingFace + PEFT LoRA training |
| `tinker.SamplingClient` | `LocalSamplingClient` | HTTP client to vLLM OpenAI-compatible API |

**Key Architecture Decision**: We run a **standalone vLLM V1 server** that the training loop communicates with via HTTP. The vLLM process is managed separately from the RL training loop.

Benefits:
- **vLLM V1 chunked prefill** — prevents OOM on 32K sequences with KL penalty computation
- **LoRA hot-reload** — runtime adapter loading via `/v1/load_lora_adapter` endpoint
- **Independent environments** — no version conflicts between vLLM and training dependencies

## Troubleshooting

### vLLM OOM during KL penalty computation

**Symptom**: `torch.cuda.OutOfMemoryError` during `incorporate_kl_penalty` step

**Root cause**: KL penalty uses `echo=True` to compute logprobs for entire sequences (prompt + generation). For 32K sequences, this creates large tensors that exhaust GPU memory.

**Solutions** (in order of preference):
1. Lower `gpu_memory_utilization` in `start_vllm.sh` (default 0.9 → try 0.7 or 0.8)
2. Add `--max-num-seqs 2` to vLLM startup to limit concurrent sequences
3. Reduce `max_model_len` in both vLLM and task config (32768 → 24576)

### Training OOM

**Symptom**: `torch.cuda.OutOfMemoryError` during training step

**Fix**: Training already uses gradient checkpointing and flash-attn. If still OOM, reduce `max_model_len` in both vLLM and task config.

### LoRA adapter not loaded

**Symptom**: vLLM returns `400 Bad Request: LoRA adapter 'lora_v1' not loaded`

**Fix**: Set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` when starting vLLM (already done in `start_vllm.sh`).

### AHC Container Issues

For AHC tasks, the job must run inside `yimjk/ale-bench:cpp20-202301` container. Common issues:

- **PID limit**: Add `--pids-limit=-1` to podman run
- **Shared memory**: Add `--shm-size=16g` to podman run
- **GPU access**: Use `--device nvidia.com/gpu=all`

## Additional Documentation

- **[CLAUDE.md](CLAUDE.md)** — Detailed architecture, development guide, and best practices
- **[Original README](discover.README.md)** — Original project documentation
- **[Paper](docs/ttt_discover_paper.pdf)** — TTT-Discover paper (arXiv:2601.16175)

## Key Differences from Original Codebase

| Aspect | Original | This Fork |
|--------|----------|-----------|
| **Model** | gpt-oss-120b | Qwen3-8B |
| **PyTorch** | N/A | 2.11.0+cu129 |
| **vLLM** | Tinker remote service | vLLM 0.23.0 local server |
| **FlashInfer** | N/A | 0.6.12 (cu129/torch2.11) |
| **Flash-Attn** | N/A | 2.8.3.post1 |
| **CUDA Driver** | N/A | 12.9+ |
| **Inference** | Tinker auto-managed | Manual vLLM server (TP=4) |
| **Training** | Tinker remote | Local PEFT LoRA |
| **Phase 2 marker** | Channel tokens | `</think>` tag |
| **Max seq len** | Unlimited | 32768 (32K) |

## Citation

If you use this codebase, please cite the original TTT-Discover paper:

```bibtex
@article{polu2025ttt,
  title={TTT-Discover: Test-Time Training for LLM Discovery},
  author={Polu, Stanislas and others},
  journal={arXiv preprint arXiv:2601.16175},
  year={2025}
}
```
