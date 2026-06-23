# TTT-Discover Local Reproduction

This is an adapted version of [TTT-Discover](https://arxiv.org/abs/2601.16175) for local reproduction without the Tinker platform. We replace Tinker's remote LLM training service with a local backend using **Qwen3-8B + vLLM + PEFT LoRA**.

For the original project documentation, see [discover.README.md](discover.README.md).
For detailed reproduction notes, troubleshooting, and lessons learned, see [reproduce.md](reproduce.md).

## What Changed

We created `ttt_discover/local_backend/` — a drop-in adapter layer that replaces Tinker's client interfaces with local implementations:

| Tinker Component | Local Replacement | Implementation |
|---|---|---|
| `tinker.ServiceClient` | `LocalServiceClient` | Orchestrates inference/training clients |
| `tinker.TrainingClient` | `LocalTrainingClient` | HuggingFace + PEFT LoRA training |
| `tinker.SamplingClient` | `LocalSamplingClient` | vLLM inference engine |

The RL algorithm (GRPO with entropic adaptive beta, PUCT state reuse) is **unchanged** from the original codebase.

## Hardware Requirements

- GPU: 2x NVIDIA H100 80GB (minimum) — 1 for inference, 1 for training
- CPU: Multi-core for sandbox code evaluation
- CUDA Driver: 12.4+
- Disk: ~50GB per task for checkpoints

## Quick Start

### 1. Environment Setup

Each task has its own conda environment and combined requirements file (`*-local.txt`) that includes both task-specific and RL backend dependencies:

```bash
# Create env (Python 3.11 for all tasks)
conda create -n <env_name> python=3.11 -y
conda activate <env_name>

# Install all dependencies (task + RL backend) in one step
pip install -r requirements/requirements-<task>-local.txt

# These two must be installed manually (require special flags)
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

Available requirements files:

| File | Tasks |
|---|---|
| `requirements-math-local.txt` | Erdős, AC Inequalities, Circle Packing |
| `requirements-denoising-local.txt` | Denoising |
| `requirements-gpumode-local.txt` | GPU Kernels (trimul, mla_decode) |
| `requirements-ahc-local.txt` | AHC (ahc039, ahc058) |

### 2. Download Model

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/models/Qwen3-8B
```

### 3. Run a Task

All tasks use `--local` flag to activate the local backend:

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m examples.<task>.env --local
```

## Tasks

### Task 1: Denoising (Single-Cell RNA-seq)

**Env:** `discover_denoising` | **Requirements:** `requirements/requirements-denoising-local.txt`

Extra dependencies:
```bash
pip install git+https://github.com/czbiohub/simscity.git
pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git
git clone https://github.com/openproblems-bio/openproblems.git
cd openproblems && git checkout v1.0.0 && git apply ../requirements/denoising/openproblems_api_fix.patch && cd ..
pip install --no-deps -e ./openproblems
```

Launch:
```bash
python -m examples.denoising.env --local
```

### Task 2: Mathematics

**Env:** `discover_math` | **Requirements:** `requirements/requirements-math-local.txt`

Launch:
```bash
# Erdos Minimum Overlap
python -m examples.erdos_min_overlap.env --local

# AC Inequalities
python -m examples.ac_inequalities.env --local          # AC1 (minimize upper bound)
python -m examples.ac_inequalities.env --local --ac2    # AC2 (maximize lower bound)

# Circle Packing
python -m examples.circle_packing.env --local           # 26 circles
python -m examples.circle_packing.env --local 32        # 32 circles
```

### Task 3: GPU Kernels

**Env:** `discover_gpumode_local` | **Requirements:** `requirements/requirements-gpumode-local.txt`

Requires [Modal](https://modal.com) account for remote kernel evaluation on cloud GPUs.

Extra setup:
```bash
pip install modal
python3 -m modal setup

# Deploy Modal apps (required before first run)
# For trimul: set TASK="trimul" in examples/gpu_mode/lib/runners/modal_runner_archs.py
conda run -n discover_gpumode python3 -c "cd examples/gpu_mode/lib && modal deploy runners/modal_runner_archs.py"

# For mla_decode: set TASK="mla_decode_nvidia", deploy with Python 3.12
conda run -n deploy_tmp python3 -c "cd examples/gpu_mode/lib && modal deploy runners/modal_runner_archs.py"
```

Launch:
```bash
PYTHONPATH="examples/gpu_mode/lib:$PYTHONPATH" \
    python -m examples.gpu_mode.env --local           # trimul (H100)
    python -m examples.gpu_mode.env --local --mla     # mla_decode (H200)
```

### Task 4: AtCoder (AHC)

**Env:** `discover_ale` | **Requirements:** `requirements/requirements-ahc-local.txt`

Runs inside `yimjk/ale-bench:cpp20-202301` container for C++20 compilation environment.

Extra setup:
```bash
# Pull container image
podman pull docker.io/yimjk/ale-bench:cpp20-202301

# Download test data
bash examples/ahc/get_cache.sh
```

Launch:
```bash
podman run --rm --device nvidia.com/gpu=all \
    --shm-size=16g --pids-limit=-1 \
    -v /workspace:/workspace \
    -w /workspace/home/asherding/code/discover \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -e WANDB_MODE=offline \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    docker.io/yimjk/ale-bench:cpp20-202301 \
    /path/to/conda/envs/discover_ale/bin/python -m examples.ahc.env --local
```

## Key Configuration Parameters

All tasks use `DiscoverConfig` with these local backend parameters:

```python
DiscoverConfig(
    # Model
    model_name="Qwen/Qwen3-8B",
    local_model_path="/path/to/models/Qwen3-8B",
    renderer_name="qwen3",

    # Local backend
    use_local_backend=True,
    inference_gpu_id=0,          # vLLM uses cuda:0 through cuda:0+tp_size-1
    training_gpu_id=4,           # Training on the GPU after inference GPUs
    inference_tp_size=4,         # Tensor parallelism (1, 2, or 4)
    max_model_len=8192,          # vLLM KV cache max length (lower = more concurrent samples)

    # RL hyperparameters (match paper Table 9)
    lora_rank=32,
    learning_rate=4e-5,
    temperature=1.0,
    kl_penalty_coef=0.1,         # 0.01 for AHC tasks

    # Scale parameters
    group_size=64,               # Completions per prompt
    groups_per_batch=8,          # Different prompts per step
    num_epochs=50,               # Training steps
    phase1_max_tokens=6000,      # Prompt + thinking budget
    save_every=2,                # Checkpoint frequency
)
```

### Per-Task Overrides

| Parameter | Default | AHC039 | AHC058 | Denoising |
|---|---|---|---|---|
| `phase1_max_tokens` | 26000 | 22000 | 25000 | 26000 |
| `kl_penalty_coef` | 0.1 | 0.01 | 0.01 | 0.1 |
| `eval_timeout` | 1000 | 530 | 530 | 530 |
| `learning_rate` | 4e-5 | 4e-5 | 2e-5 | 4e-5 |
| `num_cpus_per_task` | 0 | 2 | 2 | 1 |

## Output Structure

```
tinker_log/{experiment_name}/
├── metrics.jsonl                    # Per-step scores, rewards, advantages
├── checkpoints.jsonl                # Checkpoint path index
├── config.json                      # Training config snapshot
├── puct_sampler_step_*.json         # PUCT state sampler snapshots
└── train.log                        # Python logs

tinker_log/local_checkpoints/{experiment_name}/
├── state_{step}/                    # LoRA weights + optimizer (for resume)
├── sampler_{step}/                  # LoRA weights only (for evaluation)
└── latest_sampler/                  # Current LoRA weights (vLLM hot-reload)
```

## GPU Allocation

vLLM always uses `cuda:0` through `cuda:tp_size-1`. Training uses the next GPU. Set `CUDA_VISIBLE_DEVICES` to control which physical GPUs are used.

```bash
# TP=1: 2 GPUs total (GPU 0 inference, GPU 1 training)
CUDA_VISIBLE_DEVICES=0,1 ... inference_gpu_id=0, training_gpu_id=1, inference_tp_size=1

# TP=2: 3 GPUs total (GPU 0,1 inference, GPU 2 training)
CUDA_VISIBLE_DEVICES=0,1,2 ... inference_gpu_id=0, training_gpu_id=2, inference_tp_size=2

# TP=4: 5 GPUs total (GPU 0-3 inference, GPU 4 training)
CUDA_VISIBLE_DEVICES=0,1,2,3,4 ... inference_gpu_id=0, training_gpu_id=4, inference_tp_size=4
```

## Known Limitations

- **Model capability gap**: Qwen3-8B (8B) vs paper's gpt-oss-120b (120B). Code generation quality will be lower.
- **Training sequence truncation**: `max_train_seq_len=8192` truncates long sequences during training. Affects AHC tasks (~22K total). Solvable with TP=2 training.
- **Serial Phase 2**: When Phase 1 thinking exhausts tokens, Phase 2 completions are processed one-by-one (different prefill per sample). Phase 1 is fully batched.
- **vLLM custom all-reduce disabled**: `disable_custom_all_reduce=True` due to compatibility issues. Uses NCCL standard communication instead (minimal performance impact with NVLink).
