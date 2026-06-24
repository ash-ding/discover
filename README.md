# TTT-Discover Local Reproduction

This is an adapted version of [TTT-Discover](https://arxiv.org/abs/2601.16175) for local reproduction without the Tinker platform. We replace Tinker's remote LLM training service with a local backend using **Qwen3-8B + vLLM + PEFT LoRA**.

For the original project documentation, see [discover.README.md](discover.README.md).
For detailed reproduction notes, troubleshooting, and lessons learned, see [reproduce.md](reproduce.md).

## What Changed

We created `ttt_discover/local_backend/` — a drop-in adapter layer that replaces Tinker's client interfaces with local implementations:

| Tinker Component | Local Replacement | Implementation |
|---|---|---|
| `tinker.ServiceClient` | `LocalServiceClient` | Manages standalone vLLM process + training client |
| `tinker.TrainingClient` | `LocalTrainingClient` | HuggingFace + PEFT LoRA training |
| `tinker.SamplingClient` | `LocalSamplingClient` | HTTP client to vLLM OpenAI-compatible API |

**Architecture**: We run a **standalone vLLM V1 server** (with LoRA hot-reload support) that the training loop communicates with via HTTP. This enables:
- **vLLM V1 chunked prefill** — prevents OOM on long sequences (32K tokens) with KL penalty computation
- **Independent vLLM/training environments** — no version conflicts
- **Runtime memory tuning** — `gpu_memory_utilization` and `max_num_seqs` configuration

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

### 3. Start vLLM Server

**Start a standalone vLLM V1 server** before running any task. The training loop will connect to it via HTTP.

**Recommended (TP=4, paper configuration):**
```bash
# 4-GPU inference (TP=4) + 1 GPU training = 5 GPUs total
CUDA_VISIBLE_DEVICES=0,1,2,3,4 VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    python -m vllm.entrypoints.openai.api_server \
    --model /path/to/models/Qwen3-8B \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-lora \
    --max-lora-rank 64 \
    --gpu-memory-utilization 0.90 \
    --disable-custom-all-reduce
```

**Alternative configurations** (fewer GPUs):
```bash
# TP=2: 3 GPUs total (GPUs 0-1 inference, GPU 2 training)
CUDA_VISIBLE_DEVICES=0,1,2 VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    python -m vllm.entrypoints.openai.api_server \
    --model /path/to/models/Qwen3-8B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --enable-lora \
    --max-lora-rank 64 \
    --gpu-memory-utilization 0.90 \
    --disable-custom-all-reduce

# TP=1: 2 GPUs total (GPU 0 inference, GPU 1 training)
CUDA_VISIBLE_DEVICES=0,1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    python -m vllm.entrypoints.openai.api_server \
    --model /path/to/models/Qwen3-8B \
    --port 8000 \
    --max-model-len 32768 \
    --enable-lora \
    --max-lora-rank 64 \
    --gpu-memory-utilization 0.90 \
    --disable-custom-all-reduce
```

**Key parameters**:
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` — enables `/v1/load_lora_adapter` endpoint for hot-reload
- `gpu_memory_utilization=0.90` — paper setting (allocate 90% of GPU memory to KV cache)
- `max-model-len=32768` — maximum sequence length (must match DiscoverConfig.max_model_len)
- `disable-custom-all-reduce` — required for compatibility

**Verify startup:**
Wait for `"Application startup complete"`, then test:
```bash
curl http://localhost:8000/v1/models
# Should return: {"data": [{"id": "Qwen/Qwen3-8B", ...}]}
```

### 4. Run a Task

In a **separate terminal**, launch the training loop with `--local` flag:

```bash
# For TP=4 (5 GPUs): inference on 0-3, training on 4
CUDA_VISIBLE_DEVICES=0,1,2,3,4 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    python -m examples.<task>.env --local
```

**GPU allocation:** Training uses the GPU *after* the inference GPUs:
- TP=1 (2 GPUs): `CUDA_VISIBLE_DEVICES=0,1` → inference on GPU 0, training on GPU 1
- TP=2 (3 GPUs): `CUDA_VISIBLE_DEVICES=0,1,2` → inference on GPUs 0-1, training on GPU 2
- TP=4 (5 GPUs): `CUDA_VISIBLE_DEVICES=0,1,2,3,4` → inference on GPUs 0-3, training on GPU 4

**Estimated runtime (circle packing, 50 epochs):**
- ~8-12 hours on 5×H100 80GB (TP=4)
- ~15-20 hours on 3×H100 80GB (TP=2)

## Quick Validation (5 minutes)

Before launching a full 50-epoch run, verify your setup works:

**Method 1: Modify config temporarily**
```python
# In examples/circle_packing/env.py, temporarily change:
num_epochs=1,
groups_per_batch=2,
```

Then run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 WANDB_MODE=offline \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    python -m examples.circle_packing.env --local
```

**Expected result:** 1 epoch completes in ~3-5 minutes with no OOM errors. You should see:
- Rollout progress bars completing (2 groups × 64 completions)
- KL penalty computation finishing without errors
- Training step completing
- Checkpoint saved to `tinker_log/circle-packing-26/`

**Method 2: Use existing validation config** (if available in your fork)

Revert the temporary changes before launching the full run.

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

# Circle Packing (paper-config: 64x8, 50 epochs, KL=0.1, TP=2)
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

    # Local backend (Note: vLLM server started separately, see section 3)
    use_local_backend=True,
    inference_gpu_id=0,          # Must match vLLM server's first GPU
    training_gpu_id=2,           # Training on the GPU after inference GPUs (TP=2 → GPU 2)
    inference_tp_size=2,         # Tensor parallelism (must match vLLM --tensor-parallel-size)
    max_model_len=32768,         # Must match vLLM --max-model-len

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

## Troubleshooting

### vLLM OOM during KL penalty computation
**Symptom:** `torch.cuda.OutOfMemoryError: Tried to allocate X.XX GiB` during `incorporate_kl_penalty` step

**Cause:** KL penalty computation requires computing logprobs for entire sequences with `echo=True`, creating large logits tensors `[seq_len, vocab_size]`. With many concurrent requests, this can exceed available GPU memory.

**Fix:**
1. **Lower gpu_memory_utilization** in vLLM startup command:
   ```bash
   --gpu-memory-utilization 0.80  # or 0.70 if still OOM
   ```
   This trades KV cache capacity for logits computation memory.

2. **Reduce concurrent sequences** (if lowering GPU utilization isn't enough):
   ```bash
   --max-num-seqs 2  # limit concurrent requests
   ```

3. **Reduce sequence length** in `DiscoverConfig`:
   ```python
   phase1_max_tokens=20000,  # from 26000
   max_model_len=24576,      # from 32768
   ```

### Training OOM
**Symptom:** `torch.cuda.OutOfMemoryError` during the training step (after rollouts complete)

**Cause:** Training GPU ran out of memory when processing long sequences.

**Fix:**
1. **Use a larger training GPU** (H100 80GB recommended for 32K sequences)
2. **Lower max_model_len**:
   ```python
   max_model_len=16384,  # from 32768
   ```
3. **Check gradient checkpointing is enabled** (should be automatic in LocalTrainingClient)

### "LoRA adapter not loaded" errors
**Symptom:** vLLM returns `400 Bad Request: LoRA adapter 'lora_v1' not loaded`

**Cause:** `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` environment variable not set when starting vLLM.

**Fix:**
1. Stop vLLM server (Ctrl+C)
2. Restart with the environment variable:
   ```bash
   VLLM_ALLOW_RUNTIME_LORA_UPDATING=true python -m vllm.entrypoints.openai.api_server ...
   ```

### vLLM server doesn't start
**Symptom:** `RuntimeError: CUDA error: out of memory` when starting vLLM

**Cause:** Not enough GPU memory for the model with current settings.

**Fix:**
1. **Check GPUs are actually visible:**
   ```bash
   nvidia-smi
   ```
2. **Lower gpu_memory_utilization:**
   ```bash
   --gpu-memory-utilization 0.80
   ```
3. **Use more tensor parallelism** (if you have more GPUs):
   ```bash
   --tensor-parallel-size 4  # instead of 2
   ```

### Rollouts are very slow
**Symptom:** Rollout progress bars take >30 seconds per completion

**Cause:** vLLM not utilizing tensor parallelism efficiently, or wrong GPU allocation.

**Check:**
1. **Verify TP size matches:**
   - vLLM command: `--tensor-parallel-size 4`
   - Config: `inference_tp_size=4`
2. **Check GPU utilization:**
   ```bash
   nvidia-smi dmon -s u
   ```
   All inference GPUs should show >70% utilization during rollouts.

### "Connection refused" errors
**Symptom:** `aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host localhost:8000`

**Cause:** vLLM server not running or not ready yet.

**Fix:**
1. **Check vLLM is running:**
   ```bash
   curl http://localhost:8000/v1/models
   ```
2. **Wait for startup:** vLLM takes 30-60s to initialize. Look for `"Application startup complete"` in vLLM logs.

## Known Limitations

- **Model capability gap**: Qwen3-8B (8B) vs paper's gpt-oss-120b (120B). Code generation quality will be lower.
- **Serial Phase 2**: When Phase 1 thinking exhausts tokens, Phase 2 completions are processed one-by-one (different prefill per sample). Phase 1 is fully batched.
- **vLLM custom all-reduce disabled**: `disable_custom_all_reduce=True` due to compatibility issues. Uses NCCL standard communication instead (minimal performance impact with NVLink).
- **Standalone vLLM server**: Must be started manually before running tasks. Not auto-managed by the training loop.
