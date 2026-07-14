# GPU Mode (GPU Kernel Optimization)

## Overview

Generate and optimize fast CUDA/Triton GPU kernels for various computational tasks. The model learns to write optimized kernel code that minimizes execution time.

## Prerequisites

```bash
conda activate verl_discover
export WANDB_MODE=offline
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-8B
```

## Evaluation Setup

GPU Mode requires a **separate GPU** for kernel evaluation (training uses all 8 GPUs via VERL colocate). Two options:

### Option A: Remote Eval Server (recommended for multi-node)

Run the eval server on a separate node (e.g., Node 0) with dedicated GPUs:

```bash
# On eval node (Node 0), start eval server with 2 GPUs
conda activate verl_discover
python examples/gpu_mode/eval_server.py --port 8890 --num-gpus 2 --gpu-ids 0,1

# On training node, set the server address
export GPU_EVAL_SERVER=http://10.241.128.30:8890
```

### Option B: Local Eval (single-node, uses training GPUs)

```bash
# WARNING: Kernel eval runs on GPU 0 by default, which is also used by vLLM.
# This may cause GPU memory conflicts during inference.
export KERNEL_EVAL_GPU=0
```

### Container Isolation (recommended)

Build the eval container for crash protection (one-time, ~5 min):

```bash
cd examples/gpu_mode/local_evaluator
bash build_container.sh
```

To disable container isolation:
```bash
export KERNEL_EVAL_USE_CONTAINER=false
```

## Launch (VERL Colocate Mode)

```bash
# Smoke test (~5 min)
TOTAL_EPOCHS=1 ROLLOUT_N=4 TRAIN_BATCH_SIZE=2 bash run_verl.sh gpu_mode

# With remote eval server
GPU_EVAL_SERVER=http://10.241.128.30:8890 TOTAL_EPOCHS=1 bash run_verl.sh gpu_mode

# Full training
GPU_EVAL_SERVER=http://10.241.128.30:8890 TOTAL_EPOCHS=50 bash run_verl.sh gpu_mode
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 4e-5 | Standard |
| KL coefficient | **0.01** | Lower than standard 0.1 |
| Eval timeout | 530s | Per-kernel evaluation |
| CPUs per task | 1 | |
| Data file | `data/gpu_mode_trimul_train.parquet` | |

## GPU Mode Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_EVAL_SERVER` | (empty) | HTTP eval server URL. If set, uses remote eval |
| `KERNEL_EVAL_GPU` | `0` | GPU ID for local kernel evaluation |
| `KERNEL_EVAL_TIMEOUT` | `1200` | Per-kernel eval timeout (seconds) |
| `KERNEL_EVAL_RETRIES` | `2` | Number of eval retries on failure |
| `KERNEL_EVAL_USE_CONTAINER` | `true` | Use Docker/Podman isolation |

## Monitoring

- **Metric**: `env/all/raw_score/min` (**minimization** task)
- **Target**: Minimize kernel runtime (microseconds)
- **Lower is better**: Faster kernel execution
- Reward formula: `reward = SCORE_SCALE / score_us` (trimul: SCORE_SCALE=1500)

## Evaluation Details

- Kernels run 18 correctness tests + 7 performance benchmarks
- `score_us` = geometric mean of 7 benchmark runtimes
- Container isolation (Podman/Docker) prevents GPU crashes from affecting training
- Tests and benchmarks defined in `lib/bioml/trimul/task.yml`

## Resume Training

```bash
GPU_EVAL_SERVER=http://10.241.128.30:8890 \
TOTAL_EPOCHS=50 RESUME_DIR=checkpoints/ttt-discover/<experiment> INPLACE=true \
bash run_verl.sh gpu_mode
```
