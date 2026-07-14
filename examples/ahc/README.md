# AHC (AtCoder Heuristic Contest)

## Overview

Solve competitive programming heuristic optimization problems from AtCoder Heuristic Contest. The model generates C++ code implementing heuristic algorithms (simulated annealing, beam search, etc.).

## Prerequisites

```bash
conda activate verl_discover
export WANDB_MODE=offline
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-8B
```

## Container Setup (Required)

AHC evaluation must run inside the `yimjk/ale-bench:cpp20-202301` container.

```bash
podman run -it \
  --device nvidia.com/gpu=all \
  --shm-size=16g \
  --pids-limit=-1 \
  -v /workspace:/workspace \
  yimjk/ale-bench:cpp20-202301 \
  bash

# Inside container:
conda activate verl_discover
```

**Required container flags**:
- `--pids-limit=-1` — prevents PID limit errors during multi-process evaluation
- `--shm-size=16g` — adequate shared memory for Ray workers
- `--device nvidia.com/gpu=all` — GPU access inside container

## Launch (VERL Colocate Mode)

**Important**: AHC requires `SP_SIZE=2` (sequence parallelism) to avoid OOM from long prompts.

```bash
# Smoke test (~5 min)
TOTAL_EPOCHS=1 ROLLOUT_N=4 TRAIN_BATCH_SIZE=2 SP_SIZE=2 bash run_verl.sh ahc039

# Validation (1 epoch)
TOTAL_EPOCHS=1 SP_SIZE=2 bash run_verl.sh ahc039

# Full training
TOTAL_EPOCHS=50 SP_SIZE=2 bash run_verl.sh ahc039
```

## Key Parameters

AHC uses **different hyperparameters** from other tasks:

| Parameter | Value | vs Standard | Notes |
|-----------|-------|-------------|-------|
| Learning rate | **2e-5** | 4e-5 | Lower for stability |
| KL coefficient | **0.01** | 0.1 | More exploration |
| Phase1 max tokens | **22000** | 26000 | Shorter thinking budget |
| Eval timeout | 600s | 530s | Longer for C++ compile+run |
| CPUs per task | 2 | 1 | |
| SP_SIZE | **2** | 1 | Required to avoid OOM |
| Data file | `data/ahc_039_train.parquet` | | |

## Monitoring

- **Metric**: `env/all/raw_score/max` (maximization task)
- **Higher is better**: Higher contest scores = better solutions

## AHC Environment Variables

| Variable | Purpose |
|----------|---------|
| `ALE_BENCH_PROBLEM_ID` | Override problem ID |
| `ALE_BENCH_LOG_DIR` | Override log directory |
| `ALE_BENCH_CACHE` | Cache directory for problem data |
| `ALE_BENCH_DATA` | Data directory for problem definitions |

## Resume Training

```bash
TOTAL_EPOCHS=50 SP_SIZE=2 \
RESUME_DIR=checkpoints/ttt-discover/<experiment> INPLACE=true \
bash run_verl.sh ahc039
```
