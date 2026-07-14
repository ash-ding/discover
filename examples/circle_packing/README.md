# Circle Packing

## Overview

Pack circles in a unit square to maximize the sum of their radii.

This is a classic optimization problem in computational geometry where the goal is to pack non-overlapping circles into a unit square such that the total sum of radii is maximized.

## Variants

- **26 circles** (`circle_packing` / `cp26`): Target sum of radii >= 2.636
- **32 circles** (`cp32`): Target sum of radii >= 2.940

## Prerequisites

```bash
conda activate verl_discover
export WANDB_MODE=offline
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-8B
```

## Launch (VERL Colocate Mode)

```bash
# Smoke test (~5 min, 8 samples)
TOTAL_EPOCHS=1 ROLLOUT_N=4 TRAIN_BATCH_SIZE=2 bash run_verl.sh circle_packing

# Validation (1 epoch, 512 samples)
TOTAL_EPOCHS=1 bash run_verl.sh circle_packing

# Full training (50 epochs, 512 samples/step)
TOTAL_EPOCHS=50 bash run_verl.sh circle_packing

# 32 circles variant
TOTAL_EPOCHS=50 bash run_verl.sh cp32
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 4e-5 | Standard |
| KL coefficient | 0.1 | Standard |
| Eval timeout | 530s | Per-sample code execution |
| CPUs per task | 1 | Sandbox evaluation |
| Data file | `data/circle_packing_train.parquet` | |

## Monitoring

- **Metric**: `env/all/raw_score/max` (maximization task)
- **Target**: >= 2.636 (26 circles) or >= 2.940 (32 circles)
- **Higher is better**: Larger sum of radii = better packing

## Resume Training

```bash
# Resume from checkpoint (same directory)
TOTAL_EPOCHS=50 RESUME_DIR=checkpoints/ttt-discover/<experiment> INPLACE=true bash run_verl.sh circle_packing

# Resume into new directory
TOTAL_EPOCHS=50 RESUME_DIR=checkpoints/ttt-discover/<experiment> bash run_verl.sh circle_packing
```
