# Erdos Minimum Overlap Problem

## Overview

This task tackles the Erdos minimum overlap problem - finding an upper bound for the constant C5 that appears in harmonic analysis.

Find a step function h: [0, 2] -> [0, 1] that **minimizes** the overlap integral:

```
C5 = max_k integral h(x)(1 - h(x+k)) dx
```

**Constraints**: h(x) in [0, 1], integral_0^2 h(x) dx = 1

**Current record**: C5 <= 0.38092. **Target**: C5 <= 0.38080.

## Prerequisites

```bash
conda activate verl_discover
export WANDB_MODE=offline
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-8B
```

## Launch (VERL Colocate Mode)

```bash
# Smoke test (~5 min)
TOTAL_EPOCHS=1 ROLLOUT_N=4 TRAIN_BATCH_SIZE=2 bash run_verl.sh erdos

# Validation (1 epoch)
TOTAL_EPOCHS=1 bash run_verl.sh erdos

# Full training
TOTAL_EPOCHS=50 bash run_verl.sh erdos
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 4e-5 | Standard |
| KL coefficient | 0.1 | Standard |
| Eval timeout | 1100s | Longer than default |
| CPUs per task | 1 | |
| Data file | `data/erdos_min_overlap_train.parquet` | |

## Monitoring

- **Metric**: `env/all/raw_score/min` (**minimization** task)
- **Target**: C5 <= 0.38080
- **Lower is better**: Smaller C5 = tighter upper bound

## Discretization

The continuous function h is discretized as n_points samples over [0, 2]:
- dx = 2.0 / n_points
- 0 <= h[i] <= 1 for all i
- sum(h) * dx = 1
- Evaluation: C5 = max(np.correlate(h, 1-h, mode="full") * dx)

## Resume Training

```bash
TOTAL_EPOCHS=50 RESUME_DIR=checkpoints/ttt-discover/<experiment> INPLACE=true bash run_verl.sh erdos
```
