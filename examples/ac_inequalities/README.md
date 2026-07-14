# AC Inequalities (Autocorrelation Inequalities)

## Overview

This task optimizes autocorrelation inequalities - a mathematical problem in harmonic analysis. The goal is to find sequences of non-negative numbers that minimize certain autocorrelation-based evaluation functions.

### Variants

- **AC1**: Based on ratio `2n * max(b) / sum(a)^2` — maximize this ratio
- **AC2**: Based on L2/L1/Linf norms of autocorrelation — maximize the lower bound C

## Prerequisites

```bash
conda activate verl_discover
export WANDB_MODE=offline
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-8B
```

## Launch (VERL Colocate Mode)

```bash
# Smoke test (~5 min)
TOTAL_EPOCHS=1 ROLLOUT_N=4 TRAIN_BATCH_SIZE=2 bash run_verl.sh ac1

# Validation (1 epoch)
TOTAL_EPOCHS=1 bash run_verl.sh ac1

# Full training
TOTAL_EPOCHS=50 bash run_verl.sh ac1    # AC1 variant
TOTAL_EPOCHS=50 bash run_verl.sh ac2    # AC2 variant
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 4e-5 | Standard |
| KL coefficient | 0.1 | Standard |
| Eval timeout | 1100s | Longer than default (CPU-intensive eval) |
| CPUs per task | 2 | Higher than default |
| Data files | `data/ac_inequalities_ac1_train.parquet` | AC2 uses `_ac2_` variant |

## Monitoring

- **Metric**: `env/all/raw_score/max` (maximization for both AC1 and AC2)
- **Higher is better**

## Resume Training

```bash
TOTAL_EPOCHS=50 RESUME_DIR=checkpoints/ttt-discover/<experiment> INPLACE=true bash run_verl.sh ac1
```
