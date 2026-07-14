# Denoising (Single-Cell RNA Sequencing)

## Overview

Denoise gene expression data from single-cell RNA sequencing experiments. The goal is to remove technical noise from scRNA-seq data while preserving true biological signal.

## Prerequisites

```bash
conda activate verl_discover
export WANDB_MODE=offline
export MODEL_PATH=/workspace/home/asherding/models/Qwen3-8B
```

**Special dependency**: Requires a patched version of the `openproblems` library. See `requirements/denoising/README.md` for installation details:

```bash
pip install openproblems[pytorch,rapids]
```

## Launch (VERL Colocate Mode)

```bash
# Smoke test (~5 min)
TOTAL_EPOCHS=1 ROLLOUT_N=4 TRAIN_BATCH_SIZE=2 bash run_verl.sh denoising

# Validation (1 epoch)
TOTAL_EPOCHS=1 bash run_verl.sh denoising

# Full training
TOTAL_EPOCHS=50 bash run_verl.sh denoising
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 4e-5 | Standard |
| KL coefficient | 0.1 | Standard |
| Eval timeout | 530s | |
| CPUs per task | 1 | |
| Data file | `data/denoising_train.parquet` | |

## Monitoring

- **Metric**: `env/all/raw_score/max` (maximization task)
- **Higher is better**: Better denoising quality
- Evaluation involves biological simulations and correlation metrics from the openproblems benchmark

## Performance Notes

- **CPU usage**: High (biological simulations are CPU-intensive)
- Dataset is automatically downloaded/cached on first run

## Resume Training

```bash
TOTAL_EPOCHS=50 RESUME_DIR=checkpoints/ttt-discover/<experiment> INPLACE=true bash run_verl.sh denoising
```
