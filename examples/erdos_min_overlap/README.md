# Erdős Minimum Overlap Problem

## Overview

This task tackles the Erdős minimum overlap problem - finding an upper bound for the constant C₅ that appears in harmonic analysis.

## Problem Statement

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

```
C₅ = max_k ∫ h(x)(1 - h(x+k)) dx
```

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Lower is better** - smaller C₅ values provide tighter upper bounds on the Erdős constant.

## Current Record

- Paper record: C₅ ≤ 0.38092
- Target goal: C₅ ≤ 0.38080

## Running

```bash
# Quick validation (1 epoch)
bash run.sh validate

# Full training (50 epochs)
bash run.sh full
```

## Configuration

- **Paper config**: `config_paper.yaml` - 50 epochs, full training
- **Validation config**: `config_validate.yaml` - 1 epoch, quick test

Key parameters (from paper Table 9):
- group_size: 64
- groups_per_batch: 8
- num_epochs: 50 (paper) / 1 (validate)
- learning_rate: 4e-5
- kl_penalty_coef: 0.1

## Environment

- **CPUs per task**: 1
- **Timeout**: 1100s
- **Search budget**: 1000s in code

## Discretization

The continuous function h is discretized as n_points samples over [0, 2]:
- dx = 2.0 / n_points
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

## Output

Logs to `tinker_log/erdos-{paper,validate}/`

Monitor `env/all/raw_score/min` in WandB - **lower is better** (minimization task).
