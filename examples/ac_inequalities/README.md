# AC Inequalities (Autocorrelation Inequalities)

## Overview

This task optimizes autocorrelation inequalities - a mathematical problem in harmonic analysis. The goal is to find sequences of non-negative numbers that minimize certain autocorrelation-based evaluation functions.

## Task Types

### AC1 (First AC Inequality)
- **Metric**: `env/all/raw_score/max` (maximization)
- **Evaluation**: Based on ratio `2n * max(b) / sum(a)²`
- **Goal**: Find sequences that maximize this ratio

### AC2 (Second AC Inequality)  
- **Metric**: `env/all/raw_score/max` (maximization)
- **Evaluation**: Based on L2/L1/Linf norms of autocorrelation
- **Goal**: Find sequences that maximize the lower bound C

## Running

```bash
# Quick validation (1 epoch, AC1)
bash run.sh validate ac1

# Full training (50 epochs, AC1)
bash run.sh full ac1

# Quick validation (1 epoch, AC2)
bash run.sh validate ac2

# Full training (50 epochs, AC2)
bash run.sh full ac2
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

- **CPUs per task**: 2
- **Timeout**: 1100s
- **Search budget**: Configurable in prompt

## Output

Logs to `tinker_log/ac{1,2}-{paper,validate}/`

Monitor `env/all/raw_score/max` in WandB - higher is better.
