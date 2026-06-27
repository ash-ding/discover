# Circle Packing

## Overview

Pack circles in a unit square to maximize the sum of their radii.

This is a classic optimization problem in computational geometry where the goal is to pack non-overlapping circles into a unit square such that the total sum of radii is maximized.

## Variants

- **26 circles** (default): Target sum of radii ≥ 2.636
- **32 circles**: Target sum of radii ≥ 2.940

The problem difficulty increases significantly with more circles due to the exponentially larger search space.

## Configuration

- **Paper config**: `config_paper.yaml` (50 epochs, full training)
- **Validation config**: `config_validate.yaml` (1 epoch, quick test)

## Running

```bash
# Quick validation (1 epoch, 26 circles)
bash run.sh validate

# Full training (50 epochs, 26 circles)
bash run.sh full

# Custom config
export TTT_CONFIG_PATH=path/to/custom_config.yaml
bash run.sh full
```

## Monitoring

Track progress in WandB:
- **Metric**: `env/all/raw_score/max` (maximization task)
- **Target**: ≥ 2.636 (26 circles) or ≥ 2.940 (32 circles)
- **Higher is better**: Larger sum of radii means better packing

## Performance Notes

- **CPU usage**: Moderate (1 CPU per evaluation)
- **GPU memory**: TP=4 inference (GPUs 0-3) + 1 training GPU (GPU 4)
- **Expected runtime**:
  - Validation (1 epoch): ~30-60 minutes
  - Full training (50 epochs): ~40-50 hours

## Environment

```bash
conda activate discover_math
```

## Algorithm Approach

The model learns to generate Python code that implements circle packing algorithms. Common strategies discovered:
- Greedy placement with local optimization
- Simulated annealing or genetic algorithms
- Grid-based initialization with gradient descent
- Constraint satisfaction approaches

Best results typically come from combining multiple algorithmic ideas and parameter tuning.
