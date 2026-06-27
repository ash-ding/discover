# AHC (AtCoder Heuristic Contest)

## Overview

Solve competitive programming heuristic optimization problems from AtCoder Heuristic Contest.

This task applies TTT-Discover to heuristic contest problems that require finding good (not necessarily optimal) solutions to combinatorial optimization problems under time and resource constraints.

## Special Requirements

**Must run in Docker/Podman container**:
```bash
# Container image
yimjk/ale-bench:cpp20-202301

# Recommended system
# - HPC-grade CPUs strongly recommended
# - High core count helps with parallel search
```

## AHC-Specific Parameters

**Critical**: AHC uses different hyperparameters than other tasks:

```yaml
phase1_max_tokens: 22000      # Lower than standard 26000
kl_penalty_coef: 0.01         # Lower than standard 0.1
learning_rate: 2.0e-5         # Lower than standard 4.0e-5
```

These parameters are optimized for the competitive programming domain and differ from the standard Table 9 paper configuration.

## Configuration

- **Paper config**: `config_paper.yaml` (50 epochs, AHC-specific params)
- **Validation config**: `config_validate.yaml` (1 epoch, AHC-specific params)

## Running

**Inside the container**:
```bash
# Quick validation (1 epoch)
bash run.sh validate

# Full training (50 epochs)
bash run.sh full
```

**Container setup** (see `setup.md` for detailed instructions):
```bash
podman run -it \
  --device nvidia.com/gpu=all \
  --shm-size=16g \
  --pids-limit=-1 \
  -v /workspace:/workspace \
  yimjk/ale-bench:cpp20-202301 \
  bash
```

## Monitoring

Track progress in WandB:
- **Metric**: `env/all/raw_score/max` (maximization task)
- **Target**: Depends on specific contest problem
- **Higher is better**: Higher scores indicate better solutions

## Performance Notes

- **CPU usage**: High (competitive programming often requires CPU-intensive search)
- **GPU memory**: TP=4 inference + 1 training GPU
- **Expected runtime**:
  - Validation (1 epoch): ~40-90 minutes (slower eval than most tasks)
  - Full training (50 epochs): ~60-100 hours
- **Evaluation timeout**: 530 seconds

## Environment

```bash
# Inside container
conda activate discover_ale
```

## Container Configuration

**Important flags**:
- `--pids-limit=-1`: Prevents PID limit errors during multi-process evaluation
- `--shm-size=16g`: Adequate shared memory for Ray workers
- `--device nvidia.com/gpu=all`: GPU access inside container

## Algorithm Approach

The model learns to generate C++ code that implements heuristic algorithms:
- Greedy construction with local search
- Simulated annealing
- Beam search
- Genetic algorithms
- Tabu search
- Hybrid metaheuristics

Best results come from domain-specific insights and efficient implementation in C++.

## Why Different Parameters?

- **Lower `phase1_max_tokens`**: Competitive programming solutions tend to be more concise
- **Lower `kl_penalty_coef`**: Allow more exploration in the heuristic search space
- **Lower `learning_rate`**: More stable training for algorithmic code generation

These adjustments were empirically found to work better for this domain.
