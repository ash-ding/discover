# TTT-Discover Examples

This directory contains all example tasks for TTT-Discover. Each task demonstrates test-time training for LLM discovery in different domains.

## Available Tasks

### 📐 Mathematics

#### 1. Circle Packing (`circle_packing/`)
- **Domain**: Computational geometry
- **Goal**: Maximize sum of circle radii packed in unit square
- **Metric**: `env/all/raw_score/max` (maximization)
- **Problems**: 26 or 32 circles
- **Paper target**: 2.636 (26 circles), 2.940 (32 circles)

#### 2. AC Inequalities (`ac_inequalities/`)
- **Domain**: Harmonic analysis - Autocorrelation inequalities
- **Goal**: Find sequences maximizing autocorrelation-based bounds
- **Metric**: `env/all/raw_score/max` (maximization)
- **Variants**: AC1, AC2
- **CPU intensive**: 2 CPUs per task

#### 3. Erdős Minimum Overlap (`erdos_min_overlap/`)
- **Domain**: Harmonic analysis - Erdős constant C₅
- **Goal**: Find step functions minimizing overlap integral
- **Metric**: `env/all/raw_score/min` (minimization)
- **Current record**: C₅ ≤ 0.38092
- **Target**: C₅ ≤ 0.38080

### 🖥️ GPU Programming

#### 4. GPU Mode (`gpu_mode/`)
- **Domain**: GPU kernel optimization
- **Goal**: Generate fast CUDA/Triton kernels
- **Metric**: `env/all/raw_score/min` (minimization - runtime)
- **Requires**: Modal account for cloud execution

### 🧬 Biology

#### 5. Denoising (`denoising/`)
- **Domain**: Single-cell RNA sequencing
- **Goal**: Denoise gene expression data
- **Metric**: Custom (requires special processing)
- **Requires**: openproblems patch (see `requirements/denoising/`)

### 🏆 Algorithms

#### 6. AHC - AtCoder Heuristic Contest (`ahc/`)
- **Domain**: Algorithmic problem solving
- **Goal**: Solve competitive programming problems
- **Metric**: `env/all/raw_score/max` (maximization)
- **Requirements**: 
  - Must run in container: `yimjk/ale-bench:cpp20-202301`
  - HPC-grade CPUs recommended
  - See `ahc/setup.md` for details

## Task Categories Summary

| Category | Tasks | Type |
|----------|-------|------|
| **Math** | Circle Packing, AC Inequalities, Erdős | Maximization / Minimization |
| **GPU** | GPU Mode | Minimization (runtime) |
| **Bio** | Denoising | Custom metric |
| **Algo** | AHC | Maximization |

## Running Tasks

Each task follows the same structure:

```bash
cd examples/<task_name>

# Smoke test (ultra-fast, ~5-10 min)
bash run.sh config_smoke_test.yaml

# Quick validation (1 epoch, ~30-60 min)
bash run.sh config_validate.yaml

# Full training (50 epochs, paper config, ~40-60 hours)
bash run.sh config_paper.yaml

# Custom configuration
bash run.sh /path/to/custom_config.yaml
```

**Configuration Comparison:**

| Config | group_size | groups_per_batch | samples/step | Runtime | Purpose |
|--------|------------|------------------|--------------|---------|---------|
| `config_smoke_test.yaml` | 8 | 2 | 16 | ~5-10 min | Pipeline verification |
| `config_validate.yaml` | 64 | 8 | 512 | ~30-60 min | Pre-experiment check |
| `config_paper.yaml` | 64 | 8 | 512 | ~40-60 hrs | Full training |

**Note**: You must specify the configuration file explicitly. The run.sh script requires a config file path as the first argument.

## Configuration Files

Each task contains:
- `config_paper.yaml` - Full 50-epoch training (paper parameters)
- `config_validate.yaml` - 1-epoch quick validation (512 samples/step)
- `config_smoke_test.yaml` - Ultra-fast smoke test (16 samples/step)
- `run.sh` - Runner script
- `env.py` - Environment and reward evaluator
- `prompt.py` (if needed) - Prompt templates
- `README.md` - Task-specific documentation

## Common Parameters (Table 9 from Paper)

All tasks use these default training parameters:

```yaml
group_size: 64                # Completions per prompt
groups_per_batch: 8           # Different prompts per step
num_epochs: 50                # Training epochs (1 for validation)
learning_rate: 4e-5           # Adam learning rate
kl_penalty_coef: 0.1          # KL penalty coefficient
lora_rank: 32                 # LoRA rank
phase1_max_tokens: 26000      # Prompt + thinking budget
temperature: 1.0              # Sampling temperature
```

## Task-Specific Parameter Overrides

Most tasks use standard Table 9 parameters. **Exceptions:**

| Task | phase1_max_tokens | kl_penalty_coef | learning_rate | Reason |
|------|-------------------|-----------------|---------------|--------|
| **AHC** | **22000** | **0.01** | **2.0e-5** | Competitive programming domain |
| **GPU Mode** | 26000 | **0.01** | 4.0e-5 | Kernel optimization task |
| Circle Packing | 26000 | 0.1 | 4.0e-5 | Standard paper config |
| AC Inequalities | 26000 | 0.1 | 4.0e-5 | Standard paper config |
| Erdős Min Overlap | 26000 | 0.1 | 4.0e-5 | Standard paper config |
| Denoising | 26000 | 0.1 | 4.0e-5 | Standard paper config |

**Bold** = differs from standard values. These overrides are empirically optimized for each task domain.

## Task Documentation

Each task has detailed documentation:

- **[Circle Packing](circle_packing/README.md)** - Geometry optimization, 26/32 circles
- **[AC Inequalities](ac_inequalities/README.md)** - Harmonic analysis, AC1/AC2 variants
- **[Erdős Minimum Overlap](erdos_min_overlap/README.md)** - C₅ constant minimization
- **[GPU Mode](gpu_mode/README.md)** - CUDA/Triton kernel optimization
- **[Denoising](denoising/README.md)** - Single-cell RNA-seq denoising
- **[AHC](ahc/README.md)** - AtCoder Heuristic Contest problems

## Hardware Requirements

### Minimum Setup (All Tasks)
- 2× NVIDIA H100 80GB (1 inference, 1 training)

### Paper Configuration
- 5× NVIDIA H100 80GB (4 for TP=4 inference, 1 for training)

### Special Requirements

**CPU-Intensive Tasks** (Math, AHC):
- Circle Packing: Standard CPUs acceptable
- AC Inequalities: 2 CPUs per task
- AHC: **HPC-grade CPUs strongly recommended**
- Erdős: 1 CPU per task

**Container Requirements** (AHC only):
- Must run inside `yimjk/ale-bench:cpp20-202301` container

## Output Structure

All tasks log to `tinker_log/<experiment_name>/`:

```
tinker_log/<experiment_name>/
├── metrics.jsonl              # Per-step scores, rewards, advantages
├── checkpoints.jsonl          # Checkpoint path index
├── config.json               # Training config snapshot
└── train.log                 # Python logs

tinker_log/local_checkpoints/<experiment_name>/
├── state_<step>/             # LoRA weights + optimizer
├── sampler_<step>/           # LoRA weights only
└── latest_sampler/           # Current LoRA (hot-reloaded by vLLM)
```

## Monitoring Progress

Track metrics in WandB:

### Maximization Tasks
- Circle Packing: `env/all/raw_score/max`
- AC Inequalities: `env/all/raw_score/max`
- AHC: `env/all/raw_score/max`

### Minimization Tasks  
- Erdős: `env/all/raw_score/min`
- GPU Mode: `env/all/raw_score/min`

### Special Processing
- Denoising: Requires post-processing (see task README)

## Task Selection Guide

Choose based on your research interest:

- **Geometry/Optimization**: Circle Packing
- **Harmonic Analysis**: AC Inequalities, Erdős
- **Systems/Performance**: GPU Mode
- **Bioinformatics**: Denoising
- **Algorithms**: AHC

All tasks demonstrate the same core TTT-Discover workflow with different reward functions and environments.
