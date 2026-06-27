# GPU Mode (GPU Kernel Optimization)

## Overview

Generate and optimize fast CUDA/Triton GPU kernels for various computational tasks.

This task uses TTT-Discover to create high-performance GPU kernels that minimize execution time for specific operations. The model learns to write optimized CUDA or Triton code.

## Evaluation Backend

**Local GPU evaluation only** (Modal cloud support removed):

- **Cost**: $0 (uses local GPU)
- **Speed**: Fast (no network latency)
- **Requires**: 6th GPU for kernel evaluation (GPU 5)
- **Container isolation**: Optional Podman/Docker for crash protection

**Setup Guide**: [LOCAL_EVALUATION_GUIDE.md](./LOCAL_EVALUATION_GUIDE.md)

## Quick Start

```bash
# 1. Optional: Build container image (one-time, 5-10 minutes)
#    Requires 8GB+ disk space - skip if disk limited
cd local_evaluator
bash build_container.sh

# 2. Configure environment (container disabled by default)
export KERNEL_EVAL_GPU="5"
export KERNEL_EVAL_USE_CONTAINER="false"

# 3. Test installation
python local_evaluator/test_evaluator.py

# 4. Run smoke test
bash run.sh config_smoke_test.yaml
```

## Configuration

- **Paper config**: `config_paper.yaml` (50 epochs)
- **Validation config**: `config_validate.yaml` (1 epoch)

**GPU Mode-specific parameters** (differ from standard tasks):
```yaml
kl_penalty_coef: 0.01      # Lower than standard 0.1
phase1_max_tokens: 26000   # Standard value
learning_rate: 4.0e-5      # Standard value
```

## Running

```bash
# Smoke test (5-10 minutes, 16 samples)
bash run.sh config_smoke_test.yaml

# Validation (30-90 minutes, 512 samples, 1 epoch)
bash run.sh config_validate.yaml

# Full training (20-40 hours, 512 samples, 50 epochs)
bash run.sh config_paper.yaml
```

## Monitoring

Track progress in WandB:
- **Metric**: `env/all/raw_score/min` (minimization task)
- **Target**: Minimize kernel runtime (microseconds)
- **Lower is better**: Faster kernel execution

## Performance Notes

- **CPU usage**: Low (kernel execution is GPU-bound)
- **GPU requirements**: 6 GPUs total
  - GPUs 0-3: Inference (TP=4)
  - GPU 4: Training
  - GPU 5: Kernel evaluation
- **Expected runtime**:
  - Validation (1 epoch): ~20-40 minutes
  - Full training (50 epochs): ~30-40 hours
- **Evaluation timeout**: 300 seconds per kernel

## Environment

```bash
conda activate discover_gpumode
```

## Evaluation Details

- Kernels executed on dedicated local GPU (GPU 5 by default)
- Runs 18 correctness tests + 7 performance benchmarks
- Optional container isolation (Podman/Docker) prevents GPU crashes from affecting training
- Tests and benchmarks defined in `lib/bioml/trimul/task.yml` or `lib/mla-decode/task.yml`

## Algorithm Approach

The model learns to generate optimized GPU code using:
- CUDA kernels with shared memory and coalesced access
- Triton language for high-level kernel programming
- Loop tiling and register optimization
- Work distribution strategies across GPU cores

Best results come from understanding GPU architecture (memory hierarchy, warp execution, etc.).
