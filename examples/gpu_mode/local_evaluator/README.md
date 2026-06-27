# Local GPU Kernel Evaluator

Container-isolated local evaluation for GPU kernels using **Podman (preferred)** or Docker, replacing Modal for cost-free operation.

## Features

- ✅ **Container isolation** - GPU crashes don't affect training (Podman or Docker)
- ✅ **Automatic recovery** - GPU reset and retry on failure
- ✅ **Never crashes training** - Returns penalty reward instead of raising exceptions
- ✅ **Cost-free** - No Modal API costs
- ✅ **Full compatibility** - Runs same 18 tests + 7 benchmarks as Modal
- ✅ **Auto-detection** - Automatically prefers Podman > Docker > subprocess

## Quick Start

### 1. Build Container Image

```bash
cd examples/gpu_mode/local_evaluator
bash build_container.sh  # Auto-detects Podman or Docker
```

**Storage**: Images are stored in `/workspace/.../discover/.podman_storage/` (404GB available) to avoid disk space issues on root partition (only 8GB free). See [BUILD_INSTRUCTIONS.md](./BUILD_INSTRUCTIONS.md) for details.

This builds `gpu-kernel-evaluator:latest` with:
- CUDA 12.1
- Python 3.11
- PyTorch 2.11.0
- Triton 3.3.1

**Runtime Detection**: The script automatically uses **Podman** if available (preferred), falls back to Docker.

### 2. Configure Environment

```bash
# Enable local evaluation
export GPU_EVAL_MODE="local"          # local | modal | hybrid

# Local evaluation settings
export KERNEL_EVAL_GPU="5"            # GPU ID for kernel evaluation
export KERNEL_EVAL_TIMEOUT="1200"     # Timeout in seconds (20 minutes)
export KERNEL_EVAL_RETRIES="2"        # Retry count on GPU crash
export KERNEL_EVAL_USE_CONTAINER="true"  # Use container isolation (auto-detects Podman/Docker)
# Deprecated: KERNEL_EVAL_USE_DOCKER (still works for backward compatibility)
```

### 3. Run Training

```bash
cd examples/gpu_mode
bash run.sh config_smoke_test.yaml
```

## Evaluation Modes

### Mode 1: Local Only (`GPU_EVAL_MODE=local`)

All evaluations run locally on GPU 5 (or specified GPU).

**Pros:**
- $0 cost
- Fast (no network latency)

**Cons:**
- Requires 6th GPU
- Need to build Docker image

**Use for:**
- Smoke tests
- Validation runs
- Development

### Mode 2: Modal Only (`GPU_EVAL_MODE=modal`)

All evaluations run on Modal (original behavior).

**Pros:**
- Standardized H100/H200 hardware
- Professional benchmark infrastructure
- No local GPU needed

**Cons:**
- $1-2 per smoke test
- $30-50 per validation
- $1500-2500 per full training

**Use for:**
- Final benchmarking
- Reproducible results

### Mode 3: Hybrid (`GPU_EVAL_MODE=hybrid`)

Try local first, fallback to Modal on failure.

**Pros:**
- Best of both worlds
- Most cost-effective

**Cons:**
- Need both local GPU and Modal

**Use for:**
- Production training

## Architecture

```
Training Process (GPU 0-4)
    ↓
LocalKernelEvaluator
    ↓
Docker Container (GPU 5)
    ↓ eval_worker.py
Load task.yml → Run 18 tests → Run 7 benchmarks → Return geom_mean
    ↓
Result or Penalty (-1,000,000)
```

## File Structure

```
local_evaluator/
├── __init__.py         # Package exports
├── evaluator.py        # LocalKernelEvaluator class
├── eval_worker.py      # Worker script (runs in container)
├── Dockerfile          # Container definition
├── build_docker.sh     # Docker build script
└── README.md           # This file
```

## How It Works

### 1. Isolation Strategy

**Docker Container:**
- Runs with `--gpus device=5` (only GPU 5 visible)
- `--network none` (no network access)
- `--memory 32g` and `--cpus 4` (resource limits)
- Auto-removed after run (`--rm`)

**Benefits:**
- GPU crash → container dies, host unaffected
- Clean environment every run
- Resource limits prevent runaway processes

### 2. Evaluation Flow

```python
# 1. Write submission code to temp file
tmpdir/submission.py

# 2. Create config.json
{
  "submission_file": "tmpdir/submission.py",
  "task_name": "trimul",
  "gpu_type": "H100",
  "result_file": "tmpdir/result.json"
}

# 3. Start Docker container
docker run \
  --gpus device=5 \
  -v tmpdir:/workspace/data \
  -v lib:/workspace/lib:ro \
  gpu-kernel-evaluator:latest \
  /workspace/data/config.json

# 4. Inside container: eval_worker.py
- Load task.yml (18 tests + 7 benchmarks)
- Import submission.py
- Run all tests (fail fast if any fails)
- Run all benchmarks (100 iterations each)
- Compute geometric mean
- Write result.json

# 5. Read result.json
{
  "success": true,
  "score_us": 1234.56,
  "error": null
}

# 6. Convert to reward
reward = 1500 / 1234.56  # TriMul scale
```

### 3. Fault Tolerance

**Level 1: Timeout Protection**
- 1200s timeout (configurable)
- Container killed if exceeded
- Returns penalty reward

**Level 2: GPU Crash Recovery**
```python
for attempt in range(max_retries):
    try:
        result = run_evaluation()
        if success or not gpu_crash:
            return result
        # GPU crash detected
        recover_gpu()  # nvidia-smi --gpu-reset
        sleep(2)
    except Exception:
        continue
```

**Level 3: Never Raise Exception**
```python
try:
    result = evaluate(code)
except Exception as e:
    # Never let exception propagate to training
    return penalty_reward
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_EVAL_MODE` | `modal` | Evaluation backend: `local`, `modal`, or `hybrid` |
| `KERNEL_EVAL_GPU` | `5` | GPU device ID for evaluation |
| `KERNEL_EVAL_TIMEOUT` | `1200` | Timeout in seconds (20 minutes) |
| `KERNEL_EVAL_RETRIES` | `2` | Retry count on GPU crash |
| `KERNEL_EVAL_USE_DOCKER` | `true` | Use Docker isolation (vs subprocess) |

## Troubleshooting

### Docker image not found

```bash
cd examples/gpu_mode/local_evaluator
bash build_docker.sh
```

### Docker not available

Evaluator automatically falls back to subprocess mode (less safe).

To enable Docker:
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Enable GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### GPU 5 not found

Check available GPUs:
```bash
nvidia-smi --query-gpu=index,name --format=csv
```

Set different GPU:
```bash
export KERNEL_EVAL_GPU="6"  # Use GPU 6 instead
```

### Evaluation always returns penalty

Check worker logs:
```bash
# Run Docker manually to see logs
docker run --rm --gpus device=5 \
  -v $(pwd)/test_data:/workspace/data \
  -v $(pwd)/../lib:/workspace/lib:ro \
  gpu-kernel-evaluator:latest \
  /workspace/data/config.json
```

### Worker crashes without result

Likely GPU crash. Check:
```bash
# GPU status
nvidia-smi -i 5

# GPU errors
dmesg | grep -i "gpu\|nvidia\|cuda"

# Reset GPU
sudo nvidia-smi --gpu-reset -i 5
```

## Performance

### Evaluation Time

Single kernel evaluation:
- 18 tests: ~2 seconds
- 7 benchmarks (100 runs each): ~8 seconds
- **Total: ~10 seconds per kernel**

One epoch (512 samples):
- 512 kernels × 10s = **~85 minutes**

### Cost Comparison

| Config | Modal Cost | Local Cost | Savings |
|--------|-----------|------------|---------|
| Smoke (16 samples, 1 epoch) | $1-2 | $0 | 100% |
| Validate (512 samples, 1 epoch) | $30-50 | $0 | 100% |
| Full (512 samples, 50 epochs) | $1500-2500 | $0 | 100% |

**Requirement:** 6th GPU dedicated to kernel evaluation.

## Integration

The evaluator is automatically used when:

```bash
export GPU_EVAL_MODE="local"
cd examples/gpu_mode
bash run.sh config_smoke_test.yaml
```

No code changes needed - `GpuModeRewardEvaluator` routes to local backend based on environment variable.

## Development

### Testing Evaluator

```python
from examples.gpu_mode.local_evaluator import LocalKernelEvaluator

evaluator = LocalKernelEvaluator(gpu_id=5, use_docker=True)

code = """
@triton.jit
def my_kernel(...):
    # Your kernel code
"""

result = evaluator.evaluate(code, task_name="trimul")
print(result)
# {'success': True, 'score_us': 1234.56, 'error': None}
```

### Running Worker Directly

```bash
# Create test config
cat > /tmp/config.json << EOF
{
  "submission_file": "/tmp/submission.py",
  "task_name": "trimul",
  "gpu_type": "H100",
  "result_file": "/tmp/result.json"
}
EOF

# Write test kernel
cat > /tmp/submission.py << EOF
@triton.jit
def custom_kernel(data):
    # Trivial kernel for testing
    return data
EOF

# Run worker
python eval_worker.py /tmp/config.json

# Check result
cat /tmp/result.json
```

## Future Enhancements

Potential improvements:

1. **Benchmark caching** - Cache benchmark results for identical kernels
2. **Parallel evaluation** - Run multiple evaluations concurrently
3. **Warmup optimization** - Reuse warmed-up GPU state
4. **Result database** - Store all evaluations for analysis
5. **A/B testing** - Compare local vs Modal results

## License

Same as TTT-Discover project.
