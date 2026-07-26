# Triton Kernel Evaluation Pipeline

Complete flow of how a Triton kernel submission is evaluated in the GPU Mode task.

## Architecture Overview

```
Training Node (Node 0)                    Eval Server (Node 1)
┌──────────────────────┐                  ┌──────────────────────────────┐
│  VERL Agent Loop     │                  │  eval_server.py (Flask)      │
│  (verl_reward.py)    │   HTTP POST      │                              │
│                      │ ───────────────> │  SharedEvalPool              │
│  submission_code +   │   /evaluate      │  ├─ queue.Queue              │
│  task_name           │                  │  ├─ Worker Thread (GPU 0)    │
│                      │ <─────────────── │  ├─ Worker Thread (GPU 1)    │
│  {score, success,    │   JSON response  │  ├─ Worker Thread (GPU 2)    │
│   error, ...}        │                  │  └─ Worker Thread (GPU 3)    │
└──────────────────────┘                  └──────────────────────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────────────┐
                                          │  PooledKernelEvaluator   │
                                          │  (per-GPU instance)      │
                                          │                          │
                                          │  Podman Container        │
                                          │  ┌────────────────────┐  │
                                          │  │  eval_worker.py    │  │
                                          │  │  └─ libkernelbot   │  │
                                          │  │     └─ run_config() │  │
                                          │  └────────────────────┘  │
                                          └──────────────────────────┘
```

## Component Details

### 1. Eval Server (`examples/gpu_mode/eval_server.py`)

Flask HTTP server on Node 1 (port 8890). Accepts POST requests with `{code, task_name, timeout}`.

**SharedEvalPool**: Maintains a `queue.Queue` and one worker thread per GPU. Each thread pulls tasks when idle, ensuring fair GPU utilization without explicit scheduling.

### 2. PooledKernelEvaluator (`examples/gpu_mode/local_evaluator/evaluator.py`)

One instance per GPU. Manages container lifecycle and health tracking.

**Container runtime** (auto-detected, priority order):
1. Podman (preferred)
2. Docker
3. Subprocess (no isolation, fallback)

**Container image**: `localhost/gpu-kernel-evaluator:latest`

**Container setup**:
```bash
podman run --rm \
  --device nvidia.com/gpu=$GPU_ID \
  --pids-limit=-1 \
  --shm-size=16g \
  -v $TMPDIR:/workspace/data \
  localhost/gpu-kernel-evaluator:latest \
  python eval_worker.py --config /workspace/data/config.json
```

**Health-aware recovery**: Each GPU tracks consecutive failures. On crash (`gpu_crash=true`), the evaluator transitions to RECOVERING state, restarts the container, and retries. After too many consecutive failures, the GPU is marked FAILED and removed from the pool.

**Data exchange**: Uses a temp directory mounted at `/workspace/data`:
- Input: `submission.py` (the kernel code), `config.json` (task name, GPU type, paths)
- Output: `result.json` (score, success, error, benchmark data)

### 3. Eval Worker (`examples/gpu_mode/local_evaluator/eval_worker.py`)

Runs inside the container. Entry point that calls `libkernelbot.run_eval.run_config()`.

Two-phase evaluation:
1. **Test phase**: correctness verification (all 7 test cases must pass)
2. **Benchmark phase**: timing measurement (only runs if all tests pass)

Test timeout is capped at `max(timeout // 3, 60)` seconds. The remaining time budget goes to benchmarking.

Returns:
```json
{
  "success": true/false,
  "score_us": 1234.5,     // benchmark score in microseconds (lower = better), or penalty
  "error": "...",          // error message if failed
  "benchmark_count": 7,   // number of benchmark runs completed
  "gpu_crash": false       // whether GPU crashed (triggers container restart)
}
```

### 4. libkernelbot Eval Logic (`examples/gpu_mode/lib/bioml/trimul/eval.py`)

The core evaluation library. Handles test execution and benchmarking.

#### Test Phase: `run_testing()`

Runs all test cases **serially** via `multiprocessing.Pool(1)` (single worker subprocess).

Each test (`_run_single_test()`) executes in a subprocess:
1. Write `submission.py` to a temp directory
2. `importlib.import_module("submission")` to load the kernel
3. Call `submission.custom_kernel(data)` with generated input tensors
4. Compare output against `reference.ref_kernel(data)` using `torch.allclose`
5. Return pass/fail with error details

**Why subprocess?** Triton compilation can corrupt the process (segfaults, CUDA errors). Running each test in a subprocess isolates failures and prevents one bad kernel from killing the evaluator.

#### Benchmark Phase: `run_benchmarking()` (only if all tests pass)

Runs timing benchmarks on all test configurations:
1. Generate input data
2. Warm-up runs
3. Timed runs with `torch.cuda.Event` timing
4. Report median time in microseconds

The final score is the **worst-case (max) time** across all test configurations.

### 5. Reference Implementation (`examples/gpu_mode/lib/bioml/trimul/reference.py`)

PyTorch `TriMul` module used as ground truth:
```python
# Simplified core operation:
left  = self.ln_left(x)                      # LayerNorm
right = self.ln_right(x)                      # LayerNorm
left  = left.view(B, N, K, D)                 # reshape
right = right.view(B, N, K, D)                # reshape
out   = einsum('...ikd,...jkd->...ijd', left, right)  # contraction
out   = F.sigmoid(self.g) * out + (1 - F.sigmoid(self.g)) * x  # gated residual
```

## Test Configurations (TriMul)

Defined in `task.py` as `TestSpec` entries. 7 test cases with varying dimensions:

| # | seqlen | batch_size | dim | hidden_dim | Notes |
|---|--------|-----------|-----|-----------|-------|
| 1 | 256    | 1         | 128 | 128       | Small, power-of-2 dim |
| 2 | 512    | 1         | 128 | 128       | Medium |
| 3 | 256    | 2         | 128 | 128       | Batched |
| 4 | 512    | 2         | 128 | 128       | Batched medium |
| 5 | 256    | 1         | 384 | 128       | Non-power-of-2 dim |
| 6 | 512    | 1         | 384 | 128       | Non-power-of-2 dim, medium |
| 7 | 1024   | 1         | 384 | 128       | Non-power-of-2 dim, large |

Cases 5-7 (`dim=384`) are critical: `next_power_of_2(384) = 512`, causing 128 padding positions in Triton block operations. Kernels must mask padded elements in variance/reduction operations.

## Triton Compilation Pipeline

The compilation from Python to GPU binary is **entirely CPU-bound**. The GPU is only used for kernel execution.

```
Python source code (@triton.jit decorated)
    │
    ▼
Triton Frontend (Python AST → Triton IR)     ← CPU
    │
    ▼
Triton Optimizer (Triton IR optimization)     ← CPU
    │
    ▼
LLVM Backend (Triton IR → LLVM IR → PTX)     ← CPU
    │
    ▼
ptxas (PTX assembly → CUBIN binary)           ← CPU (NVIDIA tool)
    │
    ▼
CUBIN loaded to GPU for execution             ← GPU (first use only)
```

**Compilation caching**: Triton caches compiled kernels by default (`~/.triton/cache`). However, each test runs in a **separate subprocess** (`multiprocessing.Pool`), so cache sharing within a single evaluation depends on filesystem persistence. First-time compilation of a kernel can take 5-30 seconds depending on complexity.

## Timing Breakdown by Failure Type

Based on measured data from 2-step advisor Step 0 (357 evals):

| Failure Type | Typical Time | What Happens |
|-------------|-------------|-------------|
| Import/syntax error | ~2-3s | `import submission` fails immediately |
| Missing `@triton.jit` | ~2-3s | No kernel found, fast abort |
| Triton compile error | ~5-10s | Compilation starts but fails (type errors, invalid ops) |
| Runtime error (CUDA) | ~4-5s | Kernel compiles but crashes on execution |
| Correctness failure | ~170-180s | All 7 tests run, output doesn't match reference |
| Success (all tests pass) | ~180-200s | All 7 tests + benchmark phase |

The bimodal distribution (fast failures ~4s vs slow failures ~177s) is the key driver of eval wall time. A batch with 26% correctness failures spends 92.6% of GPU time on those failures.
