---
name: start-vllm
description: |
  Start, restart, or verify the vLLM inference server for TTT-Discover experiments.
  Use this skill whenever the user wants to start vLLM, launch the inference server, restart
  after a crash, or check if vLLM is running. Also use when an experiment fails with connection
  errors to localhost:8888 — that means vLLM isn't up. vLLM takes 3-5 minutes to initialize
  (model loading, FlashInfer JIT compilation, multi-GPU NCCL setup), so patience and proper
  waiting are critical. Declaring timeout before 5 minutes is the most common false-failure.
---

# Start vLLM Server

vLLM initialization is slow (3-5 min) because it JIT-compiles FlashInfer GPU kernels and sets up NCCL for multi-GPU communication. Cutting the wait short leads to false "server not responding" reports.

## Instructions

### Step 1: Check current state

```bash
curl -s http://localhost:8888/v1/models >/dev/null 2>&1 && echo "ALREADY RUNNING" || echo "NOT RUNNING"
```

If already running, ask the user whether to keep it or restart. To restart:
```bash
bash stop_vllm.sh
sleep 5
```

Before starting, verify GPUs are clear — use the `gpu-experiment` skill or quickly run:
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
```

### Step 2: Launch

```bash
bash start_vllm.sh
```

The script sets `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` automatically — without this, LoRA hot-reload fails during training. Override defaults with env vars if needed:
```bash
TENSOR_PARALLEL=2 GPU_MEMORY_UTIL=0.8 bash start_vllm.sh
```

Run in background — vLLM is a persistent server process.

### Step 3: Wait up to 5 minutes

```bash
for i in $(seq 1 60); do
    sleep 5
    if curl -s http://localhost:8888/v1/models >/dev/null 2>&1; then
        echo "vLLM ready after $((i*5))s"
        curl -s http://localhost:8888/v1/models | python3 -c "import sys,json; print('Model:', json.load(sys.stdin)['data'][0]['id'])"
        break
    fi
    [ $((i % 6)) -eq 0 ] && echo "Still starting... ($((i*5))s / 300s)"
done
```

### Step 4: Handle failure

If timeout reached, check the startup logs for these common patterns:

| Error pattern | Cause | Fix |
|---------------|-------|-----|
| `Free memory on device` | GPU not cleared | Kill stale processes, retry |
| `Ninja build failed` | FlashInfer cache corrupt | `rm -rf ~/.cache/flashinfer/*`, retry |
| `Address already in use` | Port 8888 occupied | `lsof -ti :8888 \| xargs kill -9` |
| `libcudart.so not found` | Wrong conda env | Activate discover_math (most complete) |

For deeper diagnosis, use the `troubleshoot-vllm` skill.
