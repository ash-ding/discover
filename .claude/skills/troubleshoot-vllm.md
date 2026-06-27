---
name: troubleshoot-vllm
description: |
  Diagnose and fix vLLM server failures, training crashes, OOM errors, and LoRA adapter issues
  in TTT-Discover. Use this skill when the user reports any error during training or inference:
  OOM errors, LoRA adapter not loaded, vLLM not responding, training crashes, Ray worker failures,
  CUDA errors, or slow inference. Also use when the user pastes an error traceback and needs help
  understanding it. This is the go-to skill for any "it's broken" situation that the simpler
  start-vllm or gpu-experiment skills can't resolve.
---

# Troubleshooting Decision Tree

Match the symptom, follow the branch. Each fix addresses the root cause — avoid shotgun debugging.

## vLLM won't start

1. **GPU memory occupied**: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`
   - Fix: `pkill -9 -f "vllm|ray::"`, wait 5s, retry
   - Why: crashed runs leave zombie GPU processes that block memory allocation
2. **Port 8888 in use**: `lsof -i :8888`
   - Fix: `lsof -ti :8888 | xargs kill -9`
3. **FlashInfer JIT failure** (error: "Ninja build failed"):
   - Fix: `rm -rf ~/.cache/flashinfer/* ~/.cache/torch_extensions/*`, retry
   - Why: corrupted compilation cache from interrupted startup
4. **Missing CUDA library** (error: "libcudart.so"):
   - Fix: switch to `discover_math` conda env (most complete CUDA deps)

## LoRA adapter errors

- **"LoRA adapter 'lora_v1' not loaded"**: `VLLM_ALLOW_RUNTIME_LORA_UPDATING` must be `true` at vLLM startup. Can't fix without restart — `start_vllm.sh` sets this automatically.
- **Adapter files missing**: check `tinker_log/local_checkpoints/<experiment>/latest_sampler/` for `adapter_config.json` + `adapter_model.safetensors`. If missing, training hasn't saved a checkpoint yet.

## KL penalty OOM

This is the most common training OOM. The `incorporate_kl_penalty` step uses `echo=True` to recompute logprobs for full sequences — each 32K-token sequence creates a 32K x 152K vocab tensor (~9GB). Multiple concurrent requests exhaust GPU memory.

Fixes (in order — each is progressively more impactful but trades off quality):
1. Lower vLLM memory reservation: `GPU_MEMORY_UTIL=0.7 bash start_vllm.sh`
2. Limit concurrent sequences: add `--max-num-seqs 2` in start_vllm.sh
3. Reduce context: set `max_model_len: 16384` in config YAML (truncates long sequences)

## Training step OOM

OOM during forward/backward pass (not KL penalty). Less common because gradient checkpointing is enabled.

Fix: reduce `max_train_seq_len` in config YAML (e.g. 32768 -> 16384). This truncates the prompt prefix while keeping the full response.

## Emergency full reset

When multiple things are broken and you need a clean slate:
```bash
pkill -9 -f "vllm|ray::|python.*train"
sleep 10
nvidia-smi  # verify <100MB per GPU
rm -rf ~/.cache/flashinfer/*
bash start_vllm.sh
```

Wait 5 minutes for vLLM to come back, then retry the experiment.
