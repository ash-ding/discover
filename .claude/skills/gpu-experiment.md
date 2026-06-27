---
name: gpu-experiment
description: |
  Pre-flight GPU check, cleanup, and experiment launch for TTT-Discover training runs.
  Use this skill whenever the user wants to run an experiment, start training, launch a task,
  or when they mention GPU problems like "GPU busy", "CUDA out of memory", "process stuck",
  or "nvidia-smi shows something". This skill should also be used proactively before any
  command that will allocate GPU memory — stale processes from previous runs are the #1 cause
  of startup failures and are easily avoided with a quick check.
---

# GPU Pre-flight Check & Experiment Launch

Stale vLLM workers and Ray processes from crashed runs are the most common reason experiments fail to start. A 5-second GPU check prevents minutes of debugging.

## Instructions

### Step 1: Check GPU state

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
```

Any GPU showing >500MB likely has a leftover process. Identify it:
```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
```

### Step 2: Clean up stale processes

Only kill processes after confirming with the user — they might have an intentional job running.

```bash
pkill -9 -f "vllm.entrypoints"
pkill -9 -f "ray::"
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
```

All target GPUs should show <100MB after cleanup.

### Step 3: Verify GPU allocation matches config

Read the experiment's YAML config and check:
- Inference needs GPUs 0 through `inference_tp_size - 1`
- Training needs `training_gpu_id` (must be >= inference_tp_size)
- Total GPUs required = `inference_tp_size + 1`

If the config asks for more GPUs than available, tell the user before they waste time starting.

### Step 4: Launch experiment

```bash
cd examples/<task>
bash run.sh <config_file>.yaml
```

For anything longer than a smoke test (~5 min), run in background and give the user the monitoring command:
```bash
tail -f tinker_log/<experiment_name>/train.log
```

Report what was checked, what was cleaned, and the experiment status.
