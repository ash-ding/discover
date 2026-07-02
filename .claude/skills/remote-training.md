---
name: remote-training
description: Launch and monitor training on a remote node (Node 1). Covers SSH setup, common pitfalls, and monitoring.
---

# Remote Node Training

## Node Info

| Node | Hostname | Internal IP | External IP |
|------|----------|-------------|-------------|
| Node 0 (head) | ai-innovation-h100-10-preserve | 10.241.128.30 | 169.62.23.172 |
| Node 1 (worker) | ai-innovation-h100-11-preserve | 10.241.128.16 | 169.62.18.122 |

SSH uses internal IPs. Both nodes must have passwordless SSH configured.

## Pre-flight Checklist

Before launching on Node 1:

1. **Code synced**: `bash scripts/sync_worker.sh`
2. **No stale Ray**: `ssh asherding@10.241.128.16 "ray stop --force 2>/dev/null; pkill -9 -f 'ray|vllm' 2>/dev/null"`
3. **GPUs free**: `ssh asherding@10.241.128.16 "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader"`
4. **No stale Ray on Node 0**: `ray stop --force` (leftover multi-node clusters conflict)

## Launch Training on Node 1

**CRITICAL**: Cannot simply `ssh node "bash run_verl.sh"` — the SSH session starts in `$HOME`, not the project directory. Must use a launcher script approach:

```bash
# Step 1: Create launcher script on Node 1
ssh asherding@10.241.128.16 "cat > /tmp/launch_training.sh << 'SCRIPT'
#!/bin/bash
set -e
cd /workspace/home/asherding/code/discover
source /workspace/home/asherding/.conda/etc/profile.d/conda.sh
conda activate verl_discover
export WANDB_MODE=disabled
export TOTAL_EPOCHS=50
exec bash run_verl.sh circle_packing 'trainer.logger=[\"console\"]'
SCRIPT
chmod +x /tmp/launch_training.sh"

# Step 2: Launch with nohup (survives SSH disconnect)
ssh asherding@10.241.128.16 \
  'nohup bash /tmp/launch_training.sh > /tmp/cp_50epoch_node1.log 2>&1 & echo PID=$!'
```

To change task/epochs, edit the SCRIPT block (e.g., replace `circle_packing` with `ac1`, change `TOTAL_EPOCHS`).

## Common Pitfalls

### 1. WandB API key missing on Node 1
**Symptom**: `wandb.errors.errors.UsageError: No API key configured`
**Fix**: Use `export WANDB_MODE=disabled` AND `'trainer.logger=["console"]'`

### 2. Working directory wrong
**Symptom**: `bash: run_verl.sh: No such file or directory`
**Fix**: Must `cd /workspace/home/asherding/code/discover` inside nohup's bash. Use the launcher script approach above — direct nohup commands lose the `cd`.

### 3. Stale Ray cluster from multi-node test
**Symptom**: `More than N types of tasks seen, this may reduce performance` followed by crash
**Fix**: Before launching single-node training, stop all Ray on BOTH nodes:
```bash
# On Node 0
ray stop --force
# On Node 1
ssh asherding@10.241.128.16 "source ~/.conda/etc/profile.d/conda.sh && conda activate verl_discover && ray stop --force"
```

### 4. GPU occupied by other users
**Symptom**: `Free memory on device cuda:X is less than desired GPU memory utilization`
**Fix**: Check with `nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader` and wait or contact admin.

### 5. SSH quoting hell
Shell quoting through SSH + nohup + bash -c is fragile. The launcher script approach avoids this entirely. Never try to pass complex arguments through nested SSH quoting.

## Monitoring

```bash
# One-time check
bash scripts/monitor.sh 10.241.128.16

# Auto-refresh every 5 minutes
watch -n 300 bash scripts/monitor.sh 10.241.128.16

# Quick step count
ssh asherding@10.241.128.16 "grep -c 'step:' /tmp/cp_50epoch_node1.log 2>/dev/null"

# Tail live log
ssh asherding@10.241.128.16 "tail -f /tmp/cp_50epoch_node1.log"
```

## Stop Training on Node 1

```bash
ssh asherding@10.241.128.16 "pkill -f main_ppo; ray stop --force 2>/dev/null; pkill -9 -f 'ray|vllm' 2>/dev/null"
```

## Checkpoints

Saved on Node 1 at: `/workspace/home/asherding/code/discover/checkpoints/ttt-discover/<experiment_name>/`

To copy checkpoints to Node 0:
```bash
rsync -avz asherding@10.241.128.16:/workspace/home/asherding/code/discover/checkpoints/ ./checkpoints/
```
