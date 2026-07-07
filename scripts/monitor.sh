#!/usr/bin/env bash
# Monitor training progress on local or remote node.
#
# Usage:
#   bash scripts/monitor.sh                           # Monitor local node
#   bash scripts/monitor.sh 10.241.128.16              # Monitor remote node
#   bash scripts/monitor.sh 10.241.128.16 /tmp/cp.log  # Monitor specific log file
#   watch -n 60 bash scripts/monitor.sh 10.241.128.16  # Auto-refresh every 60s

NODE=${1:-local}
LOG_FILE=${2:-/tmp/cp_50epoch_node1.log}

run_cmd() {
    if [ "$NODE" = "local" ]; then
        eval "$1"
    else
        ssh -o ConnectTimeout=5 asherding@${NODE} "$1" 2>/dev/null
    fi
}

echo "========================================="
echo "Training Monitor — $(date)"
echo "Node: ${NODE}"
echo "========================================="

# Check if process is alive
PROCS=$(run_cmd "ps aux | grep main_ppo | grep -v grep | wc -l")
if [ "$PROCS" -gt 0 ]; then
    echo "Status: RUNNING (${PROCS} process(es))"
else
    echo "Status: NOT RUNNING"
fi

# Show completed steps
echo ""
echo "--- Completed Steps ---"
STEPS=$(run_cmd "grep 'step:' ${LOG_FILE} 2>/dev/null" | while IFS= read -r line; do
    step=$(echo "$line" | grep -oP 'step:\K\d+' | head -1)
    score_mean=$(echo "$line" | grep -oP 'critic/score/mean:\K[\d.eE+-]+')
    score_max=$(echo "$line" | grep -oP 'critic/score/max:\K[\d.eE+-]+')
    time=$(echo "$line" | grep -oP 'timing_s/step:\K[\d.]+')
    loss=$(echo "$line" | grep -oP 'actor/pg_loss:np\.float64.\K[^)]+')
    printf "step=%2s | %6.0fs | mean=%.6f | max=%.6f | loss=%s\n" \
        "$step" "$time" "$score_mean" "$score_max" "$loss"
done)

if [ -z "$STEPS" ]; then
    echo "(no steps completed yet)"
else
    echo "$STEPS"
fi

# GPU status
echo ""
echo "--- GPU Usage ---"
run_cmd "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader" | head -8

# Disk
echo ""
echo "--- Disk ---"
run_cmd "df -h /workspace | tail -1"
echo "========================================="
