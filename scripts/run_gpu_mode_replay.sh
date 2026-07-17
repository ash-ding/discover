#!/bin/bash
# Full GPU mode (trimul) Claude replay pipeline: 50 steps
#
# Usage:
#   nohup bash scripts/run_gpu_mode_replay.sh > /tmp/gpu_mode_replay.log 2>&1 &
#
# Prerequisites:
#   - checkpoints/gpu_mode_prompts_all_steps.json (extracted from checkpoint)
#   - Eval server running on Node 1 GPUs 4-7 (port 8891)

set -e

export TASK=gpu_mode
export CONCURRENCY=64
export ROLLOUTS_PER_PROMPT=64
export GPU_EVAL_SERVER=http://10.241.128.16:8891
export EVAL_TIMEOUT=600
export EVAL_WORKERS=4

TOTAL_STEPS=50

echo "=== GPU Mode (TriMul) Claude Replay Pipeline ==="
echo "Steps: 1-${TOTAL_STEPS}"
echo "Rollouts per prompt: ${ROLLOUTS_PER_PROMPT}"
echo "Concurrency: ${CONCURRENCY}"
echo "Eval server: ${GPU_EVAL_SERVER}"
echo "Eval timeout: ${EVAL_TIMEOUT}s"
echo "Started: $(date)"
echo ""

for step in $(seq 1 ${TOTAL_STEPS}); do
    echo "========================================"
    echo "Step ${step}/${TOTAL_STEPS} - $(date)"
    echo "========================================"

    # Check if already complete
    scored_file="checkpoints/claude_gpu_mode_step${step}_scored.jsonl"
    if [ -f "$scored_file" ]; then
        count=$(wc -l < "$scored_file")
        echo "Step ${step}: scored file exists (${count} entries), skipping"
        continue
    fi

    # Phase 1: Replay with Claude
    echo "--- Replay ---"
    STEP=${step} python scripts/replay_claude.py

    # Phase 2: Evaluate
    echo "--- Evaluate ---"
    STEP=${step} python scripts/evaluate_claude.py

    echo "Step ${step} complete"
    echo ""
done

echo ""
echo "=== Pipeline Complete ==="
echo "Finished: $(date)"
