# Claude Replay Framework

Universal framework for replaying TTT-Discover experiments with Claude Opus 4.6.

## Quick Start

### Erdos Min Overlap

```bash
# Single step
TASK=erdos STEP=1 CONCURRENCY=64 python scripts/replay_claude.py
TASK=erdos STEP=1 python scripts/evaluate_claude.py

# Full pipeline (steps 1-50, skip 43)
for step in $(seq 1 50); do
    [ "$step" = "43" ] && continue
    TASK=erdos STEP=$step CONCURRENCY=64 python scripts/replay_claude.py
    TASK=erdos STEP=$step python scripts/evaluate_claude.py
done
```

### GPU Mode (trimul)

```bash
# With remote eval server
GPU_EVAL_SERVER=http://10.241.128.30:8890 TASK=gpu_mode STEP=1 python scripts/replay_claude.py
GPU_EVAL_SERVER=http://10.241.128.30:8890 TASK=gpu_mode STEP=1 python scripts/evaluate_claude.py

# With local GPU
TASK=gpu_mode STEP=1 KERNEL_EVAL_GPU=0 python scripts/replay_claude.py
TASK=gpu_mode STEP=1 KERNEL_EVAL_GPU=0 python scripts/evaluate_claude.py
```

## Prerequisites

1. **Prompt extraction**: Extract PUCT parent states from Qwen3 rollouts:
   ```bash
   # Create checkpoints/{task}_prompts_all_steps.json
   python scripts/extract_prompts.py --task erdos --checkpoint checkpoints/ttt-discover/erdos-...
   ```

2. **Anthropic Vertex AI credentials**:
   ```bash
   export CLAUDE_PROJECT_ID=itpc-gcp-ai-eng-claude
   export CLAUDE_REGION=us-east5
   # Ensure gcloud auth is configured
   ```

3. **Dependencies**:
   ```bash
   pip install anthropic numpy  # Core
   pip install requests torch triton  # For GPU mode
   ```

## Environment Variables

### Replay Script (`replay_claude.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TASK` | *required* | Task name (`erdos`, `gpu_mode`, etc.) |
| `STEP` | `1` | Training step to replay (1-50) |
| `CONCURRENCY` | `64` | Concurrent API calls |
| `ROLLOUTS_PER_PROMPT` | `64` | Rollouts per PUCT parent state |
| `CLAUDE_PROJECT_ID` | `itpc-gcp-ai-eng-claude` | GCP project ID |
| `CLAUDE_REGION` | `us-east5` | Vertex AI region |

### Evaluation Script (`evaluate_claude.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TASK` | *required* | Task name (must match replay) |
| `STEP` | `1` | Training step to evaluate |
| `EVAL_TIMEOUT` | `120` | Hard timeout per rollout (seconds) |
| `EVAL_WORKERS` | `32` | Parallel evaluation workers |

**GPU Mode specific**:
| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_EVAL_SERVER` | *(empty)* | Remote eval server URL |
| `KERNEL_EVAL_GPU` | `0` | GPU ID for local eval |

## File Structure

```
checkpoints/
├── erdos_prompts_all_steps.json          # Input: extracted PUCT prompts
├── claude_erdos_step1.jsonl              # Output: rollout data
├── claude_erdos_step1_scored.jsonl       # Output: evaluation results
├── gpu_mode_prompts_all_steps.json       # Input: GPU mode prompts
├── claude_gpu_mode_step1.jsonl           # Output: GPU mode rollouts
└── claude_gpu_mode_step1_scored.jsonl    # Output: GPU mode scores
```

## Output Format

### Rollout File (`claude_{task}_step{N}.jsonl`)

Each line contains:
```json
{
  "prompt_idx": 0,           // Which PUCT parent state (0-7)
  "rollout_idx": 0,          // Which rollout for this prompt (0-63)
  "output": "...",           // Full Claude output (thinking + code)
  "code": "...",             // Extracted Python code
  "p1_tokens": 12500,        // Phase 1 output tokens
  "p2_tokens": 0,            // Phase 2 output tokens (0 if not needed)
  "phase2": false,           // Whether Phase 2 was triggered
  "p1_stop": "end_turn",     // Phase 1 stop reason
  "has_run": true,           // Whether code contains `def run()`
  "time": 8.2,               // Generation time (seconds)
  "output_len": 15000,       // Output character count
  "code_len": 10000          // Code character count
}
```

### Scored File (`claude_{task}_step{N}_scored.jsonl`)

Adds evaluation results:
```json
{
  ... (all rollout fields),
  "score": 2.624,            // Task-specific score (reward for erdos, TFLOPs for gpu_mode)
  "eval_status": "c5=0.381"  // Evaluation status or error message
}
```

## System Prompts

The replay script uses task-specific Claude system prompts matching the TTT Advisor
codebase exactly. These are passed via the `system` parameter in the Claude API call.

| Task | System Prompt Summary |
|------|----------------------|
| Erdos | Expert in harmonic analysis and numerical optimization, Erdős min overlap |
| GPU Mode | Expert Triton kernel engineer, requires @triton.jit |
| (others) | Generic expert problem solver (fallback) |

The system prompts guide Claude's approach and strategy for each task, while the
user message contains the full problem description and state context extracted from
Qwen3 rollouts.

## Two-Phase Generation

The replay script uses a two-phase approach matching Qwen3's pipeline:

**Phase 1** (max_tokens=25300):
- Claude generates freely (analysis + code)
- If completes naturally → done
- If hits token limit BUT has complete code → done

**Phase 2** (max_tokens=6700):
- Triggered ONLY if: Phase 1 hit limit AND no `def run()` found
- Sends continuation request asking for code completion
- Concatenates Phase 2 output to Phase 1

This mirrors Qwen3's training setup where:
- Phase 1 budget = prompt_tokens → 26000 (thinking)
- Phase 2 budget = remaining context → 32768 (answer)

## Task-Specific Notes

### Erdos Min Overlap
- **Metric**: Reward = 1/C₅ (higher is better)
- **Goal**: Minimize C₅ bound (target: 0.3808)
- **Evaluation**: Sandboxed Python execution with 120s timeout
- **Success criteria**: Valid h∈[0,1], ∫h dx = 1, returns (h_values, c5, n_points)

### GPU Mode (trimul)
- **Metric**: TFLOPs (higher is better)
- **Goal**: Optimize matrix multiplication kernel performance
- **Evaluation**: Remote server or local GPU with Triton
- **Success criteria**: Valid Triton kernel, returns TFLOPs measurement

## Adding New Tasks

1. **Extract prompts** from Qwen3 training rollouts:
   ```python
   # scripts/extract_prompts.py
   # Output: checkpoints/{task}_prompts_all_steps.json
   ```

2. **Add task-specific evaluator** to `evaluate_claude.py`:
   ```python
   def evaluate_mytask_worker(idx, code, timeout, result_queue):
       # Task-specific evaluation logic
       ...
   ```

3. **Update worker selection** in `evaluate_with_timeout()`:
   ```python
   elif task == "mytask":
       worker = evaluate_mytask_worker
   ```

4. **Run replay**:
   ```bash
   TASK=mytask STEP=1 python scripts/replay_claude.py
   TASK=mytask STEP=1 python scripts/evaluate_claude.py
   ```

## Common Issues

### API Rate Limits
- Reduce `CONCURRENCY` (default 64 → 32 or 16)
- Add retry logic with exponential backoff

### Out of Memory (Evaluation)
- Reduce `EVAL_WORKERS` (default 32 → 16 or 8)
- Increase `EVAL_TIMEOUT` if processes are being killed prematurely

### Missing Prompts File
```
Error: Prompt file not found: checkpoints/{task}_prompts_all_steps.json
```
→ Run prompt extraction script first (not implemented yet — extract from Qwen3 rollouts manually)

### Phase 2 Not Triggering
- Check that Phase 1 actually hits `max_tokens` limit
- Verify code extraction regex matches your code format
- Look at `phase2` field in rollout file to confirm

## Performance

**Replay speed** (Claude Opus 4.6, concurrency=64):
- ~8-12 seconds per rollout (includes API latency + generation)
- 512 rollouts (8×64) ≈ 50-80 minutes per step
- 50 steps ≈ 40-65 hours total

**Evaluation speed** (32 workers, 120s timeout):
- Erdos: ~30-40 minutes per step (CPU-bound)
- GPU mode: ~10-20 minutes per step (GPU + network)

**Cost estimate** (Claude Opus 4.6 via Vertex AI):
- Input: ~500 tokens/rollout × 512 = 256K tokens/step
- Output: ~12K tokens/rollout × 512 = 6M tokens/step
- 50 steps ≈ 300M output tokens (~$4,500 at $15/M output tokens)

## Comparison with Test Scripts

Old test scripts (`test_two_phase_claude.py`):
- ❌ Fixed 3 rollouts, single prompt
- ❌ No concurrency
- ❌ No resume support
- ❌ Terminal output only
- ❌ Used `PREFILL` insertion for Phase 2

New production scripts:
- ✅ 512 rollouts (8 prompts × 64)
- ✅ Async concurrency (configurable)
- ✅ Resume from interruption
- ✅ JSONL persistence
- ✅ Clean Phase 2 continuation request
- ✅ Multi-task support
- ✅ Comprehensive metadata
