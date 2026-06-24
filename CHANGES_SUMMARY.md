# Summary of Changes for Paper Configuration

## Changes Made (2024-06-24)

### 1. Removed Python-level KL Penalty Batching ✓

**File:** `ttt_discover/rl/train.py`

**Before:**
```python
# Batch KL penalty requests to avoid OOM with many trajectories
# Using batch_size=1 (fully sequential) to prevent vLLM internal batching
KL_PENALTY_BATCH_SIZE = 1
base_logprobs_D = []
for i in range(0, len(full_sequence_inputs_D), KL_PENALTY_BATCH_SIZE):
    batch = full_sequence_inputs_D[i:i+KL_PENALTY_BATCH_SIZE]
    batch_results = await asyncio.gather(
        *[
            base_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in batch
        ]
    )
    base_logprobs_D.extend(batch_results)
```

**After:**
```python
base_logprobs_D = await asyncio.gather(
    *[
        base_sampling_client.compute_logprobs_async(sequence_input)
        for sequence_input in full_sequence_inputs_D
    ]
)
```

**Rationale:** Removed debugging workaround. Now matches paper implementation (all KL requests concurrent).

---

### 2. Updated Circle Packing to Paper Configuration ✓

**File:** `examples/circle_packing/env.py`

**Changes:**
| Parameter | Before | After | Notes |
|-----------|--------|-------|-------|
| `max_model_len` | 8192 | **32768** | Paper config |
| `phase1_max_tokens` | 4000 | **26000** | Paper config (Table 9) |
| `experiment_name` | `circle-packing-{n}-fast-validation` | `circle-packing-{n}` | Production naming |

Other parameters already matched paper (group_size=64, groups_per_batch=8, num_epochs=50, kl_penalty_coef=0.1).

---

### 3. Updated README for Production Settings ✓

**File:** `README.md`

**Key changes:**

1. **vLLM gpu_memory_utilization:** 0.70 → **0.90** (paper setting)
2. **Removed max_num_seqs=4** (use vLLM defaults)
3. **Reorganized vLLM startup commands:**
   - Lead with TP=4 (recommended paper config)
   - Moved TP=1/TP=2 to "Alternative configurations"
   - Added startup verification step
4. **Added "Quick Validation" section** — 5-minute test before full run
5. **Added comprehensive "Troubleshooting" section** with 7 common issues:
   - vLLM OOM during KL penalty
   - Training OOM
   - LoRA adapter errors
   - vLLM startup issues
   - Slow rollouts
   - Connection errors
6. **Improved GPU allocation explanation** with runtime estimates

---

## Verification Checklist

- [x] No `KL_PENALTY_BATCH_SIZE` in codebase
- [x] `examples/circle_packing/env.py` shows:
  - `phase1_max_tokens=26000`
  - `max_model_len=32768`
  - `experiment_name="circle-packing-{num_circles}"`
- [x] README shows `gpu_memory_utilization=0.90` in all vLLM examples
- [x] Quick Validation section added
- [x] Troubleshooting section added

---

## How to Run (Quick Reference)

### 1. Start vLLM (TP=4, recommended)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    python -m vllm.entrypoints.openai.api_server \
    --model /path/to/models/Qwen3-8B \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-lora \
    --max-lora-rank 64 \
    --gpu-memory-utilization 0.90 \
    --disable-custom-all-reduce
```

### 2. Launch Circle Packing (paper config)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 WANDB_MODE=offline \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    python -m examples.circle_packing.env --local
```

### 3. Quick Validation (optional, 5 min test)
Temporarily edit `env.py`:
```python
num_epochs=1,
groups_per_batch=2,
```

Run same command as step 2. Should complete in ~3-5 min with no errors.

---

## Files Modified

1. `ttt_discover/rl/train.py` — removed KL batching
2. `examples/circle_packing/env.py` — paper config (max_model_len, phase1_max_tokens, name)
3. `README.md` — updated vLLM commands, added Quick Validation + Troubleshooting

**No breaking changes.** Code is backward compatible with existing checkpoints.
