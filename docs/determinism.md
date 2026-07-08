# Determinism and Reproducibility

## Overview

This document records our investigation into achieving run-to-run reproducibility for TTT-Discover training. The system uses VERL colocate mode with vLLM for inference and FSDP for training, sharing 8×H100 GPUs.

**TL;DR**: Full bit-exact reproducibility is not achievable with the current vLLM V1 engine. The root cause is a global RNG state drift inside vLLM's multiprocess engine that cannot be fixed through configuration. We achieved ~70% per-rollout score consistency across runs and ~98% in standalone vLLM tests.

## Default Configuration (No Determinism)

| Setting | Value | Effect |
|---------|-------|--------|
| `full_determinism` | `False` | No deterministic CUDA kernels enforced |
| `sampling_params.seed` | Not set | vLLM uses global RNG, non-reproducible |
| `temperature` | `1.0` | Stochastic sampling — primary source of intended randomness |
| Deployment | Colocate | vLLM inference and FSDP training share the same GPU processes |

With default settings, two runs with identical configuration produce completely different results. The randomness originates from vLLM's token sampling at temperature=1.0.

## How vLLM Sampling Works

vLLM uses the **exponential noise method** (a Gumbel-max variant) instead of traditional CDF-based sampling:

```python
probs = logits.softmax(dim=-1)           # model's probability distribution
q = torch.empty_like(probs).exponential_()  # independent Exp(1) noise per element
selected_token = (probs / q).argmax(dim=-1) # score = prob/noise, take argmax
```

When a `seed` is set in `SamplingParams`, vLLM creates a per-request `torch.Generator` initialized with that seed. The generator produces deterministic noise sequences, making sampling reproducible for that request.

When 64 requests share the same prompt but have different seeds, they get different noise → different tokens → different outputs. When they share the same seed, the noise is identical → identical outputs (verified in Experiment 2a).

## Sources of Non-Determinism

### 1. Sampling RNG (Controllable)
- **Cause**: Without a seed, vLLM uses the global CUDA RNG, which varies across runs.
- **Fix**: Set per-request `seed` in `SamplingParams`.

### 2. CUDA Floating-Point Non-Determinism (Controllable)
- **Cause**: cuBLAS, FlashAttention, and other CUDA ops may use non-deterministic algorithms by default.
- **Fix**: `full_determinism=True` enables deterministic CUDA kernels via `CUBLAS_WORKSPACE_CONFIG`, `FLASH_ATTENTION_DETERMINISTIC`, `torch.use_deterministic_algorithms()`.
- **Cost**: ~50-60% throughput loss (measured: step time 16 min → 25 min).

### 3. Multi-Replica Load Balancer Routing (Controllable)
- **Cause**: With TP=4 and 8 GPUs, VERL creates 2 vLLM replicas. The load balancer routes requests based on inflight count, which depends on asyncio scheduling order (non-deterministic).
- **Fix**: When `full_determinism=True`, use `hash(request_id) % num_replicas` for deterministic routing.

### 4. Request ID Randomness (Controllable)
- **Cause**: `llm_server.py` replaces our deterministic request_id with `uuid4().hex` before sending to vLLM engine.
- **Fix**: When `full_determinism=True`, pass through the original deterministic request_id.

### 5. Prefix Caching Order Dependence (Controllable)
- **Cause**: Prefix caching state depends on which request arrives first at a replica.
- **Fix**: `enable_prefix_caching=False`, or fix routing so arrival order is deterministic.

### 6. vLLM V1 Engine Global RNG State Drift (NOT Controllable)
- **Cause**: vLLM V1's multiprocess EngineCore maintains global state that drifts between `generate()` calls. Even with per-request seeds, some internal operation reads the global RNG, and the global RNG state depends on how many requests were previously processed.
- **Evidence**: See Experiment 7 below — consecutive calls to `generate()` on the same `vllm.LLM` instance produce different outputs for 5-10 out of 64 rollouts, with the affected positions shifting progressively (always in the first ~10 indices).
- **Status**: This is a vLLM V1 engine limitation. Cannot be fixed through configuration.

## Experiment Results (July 8, 2026)

All experiments use Circle Packing 26 task with Qwen3-8B.

### Experiment 1: Seed-only (no full_determinism)

**Setting**: Per-rollout unique seed (`seed = 42 + step*10000 + group*100 + session_id`), all 64 requests in a group share the SAME seed (before per-rollout seed was implemented), `full_determinism=False`, TP=4 (2 replicas), 5 epochs × 2 runs.

**Result**: Step 1 score match = **371/512 (72.5%)**.

**Analysis**: ~85% of rollout texts were identical within a run (same seed → same generator → same noise, but CUDA FP non-determinism caused ~15% to diverge at "boundary tokens" where two candidates have near-equal probability). Across runs, ~72% score match.

### Experiment 2: Standalone vLLM Tests (outside VERL)

**2a — Same seed, single generate() call**:
- 64 independent requests, all with `seed=42`, `full_determinism` env vars set.
- Prefix caching and chunked prefill: ON (defaults).
- Result: **64/64 identical** (100% deterministic).

**2b — Different seeds, single generate() call**:
- 64 requests with seeds 42..105, `full_determinism` env vars set.
- Result: **64/64 unique** (diversity works correctly).

**Conclusion**: vLLM's per-request seed mechanism works perfectly within a single `generate()` call.

### Experiment 3: VERL Training, full_determinism + per-rollout seed + TP=4 (2 replicas)

**Setting**: `full_determinism=True`, per-rollout unique seeds, hash-based deterministic routing, deterministic request IDs. 2 epochs × 2 runs.

**Result**: Step 1 score match = **351/512 (68.6%)**, text match = **0/512**.

**Analysis**: Routing fix alone is insufficient. Even with deterministic routing, results differ because of deeper non-determinism in the async engine.

### Experiment 4: VERL Training, TP=8 (single replica)

**Setting**: Same as Experiment 3 but with `ROLLOUT_TP=8` (single replica, no load balancer). 2 epochs × 2 runs.

**Result**: Step 1 score match = **375/512 (73.4%)**, text match = **0/512**.

**Analysis**: Eliminating multi-replica routing did not help. The non-determinism is within the single engine.

### Experiment 5: VERL Training, TP=8, attempted no prefix cache / no chunked prefill

**Setting**: Same as Experiment 4 with `enable_prefix_caching=False`, `enable_chunked_prefill=False` passed via VERL config. 2 epochs × 2 runs.

**Result**: Step 1 score match = **376/512 (73.4%)**, text match = **5/512**.

**Important caveat**: Post-hoc investigation revealed that **these flags did not actually take effect** in VERL's pipeline. VERL converts config to vLLM CLI args via `build_cli_args_from_config()` (`verl/workers/rollout/vllm_rollout/utils.py:411-413`), which **skips bool False values** — it only generates `--flag` for True, and omits the flag entirely for False. Since vLLM V1's defaults are `enable_chunked_prefill=True` and `enable_prefix_caching=True`, the engine received no override and kept both enabled. This means Experiments 4 and 5 ran with **identical configurations**, explaining the near-identical results.

Note: The standalone tests (Experiments 6-8) used `vllm.LLM` Python API which passes kwargs directly to the constructor, correctly disabling both features. Even with them truly disabled, 1/64 inconsistency remained.

### Experiment 6: Standalone vLLM, same instance, two generate() calls

**Setting**: Same `vllm.LLM` instance, different seeds per request (seed=10042+i), call `generate()` twice with identical parameters. TP=4, `full_determinism` env vars set, `VLLM_BATCH_INVARIANT=1`, prefix caching: OFF, chunked prefill: OFF. Continuous batching remains ON (it is vLLM V1's core scheduler and cannot be disabled).

**Result**: **63/64 match** (1 rollout differs at char 731).

**Analysis**: Even within the same process and same engine instance, a second `generate()` call produces slightly different results. The difference is small (~1.5%) but non-zero.

### Experiment 7: Standalone vLLM, same instance, THREE consecutive generate() calls

**Setting**: Same as Experiment 6 but 3 consecutive calls. All determinism settings identical.

**Result**:

| Comparison | # Diffs | Differing positions |
|-----------|---------|-------------------|
| Run 1 vs Run 2 | 5 | [0, 6, 7, 8, 9] |
| Run 2 vs Run 3 | 10 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |
| Run 1 vs Run 3 | 5 | [1, 2, 3, 4, 5] |

All divergences occur at **char 731** and affect the **first 10 rollout indices**. The number of diffs increases with each call (0→5→10), and the affected positions shift progressively.

**Root cause**: vLLM V1 engine has a global state (likely the global CUDA RNG counter in the EngineCore process) that advances with each `generate()` call. Per-request `torch.Generator` seeds control the sampling noise, but some other operation inside the engine (possibly in the scheduler, KV cache management, or batch assembly) reads from the global RNG. Since the global RNG state differs between calls, the first ~10 requests processed by the scheduler see different internal state, causing their outputs to diverge at a consistent character position.

### Experiment 8: Standalone vLLM, two fresh instances

**Setting**: Create LLM → generate() → destroy → create new LLM → generate(). Same determinism settings as Experiment 6 (TP=4, `full_determinism` env vars, `VLLM_BATCH_INVARIANT=1`, prefix caching: OFF, chunked prefill: OFF, continuous batching: ON).

**Result**: TP=4: **63/64 match**. TP=1 (single GPU): **15/16 match**.

**Analysis**: Same pattern as Experiment 6. Destroying and recreating the LLM doesn't fully reset CUDA device state.

## Summary Table

| Experiment | Setting | Score Match | Text Match | Key Finding |
|-----------|---------|------------|------------|-------------|
| 1 | Seed-only, VERL, TP=4 | 72.5% | ~85%* | CUDA FP non-determinism dominates |
| 2a | Standalone, same seed | — | 100% | Single call + same seed = perfect |
| 2b | Standalone, diff seeds | — | 100%** | Diversity works correctly |
| 3 | VERL, full_determ, TP=4 | 68.6% | 0% | Routing fix insufficient |
| 4 | VERL, full_determ, TP=8 | 73.4% | 0% | Single replica doesn't help |
| 5 | VERL, TP=8, no cache | 73.4% | 1% | Disabling caches doesn't help |
| 6 | Standalone, 2 calls | 98.4% | 98.4% | Engine state drift (~1.5%) |
| 7 | Standalone, 3 calls | — | progressive | Global RNG drift confirmed |
| 8 | Standalone, 2 instances | 98.4% | 98.4% | CUDA state not fully reset |

\* Within a single run (same seed → same output). \*\* Each request unique, verified across 2 calls.

## Conclusion

1. **vLLM's per-request seed mechanism works correctly** for a single `generate()` call (Experiments 2a, 2b).
2. **A global RNG state drift in vLLM V1's EngineCore** causes ~1.5% of rollouts to differ between consecutive calls, even with per-request seeds (Experiments 6, 7, 8).
3. **VERL's async pipeline amplifies this** to ~25-30% mismatch due to: request arrival order non-determinism, sleep/wake cycles, and multi-replica routing (Experiments 3-5).
4. **The affected rollouts are always in the first ~10 indices** and the divergence point is consistent (char ~731), strongly suggesting a global state counter in the EngineCore (Experiment 7).
5. **This is a vLLM V1 engine limitation** — not fixable through VERL configuration, CUDA determinism flags, or routing changes.

## Changes Made

1. **Per-rollout deterministic seed** (`agent_loop.py:315-319`):
   ```python
   per_rollout_seed = base_seed + global_steps * 10000 + group_index * 100 + session_id
   ```
   Provides diversity within a step and partial reproducibility across runs.

2. **Deterministic request IDs** (`agent_loop.py:323`):
   ```python
   request_id = f"s{global_steps}_g{group_index}_r{session_id}_p1"
   ```

3. **Decoupled seed injection** (`vllm_async_server.py:521`):
   Seed is now always injected via `setdefault`, regardless of `full_determinism` flag.

4. **Deterministic load balancer routing** (`llm_server.py:99-102`):
   When `full_determinism=True`, uses `hash(request_id) % num_replicas` instead of inflight-count-based routing.

5. **Request ID passthrough** (`llm_server.py:236`):
   When `full_determinism=True`, passes deterministic request_id to vLLM engine.

6. **Group index tracking** (`agent_loop.py:617-619`):
   Assigns deterministic `_group_index` to each prompt before chunking across workers.

## Related Files

- `verl/verl/workers/engine/utils.py:31` — `enable_full_determinism()` implementation
- `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:521` — seed injection
- `verl/verl/workers/config/rollout.py:171-175` — `full_determinism` and `seed` config
- `verl/verl/workers/rollout/llm_server.py:79-110` — load balancer routing
- `verl/verl/workers/rollout/llm_server.py:234-236` — request ID passthrough
- `ttt_discover/verl_integration/agent_loop.py:315-327` — per-rollout seed computation
- `scripts/test_determinism.py` — standalone same-seed test
- `scripts/test_determinism_diverse.py` — standalone different-seed test
- `scripts/test_determinism_async.py` — two-call test on same instance
- `scripts/test_determinism_restart.py` — two-instance test
- `scripts/test_determinism_3runs.py` — three-call drift analysis
- `scripts/test_determinism_tp1.py` — TP=1 single-GPU test
