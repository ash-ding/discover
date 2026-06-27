# Code Review Report: Fork Diff Algorithmic Fidelity & Bug Audit

**Date**: 2026-06-27
**Scope**: All `.py` changes from fork point `6e5e15d` to current HEAD
**Goal**: Verify the fork ONLY replaces the Tinker backend with local vLLM + PEFT LoRA, with no unintended algorithmic changes or bugs

---

## Executive Summary

The fork is largely faithful to the original algorithm. Core RL logic (advantage computation, loss functions, data processing, PUCT sampling) is **unchanged**. However, the review identified:

- **2 algorithmic differences** that deviate from the original behavior
- **1 configuration bug** affecting paper reproduction
- **1 silent correctness issue** in KL penalty computation
- **1 gradient scaling inconsistency** when `training_batch_size > 1`

---

## Part 1: Algorithmic Differences

### 1.1 KL Penalty Logprob Alignment [HIGH — Algorithmic Change]

**File**: `ttt_discover/rl/train.py:68-95` (`incorporate_kl_penalty`)

**Original code** (commit `6e5e15d`):
```python
logprob_diffs = [
    (sampled_logprobs - torch.tensor(base_logprobs[1:])) * mask
    for base_logprobs, sampled_logprobs, mask in safezip(...)
]
```
Simple `[1:]` slice — assumes `len(base_logprobs) == len(sampled_logprobs) + 1` always.

**Current code**: 5-branch alignment logic handling various length mismatches (exact match, off-by-one, padding with 0.0, truncation).

**Root cause**: `sampling_client.py:228` decodes token IDs to text before sending to vLLM with `echo=True`. The decode → re-encode roundtrip can produce different token counts than the original sequence. The original Tinker SDK sent token IDs directly, so this never happened.

**Impact**: When lengths don't match:
- Padding with `0.0` logprob implies `p=1.0`, which is incorrect and biases the KL estimate
- Truncation discards tokens from the alignment

**Recommendation**: Send token IDs directly to vLLM instead of decoded text. The vLLM `/v1/completions` API accepts `"prompt": [token_ids]` format. If echo=True doesn't support token ID prompts in vLLM V1, the current workaround should add an assertion/warning when alignment triggers (to quantify how often it happens).

---

### 1.2 Batched Rollout Uses First Env's Observation Only [MEDIUM — Algorithmic Change]

**File**: `ttt_discover/rl/rollouts.py:18-49` (`_do_batched_group_rollout`)

**Original code**: Each env generates its own observation independently, then `asyncio.gather` runs N independent sampling calls.

**Current code** (when policy has `batch_call`):
```python
obs_and_stops = [await env.initial_observation() for env in envs_G]
ob = obs_and_stops[0][0]  # Only first env's observation used
all_completions = await policy.batch_call(ob, stop_condition, num_samples=len(envs_G))
```

**Analysis**: This is safe IF all envs in a group produce identical initial observations. Current task designs (same prompt per group via `EnvGroupBuilder`) satisfy this. But the assumption is implicit and would silently break for any future task where group members have different prompts.

**Recommendation**: Add a runtime assertion:
```python
for obs, stop in obs_and_stops[1:]:
    assert obs.length == ob.length, "Batched rollout requires identical observations within a group"
```

---

## Part 2: Confirmed Bugs

### 2.1 `training_batch_size: 2` in All Paper Configs [HIGH — Config Bug]

**Files**: All 6 `examples/*/config_paper.yaml`

Every `config_paper.yaml` sets `training_batch_size: 2`, but the paper uses serial training (`batch_size=1`). The circle_packing config even documents this:
```yaml
training_batch_size: 2  # Batch size for training (1=serial, 2=batch, paper uses 1)
```

**Impact**: With `batch_size=2`, gradients are scaled by an extra `1/2` factor compared to `batch_size=1` (see 2.2 below). This changes the effective learning rate, affecting paper reproduction fidelity.

**Fix**: Change all `config_paper.yaml` files to `training_batch_size: 1`.

---

### 2.2 Gradient Scaling Mismatch for `training_batch_size > 1` [MEDIUM — Bug]

**File**: `ttt_discover/local_backend/training_client.py:319-322`

When `training_batch_size > 1`, the batch path computes:
```python
avg_loss = batch_loss / actual_bs
avg_loss.backward()
```

This divides gradients by `actual_bs` before backprop. The serial path (`batch_size=1`) does:
```python
loss.backward()  # No division, each datum's gradient accumulates
```

**Result**: For N data items with `batch_size=B`:
- Serial: total gradient ∝ N
- Batch: total gradient ∝ N/B

The batch path produces gradients `B` times smaller, effectively reducing the learning rate by `1/B`.

**Impact**: Only affects runs with `training_batch_size > 1`. Default is 1, so paper reproduction is unaffected if config_paper.yaml is fixed (see 2.1). But if anyone uses `batch_size > 1`, they need to scale learning_rate by `B` to compensate.

**Fix**: Either remove the division (`batch_loss.backward()` instead of `avg_loss.backward()`) or document the LR scaling requirement.

---

## Part 3: Silent Correctness Issues

### 3.1 `compute_logprobs_async` Decode→Encode Roundtrip [HIGH — Root Cause of 1.1]

**File**: `ttt_discover/local_backend/sampling_client.py:228`

```python
prompt_text = tokenizer.decode(token_ids, skip_special_tokens=False)
payload = {"prompt": prompt_text, "echo": True, ...}
```

This is the root cause of the KL alignment issue in 1.1. The decode→encode roundtrip is lossy:
- Special token handling may differ between decode and vLLM's re-encode
- Unicode normalization can change token boundaries
- BOS/EOS tokens may be added/removed

**Note**: Commit `627a99d` documents this was intentional ("vLLM V1 /v1/completions endpoint with echo=True requires text prompts, not token IDs"). This claim should be re-verified with the current vLLM version — token ID prompts may work now.

---

## Part 4: Confirmed No Impact (Algorithm Unchanged)

| Component | Status | Notes |
|-----------|--------|-------|
| `data_processing.py` | Zero changes | Advantage computation, trajectory assembly intact |
| `rl/types.py` | Zero changes | Type definitions unchanged |
| Loss functions (`loss.py`) | Correct | `importance_sampling_loss` and `ppo_clip_loss` are standard implementations |
| All task env classes | Unchanged | CirclePackingEnv, AutoCorrInequalityEnv, etc. — zero changes to prompts, rewards, or evaluation |
| All task prompts | Unchanged | Verified by diff |
| `TwoPhaseTokenCompleter` | Unchanged | Original GPT-OSS completer preserved |
| `Qwen3TwoPhaseTokenCompleter` | Expected adaptation | Structurally equivalent to original, with Qwen3-specific phase markers (`</think>` vs `<|channel|>final`) |
| WandB logger `>= 2` → `>= 3` | Bug fix | Original code would IndexError on `loggers[2]` with only 2 loggers |
| WandB offline mode | Infrastructure only | No algorithm impact |
| `tqdm.gather` | Cosmetic | Same behavior as `asyncio.gather`, adds progress bar |
| `context_window` parameter | No impact at default | Default 32768 matches original hardcoded value |
| `adv_estimator_beta=2.0` comment | Correct | Parameter is indeed unused with `entropic_adaptive_beta` estimator |

---

## Part 5: Local Backend Implementation Notes

These parameters are configured in `training_client.py` but cannot be verified against the original Tinker implementation (closed source):

| Parameter | Local Value | Confidence |
|-----------|-------------|------------|
| Optimizer | AdamW (beta1=0.9, beta2=0.95, eps=1e-8) | HIGH — matches `AdamParams` passed from `train.py` |
| Gradient clipping | `max_norm=1.0` | MEDIUM — standard value, but Tinker's default unknown |
| LoRA alpha | `= lora_rank` (scaling=1.0) | MEDIUM — common default, but Tinker's config unknown |
| LoRA target_modules | q/k/v/o/gate/up/down_proj | HIGH — covers all linear layers in Qwen architecture |
| LoRA dropout | 0.0 | HIGH — standard for RL fine-tuning |
| Gradient checkpointing | Enabled (non-reentrant) | N/A — memory optimization, doesn't affect numerics |
| Flash attention | flash_attention_2 (fallback: sdpa) | N/A — same numerics, different performance |

**Note**: `training_client.py` imports the **real** `tinker` SDK (`import tinker` at line 6) for type definitions like `ForwardBackwardOutput` and `TensorData`. The local `types.py` is used by other modules. This means some of the type-mismatch concerns (missing `loss_fn_output_type` field, missing `metrics` field in `OptimStepResponse`) are **false positives** — the real tinker types are used at runtime.

---

## Part 6: Recommended Actions (Priority Order)

### Must Fix for Paper Reproduction

1. **Fix `training_batch_size`** in all `config_paper.yaml` files: change `2` → `1`

### Should Fix

2. **Fix `compute_logprobs_async`**: Try sending token IDs directly (`"prompt": token_ids`) instead of decoded text. If vLLM rejects it, add logging/assertion to the alignment code to track mismatch frequency.

3. **Fix gradient scaling for batch_size > 1**: Use `batch_loss.backward()` instead of `(batch_loss/actual_bs).backward()`, or clearly document that LR must be scaled by B.

### Nice to Have

4. **Add assertion in batched rollout**: Verify all envs produce the same observation before using `batch_call`.

5. **Document training recipe assumptions**: Record that gradient clipping=1.0 and LoRA alpha=rank are assumed to match Tinker defaults, with note that this cannot be verified.
