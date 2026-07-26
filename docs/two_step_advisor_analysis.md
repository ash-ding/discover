# Two-Step Advisor Pipeline: Claude Draft Failure Analysis & Eval Speed Impact

Analysis from the first 2-step advisor experiment on GPU Mode TriMul task (`gpu-mode-trimul_verl_20260717_1819`).

## 1. Claude Draft Failure Root Causes

All 8 Claude (Opus) drafts in Step 0 scored 0. The failures split into two distinct groups:

### Group A: Wrong einsum contraction (Drafts 1, 2, 4, 7)

The TriMul reference implementation uses:
```python
out = einsum('... i k d, ... j k d -> ... i j d', left, right)
```

Converting to `torch.bmm` requires transposing the right operand:
```python
# After permute [B,N,N,D] → [B,D,N,N] → reshape [B*D, N, N]:
out = torch.bmm(left_bmm, right_bmm.transpose(1, 2))  # correct
```

These 4 drafts omitted `.transpose(1, 2)`:
```python
out = torch.bmm(left_bmm, right_bmm)  # WRONG: computes einsum('bikd,bkjd->bijd')
```

This produces mathematically incorrect output on **all 7 test cases**.

### Group B: LayerNorm variance bug on non-power-of-2 dimensions (Drafts 3, 5, 6, 8)

These 4 drafts had the correct bmm transpose but shared a Triton LayerNorm bug.

All drafts use `BLOCK_N = next_power_of_2(dim)` as the kernel block size. When `dim=384` (not a power of 2), `BLOCK_N=512`, and 128 positions are zero-padded. The variance computation includes the padded elements:

```python
x = tl.load(x_ptr + cols, mask=cols < N, other=0.0)
mean = tl.sum(x) / N                    # correct (zeros don't affect sum)
x_centered = x - mean                   # x[384..511] = 0 - mean = -mean  ← problem
var = tl.sum(x_centered * x_centered) / N  # includes 128 * mean² extra
```

The fix requires masking the padded elements:
```python
var = tl.sum(tl.where(cols < N, x_centered * x_centered, 0.0)) / N
```

3 of 7 test cases use `dim=384`, so the inflated variance produces incorrect LayerNorm output, failing correctness checks.

### Summary Table

| Draft | bmm transpose | LN variance masked | Failure mode |
|-------|--------------|-------------------|-------------|
| 1     | missing      | no                | Runtime crash (dtype issue) + wrong einsum |
| 2     | missing      | no                | Wrong einsum (all 7 cases) |
| 4     | missing      | no                | Wrong einsum (all 7 cases) |
| 7     | missing      | no                | Wrong einsum (all 7 cases) |
| 3     | correct      | no                | LN variance bug (3/7 cases) |
| 5     | correct      | no                | LN variance bug (3/7 cases) |
| 6     | correct      | no                | LN variance bug (3/7 cases) |
| 8     | correct      | no                | LN variance bug (3/7 cases) |

### Potential System Prompt Improvements

Adding these hints to the Claude system prompt could improve draft quality:

1. **einsum → bmm**: "When converting `einsum('...ikd,...jkd->...ijd')` to bmm, you MUST transpose the right operand: `bmm(left, right.transpose(1,2))`"
2. **Triton LayerNorm**: "When using `next_power_of_2` for block size, mask zero-padded elements in variance: `tl.where(mask, x_centered², 0.0)`"
3. **Non-power-of-2 dims**: "dim can be 384 (not a power of 2) — test your LayerNorm with non-power-of-2 dimensions"

## 2. Eval Speed Impact: Failure Mode Shift

### Observation

2-step Step 0 takes ~85-90 minutes vs ~25 minutes for standard discover early steps, despite using the same eval infrastructure (4 GPUs on Node 1).

### Root Cause

The Claude draft shifts Qwen's output from **fast failures** to **slow failures**.

Triton kernel evaluation has bimodal timing:
- **Fast failures** (~4s): syntax errors, missing `@triton.jit`, import failures, Triton compilation errors — evaluator detects failure immediately
- **Slow failures** (~177s): code compiles and runs, but output doesn't match PyTorch reference — evaluator must run all 7 test configurations to determine failure

**Without Claude draft** (standard discover Step 0): Qwen writes Triton code from scratch. Most attempts have basic structural errors → fast failures dominate.

**With Claude draft** (2-step Step 0): Qwen sees a structurally correct template (valid `@triton.jit`, proper function signatures, reasonable kernel architecture). Qwen's output compiles and runs but has numerical errors → correctness failures dominate.

### Measured Data (Step 0, 357 Qwen evals)

| Failure Type | Count | % | Per-eval GPU time | Total GPU time | % of GPU time |
|-------------|-------|---|-------------------|---------------|---------------|
| Crash/runtime error | 231 | 64.7% | 4.4s | 16.9 min | 5.7% |
| Correctness failure | 94 | 26.3% | 177s | 277.3 min | **92.6%** |
| Compile/other | 32 | 9.0% | ~10s | 5.3 min | 1.8% |
| **Total** | **357** | | **avg 54.6s** | **299.6 min** | |

Wall time with 4 eval GPUs: **~75 minutes** (eval only).

### Comparison

| Scenario | Eval wall time (4 GPUs) | Total step time |
|----------|------------------------|-----------------|
| Standard discover Step 0 (mostly fast failures) | ~6.5 min | ~15-20 min |
| 2-step Step 0 (26% correctness failures) | ~75 min | ~85-90 min |
| Standard discover Step 30+ (training improves, more correctness tests) | ~50-60 min | ~45-55 min |

### Implications

This is **not a bug** — it's an inherent tradeoff of the 2-step design. The Claude draft improves Qwen's code structure quality, which is the desired outcome, but the side effect is slower evaluation because structurally correct code requires full test suites to evaluate.

As training progresses:
- Standard discover also slows down (Step 30+ takes 45-55 min) as Qwen learns to produce compilable code
- The 2-step advantage is that Qwen starts with better structural quality from Step 0, potentially reaching higher scores faster despite slower per-step eval
- The per-step time gap between 2-step and standard will **narrow** in later steps as both converge to correctness-failure-dominated eval

### Possible Mitigations

1. **Early-exit eval**: If the first test case fails correctness, skip remaining 6 test cases (currently all 7 run regardless). This would reduce correctness failure time from ~177s to ~25s.
2. **More eval GPUs**: Scale from 4 to 8 eval GPUs to halve wall time.
3. **Tiered eval**: Run a single cheap test case first as a filter, only run the full suite on passing kernels.
