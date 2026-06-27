# Training Efficiency Analysis: Current Bottleneck and Potential Improvements

**Date**: 2026-06-27
**Scope**: Analyze single-GPU training bottleneck in TTT-Discover local backend, evaluate RL framework alternatives

---

## 1. Current Architecture & Bottleneck

### Architecture Overview

```
┌─────────────────────────┐     ┌──────────────────────┐
│  vLLM Server (TP=4)     │     │  PyTorch Training    │
│  GPUs 0-3               │     │  GPU 4 (single)      │
│  - Rollout generation   │◄───►│  - PEFT LoRA         │
│  - KL penalty logprobs  │     │  - Gradient accum    │
│  - LoRA hot-reload      │     │  - AdamW optimizer   │
└─────────────────────────┘     └──────────────────────┘
```

### Wall-Clock Breakdown (from real run data)

| Phase | Time | Percentage | GPU Utilization |
|-------|------|-----------|-----------------|
| **Sampling (inference)** | 1,084s | 24% | High (TP=4) |
| **KL penalty** | 149s | 3% | Medium |
| **Training** | 3,273s | **73%** | **Very low (~0.07% of peak)** |
| **Total per step** | ~4,511s | 100% | — |

### Why Training Is So Slow

The training step processes **512 sequences** (group_size=64 × groups_per_batch=8) **serially on a single GPU**:

1. **Sequential forward/backward**: 512 individual passes through `training_client.py`, each taking ~150ms (100ms overhead + 50ms compute)
2. **Gradient checkpointing**: Adds 30-50% compute overhead to save ~2GB memory
3. **No batch fusion**: Each sequence is an independent forward pass — no GPU kernel fusion
4. **Single GPU**: No model parallelism, no FSDP, no ZeRO for training

### Memory Constraint Analysis (H100 80GB, Qwen3-8B LoRA)

| Component | Memory |
|-----------|--------|
| Base model (bf16) | ~16 GB |
| LoRA params (rank=32) | ~0.1 GB |
| AdamW optimizer state | ~0.4 GB |
| Activations (1 seq, 32K) | ~40-50 GB |
| **Total** | **~57-67 GB** |

At 32K sequence length, a single forward pass nearly fills 80GB. Batch size > 1 is only possible with shorter sequences or gradient checkpointing (which reduces activation memory at the cost of recomputation).

---

## 2. Key Question: Is Training Actually the Bottleneck?

**For TTT-Discover with 32K context: YES, training dominates at 73%.**

However, this is unusual. Industry research shows that for reasoning models with long chain-of-thought:

> "Generating long chain-of-thought outputs can account for up to 90% of total training time." — [vLLM Blog, 2025](https://vllm.ai/blog/2025-04-23-openrlhf-vllm)

TTT-Discover's training is unusually slow because:
- Single-GPU serial processing (no parallelism)
- 512 sequences processed one-by-one (no batching)
- Gradient checkpointing overhead

If training were parallelized (even 2-4 GPUs with FSDP), it would drop from 73% to ~20-30%, making inference the bottleneck — which is the normal case for RL post-training.

---

## 3. Framework Comparison

Five frameworks were evaluated through adversarial deep research (104 claims extracted, 25 verified, 22 confirmed, 3 refuted):

### Feature Matrix

| Feature | OpenRLHF | veRL | NeMo RL | rLLM | TRL |
|---------|----------|------|---------|------|-----|
| **GRPO/REINFORCE** | GRPO, REINFORCE++, RLOO, PPO | GRPO, REINFORCE++, RLOO, DAPO, PPO | GRPO | GRPO, REINFORCE, RLOO | GRPO |
| **Custom reward** | HTTP/Python file | Function-based + Sandbox Fusion | register_env() + CodeEnvironment | Docker/Modal/local sandbox | Function-based |
| **LoRA training** | DeepSpeed LoRA/QLoRA | FSDP LoRA (Megatron LoRA WIP) | DTensor LoRA | Via veRL backend | PEFT LoRA |
| **KL penalty** | --algo.kl.init_coef | use_kl_loss + kl_loss_coef | Built-in | Via veRL | Built-in |
| **IS correction** | 3 strategies (tis, icepop, seq-mask) | Standard ratio clipping | Standard | Via veRL | Standard |
| **Multi-GPU training** | DeepSpeed ZeRO-2/3 | FSDP/FSDP2 + Megatron-LM | Megatron-Core (6D parallelism) | Via veRL | HF Accelerate |
| **Inference engine** | vLLM (native) | vLLM + SGLang | vLLM | Via veRL | HF generate |
| **Weight sync** | CUDA IPC / NCCL | In-process resharding | Ray-based | Via veRL | In-process |
| **Sandboxed exec** | No (custom reward only) | Sandbox Fusion (unverified) | CodeEnvironment (built-in) | Docker/Daytona/Modal | No |
| **Maturity** | Production (70B+ models) | Production (EuroSys 2025) | Active development | Newer | Mature but simpler |

### TTT-Discover Algorithm Compatibility Analysis

TTT-Discover uses several non-standard components that constrain framework choice:

| TTT-Discover Feature | Description | Framework Compatibility |
|----------------------|-------------|------------------------|
| **Two-phase generation** | Phase 1 (thinking) → Phase 2 (answer) with injected prefill | Requires custom TokenCompleter — none support natively |
| **PUCT state reuse** | Tree search with state caching across epochs | Requires custom sampler — none support natively |
| **Entropic adaptive beta** | Custom advantage estimator | OpenRLHF has extensible estimators; others need patching |
| **Importance sampling loss** | Not PPO clip — raw IS ratio × advantage | OpenRLHF supports IS correction; veRL uses standard GRPO |
| **SandboxRewardEvaluator** | Ray-based sandboxed code execution | NeMo RL has built-in CodeEnvironment; rLLM has Docker sandbox |
| **LoRA hot-reload** | vLLM loads new LoRA each epoch | All frameworks with vLLM support this |

---

## 4. Detailed Evaluation of Options

### Option A: Migrate to OpenRLHF

**Architecture**: Ray + DeepSpeed ZeRO-3 + vLLM

**Pros**:
- Most mature vLLM integration (CUDA IPC weight sync)
- DeepSpeed ZeRO-3 enables multi-GPU training without code changes
- Importance sampling correction already implemented (3 strategies)
- LoRA/QLoRA + gradient checkpointing built-in
- Colocated execution: training and inference can share GPUs

**Cons**:
- No built-in sandboxed execution — need external reward server
- Two-phase generation (TokenCompleter) would need custom integration
- PUCT state reuse has no equivalent — would need to be ported as custom sampler
- Entropic adaptive beta is not a standard advantage estimator

**Integration effort**: HIGH (3-4 weeks)
- Port TokenCompleter logic → OpenRLHF's generation pipeline
- Port PUCT sampler → custom dataset/experience maker
- Port entropic adaptive beta → custom advantage estimator
- Port SandboxRewardEvaluator → remote reward server
- Validate numerical equivalence against current implementation

**Training speedup**: 3-5x (DeepSpeed ZeRO-3 across 2-4 GPUs)

---

### Option B: Migrate to veRL

**Architecture**: FSDP/FSDP2 or Megatron-LM + vLLM/SGLang

**Pros**:
- Widest backend flexibility (FSDP, FSDP2, Megatron)
- vLLM integration with in-process weight resharding (no separate server needed)
- GRPO natively supported with existing recipes for Qwen models
- LoRA training on FSDP backend with adapter-only weight sync
- Active development (v0.7, EuroSys 2025 paper)

**Cons**:
- LoRA + Megatron combination not fully supported yet
- Sandbox Fusion integration for code execution was **refuted** in verification (0-3 vote)
- More complex architecture (3D-HybridEngine, weight resharding)
- PUCT and two-phase generation still need custom porting

**Integration effort**: HIGH (3-4 weeks) — similar to OpenRLHF

**Training speedup**: 3-8x (FSDP across 2-4 GPUs, or Megatron for larger models)

---

### Option C: Keep Current Architecture, Add DeepSpeed ZeRO

**Architecture**: Current code + DeepSpeed ZeRO-2/3 for training only

**Approach**: Replace the single-GPU `LocalTrainingClient` with DeepSpeed-wrapped training:

```python
# Instead of:
self.model = get_peft_model(base_model, lora_config)
self.optimizer = torch.optim.AdamW(...)

# Use:
self.model, self.optimizer, _, _ = deepspeed.initialize(
    model=get_peft_model(base_model, lora_config),
    optimizer=torch.optim.AdamW(...),
    config={"zero_optimization": {"stage": 2}},
)
```

**Pros**:
- **Minimal code changes** — only `training_client.py` needs modification
- Keeps all custom algorithm logic (PUCT, two-phase, entropic beta) intact
- Keeps vLLM server architecture unchanged
- ZeRO-2 shards optimizer state across GPUs → fits larger batch sizes
- ZeRO-3 additionally shards model weights → supports bigger models

**Cons**:
- Still need to manage GPU allocation (training GPUs vs inference GPUs)
- vLLM weight sync after training step needs custom implementation
- DeepSpeed + LoRA integration can have edge cases

**Integration effort**: LOW-MEDIUM (1-2 weeks)
- Wrap model/optimizer with `deepspeed.initialize()`
- Update `optim_step_async` for DeepSpeed's optimizer step
- Handle multi-GPU checkpoint save/load
- Test weight sync with vLLM LoRA hot-reload

**Training speedup**: 2-4x (ZeRO-2 across 2 GPUs, ZeRO-3 across 4 GPUs)

---

### Option D: Keep Current Architecture, Add PyTorch FSDP

**Architecture**: Current code + FSDP for training only

**Approach**: Similar to Option C but using native PyTorch FSDP instead of DeepSpeed:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

self.model = FSDP(
    get_peft_model(base_model, lora_config),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

**Pros**:
- Native PyTorch — no additional framework dependency
- FSDP2 (PyTorch 2.11+) has improved LoRA support
- Same minimal-change approach as Option C

**Cons**:
- FSDP + PEFT LoRA has historically been tricky (requires careful wrapping)
- Need `torch.distributed` process group management
- Less community support than DeepSpeed for RL training

**Integration effort**: MEDIUM (2-3 weeks)

**Training speedup**: 2-4x

---

### Option E: Quick Win — Batch Processing Optimization (No Framework Change)

**Approach**: Maximize single-GPU efficiency without changing the architecture:

1. **Increase training_batch_size**: With gradient scaling fix (already done), batch_size=2-4 should work for shorter sequences
2. **Use `torch.compile()`**: JIT-compile the forward pass for kernel fusion
3. **Sequence packing**: Pack multiple short sequences into one long sequence

**Important: Gradient checkpointing CANNOT be disabled at 32K sequence length.** Memory analysis on H100 80GB with Qwen3-8B LoRA:

| Sequence Length | Without Checkpointing | With Checkpointing | Status |
|----------------|----------------------|-------------------|--------|
| 4K | 32 GB | 19 GB | Both OK |
| 8K | 46 GB | 22 GB | Both OK |
| 16K | 75 GB | 27 GB | No-ckpt risky |
| 32K | **133 GB** | 36 GB | **No-ckpt = OOM** |

At 32K, activation memory alone is ~116 GB (36 layers × 3.2 GB/layer), far exceeding the 80 GB capacity. Gradient checkpointing reduces this to ~19 GB by only storing √N layer activations and recomputing the rest during backward. The 30-50% compute overhead is the price for fitting 32K sequences at all — it is not optional for the paper configuration.

Disabling gradient checkpointing is only viable if `max_train_seq_len ≤ 8192`, which requires truncating prompts and may reduce training quality.

**Pros**:
- Zero architecture change
- Can implement incrementally
- No multi-GPU coordination complexity

**Cons**:
- Limited ceiling — still single GPU
- torch.compile + LoRA compatibility may have issues
- Gradient checkpointing must stay on for paper config (32K)

**Integration effort**: LOW (2-3 days)

**Training speedup**: 1.5-2x

---

## 5. Recommendation

### Priority Order

| Priority | Option | Speedup | Effort | Risk | Recommendation |
|----------|--------|---------|--------|------|----------------|
| **1** | **E: Quick wins** | 1.5-2x | 2-3 days | Very low | **Do first** — immediate gains, no risk |
| **2** | **C: Add DeepSpeed ZeRO** | 2-4x | 1-2 weeks | Low | **Best ROI** — minimal code change, proven tech |
| **3** | A or B: Full framework migration | 3-8x | 3-4 weeks | High | **Only if ZeRO is insufficient** |

### Rationale

1. **Option E first**: The batch processing fix is already done. Adding `torch.compile()` and sequence packing are safe, incremental improvements. Note: gradient checkpointing must remain enabled for 32K sequences (disabling it causes OOM — activation memory alone reaches 116 GB on Qwen3-8B).

2. **Option C (DeepSpeed ZeRO) is the sweet spot**: It gives multi-GPU training without rewriting the algorithm layer. TTT-Discover's unique components (PUCT state reuse, two-phase generation, entropic adaptive beta) are all in the RL loop — not in the training step. DeepSpeed only wraps the training step, so all custom logic stays untouched.

3. **Full framework migration (A/B) is premature**: The main value of OpenRLHF/veRL is their complete RL pipeline — but TTT-Discover already has a complete pipeline with unique algorithms. Migrating would require porting these algorithms into a new framework's abstractions, which is 3-4 weeks of work with high risk of introducing subtle behavioral differences. This only makes sense if:
   - You plan to scale to 70B+ models (current 8B works fine with ZeRO)
   - You want async training/inference overlap
   - The current architecture becomes maintenance burden

### When to Reconsider

Migrate to OpenRLHF or veRL when:
- Scaling to 70B+ models where ZeRO-3 isn't sufficient
- Need async RL (overlapping generation and training)
- vLLM native weight transfer API matures (already available, reduces integration effort)
- TTT-Discover algorithm stabilizes and custom components can be standardized

---

## 6. vLLM Weight Transfer API (Key Enabling Technology)

vLLM now provides an official weight transfer API specifically for hybrid RL workflows:

```
Phase 1: init_weight_transfer_engine  → Establish NCCL/CUDA IPC channel
Phase 2: start_weight_update          → Prepare workers
Phase 3: update_weights               → Transfer all or subset of weights
Phase 4: finish_weight_update         → Post-processing (quantization, etc.)
```

Two backends:
- **NCCL broadcast**: For separate training/inference GPUs (current TTT-Discover setup)
- **CUDA IPC**: For colocated same-GPU setups (higher efficiency)

This API makes any "X trains, vLLM infers" architecture officially supported. OpenRLHF and veRL both use it. A DeepSpeed ZeRO integration (Option C) could leverage this instead of the current LoRA hot-reload via HTTP API.

---

## Sources

- [vLLM Native RL APIs Blog](https://vllm.ai/blog/2025-04-23-openrlhf-vllm) — Architecture details, weight transfer protocol
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF) — Ray + DeepSpeed + vLLM framework
- [veRL Documentation](https://verl.readthedocs.io/) — FSDP/Megatron + vLLM hybrid engine
- [NeMo RL Environments](https://docs.nvidia.com/nemo/rl/latest/guides/environments.html) — Custom reward registration
- [rLLM GitHub](https://github.com/rllm-org/rllm) — Sandboxed execution backends
- [vLLM Weight Transfer API](https://docs.vllm.ai/en/stable/training/weight_transfer/) — Official documentation
- [HuggingFace RL Training Landscape](https://huggingface.co/blog/async-rl-training-landscape) — Framework comparison survey
- [veRL EuroSys 2025 Paper](https://arxiv.org/abs/2405.11143) — 3D-HybridEngine architecture
