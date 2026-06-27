# Fork 前后算法差异完整分析报告

## 背景

目标：将 Tinker 远程平台替换为本地 vLLM + PEFT LoRA 后端，**不改变任何算法逻辑**。
Fork 点：`6e5e15d`（原团队最后一次 merge）。
以下分析覆盖了所有 `.py` 文件的改动，分类为"预期后端替换"和"需要关注的功能性差异"。

---

## 第一部分：需要关注的功能性差异（按严重程度排序）

### 1. KL Penalty 的 logprobs 长度对齐逻辑 [严重]

**文件**: `ttt_discover/rl/train.py:68-95`（`incorporate_kl_penalty` 函数）

**原始代码** (`6e5e15d`):
```python
logprob_diffs = [
    (sampled_logprobs - torch.tensor(base_logprobs[1:])) * mask
    for base_logprobs, sampled_logprobs, mask in safezip(...)
]
```
原始代码假设 `len(base_logprobs) == len(sampled_logprobs) + 1`，直接用 `[1:]` 切片。

**当前代码**:
增加了复杂的长度对齐分支逻辑（5 个条件分支），处理各种长度不匹配情况。

**根因**: 本地 `compute_logprobs_async` 用 **文本** 而非 token IDs 发送 prompt 到 vLLM（commit `627a99d`）。`decode → re-encode` 过程可能产生不同数量的 token，导致长度不匹配。Tinker 原始实现直接传递 token IDs，不存在此问题。

**影响**: 当长度不匹配时，logprob 对齐方式会影响 KL penalty 的计算精度。`padding [0.0]` 或截断会导致部分 token 的 KL 散度被错误估计。

**建议修复**: 让 `compute_logprobs_async` 直接发送 token IDs（如 `"prompt": token_ids`）。如果 vLLM echo=True 不支持 token ID prompt，需要调查具体原因并寻找替代方案。

---

### 2. Batched Rollout（批量采样）改变了采样结构 [中等]

**文件**: `ttt_discover/rl/rollouts.py:18-49`（新增 `_do_batched_group_rollout`）

**原始代码**:
```python
trajectories_G = await asyncio.gather(*[do_single_rollout(policy, env, step_idx) for env in envs_G])
```
每个 env 独立生成 observation → 独立采样 → 独立执行 step。

**当前代码**（当 policy 有 `batch_call` 时）:
```python
ob = obs_and_stops[0][0]  # 只用第一个 env 的 observation
all_completions = await policy.batch_call(ob, stop_condition, num_samples=len(envs_G))
```
所有 env 共享第一个 env 的 observation，一次 vLLM 调用生成所有 completions。

**分析**:
- 对于当前的 task 设计（同一 group 内所有 env 的 initial_observation 相同），这在**语义上是等价的**
- 但如果任何 task 的 EnvGroupBuilder 给不同 env 产生不同的 initial_observation，行为会**静默错误**
- 此外，Phase 2（thinking tokens 用完时的 fallback）对每个 sample 是**串行**处理的，而原始代码是**并行**的（通过 asyncio.gather）

**风险**: 低（假设同一 group 内 prompt 相同），但建议验证所有 task 的 `make_envs()` 确实返回相同 prompt。

---

### 3. Qwen3TwoPhaseTokenCompleter 的 Phase 2 逻辑差异 [中等]

**文件**: `ttt_discover/tinker_utils/completers.py:169-341`

**原始 GPT-OSS Phase 2 逻辑**:
- 检查 `<|channel|>final<|message|>` 来判断是否已进入 final channel
- Prefill: `"\n\n... okay, I am out of thinking tokens. I need to send my final message now."` + channel marker tokens

**Qwen3 Phase 2 逻辑**:
- 检查 `</think>` 来判断 thinking 是否结束
- Prefill: `"\n\n... I need to give my final answer now.\n</think>\n"`

**分析**: 这是**预期的模型适配**。不同模型使用不同的 thinking/channel 机制是正常的。但需要确认：
- Qwen3 的 `</think>` 检测在所有情况下都正确工作
- Prefill 文本不会对 Qwen3 产生意想不到的行为

---

### 4. `context_window` 参数传递 [低]

**文件**: `ttt_discover/rl/train.py:343, 565`

**原始代码**: `TwoPhaseTokenCompleter` 的 `context_window` 使用默认值 `32768`，train.py 不传递此参数。

**当前代码**: 传递 `context_window=cfg.max_model_len`。

**影响**: 只有当 `max_model_len != 32768` 时才会有差异。如果配置中 `max_model_len=32768`（当前默认），行为完全一致。

---

### 5. WandB Logger 索引检查 [低]

**文件**: `ttt_discover/rl/train.py:606`

**原始代码**: `if len(ml_logger.loggers) >= 2:` 然后访问 `ml_logger.loggers[2]`
**当前代码**: `if len(ml_logger.loggers) >= 3:`

**分析**: 原始代码是一个 **bug**（索引 [2] 需要至少 3 个 logger）。修复本身是正确的。但在 offline WandB 模式下，logger 数量可能少于 3，导致某些 WandB table logging 被跳过（原始代码中会 IndexError）。

---

## 第二部分：本地后端训练实现细节（需确认与 Tinker 一致）

以下是 local_backend 中的训练配置，无法从代码中确认是否与 Tinker 完全一致：

### 6. 训练配方 (Training Recipe)

**文件**: `ttt_discover/local_backend/training_client.py`

| 参数 | 本地实现 | Tinker（推测） | 备注 |
|------|---------|--------------|------|
| Optimizer | AdamW | AdamW | train.py 中传 `AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)` ✓ |
| 初始 LR | 硬编码 `4e-5` | 动态传入 | `optim_step_async` 会覆盖，第一步之前无影响 |
| Gradient clipping | `max_norm=1.0` | 未知 | 1.0 是标准值，但无法确认 Tinker 是否相同 |
| LoRA alpha | `= lora_rank` | 未知 | 即 scaling factor = 1.0 |
| LoRA target_modules | 7 个 (q/k/v/o/gate/up/down_proj) | 未知 | 这是 Qwen 架构的全部线性层 |
| LoRA dropout | 0.0 | 未知 | |
| Gradient checkpointing | ✓ (use_reentrant=False) | 未知 | |
| Attention implementation | flash_attention_2 (fallback: sdpa) | 未知 | |

### 7. Loss Function

**文件**: `ttt_discover/local_backend/loss.py`

- `importance_sampling_loss`: 标准 IS loss = `-sum(ratio * advantages * mask) / sum(mask)` ✓
- `ppo_clip_loss`: 标准 PPO clip = `-sum(min(ratio*adv, clip(ratio)*adv) * mask) / sum(mask)` ✓

Loss function 实现是标准的，应该与 Tinker 一致。

### 8. batch_size > 1 时的梯度缩放

**文件**: `ttt_discover/local_backend/training_client.py:319-322`

当 `training_batch_size > 1` 时，loss 在 mini-batch 内做了平均 (`avg_loss = batch_loss / actual_bs`)。这会导致总梯度比 batch_size=1 时小 `batch_size` 倍。

**影响**: 默认 `training_batch_size=1`，不影响论文复现。但如果使用 >1 需要调整学习率。

---

## 第三部分：纯后端替换（无算法影响）

以下改动确认不影响算法行为：

1. **Import 替换**: `tinker` → `ttt_discover.local_backend as tinker` ✓
2. **`await` on `create_sampling_client`**: 原始是同步调用，本地是异步 — 纯 API 适配 ✓
3. **ServiceClient 条件初始化**: `use_local_backend` flag 控制 ✓
4. **Config 新字段**: `use_local_backend`, `inference_gpu_id`, `training_gpu_id` 等 ✓
5. **Ray GPU 可见性修复**: `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` ✓
6. **模型 assertion bypass**: `use_local_backend` 时跳过 GPT-OSS 检查 ✓
7. **tqdm 进度条**: `asyncio.gather` → `tqdm.gather`，纯 UI ✓
8. **WandB offline 模式**: `ml_log.py` 中的条件处理 ✓

---

## 第四部分：Task 环境文件

**所有 6 个 task 的 env.py**（circle_packing, ac_inequalities, erdos_min_overlap, ahc, denoising, gpu_mode）：

- 原始 `discover_*()` 函数：**零改动** ✓
- 新增 `discover_*_local()` 函数：隔离的本地后端入口点
- 环境类（Env, RewardEvaluator）：**零改动** ✓
- Prompts：**零改动** ✓
- Reward 计算：**零改动** ✓

**`data_processing.py`**: **零改动** ✓（advantage 计算、trajectory 数据组装完全不变）

---

## 第五部分：总结与建议

### 需要讨论的关键问题

1. **KL Penalty 的 token ID vs 文本问题** — 这是最严重的潜在差异。建议尝试让 vLLM 接受 token ID prompt，彻底消除 decode/re-encode 带来的长度不匹配。

2. **Batched rollout** — 确认同一 group 内所有 env 的 initial_observation 是否确实相同。如果是，batch_call 是安全的效率优化。

3. **Gradient clipping / LoRA config** — 无法从代码确认 Tinker 的默认值。如果有 Tinker 文档或原作者确认，可以验证。

4. **WandB logger 修复** — 原始代码有 bug（可能从未触发因为总是有 >=3 个 logger），修复是安全的。

### 确认无影响的核心算法

- Advantage 计算 (entropic adaptive beta) ✓
- PUCT 采样与状态复用 ✓
- Loss function (IS / PPO clip) ✓
- Training data 组装 (`trajectory_to_data`) ✓
- Constant reward group 过滤 ✓
- Phase 1/Phase 2 两阶段生成架构 ✓（Qwen3 适配是模型特定的，逻辑对等）
