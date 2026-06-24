# KL Penalty 计算代码完整分析

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. 高层入口函数 (train.py)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    base_sampling_client: tinker.SamplingClient,
    kl_penalty_coef: float,
) -> Dict[str, float]:
    """
    Compute KL against base model. Adjust advantages in-place by logp_base - logp_current - avg_kl,
    where avg_kl is the average of logp_base - logp_current (which is -KL[current, base])
    """
```

**输入：**
- `data_D`: 512 个 trajectories 的数据（每个包含完整的生成序列）
- `base_sampling_client`: vLLM 的 HTTP 客户端
- `kl_penalty_coef`: KL penalty 系数（0.1）

**目标：**
计算当前 policy 和 base model 之间的 KL divergence，用于调整 advantage

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 2. 步骤 1：准备完整序列输入

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
# 为每个 trajectory 构造完整序列（prompt + generation）
full_sequence_inputs_D = [
    datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
    for datum in data_D
]
```

**作用：**
- 将每个 trajectory 的 prompt + 完整生成序列拼接
- 结果：512 个完整序列，每个包含 prompt + generation tokens

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 3. 步骤 2：批处理调用 base model 计算 logprobs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

**关键点：**
1. **分批处理**：每批 `KL_PENALTY_BATCH_SIZE` 个请求
2. **并发发送**：使用 `asyncio.gather` 并发发送同一批的请求
3. **为什么需要 batching？**
   - 512 个并发 echo=True 请求会导致 vLLM OOM
   - batch_size=1 确保每次只有 1 个请求到达 vLLM

**当前配置：**
- `batch_size=1`：完全串行，每次只发送 1 个请求
- 总计：512 次调用 `compute_logprobs_async`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 4. 步骤 3：compute_logprobs_async 实现 (sampling_client.py)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
async def compute_logprobs_async(self, sequence_input) -> list[float]:
    # 1. 将序列转换为 token IDs
    token_ids = _model_input_to_token_ids(sequence_input)
    if len(token_ids) < 2:
        return [0.0] * len(token_ids)
    
    # 2. Decode 回文本（为了 vLLM API）
    tokenizer = self._get_tokenizer()
    prompt_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    # 3. 构造 vLLM 请求 payload
    payload: dict[str, Any] = {
        "prompt": prompt_text,
        "max_tokens": 1,          # ⚠️ 只生成 1 个 token
        "echo": True,             # ⚠️ 关键！返回整个序列的 logprobs
        "logprobs": 1,            # 返回 logprobs
    }
    if self.lora_name:
        payload["model"] = self.lora_name  # 使用 base model（不加 LoRA）
    
    # 4. 发送 HTTP POST 请求到 vLLM
    session = await self._get_session()
    url = f"{self.base_url}/v1/completions"
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(
                f"vLLM logprobs failed ({resp.status}): {text}"
            )
        data = await resp.json()
    
    # 5. 解析返回的 logprobs
    choice = data["choices"][0]
    logprobs_data = choice.get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", [])
    
    # 6. 返回结果
    result = [lp if lp is not None else 0.0 for lp in token_logprobs]
    if len(result) > len(token_ids):
        result = result[: len(token_ids)]
    return result
```

**关键参数解释：**

### `"echo": True`
- **作用**：让 vLLM 返回整个 prompt 的 logprobs，而不只是生成部分
- **为什么需要？** KL penalty 需要计算整个序列的 log 概率
- **内存影响**：这是导致 OOM 的根本原因！
  - vLLM 需要计算整个序列的 logits: `[seq_len, vocab_size]`
  - 例如：`[4000, 151936]` ≈ 2.3 GB per request
  - 如果批处理多个请求：`[12000, 151936]` ≈ 7 GB → OOM!

### `"max_tokens": 1`
- **作用**：只生成 1 个新 token（实际上我们不需要生成）
- **为什么？** echo=True 会返回整个 prompt 的 logprobs
- **实际效果**：vLLM 会做一次前向传播，计算整个序列的 logits

### `"model": self.lora_name`
- **作用**：指定使用哪个模型
- **当前情况**：
  - Sampling 阶段：`lora_name = "lora_v1"` (fine-tuned model)
  - KL penalty 阶段：`lora_name = None` (base model)
  - ⚠️ **注意**：从代码看，这里用的是 `self.lora_name`
  - 需要确认 KL penalty 时是否切换到 base model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 5. 步骤 4：计算 KL divergence

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
# 获取 sampled policy 的 logprobs（在 sampling 阶段已经保存）
sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]

# 计算每个位置的 logprob 差异
logprob_diffs = [
    (sampled_logprobs - torch.tensor(base_logprobs[1:])) * mask
    for base_logprobs, sampled_logprobs, mask in safezip(
        base_logprobs_D, sampled_logprobs_D, float_masks
    )
]

# 计算平均 KL
avg_logp_diff = sum([diff.sum() for diff in logprob_diffs]) / sum(
    [mask.sum() for mask in float_masks]
)
```

**数学公式：**
```
logprob_diff[i] = log P_sampled(token[i]) - log P_base(token[i])
                = log(P_sampled / P_base)
                
KL(sampled || base) = -E[log(P_base / P_sampled)]
                    = -E[log P_base - log P_sampled]
                    = -avg_logp_diff
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 6. 步骤 5：调整 advantages

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```python
for i, datum in enumerate(data_D):
    kl_advantages = kl_penalty_coef * float_masks[i] * (avg_logp_diff - logprob_diffs[i])
    datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
        datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
    )
return {"kl_policy_base": float(avg_logp_diff)}
```

**作用：**
- 为每个 token 的 advantage 添加 KL penalty 项
- 惩罚偏离 base model 太远的 trajectories

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 7. vLLM 端的处理流程

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当收到 `echo=True` 的请求时，vLLM V1 的处理：

```python
# gpu_model_runner.py: _get_prompt_logprobs_dict()

for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
    # 1. 从 hidden states 中取出这个请求的部分
    prompt_hidden_states = hidden_states[offset:offset + num_logits]
    
    # 2. 计算 logits（内存分配发生在这里！）
    logits = self.model.compute_logits(prompt_hidden_states, None)
    # → logits shape: [num_tokens, vocab_size]
    # → 例如: [4000, 151936] × 4 bytes = 2.3 GB
    
    # 3. 计算 logprobs
    logprobs = self.sampler.compute_logprobs(logits)
    
    # 4. gather_logprobs（OOM 发生在这里！）
    token_ids, logprobs, ranks = self.sampler.gather_logprobs(
        logprobs, num_prompt_logprobs, tgt_token_ids
    )
```

**gather_logprobs 中的 OOM 代码：**
```python
# sampler.py: gather_logprobs()

# 这一行导致 OOM！
token_ranks = (logprobs >= token_logprobs).sum(-1)
# → 创建 [num_tokens, vocab_size] 的布尔 tensor
# → 需要额外内存分配
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 8. 内存分析

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 单个 echo=True 请求的内存需求：

假设序列长度 = 4000 tokens

```
1. Hidden states: [4000, 4096] × 2 bytes (fp16) = 32 MB

2. Logits tensor: [4000, 151936] × 4 bytes (fp32) = 2.3 GB ⚠️

3. Logprobs tensor: [4000, 151936] × 4 bytes = 2.3 GB

4. gather_logprobs 临时 tensor:
   - token_logprobs: [4000, 1] × 4 bytes = 16 KB
   - 比较结果: [4000, 151936] × 1 byte (bool) = 579 MB
   
总计: 约 5 GB per request
```

### 批处理的影响：

如果 vLLM 批处理 2 个请求：
```
Logits: [8000, 151936] × 4 bytes = 4.6 GB
Logprobs: [8000, 151936] × 4 bytes = 4.6 GB
临时 tensor: [8000, 151936] × 1 byte = 1.2 GB
总计: 约 10 GB → 超过可用内存 → OOM!
```

### GPU 内存布局：

```
总容量: 79.10 GiB

已使用（报错时）:
  - Model weights: ~16 GiB (TP=4, 每个 GPU 4 GiB)
  - KV cache: ~60 GiB (gpu_memory_utilization=0.75)
  - Reserved: ~10 GiB (PyTorch 预留)
  ─────────────────
  已用: 72.24 GiB

可用: 6.85 GiB

尝试分配: 7.06 GiB (2 个请求的 logits)
→ OOM!
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 9. 为什么 batch_size=2 失败？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Python 层面：**
```python
# 每批只并发 2 个请求
batch_results = await asyncio.gather(
    *[compute_logprobs_async(seq) for seq in batch]  # batch 大小 = 2
)
```

**问题：**
即使 Python 只发送 2 个并发请求，vLLM 收到后仍然会：
1. 将这 2 个请求加入队列
2. 尝试批处理它们（提高 GPU 利用率）
3. 批处理时需要分配 `[total_tokens, vocab_size]` 的 logits
4. 2 个请求 × ~6000 tokens = 12000 tokens
5. `[12000, 151936]` = 7 GB → OOM!

**解决方案：**
`batch_size=1`（完全串行）
- Python 每次只发送 1 个请求
- vLLM 无法批处理（只有 1 个请求）
- 单个请求: `[~4000, 151936]` = 2.3 GB ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 10. 性能影响

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### batch_size=2 (失败)
```
并发: 2 个请求
速度: ~2x 单个请求
结果: OOM
```

### batch_size=1 (当前测试)
```
并发: 1 个请求（完全串行）
速度: 单个请求的基准速度
512 个请求总时间: 10-20 分钟（估计）
结果: 应该不会 OOM
```

### 理想情况（如果有更多内存）
```
batch_size=16
并发: 16 个请求
速度: ~16x 单个请求
512 个请求总时间: 1-2 分钟
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 11. 潜在优化方向

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **降低 gpu_memory_utilization**
   ```
   从 0.75 → 0.60
   释放 ~12 GiB 给 logits 计算
   可以支持 batch_size=2 或更大
   ```

2. **使用 vLLM V0**
   ```
   V0 可能有不同的批处理策略
   可能更容易控制内存使用
   ```

3. **修改 vLLM 源码**
   ```
   在 gather_logprobs 中优化内存分配
   例如：分块计算，避免一次性分配大 tensor
   ```

4. **使用更大的 GPU**
   ```
   H100 80GB → A100 80GB 或更大内存的 GPU
   但这不是根本解决方案
   ```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

