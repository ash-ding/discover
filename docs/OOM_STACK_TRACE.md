# OOM 完整调用链路分析

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 完整 Stack Trace (从下往上读)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
1. multiproc_executor.py:465 - worker_busy_loop()
   ↓ 
2. torch/utils/_contextlib.py:116 - decorate_context()
   ↓
3. gpu_worker.py:268 - execute_model()
   ↓ output = self.model_runner.execute_model(scheduler_output)
   ↓
4. torch/utils/_contextlib.py:116 - decorate_context()
   ↓
5. gpu_model_runner.py:1173 - execute_model()
   ↓ prompt_logprobs_dict = self._get_prompt_logprobs_dict(...)
   ↓
6. gpu_model_runner.py:1422 - _get_prompt_logprobs_dict()
   ↓ token_ids, logprobs, ranks = self.sampler.gather_logprobs(...)
   ↓
7. sampler.py:171 - gather_logprobs()
   ↓ token_ranks = (logprobs >= token_logprobs).sum(-1)
   ↓
   ⚠️ torch.OutOfMemoryError: CUDA out of memory
      Tried to allocate 8.78 GiB
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 详细调用分析

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Level 1: Worker 主循环
```
文件: vllm/v1/executor/multiproc_executor.py:465
函数: worker_busy_loop()

作用: vLLM 多进程执行器的工作进程主循环
说明: 这是 vLLM V1 的 worker 进程，处理推理请求
```

### Level 2: GPU Worker 执行
```
文件: vllm/v1/worker/gpu_worker.py:268
函数: execute_model()
代码: output = self.model_runner.execute_model(scheduler_output)

作用: GPU worker 执行模型推理
说明: 接收调度器输出，调用 model_runner 执行实际推理
```

### Level 3: Model Runner 执行 ⭐
```
文件: vllm/v1/worker/gpu_model_runner.py:1173
函数: execute_model()
代码: prompt_logprobs_dict = self._get_prompt_logprobs_dict(...)

作用: 执行模型并计算 prompt logprobs
说明: 这里是处理 echo=True 请求的入口
      会调用 _get_prompt_logprobs_dict 来计算整个 prompt 的 logprobs
```

### Level 4: 获取 Prompt Logprobs ⭐⭐
```
文件: vllm/v1/worker/gpu_model_runner.py:1422
函数: _get_prompt_logprobs_dict()
代码: token_ids, logprobs, ranks = self.sampler.gather_logprobs(...)

上下文代码:
```python
def _get_prompt_logprobs_dict(...):
    # ... 前面省略
    
    for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
        # 获取该请求的 tokens 数量
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        
        # 获取 hidden states
        req_idx = self.input_batch.req_id_to_index[req_id]
        offset = self.query_start_loc_np[req_idx].item()
        prompt_hidden_states = hidden_states[offset:offset + num_logits]
        
        # 计算 logits (第一次大内存分配)
        logits = self.model.compute_logits(prompt_hidden_states, None)
        # → logits shape: [num_tokens, vocab_size]
        # → 例如: [15500, 151936] × 4 bytes = 8.78 GiB
        
        # 计算 logprobs
        logprobs = self.sampler.compute_logprobs(logits)
        # → logprobs shape: [num_tokens, vocab_size]
        
        # 获取目标 token IDs
        tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]
        
        # 调用 gather_logprobs (这里 OOM！)
        token_ids, logprobs, ranks = self.sampler.gather_logprobs(
            logprobs,              # [num_tokens, vocab_size]
            num_prompt_logprobs,   # 1
            tgt_token_ids          # [num_tokens]
        )
```

作用: 循环处理每个 echo=True 请求，计算其 prompt logprobs
说明: 
  - 虽然是 for 循环，但每个请求独立处理
  - 每次循环需要计算该请求的完整 logits
  - 内存分配量取决于 num_tokens (序列长度)
```

### Level 5: Gather Logprobs ⚠️ OOM HERE
```
文件: vllm/v1/sample/sampler.py:171
函数: gather_logprobs()
代码: token_ranks = (logprobs >= token_logprobs).sum(-1)

完整函数:
```python
def gather_logprobs(
    self,
    logprobs: torch.Tensor,      # Shape: [num_tokens, vocab_size]
    num_logprobs: int,            # 通常是 1
    token_ids: torch.Tensor,      # Shape: [num_tokens]
) -> LogprobsTensors:
    """
    Gather logprobs for topk and sampled/prompt token.
    """
    assert token_ids.dtype == torch.int64
    
    # Find the topK values.
    topk_logprobs, topk_indices = torch.topk(
        logprobs,
        num_logprobs,
        dim=-1
    )
    
    # Get with the logprob of the prompt or sampled token.
    token_ids = token_ids.unsqueeze(-1)              # [num_tokens, 1]
    token_logprobs = logprobs.gather(-1, token_ids)  # [num_tokens, 1]
    
    # Compute the ranks of the actual token.
    # ⚠️⚠️⚠️ OOM 发生在这里！⚠️⚠️⚠️
    token_ranks = (logprobs >= token_logprobs).sum(-1)
    #             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #             这个比较操作创建临时 tensor
    #
    # 内存分配:
    # - logprobs: [15500, 151936] (已存在)
    # - token_logprobs: [15500, 1] (很小)
    # - 广播: token_logprobs → [15500, 151936]
    # - 比较: logprobs >= token_logprobs
    #   → 创建 [15500, 151936] 的布尔 tensor
    #   → PyTorch 可能用 float32 存储中间结果
    #   → 需要分配: 15500 × 151936 × 4 = 8.78 GiB
    # - 但只有 7.75 GiB 可用
    # - → OOM!
    
    return LogprobsTensors(
        logprob_token_ids=topk_indices,
        logprobs=topk_logprobs,
        selected_token_ranks=token_ranks,
    )
```

作用: 计算每个 token 在 vocab 中的 rank
说明:
  - rank = 有多少个 token 的 logprob 大于等于当前 token
  - 这个 rank 用于统计和调试
  - 但计算 rank 需要与整个 vocab 比较
  - 对于大 vocab (151936) 和长序列，内存需求巨大
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 内存分配时间线

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
时刻 0: GPU 状态
  - Total: 79.10 GiB
  - Used: 71.33 GiB
    - Model weights: ~16 GiB (TP=4)
    - KV cache: ~60 GiB (gpu_memory_utilization=0.75)
    - PyTorch reserved: 8.01 GiB (未分配)
  - Free: 7.75 GiB

时刻 1: gpu_model_runner.py:1422 - compute_logits()
  分配: logits [15500, 151936] × 4 bytes
  需求: 8.78 GiB
  状态: 可能成功 (使用 PyTorch reserved 空间)
        或者这里已经接近极限

时刻 2: sampler.py:171 - 比较操作
  分配: 临时 tensor [15500, 151936] × 4 bytes
  需求: 8.78 GiB
  状态: ❌ OOM!
  
  原因:
  1. logits tensor 占用了大部分可用内存
  2. 再分配一个同样大小的临时 tensor 超出限制
  3. PyTorch 无法找到足够的连续内存块
  4. 内存碎片化 (8.01 GiB reserved 但未分配)
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 为什么 Epoch 1 成功，Epoch 2 失败？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Epoch 1 (成功)
```
序列长度: ~4000-5000 tokens
logits 大小: [5000, 151936] × 4 = 2.84 GiB
临时 tensor: [5000, 151936] × 4 = 2.84 GiB
总需求: ~5.7 GiB
可用内存: ~8 GiB
结果: ✅ 成功
```

### Epoch 2 (失败)
```
序列长度: ~15500 tokens
logits 大小: [15500, 151936] × 4 = 8.78 GiB
临时 tensor: [15500, 151936] × 4 = 8.78 GiB
总需求: ~17.6 GiB
可用内存: 7.75 GiB
结果: ❌ OOM

原因分析:
1. 训练后 policy 生成更长序列
   - Phase 1 更容易达到 4000 limit
   - Phase 2 也更长
   
2. 或者 vLLM 批处理了多个请求
   - 即使 Python batch_size=1
   - vLLM 可能在队列中累积了请求
   - 2 个 7500 tokens 的请求 → 15000 tokens
   
3. 内存碎片化
   - PyTorch reserved: 8.01 GiB 但未分配
   - 无法找到连续的 8.78 GiB 空间
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 多 GPU 同时 OOM

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

从日志看，4 个 GPU (rank 0-3) 都同时 OOM：

```
GPU 0: Tried to allocate 8.78 GiB, 7.75 GiB free
GPU 1: Tried to allocate 8.78 GiB, 7.39 GiB free
GPU 2: Tried to allocate 8.78 GiB, 7.39 GiB free
GPU 3: Tried to allocate 8.78 GiB, 7.70 GiB free
```

说明:
- TP=4 时，每个 GPU 都执行相同的操作
- 每个 GPU 都尝试分配自己的 logits tensor
- 由于 TP 并行，所有 GPU 几乎同时 OOM
- 这是 tensor parallel 的正常行为

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 关键代码位置总结

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **入口点**: 
   `gpu_model_runner.py:1173`
   这里决定是否计算 prompt logprobs (echo=True)

2. **循环处理**: 
   `gpu_model_runner.py:1422`
   对每个 echo=True 请求计算 logits 和 logprobs

3. **OOM 点**: 
   `sampler.py:171`
   计算 token ranks 时的比较操作

4. **根本原因**:
   需要两个 [num_tokens, vocab_size] 的大 tensor
   - logits/logprobs (必需)
   - 比较操作的临时 tensor (可优化)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

