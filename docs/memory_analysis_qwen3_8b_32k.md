# Qwen3-8B + 32K 序列训练显存详细分析

**Date**: 2026-06-27
**Hardware**: NVIDIA H100 80GB SXM5
**Model**: Qwen3-8B (7.57B params), LoRA rank=32, bfloat16
**Sequence length**: 32,768 tokens

---

## 1. Qwen3-8B 模型架构参数

| 参数 | 值 |
|------|-----|
| hidden_size | 4,096 |
| num_hidden_layers | 36 |
| num_attention_heads | 32 (GQA) |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 12,288 (SwiGLU) |
| vocab_size | 151,936 |
| 总参数量 | 7,568M (7.57B) |

LoRA (rank=32, 7 target modules × 36 layers):
- 可训练参数: 87.29M (占总参数的 1.15%)
- LoRA 权重大小: 0.175 GB (bf16)

---

## 2. 各部分显存占用详解

### 2.1 模型权重 (Model Weights)

| 组件 | 计算 | 大小 |
|------|------|------|
| 基座模型 (frozen, bf16) | 7,568M × 2 bytes | **14.36 GB** |
| LoRA A+B 矩阵 (bf16) | 87.29M × 2 bytes | **0.175 GB** |
| **小计** | | **14.53 GB** |

基座模型参数不参与梯度计算（`requires_grad=False`），但必须常驻显存用于 forward pass。

### 2.2 优化器状态 (Optimizer State)

AdamW 为每个可训练参数维护 2 个 fp32 buffer：
- First moment (m): 指数移动平均梯度
- Second moment (v): 指数移动平均梯度平方

| 组件 | 计算 | 大小 |
|------|------|------|
| First moment (fp32) | 87.29M × 4 bytes | 0.349 GB |
| Second moment (fp32) | 87.29M × 4 bytes | 0.349 GB |
| **小计** | | **0.698 GB** |

优化器状态只针对 LoRA 参数（87M），而非全部 7.57B 参数。这是 LoRA 的核心优势。

### 2.3 梯度 (Gradients)

| 组件 | 计算 | 大小 |
|------|------|------|
| LoRA 梯度 (bf16) | 87.29M × 2 bytes | **0.175 GB** |

梯度同样只存在于可训练的 LoRA 参数上。基座模型参数的梯度不会被计算或存储。

### 2.4 激活内存 (Activations) — 核心差异所在

激活内存是 forward pass 过程中产生的中间张量，需要保留到 backward pass 使用。这是 32K 序列的显存瓶颈。

#### 每层 Transformer 的激活

对于 batch_size=1, seq_len=S=32768, hidden_dim=H=4096:

| 激活组件 | 形状 | 大小 (bf16) |
|---------|------|------------|
| 输入 (layer input) | [1, S, H] | 256 MB |
| Attention Q | [1, 32, S, 128] | 256 MB |
| Attention K | [1, 8, S, 128] | 64 MB |
| Attention V | [1, 8, S, 128] | 64 MB |
| Attention scores | [1, 32, S, S] (flash-attn 不存) | ~0 (flash-attn) |
| Attention output | [1, S, H] | 256 MB |
| FFN gate_proj output | [1, S, 12288] | 768 MB |
| FFN up_proj output | [1, S, 12288] | 768 MB |
| FFN SwiGLU activation | [1, S, 12288] | 768 MB |
| FFN down_proj input | [1, S, 12288] | 768 MB |
| Layer output | [1, S, H] | 256 MB |
| **每层总计** | | **~3.2 GB** |

> 注：使用 Flash Attention 2 时，attention scores ([S, S] 矩阵) 不会被显式存储，节省了 ~8 GB/层的显存。

#### 无 Gradient Checkpointing

所有 36 层的激活全部保留：

```
总激活 = 36 层 × 3.2 GB/层 = 115.2 GB
```

**远超 H100 80GB 容量 → 必然 OOM。**

#### 有 Gradient Checkpointing

Gradient checkpointing 将 36 层划分为若干 segment（约 √36 = 6 个 checkpoint），只在 segment 边界保留激活。Backward 时重新计算每个 segment 内的激活。

**存储的激活：**
```
存储激活 = √36 × 3.2 GB ≈ 6 × 3.2 GB = 19.2 GB
```

**峰值显存（backward 重算时）：**

在反向传播中，处理某个 segment 时，需要从 checkpoint 开始重新计算 segment 内所有层的激活。峰值发生在最后一个 segment 的 backward pass：

```
峰值激活 = 存储的 checkpoint 激活 + 当前 segment 重算的激活
         = 6 × 3.2 GB + 6 × 3.2 GB
         = 19.2 + 19.2 = 38.4 GB
```

### 2.5 临时缓冲区 (Temporary Buffers)

| 组件 | 大小 | 说明 |
|------|------|------|
| Logits 输出 | [1, S, V] = [1, 32768, 151936] × bf16 | **9.5 GB** |
| Log-softmax 中间结果 | 同上 (float32) | **19.0 GB** |
| Loss 计算临时 | ~0.1 GB | 标量运算 |

> **重要**: logits 张量 [S, V] 在 32K × 152K vocab 时非常大。但在当前实现中，我们只取 `logits[prompt_len-1 : prompt_len+target_len-1]`（target 部分），不需要完整的 [S, V] 张量。实际 target_len 远小于 S，所以 log_softmax 只在 target 切片上计算。

**实际临时缓冲区**（假设 target_len ≈ 8K tokens）：
```
target_logits: [8192, 151936] × bf16 ≈ 2.4 GB
log_softmax: [8192, 151936] × float32 ≈ 4.7 GB
实际临时缓冲区 ≈ 7-10 GB (峰值)
```

### 2.6 CUDA 碎片化与系统开销

| 组件 | 大小 | 说明 |
|------|------|------|
| CUDA context | ~0.5 GB | CUDA 运行时基础开销 |
| PyTorch allocator 碎片 | 5-21 GB | 取决于分配模式 |
| cuDNN workspace | ~0.5 GB | Flash attention 工作缓冲区 |

碎片化是 OOM 的主要隐患。PyTorch 默认 CUDA allocator 使用固定大小的内存块（segments），分配/释放模式不规则时会产生大量"已分配但未使用"的碎片。

**`expandable_segments:True` 的作用**：允许 allocator 动态调整 segment 大小，减少碎片。实测从 21 GB 碎片降到 <5 GB。

---

## 3. 总显存对比

### 3.1 无 Gradient Checkpointing

| 组件 | 大小 |
|------|------|
| 基座模型权重 | 14.36 GB |
| LoRA 权重 | 0.18 GB |
| 优化器状态 | 0.70 GB |
| 梯度 | 0.18 GB |
| 激活 (36 层全部保留) | **115.2 GB** |
| 临时缓冲区 | ~7 GB |
| CUDA 开销 + 碎片 | ~5 GB |
| **总计** | **~143 GB** |

**结论: 远超 H100 80GB → 必然 OOM。无 checkpoint 无法训练 32K 序列。**

### 3.2 有 Gradient Checkpointing

| 组件 | 大小 | 备注 |
|------|------|------|
| 基座模型权重 | 14.36 GB | 常驻 |
| LoRA 权重 | 0.18 GB | 常驻 |
| 优化器状态 | 0.70 GB | 常驻 |
| 梯度 | 0.18 GB | 训练时 |
| 存储的 checkpoint 激活 | 19.2 GB | ~6 个 checkpoint |
| 重算的 segment 激活 (峰值) | 19.2 GB | backward 时 |
| 临时缓冲区 (峰值) | ~7 GB | logits + softmax |
| CUDA 开销 | ~1 GB | |
| 碎片 (expandable_segments) | ~3 GB | 优化后 |
| 碎片 (默认 allocator) | ~15-21 GB | 优化前 |
| **总计 (expandable_segments)** | **~65 GB** | 可用余量 ~15 GB |
| **总计 (默认 allocator)** | **~77 GB** | 余量 <3 GB，**极易 OOM** |

**结论**:
- **必须** 开启 gradient checkpointing
- **必须** 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 即使两者都开启，峰值仍达 ~65 GB / 80 GB = 81% 利用率
- 不同 token 长度的序列会导致不同的峰值，最长的 32K 序列最危险

---

## 4. Gradient Checkpointing 的代价

| 方面 | 无 Checkpointing | 有 Checkpointing |
|------|-----------------|-----------------|
| Forward pass | 1 次 | 1 次 |
| Backward 重算 forward | 0 次 | ~1 次（每个 segment） |
| **总计算量** | 1× forward + 1× backward | **~2× forward** + 1× backward |
| **计算开销** | 基准 | **+33%** (forward ≈ backward 的 1/2) |
| 激活内存 | 115 GB (36 层全存) | 38.4 GB (峰值) |
| **内存节省** | 基准 | **-67%** |

trade-off: 用 33% 的额外计算换取 67% 的内存节省。对于 32K 序列，这是唯一可行的方案。

---

## 5. 多 GPU 方案的显存与速度影响分析

### 5.1 当前方案: DistributedTrainingClient（多线程并行）

**架构**: 每个 GPU 持有完整的模型副本，并行处理不同的数据。

**显存影响**:

| 组件 | 单 GPU | 4 GPU (每个) | 变化 |
|------|--------|-------------|------|
| 基座模型 | 14.36 GB | 14.36 GB | 无变化（每 GPU 完整副本）|
| LoRA 权重 | 0.18 GB | 0.18 GB | 无变化 |
| 优化器 | 0.70 GB | 0.70 GB (仅 primary) | 其他 GPU 无优化器 |
| 梯度 | 0.18 GB | 0.18 GB | 无变化 |
| 激活 (峰值) | 38.4 GB | 38.4 GB | 无变化 |

**结论**: 多线程并行 **不节省每 GPU 显存**。每个 GPU 的占用与单 GPU 相同。

**速度影响**:

```
单 GPU: 512 个序列串行处理 → 512 × t_per_seq
4 GPU:  128 个序列/GPU 并行处理 → 128 × t_per_seq

理论加速比: 4×
```

实际加速受限于：
1. **梯度同步开销**: LoRA 梯度 ~175 MB，GPU 间拷贝 ~5-10ms（NVLink 900GB/s）
2. **权重广播开销**: optim_step 后同步 LoRA 权重 ~175 MB，~5-10ms
3. **数据不均衡**: 序列长度不同，最长序列的 GPU 决定整体时间
4. **GIL 竞争**: Python GIL 在非 CUDA 操作时限制并行度

实测结果（Qwen3-8B, 64 sequences, 32K, 4 GPU）:
- 训练时间: 906s
- 预估单 GPU 时间: ~3,200s（基于之前的性能数据）
- **实际加速比: ~3.5×**（考虑同步和负载不均开销）

### 5.2 DeepSpeed ZeRO-2 的理论收益

ZeRO-2 分片优化器状态和梯度，但保留完整的模型参数。

**显存影响**:

| 组件 | 单 GPU | ZeRO-2 (4 GPU, 每个) | 节省 |
|------|--------|---------------------|------|
| 基座模型 | 14.36 GB | 14.36 GB | 无 |
| LoRA 权重 | 0.18 GB | 0.18 GB | 无 |
| 优化器 | 0.70 GB | **0.175 GB** | -0.525 GB |
| 梯度 | 0.18 GB | **0.045 GB** | -0.135 GB |
| 激活 (峰值) | 38.4 GB | 38.4 GB | 无 |
| **总节省/GPU** | | | **~0.66 GB** |

**结论**: 对于 LoRA 训练，ZeRO-2 的显存节省 **微不足道**（0.66 GB / 65 GB ≈ 1%）。原因是可训练参数只有 87M（全部参数的 1.15%），优化器状态和梯度本来就很小。

ZeRO-2 的真正价值体现在 **full fine-tuning** 场景：
- Full fine-tuning 优化器状态: 7,568M × 4 × 2 = **60.5 GB** → ZeRO-2 (4 GPU): **15.1 GB/GPU**
- Full fine-tuning 梯度: 7,568M × 2 = **15.1 GB** → ZeRO-2 (4 GPU): **3.8 GB/GPU**

### 5.3 DeepSpeed ZeRO-3 的理论收益

ZeRO-3 额外分片模型参数本身。

| 组件 | 单 GPU | ZeRO-3 (4 GPU, 每个) | 节省 |
|------|--------|---------------------|------|
| 基座模型 | 14.36 GB | **3.59 GB** | -10.77 GB |
| LoRA 权重 | 0.18 GB | 0.18 GB | 无 (trainable 不分片) |
| 优化器 | 0.70 GB | 0.175 GB | -0.525 GB |
| 梯度 | 0.18 GB | 0.045 GB | -0.135 GB |
| 激活 (峰值) | 38.4 GB | 38.4 GB | 无 |
| 通信缓冲区 | 0 | ~2 GB | +2 GB |
| **总节省/GPU** | | | **~9.4 GB** |

ZeRO-3 节省显著（~9.4 GB），但引入：
- 参数收集延迟（forward/backward 中需要从其他 GPU 拉取参数）
- 更复杂的 checkpoint 保存（需要 gather 参数）
- LoRA `save_pretrained()` 需要额外处理

---

## 6. 不同方案的加速机制对比

| 方案 | 加速来源 | 显存节省 | 复杂度 | 适用场景 |
|------|---------|---------|--------|---------|
| **多线程并行 (当前)** | 数据并行: N GPU 同时处理不同数据 | 无 | 低 | LoRA 训练，GPU 数量 ≤8 |
| **DeepSpeed ZeRO-2** | 数据并行 + NCCL 通信优化 | ~1% (LoRA) | 高 | Full fine-tuning |
| **DeepSpeed ZeRO-3** | 数据并行 + 模型参数分片 | ~14% (LoRA) | 很高 | 超大模型 (70B+) |
| **Gradient checkpointing** | 无加速（反而 +33% 计算） | 67% 激活内存 | 无 | 长序列 (≥16K) |
| **expandable_segments** | 无加速 | 回收碎片 10-15 GB | 无 | 所有场景 |
| **Flash Attention 2** | ~2× attention 速度 | ~8 GB/层 (不存 scores) | 无 | 所有场景 |

---

## 7. 关键结论

1. **激活内存是绝对瓶颈**: 115 GB (无 ckpt) vs 38 GB (有 ckpt)，是模型权重的 2.5-8 倍

2. **LoRA 使优化器/梯度开销可忽略**: 0.87 GB vs full fine-tuning 的 75 GB

3. **多线程并行的价值纯粹是吞吐量**: 不省内存，但 4 GPU 给出 ~3.5× 加速

4. **ZeRO-2 对 LoRA 训练几乎无意义**: 只节省 0.66 GB（<1%），远不如 expandable_segments（节省 10-15 GB）和 gradient checkpointing（节省 77 GB）

5. **ZeRO-3 对 LoRA 有中等意义**: 节省 9.4 GB 模型权重显存，但增加通信复杂度。只在需要更大 batch_size 或更长序列时值得

6. **32K + 8B 的实际峰值**: ~65 GB (优化后) / ~77 GB (优化前)。H100 80GB 是训练此规模模型的**最小**可行硬件

---

## 8. 显存优化优先级

| 优先级 | 措施 | 节省 | 状态 |
|--------|------|------|------|
| **1** | Gradient checkpointing | -77 GB | ✅ 已启用 |
| **2** | Flash Attention 2 | -8 GB/层 (避免存 scores) | ✅ 已启用 |
| **3** | expandable_segments | -10~15 GB (碎片回收) | ✅ 已启用 |
| **4** | LoRA (vs full fine-tuning) | -75 GB (优化器+梯度) | ✅ 已使用 |
| 5 | ZeRO-3 模型参数分片 | -9.4 GB | ❌ 未实现 |
| 6 | ZeRO-2 优化器/梯度分片 | -0.66 GB | ❌ 不值得 |

前 4 项已全部启用，是当前能正常训练 32K 序列的前提条件。
