# Rollout Length Statistics Report

**Model**: Qwen/Qwen3-8B | **Context Window**: 32,768 tokens | **TP**: 4 | **Temperature**: 1.0

**Sampling Design**:
- **Round 1**: 1 prompt × 50 completions = 50 samples（初始状态，无历史上下文）
- **Round 2**: 8 prompts × 6 completions = 48 samples（PUCT state reuse，8个top states）

---

## 总览

| Task | Phase1 Max Tokens | Round 1 Prompt | Round 2 Prompt | Round 1 Gen (avg) | Round 2 Gen (avg) | Round 1 Total (avg) | Round 2 Total (avg) |
|---|---|---|---|---|---|---|---|
| Circle Packing | 26,000 | 846 | 846 / **2,406**¹ | 11,557 | 11,384 | 12,403 | 12,425 |
| AHC | **22,000** | 12,978 | 12,978 | 15,123 | 15,806 | 28,101 | 28,784 |
| GPU Mode | 26,000 | 2,722 | 2,722 | 13,225 | 12,910 | 15,947 | 15,632 |
| Denoising | 26,000 | 1,489 | 1,489 | 10,516 | 11,052 | 12,005 | 12,541 |

> ¹ Circle Packing Round 2 中有 6/48 个样本的 prompt 被扩充至 2,406 tokens（1个PUCT state携带了历史context）

---

## Context Window 利用率

> 越接近 32,768 说明模型生成越接近上限

| Task | Round 1 avg total | 利用率 | Round 1 max total | 最大利用率 |
|---|---|---|---|---|
| Circle Packing | 12,403 | 37.8% | 22,440 | 68.5% |
| GPU Mode | 15,947 | 48.7% | 20,714 | 63.2% |
| Denoising | 12,005 | 36.6% | 15,442 | 47.1% |
| **AHC** | **28,101** | **85.8%** | **32,718** | **99.8%** |

⚠️ **AHC 接近 context window 上限**：avg 28,101（85.8%），max 32,718（100%，被截断）

---

## 各 Task 详细统计

### Circle Packing

**配置**: `phase1_max_tokens=26000`，prompt 固定 846 tokens

#### Generation Length

| 统计量 | Round 1 (n=50) | Round 2 (n=48) | 变化 |
|---|---|---|---|
| Min | 7,622 | 7,754 | +132 |
| Max | 21,594 | 18,188 | -3,406 |
| **Avg** | **11,557** | **11,384** | **-173** |
| Median | 11,241 | 11,179 | -62 |
| Std | 2,460 | 1,909 | ↓ 更集中 |
| P25 | 9,644 | 10,035 | +391 |
| P75 | 12,707 | 12,262 | -445 |
| P90 | 14,349 | 13,565 | -784 |
| P95 | 15,154 | 14,546 | -608 |

#### Total Length

| 统计量 | Round 1 | Round 2 |
|---|---|---|
| Min | 8,468 | 9,604 |
| Max | 22,440 | 19,034 |
| **Avg** | **12,403** | **12,425** |
| Median | 12,087 | 12,323 |

**观察**：Round 2 的 prompt 结构：8个PUCT state中1个有历史context（prompt=2406），7个无（prompt=846）。有context的样本的generation长度因budget减少而略短。

---

### AHC

**配置**: `phase1_max_tokens=22000`，prompt 固定 12,978 tokens（题面本身很长）

> AHC prompt 占可用 budget 的 **59%**（12,978 / 22,000），留给生成的空间仅约 9,000 tokens

#### Generation Length

| 统计量 | Round 1 (n=50) | Round 2 (n=48) | 变化 |
|---|---|---|---|
| Min | 8,504 | 10,550 | +2,046 |
| Max | 19,740 | 19,740 | 0 |
| **Avg** | **15,123** | **15,806** | **+683** |
| Median | 15,410 | 16,214 | +804 |
| Std | 2,476 | 2,436 | ≈ 相同 |
| P25 | 13,973 | 14,443 | +470 |
| P75 | 16,730 | 17,467 | +737 |
| P90 | 17,484 | 18,887 | +1,403 |
| P95 | 18,773 | 19,601 | +828 |

#### Total Length

| 统计量 | Round 1 | Round 2 |
|---|---|---|
| Min | 21,482 | 23,528 |
| Max | **32,718** | **32,718** |
| **Avg** | **28,101** | **28,784** |
| Median | 28,388 | 29,192 |

**观察**：
- AHC 的 prompt 在 round1 和 round2 中完全一致（12,978），说明 round1 的 rollout 未产生被 PUCT 选中并携带 context 的 state
- Max total = 32,718 ≈ 32,768（context 上限），说明存在样本被截断
- Round 2 generation 比 Round 1 更长（avg +683），分布更向右偏移

---

### GPU Mode

**配置**: `phase1_max_tokens=26000`，prompt 固定 2,722 tokens

#### Generation Length

| 统计量 | Round 1 (n=50) | Round 2 (n=48) | 变化 |
|---|---|---|---|
| Min | 8,789 | 9,441 | +652 |
| Max | 17,992 | 17,357 | -635 |
| **Avg** | **13,225** | **12,910** | **-315** |
| Median | 13,111 | 12,605 | -506 |
| Std | 2,002 | 1,775 | ↓ 更集中 |
| P25 | 11,848 | 11,601 | -247 |
| P75 | 14,312 | 14,100 | -212 |
| P90 | 15,776 | 15,183 | -593 |
| P95 | 16,572 | 15,626 | -946 |

#### Total Length

| 统计量 | Round 1 | Round 2 |
|---|---|---|
| Min | 11,511 | 12,163 |
| Max | 20,714 | 20,079 |
| **Avg** | **15,947** | **15,632** |
| Median | 15,833 | 15,327 |

**观察**：Round 2 的 prompt 长度与 Round 1 完全相同（2,722），说明 PUCT reuse 未触发 context 追加（round1 未产生有价值的 state）。Generation 略短，分布更集中。

---

### Denoising

**配置**: `phase1_max_tokens=26000`，prompt 固定 1,489 tokens

#### Generation Length

| 统计量 | Round 1 (n=50) | Round 2 (n=48) | 变化 |
|---|---|---|---|
| Min | 7,266 | 7,234 | -32 |
| Max | 13,953 | 15,064 | +1,111 |
| **Avg** | **10,516** | **11,052** | **+536** |
| Median | 10,604 | 11,317 | +713 |
| Std | 1,520 | 1,712 | ↑ 略分散 |
| P25 | 9,491 | 9,654 | +163 |
| P75 | 11,463 | 11,965 | +502 |
| P90 | 12,788 | 13,287 | +499 |
| P95 | 13,096 | 13,745 | +649 |

#### Total Length

| 统计量 | Round 1 | Round 2 |
|---|---|---|
| Min | 8,755 | 8,723 |
| Max | 15,442 | 16,553 |
| **Avg** | **12,005** | **12,541** |
| Median | 12,093 | 12,806 |

**观察**：Denoising 的 prompt 长度 round1 和 round2 完全一致（1,489）。但 generation 长度 round2 比 round1 更长（avg +536），分布向右偏移，表明模型在相同 prompt 下产生了更多内容。

---

## 跨 Task 对比

### Round 1：Prompt 长度对比

```
Circle Packing │   846 ████ 
Denoising      │  1489 ███████
GPU Mode       │  2722 █████████████
AHC            │ 12978 ████████████████████████████████████████████████████████████████
                 0                                                              13000
```

### Round 1：Generation 长度分布（avg ± std）

```
Circle Packing │ ←────── 11,557 ─────→│ std=2,460  [7,622 ~ 21,594]
Denoising      │ ←──── 10,516 ────→│   std=1,520  [7,266 ~ 13,953]
GPU Mode       │ ←────── 13,225 ──────→│ std=2,002  [8,789 ~ 17,992]
AHC            │ ←──────────── 15,123 ────────────→│ std=2,476 [8,504 ~ 19,740]
               0     5,000    10,000    15,000    20,000
```

### Round 1 vs Round 2：Generation avg 变化

| Task | R1 avg | R2 avg | 变化 | 方向 |
|---|---|---|---|---|
| Circle Packing | 11,557 | 11,384 | -173 | ↓ 略短 |
| GPU Mode | 13,225 | 12,910 | -315 | ↓ 略短 |
| AHC | 15,123 | 15,806 | **+683** | ↑ 略长 |
| Denoising | 10,516 | 11,052 | **+536** | ↑ 略长 |

---

## 关键发现

**1. PUCT State Reuse 效果有限（仅 Circle Packing 有少量触发）**

只有 Circle Packing 的 Round 2 中出现了更长的 prompt（2,406 vs 846），对应 8 个 PUCT state 中有 1 个携带了历史 context（该 state 在 round1 中产生了有效 solution）。其余 3 个 task 的 round2 prompt 与 round1 完全相同，说明 round1 的 rollout 均未产生被 PUCT 选中并带入 context 的成功 state。

**2. AHC 的 prompt 本身占据大量 context**

AHC prompt 高达 12,978 tokens（占 max_model_len 的 39.6%，占 phase1_max_tokens 的 59.1%），留给 generation 的有效空间仅约 9,000 tokens。导致 total length 平均 28,101，逼近 32,768 上限。**建议适当增大 `max_model_len` 或缩减 AHC 的问题描述长度**。

**3. Generation 长度的任务差异显著**

- **最短**：Denoising（avg 10,516）
- **最长**：AHC（avg 15,123），比 Denoising 长约 44%
- 标准差最大的是 Circle Packing 和 AHC（~2,460），说明这两个任务的生成长度最不稳定

**4. Context Window 利用率差异大**

AHC 的平均利用率高达 85.8%（max 达 100%），而其他三个 task 均在 50% 以下。这说明 AHC 相对于其他 task 的 context 压力更大，在实际训练中可能更容易遇到 OOM 或截断问题。
