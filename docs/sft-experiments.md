# SFT-vs-RL 实验计划与调研记录

写于 2026-09-29。本文档整理"用 SFT 替代/辅助 TTT-Discover 的 RL 训练"这一系列实验的
动机、已完成的数据验证、实验分类（A1 / A2 / B1 / B2）、SFT 数据规范和相关工作。
所有数字都是从共享盘 `~/data/discover-runs/`（lumen 视角）的原始 rollout 记录中用
float64 重新计算的，不是转述。

> 事实更正（三处常见记忆偏差）：我们所有 run 用的是 **Qwen3-8B**（不是 32B）；
> 每步 512 rollouts = 8 parents × 64（这个是对的）；gpt-oss-120b 在 Erdős 第 0 步
> 的有效率是 **66–73%**（不是 90%）。

---

## 1. 核心 hypothesis

RL 训练（VERL/GRPO）在这些搜索任务上的主要作用，是把 base 模型从"不太有效的
mutator"变成"有效的 mutator"——即把产不出合法 mutation 的那部分 rollout 变成合法，
而不是提高"已经合法的 mutation"的质量。若如此，则一次拒绝采样 + SFT 就应能获得
RL 的大部分收益。

### 1.1 支持证据（已验证）

**"有效率"定义：`raw_score > 0`**，即整条判定链全通：有 code block → 沙箱执行
不崩 → 不超时（erdos 1100s 硬限）→ 返回签名正确 → h 合法（1-D、值域 [0,1]）→
通过反作弊复算。任何合法 h 的 C5 必为正，所以 reward 非零 ⇔ 产出了能进 archive
的合法构造。

有效率（前 5 步均值 → 后 5 步均值）：

| 设置 | 第 1 步 | 前 5 步 | 后 5 步 |
|---|---|---|---|
| Erdős VERL 训练 ×3 | 39–42% | 41–44% | 47–56% |
| Erdős 纯推理 ×3 | 26–44% | 33–44% | 44–50% |
| AC1 VERL 训练 ×2 | 42–51% | 43–55% | **86–89%** |
| AC1 纯推理 ×2 | 43% | 48% | 73% |
| gpt-oss Erdős 纯推理 | 66% | 73% | 33%（随 parent 变难而降） |

- AC1（训练优势决定性的任务）上训练把有效率推到 89%，净超纯推理 16pt；
  Erdős（训练优势边缘）上净差只有 ~5pt。**"RL 收益 = 有效率提升空间被兑现的
  程度"与任务排序吻合。**
- 所有 9 个 run 的 mean reward 与有效率的 Pearson r = 0.993–0.999（部分机械性：
  无效记 0 分）。
- **只看合法 rollout 的平均 raw_score，训练和纯推理的轨迹几乎相同**
  （0.47→0.39 vs 0.48→0.40）：训练没有提高成功 mutation 的质量，改善来自
  parent 变好。这是 hypothesis 最核心的一条。

### 1.2 重要混淆与失败归因（已验证）

**纯推理 run 的有效率也在涨**（模型没动）：Erdős 三个纯推理 run 前5→后5 的变化为
+17.1 / +5.5 / −0.3，均值 ≈ +7.4，与训练 run 的 +3.1 / +6.3 / +14.7（均值 ≈ +8）
**同量级同散布**。原因是 parent 分布漂移（archive 变好）。
→ **在 Erdős 上，任何时间序列前后对比都不构成训练效果的证据。**

Erdős 训练 run 的零分归因（用 `eval_msg` 逐条分类，前5步 → 后5步）：

| 类别 | orig-0909 | 1e9-repeat | 1e9-repeat2 | gradclip1(对照) |
|---|---|---|---|---|
| VALID | 43.8→46.9 | 43.7→50.0 | 41.0→55.7 | 45.6→**39.8**↓ |
| crash（真实 Python 错误） | 32.1→**19.3** | 31.4→**22.1** | 33.1→32.2 | 32.0→**40.3**↑ |
| 真超时 | 1.2→**10.8** | 2.5→**16.9** | 2.1→0.7 | 3.7→2.0 |
| h 越界 | 11.6→13.6↑ | 10.4→**2.8** | 12.2→**2.8** | 11.8→10.0 |
| 反作弊 | 6.9→4.9 | 7.5→3.8 | 6.2→5.6 | 3.6→3.1 |

- **跨 run 一致的只有两条**：VALID 上升（幅度 3–15pt 不等）、反作弊小幅下降。
  其余类别的改善是 run 特有的随机路径（"三个 run 学到不同的风格"）。
- 超时已核实为**真执行超时**（`subprocess.TimeoutExpired`，进程真跑满 1105s；
  BUG-008 排队计费修复在该 run 的 commit 中已存在；`eval_time_s` 最小值恰为
  1105）。428/432 超时程序 = 单次不可中断的 `differential_evolution` 调用，
  只有 6/432 检查墙钟。超时上涨机制推测：贵程序得分好 → 进 archive 当
  parent → 子代继续加码。
- grad_clip=1.0 对照 run 经历同样的 parent 漂移但 VALID 反降、crash 反升——
  说明有效梯度是合法性改善的必要条件（弱证据，见上面的混淆）。

**结论性方法论：唯一可靠的机制度量是"固定 parent 探针"**——同一批存档 parent、
不同模型（base / RL / SFT）、各生成 N 条、在相同 parent 分布下比较 VALID 率和
失败分解。SFT 实验中此探针为主要机制指标。

---

## 2. 实验分类

| 编号 | 设计 | 算法定位 | 回答的问题 |
|---|---|---|---|
| **A1** | 一次性 SFT → inference-only 50 步 | 拒绝采样 SFT（单轮 ReST） | 有效性能否离线教会，完全不需要 RL（**主假设**） |
| **A2** | 一次性 SFT → 继续 50 步 GRPO | SFT warm-start + 在线 RL | SFT 与 RL 的收益是否叠加；warm-start 是否加速 RL |
| **B1** | 50 步 GRPO，每 5 步插一轮 SFT | 在线 RL + 周期性自蒸馏（混合） | 自蒸馏能否加速/稳定在线 RL |
| **B2** | 无 GRPO：每 5 步纯采样 → 过滤 → SFT → 继续 | **正宗 ReST / Expert Iteration** | 迭代版 A1：纯 SFT 循环能否追平在线 RL |
| 对照 | 已有 base 纯推理 ×3、RL ×3（AC1/Erdős 各） | — | — |

B1 与 B2 互为对照：唯一区别是采样间隙是否用 RL 更新权重。
A1 → B2 → B1 构成"零 RL → 迭代离线 RL → 在线 RL + 蒸馏"的梯度，与纯 RL 对照
一起夹逼出"RL 在线成分的净贡献"。

优先级：**A1 先跑**（主假设、成本最低、对照最全）→ A2（复用 A1 checkpoint）→
B2 / B1 视 A1 结果决定（若 A1 追平 RL，B 系的增量问题自动消解）。

主任务 **AC1**（训练优势决定性、base 自产带文本数据现成）；Erdős 第二
（可加 gpt-oss 蒸馏臂）。机器：lumen2 + lumen3（各 8×H100）。

判读标准（预注册）：
- 主指标：50 步最终 best raw_score。AC1 对照：RL 1.5045–1.5051，纯推理
  1.5057–1.5074，目标 1.5030。
- 机制指标：固定 parent 探针的 VALID 率与失败分解；第 1 步有效率；全程轨迹。
- SFT ≈ RL → 机制假设成立且 SFT 是便宜替代；SFT 居中 → RL 有真在线成分；
  SFT 第 1 步即命中数据内最优构造且不再改进 → memorization，收紧数据。

---

## 3. SFT 数据

### 3.1 可用池（已盘点，Erdős 三个 1e9 训练 run 合并）

| 指标 | 数量 |
|---|---|
| VALID 且带完整 input+output 文本 | **37,856** |
| 其中严格优于 parent（improved） | 9,911 |
| 唯一 parent（= 唯一 prompt） | **1,200** |
| 前 5 步（策略≈base）的 VALID | 3,287 |
| 单条 output 长度 | 中位 ~10.3k token，p90 ~12.5k |
| 完全重复的 code | 0（temperature=1.0） |

AC1：VERL 训练 run 3 个（~2 万 VALID/run，带文本）+ 纯推理 `-repeat`/`-repeat2`
（共 ~3.5 万 VALID，带文本，**base 自产、最干净的 on-policy 数据**）。

**数据阶段决定实验含义**：全池（含后期 step）= 蒸馏 RL 成果，检验"离线 SFT 能否
复制 RL"；仅前 5 步 = 拒绝采样 base 输出，检验假设的最纯形式。两版都做
（主实验用全池，前 5 步版作机制消融）。

### 3.2 规模与配比

约束在 prompt 侧多样性（仅 1,200 个 parent），不在总量：
- **主实验 3–5k 条**：每 parent 限 3–4 条，按 raw_score 分层采样。
  ≈40M token/epoch，8×H100 LoRA rank 32（与 RL 一致）1–2 epoch，数小时。
- 消融：1k improved-only；12k 不设上限。
- **B1/B2 每轮只用最近 5 步新鲜数据**（~1.1–1.4k VALID，parent 限额后
  500–800 条），1 epoch，小学习率。新鲜度 > 数量。
  B2 另有 ReST^EM 式选择：每轮从 base 重训（默认，更稳）vs 续训上轮。
- **污染基线**：池内最优构造（Erdős ≈0.38092）必须随结果一并报告。
- A2/B1 严格对比时 SFT 数据应只来自该 arm 自己可见的历史。

### 3.3 数据可用性备注（历史坑）

- Erdős 三个**纯推理** run：老记录格式，无完整文本、无 input、
  `parent=id(state)%1000`（内存哈希，不可恢复）；`-repeat2` 连逐条分数都没有。
  失败归因（crash/超时/越界）对这些 run 不可做。
- gpt-oss 已归档 run：同样无文本。lumen1 在跑的 `gptoss-erdos-50step-repeat2`
  有 `text` 但 **reasoning（harmony analysis channel）在 #17 之前被丢弃**
  （vLLM 放在 `reasoning_content`，旧代码只读 `content`；实测 44840 out_tokens
  只存下 ~1.5k token）。蒸馏臂可用其 final answer + code。
- **discover#17（已合并，2026-09-29）**修复：每条 rollout 现在持久化
  `prompt`（精确 user message）、`reasoning`、`puct_parent_id`/`puct_parent_value`
  （真 uuid，字段名与 VERL 侧一致）、`eval_msg`/`correctness`/`result_construction`；
  `config_snapshot.json` 记录 system prompt；`preflight_puct.sh` 新增断言 1b。
  lumen2/3/node10 已同步；lumen1 等其 run 结束（~10 月 4–5 日）再同步。
- Erdős 的 base 自产 SFT 数据：待定——用训练 run 前 5 步（快，混少量已训策略）
  vs 在 lumen2/3 用 base 重放存档 parent 现生成（干净，多一天；推荐后者，
  #17 后字段齐全）。

---

## 4. 相关工作

### ReST — Reinforced Self-Training
Gulcehre et al., 2023, arXiv:2308.08998（DeepMind，机器翻译）。
两阶段交替：**Grow**（用当前策略对训练 prompt 批量采样成固定数据集）→
**Improve**（按 reward 阈值过滤，在固定数据集上做 SFT/filtered BC；同一 Grow
批次可做多轮 Improve，阈值逐轮抬高）；Grow 通常只做 2–3 轮。
自称 growing-batch **offline RL**：策略改进从不使用 on-policy 梯度，
采样与更新解耦，Improve 期间数据固定（off-policy）。
→ 对应我们的 **B2**（每 5 步采样即 Grow，过滤+SFT 即 Improve）。

### ReST^EM
Singh et al., 2023, arXiv:2312.06585 "Beyond Human Data"（Google，MATH/代码，
PaLM 2）。ReST 的 EM 形式化：E 步 = 采样 + 二值 reward 过滤（答案对/错），
M 步 = SFT。关键细节：**每轮从 base 模型重新 fine-tune**（而非续训上轮
checkpoint），显著减轻累积漂移/熵坍缩；只迭代 2–3 轮，收益就基本饱和。
→ B2 的默认配置采用"每轮从 base 重训"。

### RAFT — Reward rAnked FineTuning
Dong et al., 2023, arXiv:2304.06767。逐 prompt 采 k 条，按 reward 排序取
top 分位，SFT。与 ReST 同家族，过滤方式是排序而非阈值。
→ 我们"每 parent 限额 + 按 raw_score 分层"的采样介于两者之间。

### STaR — Self-Taught Reasoner
Zelikman et al., 2022, arXiv:2203.14465。用"答对的推理链"做 SFT 自举，
答错时给答案让模型补写 rationale（rationalization）再学。
→ 对应"是否只学 improved 子集"的消融；rationalization 变体暂不采用。

### Expert Iteration (ExIt)
Anthony, Tian & Barber, 2017, arXiv:1705.08439 "Thinking Fast and Slow with
Deep Learning and Tree Search"。框架：**搜索**（MCTS）作为 expert 产生比裸
策略更好的决策，**策略网络**对搜索结果做监督学习，两者迭代互促。
AlphaGo Zero / AlphaZero（Silver et al., 2017, Nature / arXiv:1712.01815）
是同一框架的著名实例（MCTS 访问计数作为策略目标 + 自我对弈）。
→ **我们整个"PUCT 搜索产生好 rollout → SFT 回策略"就是 LLM 版 ExIt**，
  这是论文定位的最准坐标系：A1 = 单轮 ExIt 的 SL 步；B2 = 完整 ExIt 循环；
  纯 RL 对照 = 用 policy gradient 而非 SL 消化搜索数据的 ExIt 变体。

### 与在线 RL 的关系（为什么 B1 不是 ReST）
Online RL（PPO/GRPO）每步用当前策略刚采的样本算 policy gradient。
B1 保留 GRPO、周期性插入 SFT，是"在线 RL + 自蒸馏"混合体，不属于 ReST；
文献中最接近的做法是在 RL loss 里混 SFT/BC 辅助项（如 InstructGPT 的
PPO-ptx）。B1 的已知风险：对自身高分样本反复 SFT 会压低输出熵、
伤害搜索多样性（ReST 论文用"每轮只用新鲜数据 + 少 epoch"缓解）。

---

## 5. 基础设施红线（沿用 handoff.md）

- lumen1 的 gpt-oss run 结束前（~10 月 4–5 日）：不 pull、不改代码、不碰 GPU。
- node10 仅 GPU 5–7 可用（GPU 0–4 是用户 `lab` 的作业）；只杀自己记录的 PID。
- SSH：`-o ServerAliveInterval=0`，脚本经 stdin（`bash -s`）传递。
- 每次 PUCT 启动前跑 `bash scripts/preflight_puct.sh <EXP> <RAY_TMP>`。
- 归档到 `~/data/discover-runs/<run>/`；`cp -a` 的 rc=1 不代表失败，
  按文件数+字节+md5 判断；从非写入主机验证。
