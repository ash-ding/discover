# TTT Discover 复现笔记

## 服务器硬件

- GPU: 8x NVIDIA H100 80GB HBM3
- CPU: 160 核 Intel Xeon SapphireRapids
- RAM: 1.7TB
- CUDA Driver: 550.90.07 (CUDA 12.4)
- 磁盘: 2TB (可用 ~400GB)

## 总体方案

原项目依赖 Tinker 平台（远程 LLM 训练服务）进行模型推理和 LoRA 微调。由于没有 Tinker API Key，我们创建了 `ttt_discover/local_backend/` 适配层，使用本地 Qwen3-8B 模型 + vLLM 推理 + PEFT LoRA 训练来替代。

---

## Task 1: Denoising

### 状态: 端到端验证通过

已完成最小配置 (group_size=2, 1 epoch) 的端到端运行，整个 RL 循环（采样 → 沙箱执行 → 奖励计算 → 训练 → checkpoint）均正常工作。

### Conda 环境

```bash
conda create -n discover_denoising python=3.11 -y
conda activate discover_denoising
pip install -r requirements/requirements-denoising-local.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# Denoising 额外 git 依赖
pip install git+https://github.com/czbiohub/simscity.git
pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git
git clone https://github.com/openproblems-bio/openproblems.git
cd openproblems && git checkout v1.0.0 && git apply ../requirements/denoising/openproblems_api_fix.patch && cd ..
pip install --no-deps -e ./openproblems
```

关键版本: torch 2.6.0+cu124, vllm 0.8.5.post1, flashinfer 0.2.5+cu124torch2.6, peft 0.19.1, flash-attn 2.7.4.post1

### 模型下载

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir /workspace/home/asherding/models/Qwen3-8B
```

### 运行方式

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m examples.denoising.env --local
```

### 代码改动

**新建文件 (`ttt_discover/local_backend/`):**

| 文件 | 作用 |
|---|---|
| `__init__.py` | 导出适配类 |
| `future.py` | `LocalFuture`: tinker.APIFuture 的本地替代，包装同步结果 |
| `loss.py` | importance_sampling_loss / ppo_clip_loss 的 PyTorch 实现 |
| `sampling_client.py` | `LocalSamplingClient`: 用 vLLM 替代 tinker.SamplingClient |
| `training_client.py` | `LocalTrainingClient`: 用 HuggingFace + PEFT 替代 tinker.TrainingClient |
| `service_client.py` | `LocalServiceClient`: 编排推理/训练客户端，管理 GPU 分配 |

**修改的文件:**

| 文件 | 改动 |
|---|---|
| `ttt_discover/discovery.py` | 添加 `use_local_backend`, `local_model_path`, GPU 配置字段; 移除 GPT-OSS 模型断言 |
| `ttt_discover/rl/train.py` | `main()` 中条件创建 `LocalServiceClient`; 根据模型选择 Qwen3 completer; 修复 logger 索引越界 (>= 2 → >= 3) |
| `ttt_discover/tinker_utils/completers.py` | 添加 `Qwen3TwoPhaseTokenCompleter`: Phase 2 用 `</think>` 代替 GPT-OSS 的 channel 标记 |
| `examples/denoising/env.py` | 添加 `discover_denoising_local()` 入口和 `--local` 命令行参数 |

### GPU 分配

- GPU 0: vLLM 推理引擎 (~19GB)
- GPU 1: PEFT LoRA 训练模型 (~16.5GB)
- GPU 2-7: 空闲

vLLM 引擎在程序启动时创建一次，RL 循环中复用。每次训练步骤后保存 LoRA 权重，vLLM 通过 LoRARequest 热加载新 adapter（无需重建引擎）。

### 性能数据 (group_size=2, 1 epoch)

| 阶段 | HuggingFace generate | vLLM 0.8.5 |
|---|---|---|
| Sampling | 354s | 89s (4x 加速) |
| 训练 | 4.9s | 9.0s |
| 总计 | 360s | 99s (3.6x 加速) |

vLLM 首次启动有 ~110s 的冷启动开销 (torch.compile + CUDA graph capture)，编译结果会缓存到 `~/.cache/vllm/torch_compile_cache/`，后续启动更快。

### 遇到的问题及解决方案

#### 1. vLLM + CUDA 驱动兼容性

**问题**: `pip install vllm` 默认安装最新版 (0.23.0)，拉入 torch 2.11+cu130，需要 CUDA 13.0 驱动，但服务器驱动只支持 CUDA 12.4。

**解决**: 先装 `torch==2.6.0+cu124`，再装 `vllm==0.8.5.post1`，最后装 `flashinfer-python` 从专用索引 `https://flashinfer.ai/whl/cu124/torch2.6/`。安装顺序很重要——如果先装 vLLM 再降级 torch，flashinfer 的 C++ 扩展会有 ABI 不兼容。

#### 2. vLLM 版本与模型支持

**问题**: vLLM 0.6.x 不支持 `Qwen3ForCausalLM` 架构。

**解决**: 必须用 0.8.x+。vLLM 0.8.5 支持 Qwen3。

#### 3. Ray 与 opentelemetry 版本冲突

**问题**: vLLM 0.8.5 升级了 opentelemetry 到 0.63b1，但 Ray 2.53 的 dashboard agent 期望特定版本，导致 `ImportError: cannot import name 'OtelComponentTypeValues'` 和 Raylet worker 注册失败。

**解决**: 统一安装 opentelemetry-sdk/api 1.42.x, semantic-conventions 0.63b1, exporter-prometheus 0.63b1。

#### 4. 训练前向传播维度不匹配

**问题**: `forward_backward_async` 中只传入 prompt tokens 做前向传播，但试图在 target 位置取 logits，导致 `RuntimeError: Size does not match at dimension 0`。

**解决**: 需要拼接 prompt + target tokens 作为完整输入: `full_ids = prompt_ids + target_tokens.tolist()`，然后在 `logits[prompt_len-1 : prompt_len+target_len-1]` 取 target 位置的 logits。

#### 5. Gradient Checkpointing + LoRA 梯度断裂

**问题**: 启用 gradient checkpointing 后报 `RuntimeError: element 0 of tensors does not require grad`，因为 checkpoint 段的输入没有 `requires_grad=True`。

**解决**: 调用 `model.enable_input_require_grads()` 并使用 `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`。

#### 6. OOM (GPU 显存不足)

**问题**: 8B 模型在 GPU 1 上做 5K token 序列的前向传播 OOM (模型 16GB + 激活显存)。

**解决**: 启用 gradient checkpointing 将显存使用降低到可接受范围。

#### 7. WandB Logger 索引越界

**问题**: `train.py` 第 600 行 `if len(ml_logger.loggers) >= 2:` 但访问 `ml_logger.loggers[2]`，当 WandB 未启用时 loggers 列表不足 3 个元素。

**解决**: 改为 `>= 3`。这是原项目的 bug。

### 已完成的优化

- [x] **推理批量化** — 同 group 内共享同一 prompt，用 `num_samples=N` 一次 vLLM 调用生成所有 completion。group_size=4 测试：每 sample 耗时从 ~45s 降到 ~28s (1.6x 加速)，group_size 越大加速比越高
- [x] **KL penalty 验证** — `BaseModelSamplingProxy` 临时禁用 LoRA 计算 base model logprobs。`kl_policy_base=-0.000240` 确认工作正常
- [x] **Checkpoint 按实验名隔离** — `tinker_log/local_checkpoints/{experiment_name}/`，不同实验不互相覆盖
- [x] **Flash Attention** — 显式启用 `attn_implementation="flash_attention_2"`，训练侧 attention 显存从 O(n²) 降到 O(n)
- [x] **多卡推理 (TP=2/4)** — vLLM tensor_parallel_size 可配置，TP=2 和 TP=4 均验证通过。需要 `disable_custom_all_reduce=True` 避免 P2P CUDA kernel 兼容性问题
- [x] **max_model_len 可配置** — 降低 KV cache 预留长度以提高并发数（32768→8192 时并发从 ~13 提升到 ~50+）
- [x] **Ray CUDA_VISIBLE_DEVICES 修复** — 设置 `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` 阻止 Ray 篡改 GPU 可见性
- [x] **WandB 配置** — API key 和 entity 已设置到 `~/.bashrc`

### 多卡推理性能对比 (group_size=4, phase1_max_tokens=6000, max_model_len=8192)

| 配置 | Sampling | 训练 | 总计 |
|---|---|---|---|
| TP=1 | 111s | 7.7s | ~119s |
| TP=2 | 97.6s | 7.8s | 106.5s |
| TP=4 | 84.8s | 7.8s | 93.8s |

注: group_size=4 时 TP 加速不明显（sample 少），group_size=64 时差异更显著。

### GPU 分配策略

vLLM 始终使用 cuda:0 到 cuda:tp_size-1，训练放在后面的卡上：
- TP=1: `CUDA_VISIBLE_DEVICES=0,1` → 推理 GPU 0, 训练 GPU 1
- TP=2: `CUDA_VISIBLE_DEVICES=0,1,2` → 推理 GPU 0-1, 训练 GPU 2
- TP=4: `CUDA_VISIBLE_DEVICES=0,1,2,3,4` → 推理 GPU 0-3, 训练 GPU 4

### 遇到的 TP 相关问题

#### vLLM custom_all_reduce P2P 通信失败

**问题**: TP=2 时 `Failed: Cuda error custom_all_reduce.cuh:453 'invalid argument'`，vLLM 的自定义 P2P CUDA kernel 不兼容。

**解决**: `disable_custom_all_reduce=True`，使用标准 NCCL 通信。性能影响极小（NVLink 仍然生效）。

#### Ray 篡改 CUDA_VISIBLE_DEVICES

**问题**: Ray 初始化时覆盖 `CUDA_VISIBLE_DEVICES`，破坏 vLLM 的 GPU 分配。

**解决**: `os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"`。

### 待优化项

- [ ] 升级 NVIDIA 驱动以支持更新版本的 vLLM/torch
- [ ] TP=2 训练消除 AHC 任务的序列截断 (max_train_seq_len=8192)

### 磁盘空间估算

每个 checkpoint: state ~1GB + sampler ~334MB。save_every=2, 50 步 = 25 个 checkpoint ≈ 34GB/任务。4 个任务 ≈ 136GB。磁盘剩余 ~349GB，足够。

---

## Task 2: Math (Erdos Min Overlap)

### 状态: 端到端验证通过

local_backend 适配层完全复用 Denoising 的实现，无需任何修改。仅在 env.py 中添加了 `discover_erdos_min_overlap_local()` 入口函数。

### Conda 环境

```bash
conda create -n discover_math python=3.11 -y
conda activate discover_math
pip install -r requirements/requirements-math-local.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

### 运行方式

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m examples.erdos_min_overlap.env --local
```

### 代码改动

仅修改 `examples/erdos_min_overlap/env.py`：添加 `discover_erdos_min_overlap_local()` 和 `--local` 参数支持 (~25行)。

### 与 Denoising 的关键差异

| 方面 | 差异 |
|---|---|
| eval_timeout | 1100s (Denoising 530s)，数学求解器需要更长时间 |
| State 类型 | 普通 `State`，无自定义字段（Denoising 有 mse/poisson 字段） |
| 初始状态 | 随机生成 h_values 构造（Denoising 有固定的 MAGIC 基线算法） |
| 评估指标 | 最小化 C₅ 上界（Denoising 最小化 MSE） |

### 性能数据 (group_size=2, 1 epoch)

| 阶段 | 时间 |
|---|---|
| Sampling (vLLM) | 107s |
| 训练 | 9.3s |
| 总计 | 117.6s |

### 经验总结

- **local_backend 完全任务无关**: 所有适配层代码 (sampling_client, training_client, service_client, loss, future) 零修改即可复用
- **安装流程可标准化**: torch→vllm→flashinfer→peft→opentelemetry 的安装顺序适用于所有任务
- **correctness=0% 在 group_size=2 时正常**: 数学任务对代码质量要求更高，需要更大的 group_size 才能产生有效解

### 备注
- CPU 密集型任务 (scipy/cvxpy 求解器)，评估不需要 GPU
- 文档提到需要 HPC 级 CPU 才能获得好的性能

---

## Task 2b: AC Inequalities

### 状态: 代码已修改，待验证

与 Erdős 共用 `discover_math` 环境，已添加 local 入口函数。

### 运行方式

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m examples.ac_inequalities.env --local          # AC1
    python -m examples.ac_inequalities.env --local --ac2    # AC2
```

### 备注
- 两个子问题：AC1（最小化上界）和 AC2（最大化下界）
- eval_timeout=1100s，与 Erdős 相同
- local_backend 直接复用

---

## Task 2c: Circle Packing

### 状态: 代码已修改，待验证

与 Erdős 共用 `discover_math` 环境，已添加 local 入口函数。

### 运行方式

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m examples.circle_packing.env --local           # 26 circles
    python -m examples.circle_packing.env --local 32        # 32 circles
```

### 备注
- eval_timeout=530s
- local_backend 直接复用

---

## Task 3: GPU Kernels (GPU Mode)

### 状态: 端到端验证通过

### Conda 环境

原始文档要求 Python 3.13，但 Python 3.13 + CUDA 12.4 驱动没有兼容的 torch+vllm 组合。改用 Python 3.11：

```bash
conda create -n discover_gpumode_local python=3.11 -y
conda activate discover_gpumode_local
pip install -r requirements/requirements-gpumode-local.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# Modal 认证
python3 -m modal setup

# 部署 Modal Apps (必须！否则评测会 NotFoundError)
# trimul (TASK="trimul"): 需要 Python 3.13 环境 deploy
conda run -n discover_gpumode bash -c "cd examples/gpu_mode/lib && modal deploy runners/modal_runner_archs.py"
# mla_decode (TASK="mla_decode_nvidia"): 需要 Python 3.12 环境 deploy
conda create -n deploy_tmp python=3.12 -y && conda run -n deploy_tmp pip install modal -q
conda run -n deploy_tmp bash -c "cd examples/gpu_mode/lib && modal deploy runners/modal_runner_archs.py"
```

### 运行方式

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="examples/gpu_mode/lib:$PYTHONPATH" \
    python -m examples.gpu_mode.env --local           # trimul
    python -m examples.gpu_mode.env --local --mla     # mla_decode
```

### 代码改动

| 文件 | 改动 |
|---|---|
| `examples/gpu_mode/env.py` | 添加 `discover_gpu_mode_local()` + `--local`/`--mla` 参数 |
| `examples/gpu_mode/lib/libkernelbot/report.py` | 修复 f-string 反斜杠语法（Python 3.12+ 语法不兼容 3.11） |

### 性能数据 (group_size=2, 1 epoch, trimul)

| 阶段 | 时间 |
|---|---|
| Sampling (vLLM) | 132s |
| Modal 评测 (H100) | 包含在 sampling 中 |
| 训练 | 5.0s |
| 总计 | 138s |

### 遇到的问题及解决方案

#### 1. Python 3.13 与 CUDA 12.4 驱动不兼容

**问题**: Python 3.13 需要 torch >= 2.7，但 PyTorch cu124 索引最高只有 torch 2.6.0（不支持 Python 3.13）。最新的 torch 2.11+cu130 需要 CUDA 13.0 驱动。

**解决**: 创建 Python 3.11 的新环境 `discover_gpumode_local`。Python 版本只影响本地 RL 训练，不影响 Modal 上的 kernel 评测。

#### 2. libkernelbot 模块导入失败

**问题**: `libkernelbot` 内部使用绝对导入 `from libkernelbot.xxx`，但它不在 Python path 上。

**解决**: 运行时添加 `PYTHONPATH="examples/gpu_mode/lib:$PYTHONPATH"`。

#### 3. f-string 语法不兼容

**问题**: `report.py` 中 `f">{msg.replace('\\n', '\n')}"` 在 Python 3.11 中是语法错误（f-string 表达式内不能有反斜杠，3.12+ 才支持）。

**解决**: 将 replace 操作提取到 f-string 外部。

### 备注
- 评估通过 Modal 云平台在远程 H100/H200 上执行 Triton kernel
- 需要有效的 Modal 账号（`~/.modal.toml`）
- 本地 GPU 仅用于 LLM 推理和 LoRA 训练，不用于 kernel 评测

---

## Task 4: AtCoder (AHC)

### 状态: 端到端验证通过

需要在 Docker/Podman 容器内运行，因为 C++ 编译评测依赖容器内的 g++-12 和 boost/ac-library。

### Conda 环境

```bash
conda create -n discover_ale python=3.11 -y
conda activate discover_ale
pip install -r requirements/requirements-ahc-local.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

### 前置准备

```bash
# 拉取容器镜像
podman pull docker.io/yimjk/ale-bench:cpp20-202301

# 下载测试数据
bash examples/ahc/get_cache.sh
```

### 运行方式

```bash
podman run --rm --device nvidia.com/gpu=all \
    --shm-size=16g \
    --pids-limit=-1 \
    -v /workspace:/workspace \
    -w /workspace/home/asherding/code/discover \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -e WANDB_MODE=offline \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    docker.io/yimjk/ale-bench:cpp20-202301 \
    /workspace/home/asherding/.conda/envs/discover_ale/bin/python -m examples.ahc.env --local
```

### 代码改动

| 文件 | 改动 |
|---|---|
| `examples/ahc/env.py` | 添加 `discover_ahc039_local()` + `--local` 参数 |
| `ttt_discover/local_backend/training_client.py` | 显式启用 `attn_implementation="flash_attention_2"`; 添加 `max_train_seq_len=8192` 序列截断 |

### 性能数据 (group_size=2, 1 epoch)

| 阶段 | 时间 |
|---|---|
| Sampling (vLLM) | 270s |
| 训练 (FA + gradient checkpointing) | 13.7s |
| 总计 | 284.8s |

### 遇到的问题及解决方案

#### 1. 容器内 Ray 启动失败 — PID 限制

**问题**: Podman 默认 PID 限制太低，Ray 无法 fork worker 进程，报 `Resource temporarily unavailable`。

**解决**: 添加 `--pids-limit=-1` 到 podman run 命令。

#### 2. 容器内 /dev/shm 太小

**问题**: 容器默认 /dev/shm 只有 64MB，Ray 的 object store 需要更多共享内存。

**解决**: 添加 `--shm-size=16g` 到 podman run 命令。

#### 3. Prompt 超长导致 phase1_max_tokens 不足

**问题**: AHC039 的 prompt 有 ~13K token（包含完整竞赛题目 + 初始 C++ 代码），超过了 phase1_max_tokens=4000。

**解决**: 改为 22000（与论文原始设定一致）。

#### 4. 训练序列过长导致 OOM

**问题**: prompt (13K) + response (9K) ≈ 22K token 的前向传播在单卡 80GB 上 OOM，即使有 gradient checkpointing。

**解决**: 
- 安装 `flash-attn==2.7.4.post1` 将 attention 显存从 O(n²) 降到 O(n)
- 在模型加载时显式指定 `attn_implementation="flash_attention_2"`（HuggingFace 不一定自动检测）
- 添加 `max_train_seq_len=8192` 截断：保留完整 response，截断 prompt 前部
- 这是工程妥协——论文用 120B 模型在 Tinker 平台训练（无显存限制）

#### 5. 残留 GPU 进程

**问题**: 前一次运行失败后 vLLM 进程未完全释放，占满 GPU 0 导致新启动的 vLLM 无法分配显存。

**解决**: 运行前手动清理: `kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)`

### 备注
- 文档中写的 requirements 文件名是 `requirements-ale.txt`，实际文件是 `requirements-ahc.txt`
- 代码中 `run_command_direct()` 直接在宿主机用 subprocess 编译执行 C++，但编译命令硬编码了 `g++-12` 和 `/opt/ac-library` 等容器路径，因此必须在容器内运行
- 论文设计是用 Slurm Pyxis 把整个训练进程启动在容器内，`run_command_direct()` 假设环境中已有 g++-12
- 我们用 podman 挂载宿主机 conda 环境到容器内，达到相同效果
