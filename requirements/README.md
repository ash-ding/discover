# Requirements 安装说明

**服务器**: CUDA 12.9, 8x H100 80GB  
**vLLM 版本**: 0.23.0  
**PyTorch 版本**: 2.11.0+cu129

---

## 📦 文件结构

```
requirements/
├── README.md                     # 本文件（总安装指南）
├── requirements-math.txt         # Math 任务
├── requirements-gpumode.txt      # GPU Mode 任务
├── requirements-ahc.txt          # AHC 任务
│
├── denoising/                    # Denoising 任务（独立文件夹）
│   ├── README.md                 # Denoising 详细安装说明
│   ├── requirements-denoising.txt  # Denoising 依赖
│   └── openproblems_api_fix.patch  # OpenProblems 补丁
│
└── _backup_old/                  # 旧版 requirements 备份
```

---

## 🚀 快速安装

### 1️⃣ Math 任务

```bash
conda create -n discover_math python=3.11 -y
conda activate discover_math
pip install -r requirements/requirements-math.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/
```

**包含任务**: Circle Packing (26/32 circles), AC Inequalities (AC1/AC2), Erdős Min Overlap

---

### 2️⃣ Denoising 任务

**⚠️ 特殊要求**: 需要安装 Git 依赖和 openproblems 包

```bash
conda create -n discover_denoising python=3.11 -y
conda activate discover_denoising

# 1. 安装基础依赖
pip install -r requirements/denoising/requirements-denoising.txt

# 2. 安装 FlashInfer
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/

# 3. 安装 Git 依赖
pip install git+https://github.com/czbiohub/simscity.git
pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git

# 4. 克隆并安装 openproblems（需要应用补丁）
cd /path/to/ttt-discover
git clone https://github.com/openproblems-bio/openproblems.git
cd openproblems && git checkout v1.0.0 && cd ..
cd openproblems && git apply ../requirements/denoising/openproblems_api_fix.patch && cd ..
pip install --no-deps -e ./openproblems
```

**详细说明**: 参考 [requirements/denoising/README.md](denoising/README.md)

---

### 3️⃣ GPU Mode 任务

```bash
conda create -n discover_gpumode python=3.11 -y
conda activate discover_gpumode
pip install -r requirements/requirements-gpumode.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/
```

**包含任务**: trimul, mla_decode  
**特殊要求**: 需要 Modal 账户并配置 API token

---

### 4️⃣ AHC 任务

```bash
conda create -n discover_ale python=3.11 -y
conda activate discover_ale
pip install -r requirements/requirements-ahc.txt
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/
```

**包含任务**: ahc039, ahc058  
**特殊要求**: 必须在 Docker 容器中运行（`yimjk/ale-bench:cpp20-202301`）

---

## 🔑 核心依赖版本

所有任务共享相同的 RL 后端版本：

| 组件 | 版本 |
|------|------|
| **PyTorch** | 2.11.0+cu129 |
| **vLLM** | 0.23.0 |
| **PEFT** | 0.19.1 |
| **Transformers** | 5.12.1 |
| **FlashInfer** | cu129/torch2.11 |
| **Ray** | 2.50+ |

任务特定依赖：

| 任务 | 特殊依赖 |
|------|---------|
| **Math** | — |
| **Denoising** | Scanpy, Numba 0.65, OpenProblems |
| **GPU Mode** | Modal, PyGithub, better-profanity |
| **AHC** | Docker, ahocorapy, Pillow |

---

## 🧪 验证安装

### Math
```bash
conda activate discover_math
python -c "import torch, vllm; print(f'✅ torch={torch.__version__}, vllm={vllm.__version__}')"
```

### Denoising
```bash
conda activate discover_denoising
python -c "import torch, vllm, scanpy, openproblems; print('✅ Denoising 环境完整')"
```

### GPU Mode
```bash
conda activate discover_gpumode
python -c "import torch, vllm, modal; print('✅ GPU Mode 环境完整')"
```

### AHC
```bash
conda activate discover_ale
python -c "import torch, vllm, docker; print('✅ AHC 环境完整')"
```

---

## 📝 安装原则

1. **每个任务独立环境**：避免依赖冲突
2. **vLLM 自动拉取 PyTorch**：无需手动安装 torch
3. **FlashInfer 手动安装**：必须匹配 CUDA + PyTorch 版本
4. **Flash Attention 可选**：vLLM 内置，训练回退 SDPA

---

## ❓ 常见问题

### Q: 为什么每个任务需要独立环境？

**A**: 不同任务有特殊依赖冲突：
- Denoising: `numba`, `scanpy` 单细胞分析库
- GPU Mode: `modal`, `logfire` 版本要求
- AHC: Docker 相关库较重

### Q: Flash Attention 必须安装吗？

**A**: **不必须**。vLLM 推理内置 FlashAttention v3，PEFT 训练未安装时回退 SDPA。

### Q: 为什么 Denoising 需要单独文件夹？

**A**: Denoising 需要：
1. Git 依赖（simscity, molecular-cross-validation）
2. OpenProblems 包（需要应用补丁）
3. 特殊配置（OPENPROBLEMS_CACHE_DIR）

单独文件夹方便管理补丁和文档。

---

## 📚 相关文档

- **环境配置总结**: [ENVIRONMENTS.md](../ENVIRONMENTS.md)
- **快速开始**: [QUICKSTART.md](../QUICKSTART.md)
- **升级记录**: [CHANGELOG_v0.23.0.md](../CHANGELOG_v0.23.0.md)
- **Denoising 详细说明**: [denoising/README.md](denoising/README.md)

---

**最后更新**: 2026-06-25  
**维护者**: TTT-Discover 本地复现团队
