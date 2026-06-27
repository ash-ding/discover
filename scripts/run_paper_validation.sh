#!/bin/bash
# Run paper-exact validation (1 epoch) for Circle Packing
# Prerequisites:
#   1. vLLM server running with TP=4 (scripts/start_vllm_paper_config.sh)
#   2. CUDA_VISIBLE_DEVICES=0,1,2,3,4
#   3. export VLLM_BASE_URL="http://localhost:8888"

set -e

echo "=== 🧪 Circle Packing 论文配置验证（1 Epoch）==="
echo ""
echo "配置参数（与论文 Table 9 完全一致）："
echo "  - Model: Qwen3-8B"
echo "  - Tensor Parallelism: TP=4"
echo "  - Max Tokens: 26000"
echo "  - Group Size: 64"
echo "  - Groups per Batch: 8"
echo "  - KL Penalty Coef: 0.1"
echo "  - LoRA Rank: 32"
echo "  - Learning Rate: 4e-5"
echo "  - GPU Memory Util: 0.9"
echo "  - Epochs: 1 (validation only)"
echo ""
echo "预计耗时：~30-60 分钟"
echo ""

# Check vLLM server
if ! curl -s http://localhost:8888/v1/models > /dev/null 2>&1; then
    echo "❌ 错误：vLLM 服务器未运行在 http://localhost:8888"
    echo ""
    echo "请先启动 vLLM 服务器："
    echo "  bash scripts/start_vllm_paper_config.sh"
    exit 1
fi

echo "✅ vLLM 服务器运行正常"
echo ""

# Activate environment
source ~/.bashrc
conda activate discover_math

# Set environment variables
export VLLM_BASE_URL="http://localhost:8888"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

# Run validation
echo "🚀 启动验证实验..."
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m examples.circle_packing.env --validate 26

echo ""
echo "✅ 验证完成！"
echo ""
echo "查看结果："
echo "  - 日志：tinker_log/circle-packing-26-paper-validate/train.log"
echo "  - 指标：tinker_log/circle-packing-26-paper-validate/metrics.jsonl"
echo "  - Checkpoint：tinker_log/local_checkpoints/circle-packing-26-paper-validate/"
