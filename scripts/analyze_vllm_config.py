#!/usr/bin/env python3
"""
vLLM配置分析工具
用于计算给定参数下的理论并发度和吞吐量影响

使用方法:
    python scripts/analyze_vllm_config.py --max-model-len 32768 --max-num-seqs 4
    python scripts/analyze_vllm_config.py --help
"""

import argparse


def analyze_config(
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    total_mem_gb: float = 80.0,
    tp_size: int = 2,
    model_weight_gb: float = 16.0,
    total_rollouts: int = 512,
):
    """分析vLLM配置的并发度和吞吐量影响"""

    # Qwen3-8B KV cache参数
    head_dim = 128
    num_layers = 28
    kv_heads = 4  # GQA

    # 计算单序列KV cache大小
    bytes_per_token = 2 * kv_heads * head_dim * num_layers * 2  # K+V, fp16
    kv_per_seq_gb = bytes_per_token * max_model_len / (1024**3)

    # 可用KV cache池
    model_weight_per_gpu = model_weight_gb / tp_size
    available_mem = total_mem_gb - model_weight_per_gpu
    kv_cache_pool = available_mem * gpu_memory_utilization
    temp_compute_space = available_mem * (1 - gpu_memory_utilization)

    # 理论最大并发度
    max_by_memory = int(kv_cache_pool / kv_per_seq_gb)

    # 实际并发度
    actual_concurrent = min(max_num_seqs, max_by_memory)
    bottleneck = "max_num_seqs" if actual_concurrent == max_num_seqs else "KV cache显存"

    # 吞吐量分析
    rounds_needed = (total_rollouts + actual_concurrent - 1) // actual_concurrent

    # 打印分析结果
    print("=" * 80)
    print("vLLM配置分析")
    print("=" * 80)
    print(f"\n📋 输入参数:")
    print(f"  max_model_len:           {max_model_len:,}")
    print(f"  max_num_seqs:            {max_num_seqs}")
    print(f"  gpu_memory_utilization:  {gpu_memory_utilization}")
    print(f"  tensor_parallel_size:    {tp_size}")
    print(f"  总GPU显存:               {total_mem_gb:.0f} GB")
    print(f"  模型权重:                {model_weight_gb:.0f} GB")

    print(f"\n💾 显存分配:")
    print(f"  单GPU可用显存:           {available_mem:.1f} GB")
    print(f"  KV cache池:              {kv_cache_pool:.1f} GB ({gpu_memory_utilization*100:.0f}%)")
    print(f"  临时计算空间:            {temp_compute_space:.1f} GB ({(1-gpu_memory_utilization)*100:.0f}%)")
    print(f"  单序列KV cache:          {kv_per_seq_gb:.2f} GB")

    print(f"\n🔢 并发度分析:")
    print(f"  理论最大(显存限制):      {max_by_memory} 个序列")
    print(f"  用户设置(max_num_seqs):  {max_num_seqs} 个序列")
    print(f"  实际并发度:              {actual_concurrent} 个序列")
    print(f"  瓶颈因素:                {bottleneck}")

    if max_by_memory < max_num_seqs:
        print(f"\n  ⚠️  警告: KV cache显存不足以支持max_num_seqs={max_num_seqs}")
        print(f"      建议降低max_num_seqs到{max_by_memory}以下")
        print(f"      或降低max_model_len以腾出更多KV cache空间")
    elif max_by_memory >= max_num_seqs * 2:
        print(f"\n  ✅ KV cache空间充裕 (理论最大 {max_by_memory} >> 实际 {max_num_seqs})")
        print(f"      可以安全提升max_num_seqs到{min(max_by_memory, max_num_seqs*4)}")

    print(f"\n🚀 吞吐量影响 (假设需要处理{total_rollouts}个rollouts):")
    print(f"  需要处理轮数:            {rounds_needed} 轮")
    print(f"  每轮并发:                {actual_concurrent} 个序列")
    print(f"  平均每轮处理:            {total_rollouts/rounds_needed:.1f} 个序列")

    # 对比基准配置 (max_num_seqs=4)
    baseline_rounds = (total_rollouts + 3) // 4  # 基准: max_num_seqs=4
    speedup = baseline_rounds / rounds_needed
    print(f"  相对加速 (vs max_num_seqs=4): {speedup:.1f}x")

    # KL penalty分析
    print(f"\n🔍 KL Penalty考虑:")
    # 假设chunked prefill, chunk_size=2048
    chunk_size = 2048
    vocab_size = 152064
    logits_per_chunk_gb = chunk_size * vocab_size * 2 / (1024**3)
    max_concurrent_kl = int(temp_compute_space / logits_per_chunk_gb)

    print(f"  chunked prefill chunk大小: {chunk_size} tokens")
    print(f"  每chunk logits:            {logits_per_chunk_gb:.2f} GB")
    print(f"  临时空间可并发KL计算:      {max_concurrent_kl} 个chunk")

    if max_concurrent_kl >= actual_concurrent * 4:
        print(f"  ✅ 临时计算空间充足，无OOM风险")
    elif max_concurrent_kl >= actual_concurrent:
        print(f"  ⚠️  临时计算空间紧张，建议监控KL penalty阶段的显存")
    else:
        print(f"  ❌ 临时计算空间不足，KL penalty可能OOM!")
        print(f"     建议降低gpu_memory_utilization到0.6或更低")

    print("\n" + "=" * 80)

    # 返回关键指标
    return {
        "actual_concurrent": actual_concurrent,
        "bottleneck": bottleneck,
        "speedup": speedup,
        "rounds_needed": rounds_needed,
        "kv_cache_utilization": actual_concurrent * kv_per_seq_gb / kv_cache_pool,
    }


def main():
    parser = argparse.ArgumentParser(
        description="分析vLLM配置对rollout吞吐量的影响",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析当前配置
  python scripts/analyze_vllm_config.py --max-model-len 32768 --max-num-seqs 4

  # 测试优化后的配置
  python scripts/analyze_vllm_config.py --max-model-len 32768 --max-num-seqs 16

  # 测试降低context的影响
  python scripts/analyze_vllm_config.py --max-model-len 16384 --max-num-seqs 32
        """
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM的max-model-len参数 (默认: 32768)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="vLLM的max-num-seqs参数 (默认: 4)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.70,
        help="vLLM的gpu-memory-utilization参数 (默认: 0.70)"
    )
    parser.add_argument(
        "--total-mem",
        type=float,
        default=80.0,
        help="单GPU总显存 GB (默认: 80)"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=2,
        help="tensor-parallel-size (默认: 2)"
    )
    parser.add_argument(
        "--total-rollouts",
        type=int,
        default=512,
        help="需要处理的rollout总数 (默认: 512)"
    )

    args = parser.parse_args()

    analyze_config(
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        total_mem_gb=args.total_mem,
        tp_size=args.tp_size,
        total_rollouts=args.total_rollouts,
    )


if __name__ == "__main__":
    main()
