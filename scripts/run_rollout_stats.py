#!/usr/bin/env python3
"""
Main entry point for rollout length statistics collection.

Runs all 4 tasks SEQUENTIALLY to collect rollout sample length statistics
across 2 rounds (Round 1: initial exploration, Round 2: PUCT state reuse).

Usage:
    python scripts/run_rollout_stats.py
"""

import asyncio
import json
import time
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rollout_stats.collect import collect_rollout_stats_for_task


async def main():
    """
    Run rollout statistics collection for all 4 tasks SEQUENTIALLY.

    Critical: Tasks must run one at a time to:
    1. Avoid GPU memory conflicts (vLLM uses GPUs 0-3, training uses GPU 4)
    2. Prevent PUCT sampler state pollution between tasks
    3. Ensure proper resource cleanup between tasks
    """
    tasks = [
        ("circle_packing", "examples/circle_packing/config_paper.yaml"),
        ("ahc", "examples/ahc/config_paper.yaml"),
        ("gpu_mode", "examples/gpu_mode/config_paper.yaml"),
        ("denoising", "examples/denoising/config_paper.yaml"),
    ]

    all_results = {}

    print("\n" + "="*80)
    print("Starting Rollout Length Statistics Collection")
    print("="*80)
    print(f"Total tasks: {len(tasks)}")
    print(f"Execution mode: SEQUENTIAL (one task at a time)")
    print("="*80 + "\n")

    for idx, (task_name, config_path) in enumerate(tasks, 1):
        task_start_time = time.time()

        print("\n" + "="*80)
        print(f"Task {idx}/{len(tasks)}: {task_name.upper()}")
        print("="*80)
        print(f"Config: {config_path}")
        print(f"Round 1: 1 prompt × 50 samples = 50 total")
        print(f"Round 2: 8 prompts × 6 samples = 48 total")
        print("="*80 + "\n")

        try:
            # Collect stats for this task
            stats = await collect_rollout_stats_for_task(config_path, task_name)
            all_results[task_name] = stats

            # Print summary for this task
            print_task_summary(task_name, stats)

            task_elapsed = time.time() - task_start_time
            print(f"\n✓ Task {task_name} completed in {task_elapsed:.1f}s")

            # Save intermediate results after each task
            output_path = "rollout_length_statistics.json"
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"✓ Intermediate results saved to: {output_path}")

            # Explicit resource cleanup between tasks
            print(f"\nCleaning up resources before next task...")
            import gc
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Small delay to ensure cleanup completes
            await asyncio.sleep(2)

        except Exception as e:
            print(f"\n✗ Task {task_name} FAILED with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Continuing to next task...\n")
            all_results[task_name] = {"error": str(e)}

    # Final save
    output_path = "rollout_length_statistics.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("ALL TASKS COMPLETED")
    print("="*80)
    print(f"Results saved to: {output_path}")
    successful_tasks = sum(1 for r in all_results.values() if 'error' not in r)
    print(f"Successful tasks: {successful_tasks}/{len(tasks)}")
    print("="*80 + "\n")


def print_task_summary(task_name: str, stats: dict):
    """Print summary statistics for a completed task."""
    print(f"\n{'─'*80}")
    print(f"Summary for {task_name}:")
    print(f"{'─'*80}")

    for round_name in ["round1", "round2"]:
        if round_name not in stats:
            continue

        round_data = stats[round_name]
        print(f"\n{round_name.upper()}:")
        print(f"  Samples: {round_data['num_samples']}")
        print(f"  Prompt length:     min={round_data['prompt_stats']['min']:5d}  "
              f"max={round_data['prompt_stats']['max']:5d}  "
              f"avg={round_data['prompt_stats']['avg']:7.1f}  "
              f"median={round_data['prompt_stats']['median']:7.1f}")
        print(f"  Generation length: min={round_data['generation_stats']['min']:5d}  "
              f"max={round_data['generation_stats']['max']:5d}  "
              f"avg={round_data['generation_stats']['avg']:7.1f}  "
              f"median={round_data['generation_stats']['median']:7.1f}")
        print(f"  Total length:      min={round_data['total_stats']['min']:5d}  "
              f"max={round_data['total_stats']['max']:5d}  "
              f"avg={round_data['total_stats']['avg']:7.1f}  "
              f"median={round_data['total_stats']['median']:7.1f}")

    print(f"{'─'*80}")


if __name__ == "__main__":
    asyncio.run(main())
