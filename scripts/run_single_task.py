#!/usr/bin/env python3
"""
Single task runner for rollout statistics collection.
Designed to be called from conda run with the appropriate environment.
"""

import asyncio
import json
import argparse
import time
import gc
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from scripts.rollout_stats.collect import collect_rollout_stats_for_task


def load_existing_results(output_file: str) -> dict:
    """Load existing results if file exists."""
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            return json.load(f)
    return {}


def save_results(output_file: str, results: dict):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


async def run_single_task(task_name: str, config_path: str, output_file: str):
    """Run rollout statistics collection for a single task."""

    print(f"Starting {task_name}...")
    task_start_time = time.time()

    try:
        # Collect stats
        stats = await collect_rollout_stats_for_task(config_path, task_name)

        # Load existing results and merge
        all_results = load_existing_results(output_file)
        all_results[task_name] = stats

        # Save updated results
        save_results(output_file, all_results)

        # Print summary
        print_task_summary(task_name, stats)

        task_elapsed = time.time() - task_start_time
        print(f"\n✓ Task {task_name} completed in {task_elapsed:.1f}s")
        print(f"✓ Results saved to: {output_file}\n")

        return 0

    except Exception as e:
        print(f"\n✗ Task {task_name} FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        # Save error to results
        all_results = load_existing_results(output_file)
        all_results[task_name] = {"error": str(e)}
        save_results(output_file, all_results)

        return 1

    finally:
        # Cleanup
        print(f"Cleaning up resources for {task_name}...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


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


def main():
    parser = argparse.ArgumentParser(description='Run rollout statistics for a single task')
    parser.add_argument('--task-name', required=True, help='Task name (e.g., circle_packing)')
    parser.add_argument('--config-path', required=True, help='Path to config YAML')
    parser.add_argument('--output-file', default='rollout_length_statistics.json',
                       help='Output JSON file')
    parser.add_argument('--skip', action='store_true', help='Mark task as skipped')

    args = parser.parse_args()

    if args.skip:
        # Just mark as skipped in output
        all_results = load_existing_results(args.output_file)
        all_results[args.task_name] = {"skipped": True, "reason": "Environment not available"}
        save_results(args.output_file, all_results)
        print(f"Task {args.task_name} marked as skipped")
        return 0

    # Run the task
    exit_code = asyncio.run(run_single_task(args.task_name, args.config_path, args.output_file))
    return exit_code


if __name__ == "__main__":
    exit(main())
