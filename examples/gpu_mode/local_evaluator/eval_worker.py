#!/usr/bin/env python3
"""
GPU Kernel Evaluation Worker

This script runs in an isolated process/container and may crash without
affecting the main training process.

Usage:
    python eval_worker.py <config_file>

Config format:
    {
        "submission_file": "/path/to/submission.py",
        "task_name": "trimul" or "mla_decode_nvidia",
        "gpu_type": "H100" or "H200",
        "result_file": "/path/to/result.json"
    }
"""

import sys
import json
import os
import traceback
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: eval_worker.py <config_file>", file=sys.stderr)
        sys.exit(1)

    config_file = Path(sys.argv[1])
    config = json.loads(config_file.read_text())

    submission_file = Path(config["submission_file"])
    task_name = config["task_name"]
    gpu_type = config["gpu_type"]
    result_file = Path(config["result_file"])

    try:
        # Import task utilities
        # These will be available from the mounted lib directory
        sys.path.insert(0, "/workspace/lib" if Path("/workspace/lib").exists() else str(Path(__file__).parent.parent / "lib"))

        from libkernelbot.task import make_task_definition, build_task_config
        from libkernelbot.consts import SubmissionMode, RankCriterion
        from libkernelbot.run_eval import run_config
        import math

        # Inline compute_score to avoid psycopg2 dependency
        def compute_score(result, task):
            """Compute geometric mean score from benchmark results."""
            num_benchmarks = int(result.runs["leaderboard"].run.result["benchmark-count"])

            if task.ranking_by == RankCriterion.LAST:
                # Only one benchmark, use its mean
                score = float(result.runs["leaderboard"].run.result["benchmark.0.mean"]) / 1e9
            elif task.ranking_by == RankCriterion.GEOM:
                # Geometric mean of all benchmarks
                scores = []
                for i in range(num_benchmarks):
                    scores.append(float(result.runs["leaderboard"].run.result[f"benchmark.{i}.mean"]) / 1e9)
                score = math.pow(math.prod(scores), 1.0 / num_benchmarks)
            elif task.ranking_by == RankCriterion.MEAN:
                # Arithmetic mean of all benchmarks
                scores = []
                for i in range(num_benchmarks):
                    scores.append(float(result.runs["leaderboard"].run.result[f"benchmark.{i}.mean"]) / 1e9)
                score = sum(scores) / len(scores)
            else:
                raise ValueError(f"Invalid ranking criterion {task.ranking_by}")

            return score

        # Load task definition
        task_map = {
            "trimul": "bioml/trimul/task.yml",
            "mla_decode_nvidia": "mla-decode/task.yml",
        }

        if task_name not in task_map:
            raise ValueError(f"Unknown task: {task_name}")

        # Find task.yml
        lib_base = Path("/workspace/lib") if Path("/workspace/lib").exists() else Path(__file__).parent.parent / "lib"
        task_yml_path = lib_base / task_map[task_name]

        if not task_yml_path.exists():
            raise FileNotFoundError(f"Task definition not found: {task_yml_path}")

        # Load task
        definition = make_task_definition(task_yml_path)
        task = definition.task

        # Read submission code
        submission_code = submission_file.read_text()

        # Build config for evaluation
        eval_config = build_task_config(
            task=task,
            submission_content=submission_code,
            arch=None,
            mode=SubmissionMode.LEADERBOARD,  # Run tests + benchmarks
        )

        # Run evaluation
        # This executes:
        # 1. All correctness tests
        # 2. All benchmarks
        # 3. Returns FullResult with timings
        full_result = run_config(eval_config)

        if not full_result.success:
            # Evaluation failed (e.g., syntax error, test failure)
            result = {
                "success": False,
                "score_us": -1_000_000,
                "error": full_result.error or "Unknown error",
                "full_result": None
            }
        else:
            # Check if tests passed
            if "test" not in full_result.runs:
                result = {
                    "success": False,
                    "score_us": -1_000_000,
                    "error": "No test results found"
                }
            elif not full_result.runs["test"].run.passed:
                result = {
                    "success": False,
                    "score_us": -1_000_000,
                    "error": f"Tests failed: {full_result.runs['test'].run.stderr}"
                }
            elif "leaderboard" not in full_result.runs:
                result = {
                    "success": False,
                    "score_us": -1_000_000,
                    "error": "No leaderboard results found"
                }
            else:
                # Success - compute score
                try:
                    # compute_score returns geometric mean in seconds
                    score_seconds = compute_score(full_result, task)
                    score_us = score_seconds * 1_000_000  # Convert to microseconds

                    result = {
                        "success": True,
                        "score_us": score_us,
                        "error": None,
                        "benchmark_count": len(full_result.runs["leaderboard"].run.benchmarks)
                    }
                except Exception as e:
                    result = {
                        "success": False,
                        "score_us": -1_000_000,
                        "error": f"Failed to compute score: {e}"
                    }

        # Write result
        result_file.write_text(json.dumps(result, indent=2))
        sys.exit(0)

    except Exception as e:
        # Worker crashed - write error but don't write result file
        # (parent will detect missing result file and treat as crash)
        print(f"Worker crashed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        # Try to write minimal error result
        try:
            error_result = {
                "success": False,
                "score_us": -1_000_000,
                "error": f"Worker exception: {str(e)}",
                "gpu_crash": True
            }
            result_file.write_text(json.dumps(error_result, indent=2))
        except:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
