#!/usr/bin/env python3
"""
GPU Kernel Evaluation Worker

This script runs in an isolated process/container and may crash without
affecting the main training process.

Usage (standalone):
    python eval_worker.py <config_file>

Config format:
    {
        "submission_file": "/path/to/submission.py",
        "task_name": "trimul" or "mla_decode_nvidia",
        "gpu_type": "H100" or "H200",
        "result_file": "/path/to/result.json"
    }

The run_evaluation() function can also be imported by pool_worker.py
for container pooling mode.
"""

import sys
import json
import os
import math
import traceback
from pathlib import Path


TASK_MAP = {
    "trimul": "bioml/trimul/task.yml",
    "mla_decode_nvidia": "mla-decode/task.yml",
}


def _get_lib_base():
    if Path("/workspace/lib").exists():
        return Path("/workspace/lib")
    return Path(__file__).parent.parent / "lib"


def _ensure_lib_path():
    lib = str(_get_lib_base())
    if lib not in sys.path:
        sys.path.insert(0, lib)


def _compute_score(full_result, task):
    from libkernelbot.consts import RankCriterion

    num_benchmarks = int(full_result.runs["leaderboard"].run.result["benchmark-count"])

    if task.ranking_by == RankCriterion.LAST:
        score = float(full_result.runs["leaderboard"].run.result["benchmark.0.mean"]) / 1e9
    elif task.ranking_by == RankCriterion.GEOM:
        scores = []
        for i in range(num_benchmarks):
            scores.append(float(full_result.runs["leaderboard"].run.result[f"benchmark.{i}.mean"]) / 1e9)
        score = math.pow(math.prod(scores), 1.0 / num_benchmarks)
    elif task.ranking_by == RankCriterion.MEAN:
        scores = []
        for i in range(num_benchmarks):
            scores.append(float(full_result.runs["leaderboard"].run.result[f"benchmark.{i}.mean"]) / 1e9)
        score = sum(scores) / len(scores)
    else:
        raise ValueError(f"Invalid ranking criterion {task.ranking_by}")

    return score


def load_task_definition(task_name):
    _ensure_lib_path()
    from libkernelbot.task import make_task_definition

    if task_name not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_name}")

    task_yml_path = _get_lib_base() / TASK_MAP[task_name]
    if not task_yml_path.exists():
        raise FileNotFoundError(f"Task definition not found: {task_yml_path}")

    return make_task_definition(task_yml_path)


def run_evaluation(submission_code, task_name, gpu_type, timeout=530, task_def=None):
    """Core eval logic. Returns result dict.

    Args:
        submission_code: Source code of the kernel submission.
        task_name: "trimul" or "mla_decode_nvidia".
        gpu_type: GPU type string (for metadata, not used in eval).
        timeout: Outer eval timeout in seconds.
        task_def: Pre-loaded task definition (from load_task_definition).
                  If None, loads from disk.

    Returns:
        dict with keys: success, score_us, error, and optionally benchmark_count / gpu_crash.
    """
    _ensure_lib_path()
    from libkernelbot.task import build_task_config
    from libkernelbot.consts import SubmissionMode
    from libkernelbot.run_eval import run_config

    import copy

    if task_def is None:
        task_def = load_task_definition(task_name)

    task = copy.copy(task_def.task)

    # Cap inner timeouts to prevent orphan grandchild processes
    max_phase_timeout = max(timeout // 3, 60)
    task.test_timeout = min(task.test_timeout, max_phase_timeout)
    task.benchmark_timeout = min(task.benchmark_timeout, max_phase_timeout)
    task.ranked_timeout = min(task.ranked_timeout, max_phase_timeout)

    eval_config = build_task_config(
        task=task,
        submission_content=submission_code,
        arch=None,
        mode=SubmissionMode.LEADERBOARD,
    )

    full_result = run_config(eval_config)

    if not full_result.success:
        return {
            "success": False,
            "score_us": -1_000_000,
            "error": full_result.error or "Unknown error",
        }

    if "test" not in full_result.runs:
        return {"success": False, "score_us": -1_000_000, "error": "No test results found"}

    if not full_result.runs["test"].run.passed:
        return {
            "success": False,
            "score_us": -1_000_000,
            "error": f"Tests failed: {full_result.runs['test'].run.stderr}",
        }

    if "leaderboard" not in full_result.runs:
        return {"success": False, "score_us": -1_000_000, "error": "No leaderboard results found"}

    try:
        score_seconds = _compute_score(full_result, task)
        return {
            "success": True,
            "score_us": score_seconds * 1_000_000,
            "error": None,
            "benchmark_count": int(
                full_result.runs["leaderboard"].run.result.get("benchmark-count", 0)
            ),
        }
    except Exception as e:
        return {"success": False, "score_us": -1_000_000, "error": f"Failed to compute score: {e}"}


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
        submission_code = submission_file.read_text()
        timeout = int(os.environ.get("EVAL_TIMEOUT", "530"))
        result = run_evaluation(submission_code, task_name, gpu_type, timeout=timeout)
        result_file.write_text(json.dumps(result, indent=2))
        sys.exit(0)

    except Exception as e:
        print(f"Worker crashed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        try:
            error_result = {
                "success": False,
                "score_us": -1_000_000,
                "error": f"Worker exception: {str(e)}",
                "gpu_crash": True,
            }
            result_file.write_text(json.dumps(error_result, indent=2))
        except Exception:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
