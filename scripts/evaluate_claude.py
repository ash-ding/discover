"""Universal evaluation script for Claude replays.

Supports multiple tasks with task-specific evaluation logic.

Usage:
    TASK=erdos STEP=1 python scripts/evaluate_claude.py
    TASK=gpu_mode STEP=1 GPU_EVAL_SERVER=http://host:8890 python scripts/evaluate_claude.py

Environment Variables:
    TASK: Task name (erdos, gpu_mode, etc.) [required]
    STEP: Training step to evaluate [default: 1]
    EVAL_TIMEOUT: Hard timeout per rollout in seconds [default: 120]
    EVAL_WORKERS: Number of parallel workers [default: 32]

    GPU Mode specific:
    GPU_EVAL_SERVER: Remote eval server URL (if not set, uses local GPU)
    KERNEL_EVAL_GPU: GPU ID for local eval [default: 0]
"""
import json
import os
import sys
import time
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def evaluate_erdos_worker(idx, code, timeout, result_queue):
    """Evaluate Erdos Min Overlap task."""
    try:
        exec_globals = {"__builtins__": __builtins__}
        exec("import numpy as np", exec_globals)
        exec("import time", exec_globals)

        # Setup initial conditions
        rng = np.random.default_rng(42)
        n_points = 71
        construction = np.ones(n_points) * 0.5
        perturbation = rng.uniform(-0.4, 0.4, n_points)
        perturbation = perturbation - np.mean(perturbation)
        construction = construction + perturbation
        exec_globals["initial_h_values"] = np.array(construction)

        def evaluate_erdos_solution(result):
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                return False
            h_values, c5_bound, n_pts = result
            h = np.array(h_values)
            if len(h) < 2:
                return False
            if np.any(h < -0.01) or np.any(h > 1.01):
                return False
            dx = 2.0 / len(h)
            integral = np.sum(h) * dx
            if abs(integral - 1.0) > 0.05:
                return False
            return True

        exec_globals["evaluate_erdos_solution"] = evaluate_erdos_solution

        # Execute code
        exec(code, exec_globals)

        if "run" not in exec_globals:
            result_queue.put((idx, 0.0, "no_run_func"))
            return

        result = exec_globals["run"](seed=42, budget_s=min(timeout, 120))

        if not isinstance(result, (tuple, list)) or len(result) != 3:
            result_queue.put((idx, 0.0, "bad_return"))
            return

        h_values, c5_bound, n_points_out = result
        h = np.array(h_values)

        if len(h) < 2:
            result_queue.put((idx, 0.0, "too_short"))
            return

        dx = 2.0 / len(h)
        integral = np.sum(h) * dx
        if abs(integral - 1.0) > 0.05:
            result_queue.put((idx, 0.0, f"bad_integral_{integral:.4f}"))
            return

        if np.any(h < -0.01) or np.any(h > 1.01):
            result_queue.put((idx, 0.0, "out_of_range"))
            return

        # Compute actual C5 score
        correlation = np.correlate(h, 1 - h, mode="full") * dx
        actual_c5 = float(np.max(correlation))
        score = float(1.0 / (1e-8 + actual_c5))
        result_queue.put((idx, score, f"c5={actual_c5:.6f}"))

    except Exception as e:
        result_queue.put((idx, 0.0, f"error:{type(e).__name__}"))


def evaluate_gpu_mode_worker(idx, code, timeout, result_queue):
    """Evaluate GPU Mode (trimul) task.

    Uses remote eval server if GPU_EVAL_SERVER is set, otherwise local GPU.
    """
    try:
        eval_server = os.getenv("GPU_EVAL_SERVER")

        if eval_server:
            # Remote evaluation via HTTP
            import requests

            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)

            if "run" not in exec_globals:
                result_queue.put((idx, 0.0, "no_run_func"))
                return

            # Generate kernel code
            kernel_code = exec_globals["run"](seed=42, budget_s=min(timeout, 120))

            # Send to eval server
            response = requests.post(
                f"{eval_server}/evaluate",
                json={"code": kernel_code, "timeout": timeout},
                timeout=timeout + 30
            )

            if response.status_code == 200:
                result = response.json()
                score = result.get("tflops", 0.0)
                status = result.get("status", "unknown")
                result_queue.put((idx, score, status))
            else:
                result_queue.put((idx, 0.0, f"server_error_{response.status_code}"))

        else:
            # Local evaluation (requires GPU)
            import torch
            import triton

            exec_globals = {"__builtins__": __builtins__}
            exec("import torch", exec_globals)
            exec("import triton", exec_globals)
            exec("import triton.language as tl", exec_globals)

            exec(code, exec_globals)

            if "run" not in exec_globals:
                result_queue.put((idx, 0.0, "no_run_func"))
                return

            # Run kernel and measure performance
            gpu_id = int(os.getenv("KERNEL_EVAL_GPU", "0"))
            device = torch.device(f"cuda:{gpu_id}")

            kernel_result = exec_globals["run"](seed=42, budget_s=min(timeout, 120))

            # kernel_result should contain TFLOPs measurement
            if isinstance(kernel_result, (int, float)):
                score = float(kernel_result)
                result_queue.put((idx, score, f"tflops={score:.2f}"))
            else:
                result_queue.put((idx, 0.0, "bad_return"))

    except Exception as e:
        result_queue.put((idx, 0.0, f"error:{type(e).__name__}"))


def evaluate_with_timeout(idx, code, timeout, task):
    """Spawn a process and kill it if it exceeds timeout."""
    if not code or "def run" not in code:
        return idx, 0.0, "no_code"

    q = Queue()

    # Select worker based on task
    if task == "erdos":
        worker = evaluate_erdos_worker
    elif task == "gpu_mode":
        worker = evaluate_gpu_mode_worker
    else:
        return idx, 0.0, f"unsupported_task_{task}"

    p = Process(target=worker, args=(idx, code, timeout, q))
    p.start()
    p.join(timeout=timeout + 10)

    if p.is_alive():
        p.kill()
        p.join(timeout=5)
        return idx, 0.0, "timeout_killed"

    if not q.empty():
        return q.get_nowait()
    return idx, 0.0, f"exit_code_{p.exitcode}"


def main():
    # Parse task from environment
    task = os.getenv("TASK")
    if not task:
        print("Error: TASK environment variable required", file=sys.stderr)
        print("Example: TASK=erdos STEP=1 python scripts/evaluate_claude.py", file=sys.stderr)
        sys.exit(1)

    step = os.getenv("STEP", "1")
    timeout = int(os.getenv("EVAL_TIMEOUT", "120"))
    max_workers = int(os.getenv("EVAL_WORKERS", "32"))

    input_file = Path(f"checkpoints/claude_{task}_step{step}.jsonl")
    output_file = Path(f"checkpoints/claude_{task}_step{step}_scored.jsonl")

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Evaluate Claude {task.upper()} Step {step} ===")
    print(f"Reading {input_file}...")

    rollouts = []
    with open(input_file) as f:
        for line in f:
            rollouts.append(json.loads(line))

    print(f"Total rollouts: {len(rollouts)}")
    has_code = sum(1 for r in rollouts if r.get("has_run"))
    print(f"Has def run(): {has_code}/{len(rollouts)}")

    scores = [0.0] * len(rollouts)
    statuses = ["pending"] * len(rollouts)

    print(f"\nEvaluating {len(rollouts)} rollouts (timeout={timeout}s, workers={max_workers})...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, r in enumerate(rollouts):
            code = r.get("code", "")
            fut = executor.submit(evaluate_with_timeout, i, code, timeout, task)
            futures[fut] = i

        from concurrent.futures import as_completed
        done_count = 0
        for fut in as_completed(futures):
            idx, score, status = fut.result()
            scores[idx] = score
            statuses[idx] = status
            done_count += 1
            if done_count % 50 == 0:
                nonzero_so_far = sum(1 for s in scores if s > 0)
                elapsed = time.time() - t0
                print(f"  [{done_count}/{len(rollouts)}] nonzero={nonzero_so_far} elapsed={elapsed:.0f}s", flush=True)

    # Save results
    with open(output_file, "w") as f:
        for i, r in enumerate(rollouts):
            r["score"] = scores[i]
            r["eval_status"] = statuses[i]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n=== Results ===")
    nonzero = [s for s in scores if s > 0]
    print(f"Total: {len(scores)}")
    print(f"Success: {len(nonzero)}/{len(scores)} ({100*len(nonzero)/len(scores):.1f}%)")
    print(f"Fail Rate: {100*(1-len(nonzero)/len(scores)):.1f}%")

    if nonzero:
        if task == "erdos":
            # For Erdos: score = 1/C5 (higher is better)
            c5_bounds = [1.0/s for s in nonzero]
            print(f"Reward Max: {max(nonzero):.6f}")
            print(f"Reward Mean (all): {sum(scores)/len(scores):.6f}")
            print(f"Reward Mean (success): {sum(nonzero)/len(nonzero):.6f}")
            print(f"C5 bound min (best): {min(c5_bounds):.6f}")
        elif task == "gpu_mode":
            # For GPU mode: score = TFLOPs (higher is better)
            print(f"TFLOPs Max: {max(nonzero):.2f}")
            print(f"TFLOPs Mean (all): {sum(scores)/len(scores):.2f}")
            print(f"TFLOPs Mean (success): {sum(nonzero)/len(nonzero):.2f}")

    from collections import Counter
    status_counts = Counter(statuses)
    print(f"\nStatus breakdown:")
    for status, count in status_counts.most_common(10):
        print(f"  {status}: {count}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
