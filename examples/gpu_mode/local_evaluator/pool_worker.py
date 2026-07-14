#!/usr/bin/env python3
"""
Persistent container worker for container pooling mode.

Reads eval requests from stdin (JSON lines), runs evals, writes results to stdout.
Heavy libraries are imported once at startup to avoid per-eval import overhead.

Protocol:
  - On startup, prints {"status": "ready"} to stdout
  - For each line on stdin: parse JSON request, run eval, print JSON result to stdout
  - Request: {"code": "...", "task_name": "trimul", "gpu_type": "H100", "timeout": 530}
  - Result:  {"success": bool, "score_us": float, "error": str|null, ...}
"""

import sys
import json
import os
import traceback
import tempfile

# stdout is the JSON protocol channel — eval code must not pollute it.
# Save real stdout for protocol messages, redirect sys.stdout to stderr.
_protocol_out = sys.stdout
sys.stdout = sys.stderr

from eval_worker import run_evaluation, load_task_definition, TASK_MAP


def _send(obj):
    _protocol_out.write(json.dumps(obj) + "\n")
    _protocol_out.flush()


# Pre-load task definitions once (avoids re-reading YAML per eval)
TASKS = {}
for name in TASK_MAP:
    try:
        TASKS[name] = load_task_definition(name)
    except Exception as e:
        print(f"Warning: failed to pre-load task '{name}': {e}", flush=True)

_send({"status": "ready"})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        request = json.loads(line)
    except json.JSONDecodeError as e:
        _send({
            "success": False,
            "score_us": -1_000_000,
            "error": f"Invalid JSON request: {e}",
        })
        continue

    code = request.get("code", "")
    task_name = request.get("task_name", "trimul")
    gpu_type = request.get("gpu_type", "H100")
    timeout = request.get("timeout", int(os.environ.get("EVAL_TIMEOUT", "530")))

    try:
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            task_def = TASKS.get(task_name)
            result = run_evaluation(code, task_name, gpu_type, timeout=timeout, task_def=task_def)
            os.chdir(old_cwd)
    except Exception as e:
        traceback.print_exc()
        result = {
            "success": False,
            "score_us": -1_000_000,
            "error": f"Pool worker exception: {e}",
            "gpu_crash": "CUDA" in str(e).upper(),
        }
        try:
            os.chdir(old_cwd)
        except Exception:
            pass

    _send(result)
