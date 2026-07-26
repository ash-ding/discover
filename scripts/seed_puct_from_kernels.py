#!/usr/bin/env python3
"""Seed PUCT sampler state from a kernel JSON file.

Creates a step-0 PUCT state file so that training starts from pre-existing
kernel designs instead of empty code. The agent loop's _detect_puct_resume_step()
will find the file and load it automatically.

Usage:
    # Basic: use score_us values from JSON, convert to reward
    python scripts/seed_puct_from_kernels.py \
        --kernels examples/gpu_mode/kernels/top8_kernels.json \
        --experiment-name trimul-seed-top8 \
        --score-scale 1500

    # With live re-evaluation: send each kernel to eval server first
    python scripts/seed_puct_from_kernels.py \
        --kernels examples/gpu_mode/kernels/top8_kernels.json \
        --experiment-name trimul-seed-top8 \
        --score-scale 1500 \
        --eval-server http://10.241.128.16:8890 \
        --task-name trimul
"""

import argparse
import json
import os
import sys
import uuid


def eval_kernel(server_url, code, task_name, timeout=600):
    """Evaluate a kernel via the HTTP eval server, return score_us or None."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "code": code,
        "task_name": task_name,
        "timeout": timeout,
    }).encode()

    url = f"{server_url.rstrip('/')}/eval"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp_bytes = urllib.request.urlopen(req, timeout=timeout + 60).read()
        resp = json.loads(resp_bytes)
        if resp.get("success") and resp.get("score_us", 0) > 0:
            return resp["score_us"]
        print(f"  Eval failed: {resp.get('error', 'unknown')}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Eval error: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Seed PUCT state from kernel JSON")
    parser.add_argument("--kernels", required=True, help="Path to kernel JSON file")
    parser.add_argument("--experiment-name", required=True, help="Experiment name for checkpoint dir")
    parser.add_argument("--score-scale", type=float, required=True,
                        help="Score scale for reward conversion (e.g., 1500 for trimul, 5000 for mla_decode)")
    parser.add_argument("--checkpoint-dir", default="checkpoints/ttt-discover",
                        help="Base checkpoint directory (default: checkpoints/ttt-discover)")
    parser.add_argument("--eval-server", default=None,
                        help="Eval server URL to re-evaluate kernels (e.g., http://10.241.128.16:8890)")
    parser.add_argument("--task-name", default=None,
                        help="Task name for eval server (e.g., trimul, mla_decode)")
    parser.add_argument("--eval-timeout", type=int, default=600,
                        help="Per-kernel eval timeout in seconds (default: 600)")
    args = parser.parse_args()

    if args.eval_server and not args.task_name:
        parser.error("--task-name is required when using --eval-server")

    with open(args.kernels) as f:
        kernels = json.load(f)

    states = []
    for i, k in enumerate(kernels):
        score_us = k["score_us"]

        if args.eval_server:
            print(f"Evaluating kernel {i} (file score_us={score_us:.2f}) ...", flush=True)
            live_score = eval_kernel(args.eval_server, k["code"], args.task_name, args.eval_timeout)
            if live_score is not None:
                print(f"  Live score_us={live_score:.2f} (file={score_us:.2f}, diff={live_score - score_us:+.2f})")
                score_us = live_score
            else:
                print(f"  Live eval failed, using file score_us={score_us:.2f}")

        reward = args.score_scale / score_us

        state = {
            "type": "State",
            "id": str(uuid.uuid4()),
            "timestep": -1,
            "value": reward,
            "code": k["code"],
            "construction": None,
            "parent_values": [],
            "parents": [],
            "observation": "",
        }
        states.append(state)
        print(f"  Kernel {i}: score_us={score_us:.2f} → reward={reward:.6f}")

    store = {
        "step": 0,
        "states": states,
        "initial_states": list(states),
        "puct_n": {},
        "puct_m": {},
        "puct_T": 0,
    }

    exp_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    out_path = os.path.join(exp_dir, "puct_sampler_step_000000.json")

    with open(out_path, "w") as f:
        json.dump(store, f, indent=2, sort_keys=True)

    print(f"\nSeeded {len(states)} states from {args.kernels}")
    print(f"  Score scale: {args.score_scale}")
    print(f"  Rewards: {[round(s['value'], 6) for s in states]}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
