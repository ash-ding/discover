"""Stress test the eval server using saved rollout data.

Usage:
    python scripts/stress_test_eval_server.py \
        --rollout-file checkpoints/ttt-discover/gpu-mode-trimul_verl_20260715_0259/rollouts/20.jsonl \
        --server http://10.241.128.16:8890 \
        --concurrency 512
"""

import argparse
import asyncio
import json
import re
import time
import urllib.request


def extract_code(output_text):
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, output_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def load_rollout_codes(path, max_samples=None):
    codes = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            code = extract_code(row.get("output", ""))
            if code and "@triton.jit" in code:
                codes.append((code, row.get("score", 0), row.get("uid", "")))
    if max_samples and len(codes) > max_samples:
        codes = codes[:max_samples]
    return codes


def _http_eval(url, payload, timeout):
    req = urllib.request.Request(url, data=payload,
                                headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


async def eval_one(server_url, code, task_name, timeout, idx):
    payload = json.dumps({"code": code, "task_name": task_name, "gpu_type": "H100"}).encode()
    url = f"{server_url.rstrip('/')}/"
    t0 = time.time()
    try:
        result = await asyncio.to_thread(_http_eval, url, payload, timeout)
        elapsed = time.time() - t0
        return {"idx": idx, "elapsed": elapsed, "success": result.get("success", False),
                "score_us": result.get("score_us"), "error": result.get("error", "")[:200]}
    except Exception as e:
        elapsed = time.time() - t0
        return {"idx": idx, "elapsed": elapsed, "success": False,
                "score_us": None, "error": f"HTTP: {type(e).__name__}: {e}"}


async def run_stress_test(server_url, codes, concurrency, timeout, task_name):
    print(f"Stress test: {len(codes)} requests, concurrency={concurrency}, timeout={timeout}s")
    print(f"Server: {server_url}")
    print()

    sem = asyncio.Semaphore(concurrency)

    async def limited(coro):
        async with sem:
            return await coro

    t_start = time.time()
    tasks = [limited(eval_one(server_url, code, task_name, timeout, i))
             for i, (code, _, _) in enumerate(codes)]
    results = await asyncio.gather(*tasks)
    t_total = time.time() - t_start

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    http_errors = [r for r in failures if r["error"].startswith("HTTP:")]
    eval_errors = [r for r in failures if not r["error"].startswith("HTTP:")]

    elapsed_all = [r["elapsed"] for r in results]
    elapsed_all.sort()

    print("=" * 60)
    print(f"RESULTS ({len(codes)} requests in {t_total:.1f}s)")
    print("=" * 60)
    print(f"  Success:      {len(successes)}/{len(codes)} ({100*len(successes)/len(codes):.1f}%)")
    print(f"  Eval errors:  {len(eval_errors)} (code evaluated but failed)")
    print(f"  HTTP errors:  {len(http_errors)} (never evaluated)")
    print()
    print(f"  Latency (all requests):")
    print(f"    min:    {elapsed_all[0]:.1f}s")
    print(f"    p50:    {elapsed_all[len(elapsed_all)//2]:.1f}s")
    print(f"    p90:    {elapsed_all[int(len(elapsed_all)*0.9)]:.1f}s")
    print(f"    p99:    {elapsed_all[int(len(elapsed_all)*0.99)]:.1f}s")
    print(f"    max:    {elapsed_all[-1]:.1f}s")
    print(f"    total:  {t_total:.1f}s wall clock")
    print()

    if successes:
        scores = [r["score_us"] for r in successes if r["score_us"] is not None]
        if scores:
            scores.sort()
            print(f"  Score (successful, μs):")
            print(f"    min:    {scores[0]:.1f}")
            print(f"    median: {scores[len(scores)//2]:.1f}")
            print(f"    max:    {scores[-1]:.1f}")
            print()

    if http_errors:
        print(f"  HTTP errors (first 5):")
        for r in http_errors[:5]:
            print(f"    [{r['idx']}] {r['elapsed']:.1f}s: {r['error']}")
        print()

    if eval_errors:
        from collections import Counter
        err_types = Counter()
        for r in eval_errors:
            err_preview = r["error"][:80]
            err_types[err_preview] += 1
        print(f"  Eval error breakdown (top 5):")
        for err, cnt in err_types.most_common(5):
            print(f"    {cnt}x: {err}")
        print()

    # Check pool health after test
    try:
        health = json.loads(urllib.request.urlopen(
            f"{server_url.rstrip('/')}/health", timeout=5).read())
        print(f"  Post-test server health:")
        print(f"    pool_queue_depth: {health.get('pool_queue_depth', 'N/A')}")
        for g in health.get("gpu_details", []):
            print(f"    GPU {g['gpu_id']}: {g['state']}, busy={g.get('busy')}")
    except Exception as e:
        print(f"  Post-test health check failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-file", required=True)
    parser.add_argument("--server", default="http://10.241.128.16:8890")
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--task-name", default="trimul")
    args = parser.parse_args()

    codes = load_rollout_codes(args.rollout_file, args.max_samples)
    print(f"Loaded {len(codes)} valid Triton code samples from {args.rollout_file}")
    if not codes:
        print("No valid code samples found!")
        return

    asyncio.run(run_stress_test(
        args.server, codes, args.concurrency, args.timeout, args.task_name))


if __name__ == "__main__":
    main()
