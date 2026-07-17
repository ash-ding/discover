"""Universal Claude replay script for TTT-Discover tasks.

Supports multiple tasks (erdos, gpu_mode, etc.) via environment variables.

Usage:
    TASK=erdos STEP=1 CONCURRENCY=64 python scripts/replay_claude.py
    TASK=gpu_mode STEP=1 python scripts/replay_claude.py

Environment Variables:
    TASK: Task name (erdos, gpu_mode, etc.) [required]
    STEP: Training step to replay [default: 1]
    CONCURRENCY: Number of concurrent API calls [default: 64]
    CLAUDE_PROJECT_ID: GCP project ID [default: itpc-gcp-ai-eng-claude]
    CLAUDE_REGION: Vertex AI region [default: us-east5]

Input:
    checkpoints/{task}_prompts_all_steps.json

Output:
    checkpoints/claude_{task}_step{N}.jsonl
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from anthropic import AnthropicVertex

# Model configuration
MODEL = "claude-opus-4-6"
TEMPERATURE = 1.0
PHASE1_MAX = 25300
PHASE2_MAX = 6700

# Task-specific Claude system prompts (matching TTT Advisor codebase exactly)
SYSTEM_PROMPTS = {
    "erdos": (
        "You are an expert in harmonic analysis and numerical optimization, "
        "solving the Erdős minimum overlap problem. Find a step function "
        "h: [0,2] → [0,1] with ∫h=1 that MINIMIZES C₅ = max_k ∫h(x)(1−h(x+k))dx. "
        "Your code MUST define run(seed, budget_s, **kwargs) returning "
        "(h_values, c5_bound, n_points). Use numpy and scipy. Strategy: try diverse "
        "approaches — gradient descent, simulated annealing, genetic algorithms, "
        "spectral methods, or convex relaxations. Do not get stuck on a single approach. "
        "Lower C₅ is better; current record is ≤ 0.3809."
    ),
    "gpu_mode": (
        "You are an expert GPU kernel engineer specializing in Triton. "
        "Your solution MUST include at least one @triton.jit decorated kernel "
        "function — solutions without @triton.jit will score zero. Strategy: "
        "use PyTorch for large matmuls (cuBLAS), but fuse elementwise operations "
        "(LayerNorm, sigmoid, gating, masking, multiply) into Triton kernels "
        "for maximum throughput."
    ),
}
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert problem solver. Provide a clear, concise solution to the problem. "
    "Show your reasoning and give a final answer."
)


def extract_code(text):
    """Extract the last Python code block from text."""
    pattern = re.compile(r'```python\n(.*?)(?:\n```)', re.DOTALL)
    matches = list(pattern.finditer(text))
    return matches[-1].group(1).rstrip() if matches else ""


ENTRY_FUNCTIONS = {
    "gpu_mode": "custom_kernel",
}
DEFAULT_ENTRY_FUNCTION = "run"

PHASE2_MESSAGES = {
    "gpu_mode": (
        "Your response was cut off. Please provide ONLY the complete Python code "
        "with the `custom_kernel(data)` function. "
        "No explanation needed, just the code in a ```python block."
    ),
}
DEFAULT_PHASE2_MESSAGE = (
    "Your response was cut off. Please provide ONLY the complete Python code "
    "with `def run(seed=42, budget_s=1000, **kwargs)` function. "
    "No explanation needed, just the code in a ```python block."
)


async def call_claude_two_phase(client, user_content, semaphore, system_prompt=None,
                                entry_func="run", phase2_msg=None, idx=""):
    """Two-phase Claude generation.

    Phase 1: Free generation up to PHASE1_MAX tokens
    Phase 2: Only if Phase 1 hit limit AND no complete code found

    Returns:
        dict with output, code, and metadata
    """
    if phase2_msg is None:
        phase2_msg = DEFAULT_PHASE2_MESSAGE

    async with semaphore:
        try:
            t0 = time.time()

            # Phase 1: free generation
            def _phase1():
                kwargs = dict(
                    model=MODEL,
                    max_tokens=PHASE1_MAX,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": user_content}],
                )
                if system_prompt:
                    kwargs["system"] = system_prompt
                with client.messages.stream(**kwargs) as stream:
                    return stream.get_final_message()

            r1 = await asyncio.to_thread(_phase1)
            p1_text = r1.content[0].text
            p1_tokens = r1.usage.output_tokens
            hit_limit = r1.stop_reason == "max_tokens"

            code = extract_code(p1_text)
            has_complete_code = f"def {entry_func}" in code

            phase2_done = False
            p2_tokens = 0

            if not hit_limit or has_complete_code:
                final_text = p1_text
            else:
                # Phase 2: explicit continuation request
                def _phase2():
                    kwargs = dict(
                        model=MODEL,
                        max_tokens=PHASE2_MAX,
                        temperature=TEMPERATURE,
                        messages=[
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": p1_text},
                            {"role": "user", "content": phase2_msg},
                        ],
                    )
                    if system_prompt:
                        kwargs["system"] = system_prompt
                    with client.messages.stream(**kwargs) as stream:
                        return stream.get_final_message()

                r2 = await asyncio.to_thread(_phase2)
                p2_text = r2.content[0].text
                p2_tokens = r2.usage.output_tokens
                final_text = p1_text + "\n\n" + p2_text
                phase2_done = True

            elapsed = time.time() - t0
            code = extract_code(final_text)

            return {
                "output": final_text,
                "code": code,
                "p1_tokens": p1_tokens,
                "p2_tokens": p2_tokens,
                "phase2": phase2_done,
                "p1_stop": r1.stop_reason,
                "has_entry": f"def {entry_func}" in code,
                "time": round(elapsed, 1),
                "output_len": len(final_text),
                "code_len": len(code),
            }
        except Exception as e:
            print(f"  [{idx}] API error: {type(e).__name__}: {e}", flush=True)
            return {
                "output": "",
                "code": "",
                "p1_tokens": 0,
                "p2_tokens": 0,
                "phase2": False,
                "p1_stop": "error",
                "has_entry": False,
                "time": 0,
                "error": str(e),
                "output_len": 0,
                "code_len": 0,
            }


async def main():
    # Parse task from environment
    task = os.getenv("TASK")
    if not task:
        print("Error: TASK environment variable required", file=sys.stderr)
        print("Example: TASK=erdos STEP=1 python scripts/replay_claude.py", file=sys.stderr)
        sys.exit(1)

    step = int(os.getenv("STEP", "1"))
    n_rollouts = int(os.getenv("ROLLOUTS_PER_PROMPT", "64"))
    concurrency = int(os.getenv("CONCURRENCY", "64"))

    # File paths
    prompt_file = Path(f"checkpoints/{task}_prompts_all_steps.json")
    output_file = Path(f"checkpoints/claude_{task}_step{step}.jsonl")

    if not prompt_file.exists():
        print(f"Error: Prompt file not found: {prompt_file}", file=sys.stderr)
        print(f"Expected format: checkpoints/<task>_prompts_all_steps.json", file=sys.stderr)
        sys.exit(1)

    # Initialize API client
    client = AnthropicVertex(
        project_id=os.getenv("CLAUDE_PROJECT_ID", "itpc-gcp-ai-eng-claude"),
        region=os.getenv("CLAUDE_REGION", "us-east5"),
    )

    # Load prompts
    with open(prompt_file) as f:
        all_prompts = json.load(f)

    prompts = all_prompts[str(step)]

    # Task-specific configuration (matching TTT Advisor)
    system_prompt = SYSTEM_PROMPTS.get(task, DEFAULT_SYSTEM_PROMPT)
    entry_func = ENTRY_FUNCTIONS.get(task, DEFAULT_ENTRY_FUNCTION)
    phase2_msg = PHASE2_MESSAGES.get(task, DEFAULT_PHASE2_MESSAGE)

    unique_count = len(set(p[:200] for p in prompts))
    print(f"=== Claude Replay: {task.upper()} Step {step} ===", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"System prompt: {system_prompt[:80]}...", flush=True)
    print(f"Entry function: {entry_func}", flush=True)
    print(f"Phase 1 max: {PHASE1_MAX}, Phase 2 max: {PHASE2_MAX}", flush=True)
    print(f"Groups: {len(prompts)} ({unique_count} unique), Rollouts/group: {n_rollouts}", flush=True)
    print(f"Total calls: {len(prompts) * n_rollouts}", flush=True)
    print(f"Concurrency: {concurrency}", flush=True)
    print(f"Output: {output_file}", flush=True)
    print(flush=True)

    # Resume support
    existing = 0
    if output_file.exists():
        with open(output_file) as f:
            existing = sum(1 for _ in f)

    total = len(prompts) * n_rollouts
    if existing >= total:
        print(f"Already complete ({existing} results)", flush=True)
        return
    else:
        print(f"Resuming from {existing}/{total}", flush=True)

    # Run replay with concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    completed = existing

    with open(output_file, "a") as f:
        for pi, prompt in enumerate(prompts):
            # Extract user content (remove "user\n" prefix and "assistant\n" suffix)
            user_content = prompt.split("\nassistant\n", 1)[0]
            if user_content.startswith("user\n"):
                user_content = user_content[5:]

            for batch_start in range(0, n_rollouts, concurrency):
                tasks = []
                task_ri_values = []

                for ri in range(batch_start, min(batch_start + concurrency, n_rollouts)):
                    item_idx = pi * n_rollouts + ri
                    if item_idx < existing:
                        continue

                    idx = f"p{pi}r{ri}"
                    tasks.append(call_claude_two_phase(
                        client, user_content, semaphore, system_prompt,
                        entry_func, phase2_msg, idx))
                    task_ri_values.append(ri)

                if not tasks:
                    continue

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for ri, result in zip(task_ri_values, results):
                    if isinstance(result, Exception):
                        result = {
                            "error": str(result),
                            "has_entry": False,
                            "code": "",
                            "output": "",
                            "time": 0,
                            "p1_tokens": 0,
                            "p2_tokens": 0,
                            "output_len": 0,
                            "code_len": 0,
                            "phase2": False,
                            "p1_stop": "exception"
                        }

                    entry = {
                        "prompt_idx": pi,
                        "rollout_idx": ri,
                        **result,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    completed += 1

                f.flush()
                print(f"  [{completed}/{total}] p{pi} ri {task_ri_values[0]}-{task_ri_values[-1]} done", flush=True)

    # Summary statistics
    print(f"\n=== Summary ===", flush=True)
    results = []
    with open(output_file) as f:
        for line in f:
            results.append(json.loads(line))

    has_entry_count = sum(1 for r in results if r.get("has_entry"))
    phase2_count = sum(1 for r in results if r.get("phase2"))
    errors = sum(1 for r in results if "error" in r)
    avg_time = sum(r.get("time", 0) for r in results) / len(results) if results else 0
    avg_p1 = sum(r.get("p1_tokens", 0) for r in results) / len(results) if results else 0

    print(f"Total: {len(results)}", flush=True)
    print(f"Has {entry_func}(): {has_entry_count}/{len(results)} ({100*has_entry_count/len(results):.1f}%)", flush=True)
    print(f"Needed Phase 2: {phase2_count}/{len(results)}", flush=True)
    print(f"Errors: {errors}", flush=True)
    print(f"Avg time: {avg_time:.0f}s", flush=True)
    print(f"Avg Phase 1 tokens: {avg_p1:.0f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
