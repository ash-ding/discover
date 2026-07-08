"""Test whether 64 requests with same seed + full_determinism produce identical outputs."""

import asyncio
import os
import hashlib

# Enable full determinism BEFORE importing anything else
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def build_prompt():
    """Build the same prompt as step 0 Circle Packing (empty initial state)."""
    from ttt_discover.tinker_utils.state import State
    state = State(timestep=-1, construction=None, code="", value=0.0)
    state_ctx = state.to_prompt(2.636, metric_name="sum of radii")

    prompt = f"""You are an expert mathematician specializing in circle packing problems and computational geometry.

Your task is to pack 26 circles in a unit square [0,1]×[0,1] to maximize the sum of radii.

{state_ctx}

Rules:
- You must define the run_packing function: def run_packing() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape (26, 2) and radii has shape (26,).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- No filesystem or network IO.

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""
    return prompt


async def main():
    model_name = "Qwen/Qwen3-8B"
    seed = 42
    n_requests = 64
    max_tokens = 2000  # short for speed, enough to see divergence

    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print(f"Full determinism: ON")
    print(f"Requests: {n_requests}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Build prompt
    prompt_text = build_prompt()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Format with chat template + <think>
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted += "<think>\n"
    prompt_ids = tokenizer.encode(formatted)

    print(f"Prompt length: {len(prompt_ids)} tokens")
    print()

    # Initialize vLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        max_model_len=4096,  # short context for this test
        gpu_memory_utilization=0.5,
        seed=seed,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Create sampling params with seed
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_tokens,
        seed=seed,
    )

    # Submit 64 requests with same prompt and same seed
    print(f"Submitting {n_requests} requests with seed={seed}...")

    # Use llm.generate with list of prompts (same prompt repeated)
    prompts = [{"prompt_token_ids": prompt_ids}] * n_requests
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # Analyze results
    print(f"\n=== Results ===")
    texts = [o.outputs[0].text for o in outputs]
    token_ids_list = [o.outputs[0].token_ids for o in outputs]

    # Check how many are identical
    hashes = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    unique_hashes = set(hashes)
    print(f"Unique outputs: {len(unique_hashes)}/{n_requests}")

    # Count frequency of each unique output
    from collections import Counter
    hash_counts = Counter(hashes)
    for h, count in hash_counts.most_common():
        idx = hashes.index(h)
        print(f"  Hash {h[:8]}: {count} copies, length={len(texts[idx])} chars")

    # If not all identical, find divergence points
    if len(unique_hashes) > 1:
        ref = texts[0]
        print(f"\n=== Divergence analysis (vs output 0) ===")
        for i in range(1, min(10, n_requests)):
            other = texts[i]
            min_len = min(len(ref), len(other))
            pos = min_len
            for j in range(min_len):
                if ref[j] != other[j]:
                    pos = j
                    break
            if pos < min_len:
                print(f"  Output {i:2d}: diverge at char {pos}")
            else:
                print(f"  Output {i:2d}: IDENTICAL")
    else:
        print("\nAll 64 outputs are IDENTICAL!")
        print(f"First 200 chars: {repr(texts[0][:200])}")


if __name__ == "__main__":
    asyncio.run(main())
