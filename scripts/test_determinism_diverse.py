"""Test: full_determinism + per-request unique seed → diverse but reproducible outputs."""

import asyncio
import os
import hashlib
import re

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
    from ttt_discover.tinker_utils.state import State
    state = State(timestep=-1, construction=None, code="", value=0.0)
    state_ctx = state.to_prompt(2.636, metric_name="sum of radii")
    return f"""You are an expert mathematician specializing in circle packing problems and computational geometry.

Your task is to pack 26 circles in a unit square [0,1]×[0,1] to maximize the sum of radii.

{state_ctx}

Rules:
- You must define the run_packing function: def run_packing() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape (26, 2) and radii has shape (26,).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- No filesystem or network IO.

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""


def extract_last_code_block(text):
    languages = ['python', 'cpp', 'java', 'cuda']
    languages_pattern = '|'.join(re.escape(lang) for lang in languages)
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(?!```)(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].group(2).rstrip()
    return ""


async def main():
    model_name = "Qwen/Qwen3-8B"
    base_seed = 42
    n_requests = 64
    max_tokens = 26000

    print(f"Model: {model_name}")
    print(f"Base seed: {base_seed}")
    print(f"Per-request seed: base_seed + session_id (42..105)")
    print(f"Full determinism: ON")
    print(f"Requests: {n_requests}")
    print(f"Max tokens: {max_tokens}")
    print()

    prompt_text = build_prompt()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted += "<think>\n"
    prompt_ids = tokenizer.encode(formatted)

    print(f"Prompt length: {len(prompt_ids)} tokens")
    print()

    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        max_model_len=32768,
        gpu_memory_utilization=0.5,
        seed=base_seed,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Each request gets a unique seed: base_seed + session_id
    prompts = []
    sampling_params_list = []
    for session_id in range(n_requests):
        prompts.append({"prompt_token_ids": prompt_ids})
        sampling_params_list.append(SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            seed=base_seed + session_id,
        ))

    print(f"Submitting {n_requests} requests with seeds {base_seed}..{base_seed + n_requests - 1}...")
    outputs = llm.generate(prompts, sampling_params=sampling_params_list)

    print(f"\n{'='*60}")
    print(f"=== Diversity Analysis ===")
    print(f"{'='*60}")

    texts = [o.outputs[0].text for o in outputs]

    # Uniqueness
    hashes = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    unique = len(set(hashes))
    print(f"\nUnique outputs: {unique}/{n_requests}")

    # Length distribution
    lengths = [len(t) for t in texts]
    print(f"\nOutput length (chars): min={min(lengths)} max={max(lengths)} mean={sum(lengths)//len(lengths)}")

    # Token count estimate
    tok_lens = [len(t)//3.7 for t in texts]
    print(f"Output length (est tokens): min={int(min(tok_lens))} max={int(max(tok_lens))} mean={int(sum(tok_lens)//len(tok_lens))}")

    # Extract code and evaluate scores
    print(f"\n{'='*60}")
    print(f"=== Score Distribution ===")
    print(f"{'='*60}")

    scores = []
    has_code = 0
    for i, text in enumerate(texts):
        code = extract_last_code_block(text)
        if code:
            has_code += 1
            # Try to evaluate
            try:
                exec_globals = {"np": np}
                exec(code, exec_globals)
                if "run_packing" in exec_globals:
                    result = exec_globals["run_packing"]()
                    centers, radii, sum_radii = result
                    scores.append(float(sum_radii))
                else:
                    scores.append(0.0)
            except Exception as e:
                scores.append(0.0)
        else:
            scores.append(0.0)

    nonzero = [s for s in scores if s > 0]
    print(f"\nCode extracted: {has_code}/{n_requests}")
    print(f"Execution success (score > 0): {len(nonzero)}/{n_requests}")

    if nonzero:
        print(f"\nScore distribution (nonzero only):")
        print(f"  max:  {max(nonzero):.6f}")
        print(f"  min:  {min(nonzero):.6f}")
        print(f"  mean: {sum(nonzero)/len(nonzero):.6f}")
        print(f"  top 5: {sorted(nonzero, reverse=True)[:5]}")

    # Divergence analysis
    print(f"\n{'='*60}")
    print(f"=== Divergence Analysis (vs output 0) ===")
    print(f"{'='*60}")

    ref = texts[0]
    for i in range(1, min(20, n_requests)):
        other = texts[i]
        min_len = min(len(ref), len(other))
        pos = min_len
        for j in range(min_len):
            if ref[j] != other[j]:
                pos = j
                break
        if pos < min_len:
            ctx = ref[max(0,pos-20):pos]
            print(f"  Output {i:2d}: diverge at char {pos:6d} | ...{repr(ctx[-20:])}")
        else:
            if len(ref) == len(other):
                print(f"  Output {i:2d}: IDENTICAL")
            else:
                print(f"  Output {i:2d}: same up to {min_len}, lengths differ ({len(ref)} vs {len(other)})")


if __name__ == "__main__":
    asyncio.run(main())
