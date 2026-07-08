"""Test: does vllm.LLM produce deterministic output when requests are submitted
with DIFFERENT seeds (simulating per-rollout seeds)?"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import random, numpy as np, torch, hashlib
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def build_prompt():
    from ttt_discover.tinker_utils.state import State
    state = State(timestep=-1, construction=None, code="", value=0.0)
    state_ctx = state.to_prompt(2.636, metric_name="sum of radii")
    return f"""You are an expert mathematician specializing in circle packing.
Your task is to pack 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
{state_ctx}
Rules: define run_packing() -> tuple[np.ndarray, np.ndarray, float].
Think step by step."""

def main():
    model_name = "Qwen/Qwen3-8B"
    n_requests = 64
    max_tokens = 2000  # short for speed
    base_seed = 42

    prompt_text = build_prompt()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted += "<think>\n"
    prompt_ids = tokenizer.encode(formatted)
    print(f"Prompt length: {len(prompt_ids)} tokens")

    llm = LLM(
        model=model_name, tensor_parallel_size=4, max_model_len=4096,
        gpu_memory_utilization=0.5, seed=base_seed, enforce_eager=True,
        trust_remote_code=True, enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )

    # KEY DIFFERENCE: each request gets a DIFFERENT seed (like our VERL setup)
    prompts = []
    sp_list = []
    for i in range(n_requests):
        prompts.append({"prompt_token_ids": prompt_ids})
        sp_list.append(SamplingParams(
            temperature=1.0, top_p=1.0, top_k=-1,
            max_tokens=max_tokens,
            seed=base_seed + 10000 + i,  # unique per request
        ))

    print(f"Submitting {n_requests} requests with seeds {base_seed+10000}..{base_seed+10000+n_requests-1}")

    # RUN TWICE with the same LLM instance
    for run_id in range(2):
        outputs = llm.generate(prompts, sampling_params=sp_list)
        texts = [o.outputs[0].text for o in outputs]
        hashes = [hashlib.md5(t.encode()).hexdigest()[:8] for t in texts]
        unique = len(set(hashes))
        print(f"\nRun {run_id+1}: {unique}/64 unique, first hash={hashes[0]}")
        if run_id == 0:
            saved_hashes = hashes[:]
        else:
            match = sum(1 for a,b in zip(saved_hashes, hashes) if a == b)
            print(f"Cross-run match: {match}/64")
            if match == 64:
                print("DETERMINISTIC: same LLM instance, two calls, identical!")
            else:
                print(f"NON-DETERMINISTIC even within same process!")
                for i, (a,b) in enumerate(zip(saved_hashes, hashes)):
                    if a != b:
                        print(f"  First diff at rollout {i}")
                        break

if __name__ == "__main__":
    main()
