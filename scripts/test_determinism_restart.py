"""Test: rebuild LLM instance between calls to get clean engine state."""

import os, gc
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

def run_once(prompt_ids, n=64, base_seed=42, max_tokens=2000):
    llm = LLM(
        model="Qwen/Qwen3-8B", tensor_parallel_size=4, max_model_len=4096,
        gpu_memory_utilization=0.5, seed=base_seed, enforce_eager=True,
        trust_remote_code=True, enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    prompts = [{"prompt_token_ids": prompt_ids}] * n
    sp_list = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                              max_tokens=max_tokens, seed=base_seed+10000+i)
               for i in range(n)]
    outputs = llm.generate(prompts, sampling_params=sp_list)
    texts = [o.outputs[0].text for o in outputs]
    hashes = [hashlib.md5(t.encode()).hexdigest()[:8] for t in texts]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts, hashes

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    prompt_text = build_prompt()
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted += "<think>\n"
    prompt_ids = tokenizer.encode(formatted)
    print(f"Prompt: {len(prompt_ids)} tokens, 64 requests, different seeds")

    print("\n=== Run 1 (fresh LLM) ===")
    texts1, hashes1 = run_once(prompt_ids)
    unique1 = len(set(hashes1))
    print(f"  {unique1}/64 unique, first={hashes1[0]}")

    import time; time.sleep(5)

    print("\n=== Run 2 (fresh LLM) ===")
    texts2, hashes2 = run_once(prompt_ids)
    unique2 = len(set(hashes2))
    print(f"  {unique2}/64 unique, first={hashes2[0]}")

    match = sum(1 for a,b in zip(hashes1, hashes2) if a == b)
    text_match = sum(1 for a,b in zip(texts1, texts2) if a == b)
    print(f"\nCross-run hash match: {match}/64")
    print(f"Cross-run text match: {text_match}/64")
    if text_match == 64:
        print("FULLY DETERMINISTIC!")
    else:
        for i, (a,b) in enumerate(zip(texts1, texts2)):
            if a != b:
                min_len = min(len(a), len(b))
                pos = min_len
                for j in range(min_len):
                    if a[j] != b[j]:
                        pos = j
                        break
                print(f"First diff: rollout {i}, char {pos}")
                break

if __name__ == "__main__":
    main()
