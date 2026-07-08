"""Test: TP=1, no tensor parallelism, single GPU — eliminate NCCL."""
import os, gc
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    return f"""You are an expert mathematician. Pack 26 circles in [0,1]^2. {state_ctx} Think step by step."""

def run_once(prompt_ids, n=16, base_seed=42, max_tokens=500):
    llm = LLM(model="Qwen/Qwen3-8B", tensor_parallel_size=1, max_model_len=2048,
              gpu_memory_utilization=0.9, seed=base_seed, enforce_eager=True,
              trust_remote_code=True, enable_prefix_caching=False, enable_chunked_prefill=False)
    prompts = [{"prompt_token_ids": prompt_ids}] * n
    sp_list = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                              max_tokens=max_tokens, seed=base_seed+10000+i) for i in range(n)]
    outputs = llm.generate(prompts, sampling_params=sp_list)
    texts = [o.outputs[0].text for o in outputs]
    del llm; gc.collect(); torch.cuda.empty_cache()
    return texts

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    prompt_text = build_prompt()
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>\n"
    prompt_ids = tokenizer.encode(formatted)
    print(f"Prompt: {len(prompt_ids)} tokens, 16 requests, TP=1")

    texts1 = run_once(prompt_ids); print(f"Run 1 done: {len(set(hashlib.md5(t.encode()).hexdigest() for t in texts1))}/16 unique")
    import time; time.sleep(3)
    texts2 = run_once(prompt_ids); print(f"Run 2 done: {len(set(hashlib.md5(t.encode()).hexdigest() for t in texts2))}/16 unique")

    text_match = sum(1 for a,b in zip(texts1, texts2) if a == b)
    print(f"\nText match: {text_match}/16")
    if text_match == 16:
        print("FULLY DETERMINISTIC with TP=1!")
    else:
        for i, (a,b) in enumerate(zip(texts1, texts2)):
            if a != b:
                pos = next((j for j in range(min(len(a),len(b))) if a[j]!=b[j]), min(len(a),len(b)))
                print(f"First diff: rollout {i}, char {pos}/{min(len(a),len(b))}")
                break

if __name__ == "__main__":
    main()
