"""Run generate() 3 times on same LLM instance to check which rollout differs."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    from ttt_discover.tinker_utils.state import State
    state = State(timestep=-1, construction=None, code="", value=0.0)
    state_ctx = state.to_prompt(2.636, metric_name="sum of radii")
    prompt_text = f"You are an expert mathematician. Pack 26 circles in [0,1]^2. {state_ctx} Think step by step."

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>\n"
    prompt_ids = tokenizer.encode(formatted)

    llm = LLM(model="Qwen/Qwen3-8B", tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.5, seed=42, enforce_eager=True,
              trust_remote_code=True, enable_prefix_caching=False, enable_chunked_prefill=False)

    n = 64
    prompts = [{"prompt_token_ids": prompt_ids}] * n
    sp_list = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=2000, seed=10042+i) for i in range(n)]

    all_texts = []
    for run_id in range(3):
        outputs = llm.generate(prompts, sampling_params=sp_list)
        texts = [o.outputs[0].text for o in outputs]
        all_texts.append(texts)
        print(f"Run {run_id+1} done")

    diffs_12 = [i for i in range(n) if all_texts[0][i] != all_texts[1][i]]
    diffs_23 = [i for i in range(n) if all_texts[1][i] != all_texts[2][i]]
    diffs_13 = [i for i in range(n) if all_texts[0][i] != all_texts[2][i]]

    print(f"\nRun1 vs Run2: {len(diffs_12)} diffs at positions {diffs_12}")
    print(f"Run2 vs Run3: {len(diffs_23)} diffs at positions {diffs_23}")
    print(f"Run1 vs Run3: {len(diffs_13)} diffs at positions {diffs_13}")

    # Show divergence details for first diff
    for name, d, t1, t2 in [("1v2", diffs_12, all_texts[0], all_texts[1]),
                              ("2v3", diffs_23, all_texts[1], all_texts[2])]:
        if d:
            i = d[0]
            a, b = t1[i], t2[i]
            pos = next((j for j in range(min(len(a),len(b))) if a[j]!=b[j]), min(len(a),len(b)))
            print(f"\n{name} first diff: rollout {i}, char {pos}/{min(len(a),len(b))}")

if __name__ == "__main__":
    main()
