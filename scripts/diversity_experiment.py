"""Measure rollout diversity on a real task prompt.

Answers one question: for a fixed parent state, does drawing N completions
actually buy search value over drawing 1? Reports best-of-k curves computed
exactly from order statistics, plus uniqueness counts on code / construction /
score.

    TASK=circle_packing N_PER_ARM=20 python scripts/diversity_experiment.py
"""
import asyncio, hashlib, json, math, os, re, statistics, sys, time
from pathlib import Path

sys.path.insert(0, os.getcwd())

TASK = os.getenv("TASK", "circle_packing")
N = int(os.getenv("N_PER_ARM", "20"))
CONC = int(os.getenv("CONCURRENCY", "10"))
MAXTOK = int(os.getenv("MAX_TOKENS", "16000"))
EFFORT = os.getenv("EFFORT", "high")
MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))  # scripts/ is not a package
from claude_puct_loop import (
    TASKS, SYSTEM_PROMPTS, DEFAULT_SYSTEM_PROMPT, STRATEGY_HINTS,
    extract_code, Usage)

T = TASKS[TASK]

ARMS = {
    "none":   lambda i: "",
    "rotate": lambda i: STRATEGY_HINTS[i % len(STRATEGY_HINTS)],
    "nonce":  lambda i: (f"This is independent exploration attempt #{i+1}. "
                         f"Produce your own solution from scratch."),
}


def best_of_k(scores, k):
    """Exact E[max of a uniformly random k-subset] via order statistics."""
    s = sorted(scores)
    n = len(s)
    if k >= n:
        return s[-1]
    tot = math.comb(n, k)
    return sum(s[i] * math.comb(i, k - 1) for i in range(k - 1, n)) / tot


async def run_arm(client, arm, hint_fn, system, prompt, extra, usage):
    sem = asyncio.Semaphore(CONC)

    async def one(i):
        async with sem:
            blocks = [{"type": "text", "text": prompt,
                       "cache_control": {"type": "ephemeral"}}]
            h = hint_fn(i)
            if h:
                blocks.append({"type": "text", "text": h})
            try:
                async with client.messages.stream(
                        model=MODEL, max_tokens=MAXTOK,
                        thinking={"type": "adaptive"},
                        output_config={"effort": EFFORT},
                        system=[{"type": "text", "text": system}],
                        messages=[{"role": "user", "content": blocks}]) as s:
                    r = await s.get_final_message()
                usage.add(r.usage)
                txt = "".join(b.text for b in r.content if b.type == "text")
                return extract_code(txt)
            except Exception as e:
                print(f"    [{arm}#{i}] {type(e).__name__}: {str(e)[:100]}", flush=True)
                return ""

    # warm the cache with one call, then fan out
    first = await one(0)
    rest = await asyncio.gather(*[one(i) for i in range(1, N)])
    codes = [first, *rest]

    from ttt_discover.verl_integration.verl_reward import compute_score
    async def sc(code, raw_text):
        if not code:
            return None
        return await asyncio.to_thread(
            compute_score, data_source=f"{T['cls']}_{T['problem_type']}",
            solution_str=raw_text, ground_truth=None, extra_info=extra)

    results = await asyncio.gather(*[
        sc(c, f"```python\n{c}\n```") for c in codes])

    scores, cons, ok = [], [], 0
    for c, r in zip(codes, results):
        if not r:
            continue
        v = r.get("raw_score")
        if r.get("score", 0) > 0 and v is not None:
            ok += 1
            scores.append(float(v))
            k = r.get("result_construction")
            cons.append(hashlib.md5(
                json.dumps(k, sort_keys=True, default=str).encode()).hexdigest()[:10]
                if k is not None else "none")

    code_h = [hashlib.md5(re.sub(r"\s+", " ", c).encode()).hexdigest()[:10]
              for c in codes if c]
    return dict(arm=arm, n=N, with_code=len(code_h), scored=ok,
                uniq_code=len(set(code_h)), uniq_construction=len(set(cons)),
                uniq_score=len(set(round(s, 9) for s in scores)),
                scores=scores)


async def main():
    from ttt_discover.verl_integration.puct_data_source import PUCTDataSource
    from anthropic import AsyncAnthropicVertex
    import ray
    if not ray.is_initialized():
        ray.init(_temp_dir="/tmp/ray_divexp", namespace="divexp",
                 include_dashboard=False,
                 num_cpus=int(os.getenv("RAY_NUM_CPUS", "8")),
                 ignore_reinit_error=True, logging_level="ERROR")

    out = Path("checkpoints/claude-puct/diversity_exp"); out.mkdir(parents=True, exist_ok=True)
    ds = PUCTDataSource(
        model_name=os.path.expanduser("~/models/Qwen3-8B"),
        env_module=T["module"], env_class=T["cls"], problem_type=T["problem_type"],
        groups_per_batch=1, group_size=N, log_dir=str(out / "log"),
        puct_file_path=str(out / "puct.json"),
        eval_timeout=T["eval_timeout"], num_cpus_per_task=T["cpus"])

    state = ds.sampler.sample_states(1)[0]
    prompt = ds._build_prompt(state)
    system = SYSTEM_PROMPTS.get(TASK, DEFAULT_SYSTEM_PROMPT)
    extra = dict(state=state, env_module=T["module"], env_class=T["cls"],
                 problem_type=T["problem_type"], eval_timeout=T["eval_timeout"],
                 num_cpus_per_task=T["cpus"], log_dir=str(out / "log"))

    print(f"=== diversity experiment: {TASK} / {MODEL} / effort={EFFORT} ===", flush=True)
    print(f"prompt {len(prompt)} chars, {N} rollouts x {len(ARMS)} arms, target={T['target']}", flush=True)

    client = AsyncAnthropicVertex(
        project_id=os.getenv("ANTHROPIC_VERTEX_PROJECT_ID", "lightwell-devel"),
        region=os.getenv("CLOUD_ML_REGION", "global"),
        max_retries=5, timeout=1800.0)

    usage = Usage(); rows = []
    for arm, fn in ARMS.items():
        t0 = time.time()
        r = await run_arm(client, arm, fn, system, prompt, extra, usage)
        r["secs"] = round(time.time() - t0)
        rows.append(r)
        print(f"  [{arm}] {r['secs']}s code={r['with_code']}/{N} scored={r['scored']} "
              f"uniq_code={r['uniq_code']} uniq_constr={r['uniq_construction']} "
              f"uniq_score={r['uniq_score']}  spent=${usage.cost_usd():.2f}", flush=True)

    print("\n=== best-of-k（搜索真正吃的指标）===", flush=True)
    ks = [1, 2, 4, 8, 16, N]
    print("  arm      " + "".join(f"  k={k:<9}" for k in ks) + "  max     mean    std", flush=True)
    for r in rows:
        s = r["scores"]
        if not s:
            print(f"  {r['arm']:<8}  (无有效得分)", flush=True); continue
        cells = "".join(f"  {best_of_k(s, k):<11.6f}" for k in ks)
        sd = statistics.stdev(s) if len(s) > 1 else 0.0
        print(f"  {r['arm']:<8}{cells}  {max(s):.6f}  {statistics.mean(s):.6f}  {sd:.2e}", flush=True)

    json.dump({"task": TASK, "model": MODEL, "effort": EFFORT, "n_per_arm": N,
               "target": T["target"], "rows": rows, "usage": usage.as_dict()},
              open(out / "result.json", "w"), indent=2, default=str)
    print(f"\n总花费 ${usage.cost_usd():.2f}  →  {out}/result.json", flush=True)

asyncio.run(main())
