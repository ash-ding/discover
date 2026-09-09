"""Closed-loop PUCT search driven by a self-hosted frontier model.

Same search procedure as run_verl.sh -- identical PUCT sampler, prompts, and
sandbox evaluator -- with rollouts drawn from a local vLLM fleet instead of
vLLM-inside-VERL, and no weights trained. Sibling of claude_puct_loop.py; the
only differences are the sampling backend and that sampling parameters are
ours to set again (a hosted API had removed non-default temperature).

    TASK=circle_packing bash scripts/serve_gpt_oss.sh start   # first
    TASK=circle_packing python scripts/vllm_puct_loop.py

Environment:
    TASK                 circle_packing | cp32 | erdos | ac1 | ac2 | denoising
    ENDPOINTS            comma-separated base URLs, or PORTS below
    PORTS                comma-separated ports on localhost  [8100..8107]
    MODEL_NAME           served model name       [gpt-oss-120b]
    TOTAL_STEPS          [50]
    GROUPS_PER_BATCH     parents per step        [8]
    ROLLOUTS_PER_GROUP   completions per parent  [64]
    CONCURRENCY          in-flight requests      [128]
    TEMPERATURE          [1.0]   TOP_P [1.0]
    REASONING_EFFORT     low|medium|high         [high]
    MAX_TOKENS           [32000]
    EXPERIMENT_NAME      [gptoss-{task}-{timestamp}]
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- task table
# Mirrors the case block in run_verl.sh. `minimize` is verified against each
# env: ac_inequalities.env.is_maximize() returns False for ac1 / True for ac2;
# erdos calls to_prompt(..., maximize=False); circle_packing maximizes the sum
# of radii. `target` is the paper reference value, read from each env module.
TASKS = {
    "circle_packing": dict(module="examples.circle_packing.env", cls="CirclePackingEnv",
                           problem_type="26", eval_timeout=530, cpus=1,
                           minimize=False, target=2.636, metric="sum of radii"),
    "cp32":           dict(module="examples.circle_packing.env", cls="CirclePackingEnv",
                           problem_type="32", eval_timeout=530, cpus=1,
                           minimize=False, target=2.940, metric="sum of radii"),
    "erdos":          dict(module="examples.erdos_min_overlap.env", cls="ErdosMinOverlapEnv",
                           problem_type="", eval_timeout=1100, cpus=1,
                           minimize=True, target=0.3808, metric="C5 bound"),
    "ac1":            dict(module="examples.ac_inequalities.env", cls="AutoCorrInequalityEnv",
                           problem_type="ac1", eval_timeout=1100, cpus=2,
                           minimize=True, target=1.5030, metric="upper bound"),
    "ac2":            dict(module="examples.ac_inequalities.env", cls="AutoCorrInequalityEnv",
                           problem_type="ac2", eval_timeout=1100, cpus=2,
                           minimize=False, target=0.97, metric="lower bound"),
    "denoising":      dict(module="examples.denoising.env", cls="DenoisingEnv",
                           problem_type="", eval_timeout=530, cpus=1,
                           minimize=True, target=None, metric="loss"),
}

# System prompts carried over verbatim from scripts/replay_claude.py so the two
# Claude experiments stay comparable.
SYSTEM_PROMPTS = {
    "erdos": (
        "You are an expert in harmonic analysis and numerical optimization, "
        "solving the Erdos minimum overlap problem. Find a step function "
        "h: [0,2] -> [0,1] with integral h=1 that MINIMIZES C5 = max_k "
        "integral h(x)(1-h(x+k))dx. Your code MUST define "
        "run(seed, budget_s, **kwargs) returning (h_values, c5_bound, n_points). "
        "Use numpy and scipy. Lower C5 is better; current record is <= 0.3809."
    ),
}
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert problem solver. Provide a clear, concise solution to the "
    "problem. Show your reasoning and give a final answer."
)

# Unused here: a self-hosted model gives us temperature back, which is the
# mechanism DISCOVER used for rollout diversity in the first place. Kept only
# so both loops expose the same DIVERSITY switch.
STRATEGY_HINTS = [""]

CODE_RE = re.compile(r"```python\n(.*?)(?:\n```)", re.DOTALL)


def extract_code(text: str) -> str:
    m = list(CODE_RE.finditer(text))
    return m[-1].group(1).rstrip() if m else ""


class Usage:
    """Throughput accounting. Self-hosted, so tokens are GPU time, not dollars."""
    __slots__ = ("inp", "out", "calls", "t0")

    def __init__(self):
        self.inp = self.out = self.calls = 0
        self.t0 = time.time()

    def add(self, prompt_tokens, completion_tokens):
        self.inp += prompt_tokens or 0
        self.out += completion_tokens or 0
        self.calls += 1

    def cost_usd(self):
        return 0.0          # self-hosted

    def tok_per_s(self):
        el = max(time.time() - self.t0, 1e-6)
        return self.out / el

    def as_dict(self):
        return dict(calls=self.calls, input_tokens=self.inp, output_tokens=self.out,
                    elapsed_s=round(time.time() - self.t0, 1),
                    output_tok_per_s=round(self.tok_per_s(), 1))


async def generate_one(clients, model, system, prompt, hint, sem, cfg, usage, idx):
    """One completion, round-robined across the vLLM replicas.

    Replicas are independent servers, so spreading by rollout index keeps every
    GPU busy without a proxy in front. Single call, no continuation phase --
    same reasoning as the Claude loop: truncation is rare enough that a retry
    path would be untested code, so truncations are counted and dropped.

    gpt-oss emits harmony channels; vLLM puts the answer in `content` and the
    chain of thought in `reasoning_content`. Only `content` is parsed for code,
    but reasoning tokens are still billed against max_tokens, which is why the
    cap is generous rather than tuned.
    """
    client = clients[idx % len(clients)]
    async with sem:
        t0 = time.time()
        user = prompt if not hint else f"{prompt}\n\n{hint}"
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                extra_body={"reasoning_effort": cfg["effort"]},
            )
            ch = r.choices[0]
            text = ch.message.content or ""
            u = getattr(r, "usage", None)
            usage.add(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
            return dict(ok=True, text=text, code=extract_code(text),
                        truncated=ch.finish_reason == "length",
                        out_tokens=getattr(u, "completion_tokens", 0),
                        stop=ch.finish_reason, secs=round(time.time() - t0, 1))
        except Exception as e:
            return dict(ok=False, text="", code="", truncated=False, out_tokens=0,
                        stop="error", secs=round(time.time() - t0, 1),
                        error=f"{type(e).__name__}: {str(e)[:200]}", idx=idx)


async def rollouts_for_parent(clients, model, system, prompt, sem, cfg, usage):
    """Draw ROLLOUTS_PER_GROUP completions for one parent state, all at once.

    An earlier version awaited the first rollout alone so it could populate the
    prompt cache for the rest (concurrent requests cannot read an entry still
    being written). Measured, that trade is bad: the cached prefix is ~1k
    tokens, worth ~$0.07 per parent, while the extra serial call added ~90s --
    about 10 hours across a 50-step run. cache_control is left on the blocks
    since it costs nothing and can still hit when PUCT re-samples a parent.
    """
    n = cfg["rollouts"]
    hints = (STRATEGY_HINTS if cfg["diversity"] == "rotate" else [""])
    return list(await asyncio.gather(*[
        generate_one(clients, model, system, prompt,
                     hints[i % len(hints)], sem, cfg, usage, i)
        for i in range(n)]))


# ------------------------------------------------------------------ the loop
async def main():
    task = os.getenv("TASK")
    if task not in TASKS:
        sys.exit(f"TASK must be one of {sorted(TASKS)}; got {task!r}")
    T = TASKS[task]

    cfg = dict(
        model=os.getenv("MODEL_NAME", "gpt-oss-120b"),
        steps=int(os.getenv("TOTAL_STEPS", "50")),
        groups=int(os.getenv("GROUPS_PER_BATCH", "8")),
        rollouts=int(os.getenv("ROLLOUTS_PER_GROUP", "64")),
        effort=os.getenv("REASONING_EFFORT", "high"),
        temperature=float(os.getenv("TEMPERATURE", "1.0")),
        top_p=float(os.getenv("TOP_P", "1.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "32000")),
        diversity=os.getenv("DIVERSITY", "none"),
        cache=os.getenv("PROMPT_CACHE", "1") == "1",
    )
    concurrency = int(os.getenv("CONCURRENCY", "256"))
    exp = os.getenv("EXPERIMENT_NAME") or f"gptoss-{task}-{time.strftime('%Y%m%d_%H%M')}"

    out_dir = Path("checkpoints/gptoss-puct") / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(exist_ok=True)

    # Resume: the PUCT snapshot on disk is the source of truth for progress.
    done = sorted(int(p.stem.split("_")[-1])
                  for p in out_dir.glob("puct_sampler_step_*.json"))
    resume = done[-1] if done else 0

    # The sandbox evaluator submits ray tasks and will otherwise auto-start a
    # cluster that collides with a VERL training job already running on this
    # host. Initialize first with an isolated temp dir / namespace and a capped
    # core count, matching the isolation pattern in launch_qwen0.6b_4tasks.sh.
    import ray
    if not ray.is_initialized():
        ray.init(
            _temp_dir=os.getenv("RAY_TEMP_DIR", f"/tmp/ray_gptoss_{exp}"),
            namespace=f"gptoss_puct_{exp}",
            include_dashboard=False,
            num_cpus=int(os.getenv("RAY_NUM_CPUS", "16")),
            ignore_reinit_error=True,
            logging_level="ERROR",
        )

    # Build the sampler and environment directly rather than through
    # PUCTDataSource. That module imports torch + transformers and loads the
    # policy tokenizer, none of which this loop needs -- we never sample from a
    # local model. Verified on circle_packing that get_question() with
    # renderer=None is byte-identical (md5 d4898759f5bd) to the tokenized path,
    # so prompts still match the RL run exactly.
    import importlib
    from dataclasses import dataclass
    from typing import Any
    sys.path.insert(0, os.getcwd())
    from ttt_discover.tinker_utils.sampler import PUCTSampler
    from ttt_discover.tinker_utils.state import State
    from ttt_discover.verl_integration.verl_reward import compute_score
    from openai import AsyncOpenAI

    env_cls = getattr(importlib.import_module(T["module"]), T["cls"])

    @dataclass
    class _Cfg:   # mirrors puct_data_source._DatasetConfig
        problem_type: str
        env_type: type
        batch_size: int
        group_size: int
        num_cpus_per_task: int = 1
        eval_timeout: int = 530
        log_path: str = "./tinker_log"
        timeout: float = 8000.0
        convo_prefix: Any = None

    ds_cfg = _Cfg(T["problem_type"], env_cls, cfg["groups"], cfg["rollouts"],
                  T["cpus"], T["eval_timeout"], str(out_dir / "log"), 8000.0)

    sampler = PUCTSampler(
        file_path=str(out_dir / "puct_sampler.json"),
        env_type=env_cls, problem_type=T["problem_type"],
        max_buffer_size=int(os.getenv("MAX_BUFFER_SIZE", "1000")),
        batch_size=cfg["groups"], resume_step=resume or None,
        puct_c=float(os.getenv("PUCT_C", "1.0")),
        topk_children=int(os.getenv("TOPK_CHILDREN", "2")),
    )

    def build_prompt(state):
        env = env_cls(renderer=None, initial_state=state,
                      sampler=sampler, config=ds_cfg)
        return env.get_question()
    system = SYSTEM_PROMPTS.get(task, DEFAULT_SYSTEM_PROMPT)
    eps = os.getenv("ENDPOINTS")
    if eps:
        endpoints = [e.strip().rstrip("/") for e in eps.split(",") if e.strip()]
    else:
        ports = os.getenv("PORTS", ",".join(str(8100 + i) for i in range(8)))
        endpoints = [f"http://127.0.0.1:{p.strip()}" for p in ports.split(",") if p.strip()]
    clients = [AsyncOpenAI(base_url=f"{e}/v1", api_key="EMPTY",
                           max_retries=int(os.getenv("MAX_RETRIES", "5")),
                           timeout=float(os.getenv("REQUEST_TIMEOUT", "1800")))
               for e in endpoints]
    sem = asyncio.Semaphore(concurrency)
    usage = Usage()

    json.dump({"task": task, "experiment": exp, "resumed_from": resume,
               "config": cfg, "concurrency": concurrency,
               "endpoints": endpoints,
               "target": T["target"], "minimize": T["minimize"],
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(out_dir / "config_snapshot.json", "w"), indent=2)

    print(f"=== vLLM-PUCT closed loop: {task} ===", flush=True)
    print(f"model={cfg['model']} reasoning_effort={cfg['effort']} "
          f"temp={cfg['temperature']} top_p={cfg['top_p']} max_tokens={cfg['max_tokens']}", flush=True)
    print(f"endpoints ({len(endpoints)}): {', '.join(endpoints)}", flush=True)
    print(f"{cfg['groups']} parents x {cfg['rollouts']} rollouts = "
          f"{cfg['groups'] * cfg['rollouts']} calls/step, {cfg['steps']} steps", flush=True)
    print(f"diversity={cfg['diversity']} cache={cfg['cache']} concurrency={concurrency}", flush=True)
    print(f"target={T['target']} ({'minimize' if T['minimize'] else 'maximize'} {T['metric']})", flush=True)
    print(f"output={out_dir}" + (f"  [resuming from step {resume}]" if resume else ""), flush=True)

    best_ever = None
    for step in range(resume + 1, cfg["steps"] + 1):
        t_step = time.time()
        states = sampler.sample_states(cfg["groups"])
        new_states, parents = [], []
        n_ok = n_code = n_scored = n_trunc = 0
        errors, raws, all_results = [], [], []

        extra_base = dict(env_module=T["module"], env_class=T["cls"],
                          problem_type=T["problem_type"], eval_timeout=T["eval_timeout"],
                          num_cpus_per_task=T["cpus"], log_dir=str(out_dir / "log"))

        async def score_one(r, state):
            if not r["ok"]:
                errors.append(r.get("error", "?"))
                return None
            if not r["code"]:
                return None
            return await asyncio.to_thread(
                compute_score, data_source=f"{T['cls']}_{T['problem_type']}",
                solution_str=r["text"], ground_truth=None,
                extra_info=dict(extra_base, state=state))

        async def run_parent(pi, state):
            """Generate and evaluate one parent's rollouts.

            Parents run concurrently, bounded by the shared semaphore. The
            Claude loop processes them serially because 64 rollouts already
            exceed any sane API concurrency; a local fleet has far more
            capacity than one parent can fill, so serial parents would leave
            most of it idle.
            """
            prompt = build_prompt(state)
            results = await rollouts_for_parent(
                clients, cfg["model"], system, prompt, sem, cfg, usage)
            scored = await asyncio.gather(*[score_one(r, state) for r in results])
            ok = sum(1 for r in results if r["ok"])
            tr = sum(1 for r in results if r.get("truncated"))
            med = sorted(r.get("out_tokens", 0) for r in results)[len(results) // 2]
            print(f"    step {step} parent {pi + 1}/{len(states)}: "
                  f"ok={ok}/{len(results)} trunc={tr} med_tokens={med} "
                  f"({round(time.time() - t_step)}s elapsed, {usage.tok_per_s():.0f} tok/s)",
                  flush=True)
            return state, results, scored

        per_parent = await asyncio.gather(*[run_parent(pi, st)
                                            for pi, st in enumerate(states)])

        for state, results, scored in per_parent:
            all_results.extend(dict(r, parent=id(state) % 1000) for r in results)
            for r, sc in zip(results, scored):
                n_ok += r["ok"]
                n_code += bool(r["code"])
                n_trunc += r.get("truncated", False)
                if not sc:
                    continue
                score = float(sc.get("score", 0.0) or 0.0)
                raw = sc.get("raw_score")
                if score > 0 and r["code"]:
                    n_scored += 1
                    if raw is not None:
                        raws.append(float(raw))
                    value = float(-raw if (T["minimize"] and raw is not None)
                                  else (raw if raw is not None else score))
                    new_states.append(State(timestep=state.timestep + 1,
                                            construction=sc.get("result_construction"),
                                            code=r["code"], value=value))
                    parents.append(state)
            sampler.record_expansion(state)

        if new_states:
            sampler.update_states(new_states, parents, save=True, step=step)
        sampler.flush(step=step)

        # Paper-comparable metric, recovered from the archive. The RL run logs
        # only critic/score (a transformed reward), which is not comparable to
        # the published target for minimization tasks -- so log raw_score here.
        vals = [s.value for s in sampler._states if isinstance(s.value, (int, float))]
        best = (-max(vals) if T["minimize"] else max(vals)) if vals else None
        if best is not None:
            best_ever = best if best_ever is None else (
                min(best_ever, best) if T["minimize"] else max(best_ever, best))

        rec = dict(step=step, elapsed_s=round(time.time() - t_step, 1),
                   rollouts=cfg["groups"] * cfg["rollouts"],
                   api_ok=n_ok, with_code=n_code, scored_positive=n_scored,
                   api_errors=len(errors), truncated=n_trunc,
                   raw_score_best_this_step=(min(raws) if T["minimize"] else max(raws)) if raws else None,
                   raw_score_mean_this_step=(sum(raws) / len(raws)) if raws else None,
                   archive_best=best, best_ever=best_ever,
                   archive_size=len(sampler._states), target=T["target"],
                   usage=usage.as_dict())
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(rollout_dir / f"{step}.jsonl", "w") as f:
            for r in all_results:
                f.write(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n")

        pct = ""
        if T["target"] and best:
            pct = (f"  ({T['target']/best*100:.2f}% of target)" if T["minimize"]
                   else f"  ({best/T['target']*100:.2f}% of target)")
        print(f"[step {step}/{cfg['steps']}] {round(time.time()-t_step)}s  "
              f"ok={n_ok}/{cfg['groups']*cfg['rollouts']} code={n_code} scored={n_scored} "
              f"trunc={n_trunc} err={len(errors)}  best={best}{pct}  archive={len(sampler._states)}  "
              f"{usage.tok_per_s():.0f} tok/s", flush=True)
        if errors:
            print(f"    first error: {errors[0]}", flush=True)

    print(f"=== done. best_ever={best_ever} target={T['target']} "
          f"out_tokens={usage.out} avg={usage.tok_per_s():.0f} tok/s ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
