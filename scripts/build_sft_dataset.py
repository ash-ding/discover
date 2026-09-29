"""Build an SFT dataset from archived VERL training-run rollouts.

Turns the per-rollout dumps of one or more training runs into a parquet file
that verl's MultiTurnSFTDataset can consume directly (a `messages` column of
[{role, content}, ...]), for the A1/A2 experiments in docs/sft-experiments.md.

Only VERL-side rollout records work here: they carry the full `input` prompt
and `output` text. PUCT inference-only records from before discover#17 do not
(no text, no prompt) -- pass those and the script will tell you.

The recorded `input` is the detokenized prompt with special tokens dropped:
    "user\n{question}\nassistant\n<think>\n"
The question is recovered by stripping those exact affixes, and the SFT
target is "<think>\n" + output, so the trained model learns to open its own
think block -- matching how the model behaves when served with the standard
Qwen3 chat template (no prefill) in scripts/vllm_puct_loop.py.

Selection: keep rollouts with raw_score > 0 (--filter valid, default) or
strictly better than their parent (--filter improved), then cap how many one
parent may contribute (--per-parent-cap): within a parent the kept rollouts
are spread across the quality range (best, worst, quantiles between), not
best-only, so the dataset does not collapse onto each parent's single best
answer. Prompt-side diversity is bounded by the number of distinct parents,
which is why the cap -- not the pool size -- is the knob that matters.

Usage:
    python scripts/build_sft_dataset.py --task erdos \
        --runs ~/data/discover-runs/erdos-gradclip1e9-repeat ... \
        --out data/sft/erdos_a1 \
        [--filter valid|improved] [--per-parent-cap 4] [--seed 0]
        [--tokenizer ~/models/Qwen3-8B] [--max-length 30720]

Writes <out>.parquet, <out>.stats.json and <out>.preview.jsonl. The stats
file records the best raw_score present in the kept set -- report it next to
any result from a model trained on this data (memorization baseline).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys

PREFIX = "user\n"
SUFFIX = "\nassistant\n<think>\n"

# minimize => state.value == -raw_score, and lower raw is better
TASK_MINIMIZE = {"erdos": True, "ac1": True, "ac2": False,
                 "circle_packing": False, "cp32": False}


def spread_indices(n, cap):
    """Up to `cap` indices spread evenly over range(n), always including 0
    and n-1 (best and worst of a sorted-by-quality list)."""
    if n <= cap:
        return list(range(n))
    return sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASK_MINIMIZE))
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories containing rollouts/*.jsonl")
    ap.add_argument("--out", required=True, help="output path stem")
    ap.add_argument("--filter", default="valid", choices=["valid", "improved"])
    ap.add_argument("--per-parent-cap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer path; enables exact length filtering and "
                         "prompt round-trip validation")
    ap.add_argument("--max-length", type=int, default=30720,
                    help="drop examples whose rendered conversation exceeds "
                         "this many tokens (needs --tokenizer)")
    ap.add_argument("--steps", default=None,
                    help="optional step range 'lo-hi' (inclusive), e.g. 1-5")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    minimize = TASK_MINIMIZE[args.task]

    step_lo, step_hi = None, None
    if args.steps:
        step_lo, step_hi = (int(x) for x in args.steps.split("-"))

    # ------------------------------------------------------------- collect
    by_parent = {}          # (run, parent_id) -> [record]
    skip = dict(not_selected=0, no_text=0, bad_affix=0, dup=0, no_parent=0)
    seen = set()
    for run_dir in args.runs:
        run = os.path.basename(os.path.normpath(run_dir))
        files = glob.glob(os.path.join(run_dir, "rollouts", "*.jsonl"))
        if not files:
            sys.exit(f"no rollouts/*.jsonl under {run_dir}")
        for f in files:
            step = int(os.path.basename(f).split(".")[0])
            if step_lo is not None and not (step_lo <= step <= step_hi):
                continue
            for line in open(f):
                d = json.loads(line)
                raw = d.get("raw_score")
                if not (raw and raw > 0):
                    skip["not_selected"] += 1
                    continue
                pv = d.get("puct_parent_value")
                if args.filter == "improved":
                    if pv is None or not (raw < -pv if minimize else raw > pv):
                        skip["not_selected"] += 1
                        continue
                inp, out = d.get("input") or "", d.get("output") or ""
                if not inp or not out:
                    skip["no_text"] += 1
                    continue
                if not (inp.startswith(PREFIX) and inp.endswith(SUFFIX)):
                    skip["bad_affix"] += 1
                    continue
                question = inp[len(PREFIX):-len(SUFFIX)]
                h = hashlib.md5((question + out).encode()).hexdigest()
                if h in seen:
                    skip["dup"] += 1
                    continue
                seen.add(h)
                pid = d.get("puct_parent_id")
                if not pid:
                    skip["no_parent"] += 1
                    continue
                by_parent.setdefault((run, pid), []).append(dict(
                    question=question, output=out, raw_score=float(raw),
                    parent_value=pv, run=run, step=step, uid=d.get("uid")))

    # ------------------------------------------------- cap + stratify
    kept = []
    for key, recs in by_parent.items():
        recs.sort(key=lambda r: r["raw_score"], reverse=not minimize)  # best first
        kept.extend(recs[i] for i in spread_indices(len(recs), args.per_parent_cap))
    rng.shuffle(kept)

    # ------------------------------------------------- tokenizer pass
    tok_stats, roundtrip = None, None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        lens, dropped, ok_rt, n_rt = [], [], 0, 0
        for r in kept:
            msgs = [{"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": "<think>\n" + r["output"]}]
            ids = tok.apply_chat_template(msgs, tokenize=True,
                                          add_generation_prompt=False)
            r["n_tokens"] = len(ids)
            (lens if len(ids) <= args.max_length else dropped).append(len(ids))
        kept = [r for r in kept if r["n_tokens"] <= args.max_length]
        # round-trip: does the standard chat template reproduce the recorded
        # prompt (minus special tokens, plus the '<think>\n' prefill)?
        for r in kept[:50]:
            n_rt += 1
            pids = tok.apply_chat_template(
                [{"role": "user", "content": r["question"]}],
                tokenize=True, add_generation_prompt=True)
            dec = tok.decode(pids, skip_special_tokens=True)
            if dec.rstrip("\n") + "\n<think>\n" == \
               PREFIX + r["question"] + SUFFIX or \
               dec + "<think>\n" == PREFIX + r["question"] + SUFFIX:
                ok_rt += 1
        lens.sort()
        tok_stats = dict(
            n=len(lens), dropped_over_max=len(dropped),
            p50=lens[len(lens) // 2], p90=lens[9 * len(lens) // 10],
            max=lens[-1]) if lens else dict(n=0, dropped_over_max=len(dropped))
        roundtrip = f"{ok_rt}/{n_rt}"

    if not kept:
        sys.exit("nothing selected -- check --runs/--filter/--steps")

    # ------------------------------------------------- write
    import pandas as pd
    rows = [dict(
        messages=[{"role": "user", "content": r["question"]},
                  {"role": "assistant", "content": "<think>\n" + r["output"]}],
        raw_score=r["raw_score"], run=r["run"], step=r["step"], uid=r["uid"],
    ) for r in kept]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out + ".parquet")

    best = min(r["raw_score"] for r in kept) if minimize else \
        max(r["raw_score"] for r in kept)
    per_run = {}
    for r in kept:
        per_run[r["run"]] = per_run.get(r["run"], 0) + 1
    stats = dict(
        task=args.task, filter=args.filter, per_parent_cap=args.per_parent_cap,
        steps=args.steps, seed=args.seed, runs=args.runs,
        pool_parents=len(by_parent), kept=len(kept), per_run=per_run,
        skipped=skip,
        best_raw_in_dataset=best,   # memorization baseline -- report with results
        token_stats=tok_stats, prompt_roundtrip=roundtrip,
    )
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(args.out + ".preview.jsonl", "w") as f:
        for r in rows[:3]:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
