# Handoff — TTT-Discover reproduction

Written 2026-09-28. Everything below was verified on the machines that day, not recalled.

> **Read this before `CLAUDE.md` and `EXPERIMENT_PLAN.md`.** Both are stale and contradict
> reality in ways that will cost you time — see [Stale docs](#stale-docs-do-not-trust-these).

---

## 1. Where things stand right now

**Nothing is running.** No training, no inference loop, no vLLM, no cron watchdogs, no
background watchers. All experiment records are archived to shared storage and local copies
deleted. The repo is on `main` at `321637d`, in sync with `origin`.

Four hosts, all reachable: `lumen1`, `lumen2`, `lumen3`, `node10`.

---

## 2. Infrastructure

### Hosts

| Host | GPUs | Free? | `/` free | Notes |
|---|---|---|---|---|
| lumen1 | 8×H100 80G | all 8 | 119G | `verl_discover` lives in **miniconda3**, not miniforge3 |
| lumen2 | 8×H100 80G | all 8 | 91G | |
| lumen3 | 8×H100 80G | all 8 | **26G** | tight on disk; see [Storage](#storage) |
| node10 | 8×H100 80G | **only 5,6,7** | 195G | GPUs 0–4 carry user `lab`'s `gcg`/`gedd-validation` jobs |

**node10 is shared with another user.** Do not plan an 8-GPU run there, and never kill GPU
processes by pattern — identify PIDs and confirm the owner first.

### SSH

Long-lived idle connections to the lumens fail with exit 255 under load. Always:

```bash
ssh -o ServerAliveInterval=0 -o ConnectTimeout=30 lumen1 'bash -s' <<'EOF'
...
EOF
```

Pass scripts on **stdin** (`bash -s`), not `bash -c`: a `pgrep -f <pattern>` inside `bash -c`
matches the ssh command line itself, so wait-loops never exit and `pkill -f` kills your own
session.

For anything that takes minutes, `nohup` it on the remote and poll a marker file. An SSH
transport failure does **not** stop the remote shell — absence of output never means the work
didn't happen. Check before re-running.

### Storage

Three distinct places:

| Path | What it is | Use for |
|---|---|---|
| `/` | container root, ZFS | code, conda envs |
| `/scratch` | separate dataset, 250G quota | model weights, caches (`~/models`, `~/.cache`, `~/forge_images`, `~/download` are symlinks into it) |
| `~/data` | rclone mount of `ai:ai-innovation-bucket/users/asherding` | **experiment archive** — the only genuinely off-pool storage |

**Unresolved:** `/`'s *available* figure does not respond to anything. Moving 104G off `/` and
deleting 57G on `/scratch` both left it unchanged, yet writing 5G to `/` consumes 5G and
deleting it returns 5G immediately. Accounting lag, pool sharing, and snapshots were each
hypothesized and each falsified. Plan against the number as-is; do not expect cleanup to raise
it. To settle it, ask whoever administers the Incus host for
`zfs get -r used,available,quota,refquota,reservation,refreservation,usedbysnapshots envpool`.

The platform console's "Storage N / 250 GB" tracks **`$HOME`**, which is a different quota from
`/`. Relocating to `/scratch` cut lumen3's `$HOME` from 226G to 118G — real headroom on that
quota, none on `/`.

**rclone caches directory listings per host.** A file written from lumen1 can be invisible on
lumen2 for ~5 minutes. Verify an archive from a host that did *not* write it.

`cp -a` onto `~/data` returns **rc=1** (it cannot preserve permissions) even when every byte
copies. Judge success by file count + bytes + md5, never by exit code.

### Environment

- On the **lumens** the env is `verl_discover`. On **lumen1 it lives under `~/miniconda3`**; on
  lumen2/lumen3 under `~/miniforge3`. Launch scripts resolve it with
  `ls miniforge3/... miniconda3/... | head -1` — `ls` sorts alphabetically, so `miniconda3`
  wins on lumen1, which happens to be correct there.
- On **node10 the env is `~/.conda/envs/verl_discover`** — same name as the lumens, but not
  under miniforge3 or miniconda3 (globbing those paths finds nothing and will mislead you).
  It was called `lumen` until 2026-09-28; scripts and logs older than that still say so.
  It carries vllm 0.23.0, torch 2.11.0+cu129, ray 2.56.0, openai 2.44.0.
- **transformers is pinned to 4.57.6 on node10**, downgraded in place from 5.12.1 on
  2026-09-28. This is not cosmetic: transformers 5.x routes gpt-oss's flash attention
  through the HF hub `kernels` package, which tries to fetch
  `kernels-community/vllm-flash-attn3` — that repo publishes `torch211-cu128` and
  `torch211-cu130` but **not** the `torch211-cu129` this env needs, so flash attention is
  unreachable there and the only way forward would be `eager`. 4.57.6 uses the local
  `flash_attn` directly. Do not upgrade transformers on node10 without re-checking this.
- `~/install/cuda129` is the **live CUDA 12.9 toolkit** and is what `which nvcc` resolves to.
  `.bashrc` references it. The system only has CUDA 13.2. **Do not delete it.**

### node10's repo — read before touching it

node10 had **two clones** of this repo until 2026-09-28: `~/code/discover` and
`~/code/discover-claude`. They are now **consolidated into `~/code/discover`**. Old scripts,
logs and backups still mention `discover-claude`; that path no longer exists — do not go
looking for it, and do not recreate it.

The consolidation direction was the opposite of what the evidence first suggested, which is
worth knowing if you ever redo something like it:

- every script pointed at `discover-claude`, so it looked like the live one;
- but `~/.conda/envs/lumen` has verl as an **editable install pinned to
  `~/code/discover/verl`**. Deleting that path breaks `import verl` for the whole env.
- `discover-claude` was additionally a **shallow clone with an empty `verl/`**, so it could
  not have run VERL training at all.

So `discover` was kept, the two scripts (`~/launch_gptoss_repeat.sh`, `~/node_setup.sh`) were
repointed to it, and `discover-claude` was deleted. Originals of both scripts are in
`~/.merge_backup_20260928_202100`.

**Two stashes live in that repo** and are intentionally preserved — `git stash list`:

- `stash@{0}` (Jul 2026): one line in `run_verl.sh` setting `DISCOVER_CLAUDE_SYSTEM_PROMPT`
  for erdos.
- `stash@{1}` (Jun 2026): 27 files, +392/−194 — a YAML config refactor plus a `run_verl.sh`
  rewrite.

Neither has been evaluated against current `main`. A `git gc --prune=now` together with
`git reflog expire --expire=now` **would destroy both**; the cleanup that reduced `.git` from
454M to 56M deliberately used the conservative form instead.

---

## 3. Two code paths — don't confuse them

| | RL training | Inference-only search |
|---|---|---|
| entry | `run_verl.sh` → `main_ppo` | `scripts/vllm_puct_loop.py` |
| writes to | `checkpoints/ttt-discover/` | `checkpoints/gptoss-puct/` |
| serves vLLM | colocated, managed by VERL | **you start it yourself** (`scripts/serve_gpt_oss.sh`) |
| records | `<name>.jsonl`, metrics nested under `data`, logs **reward** not raw score | `metrics.jsonl` (per step) + `rollouts/<step>.jsonl` (per rollout) |

For minimization tasks the VERL logs store `reward = 1/(1e-8 + raw)`; recover the objective as
`raw = 1/reward`. `critic/score` is float32, so an inverted value is only good to ~7 digits —
prefer the recorded raw values where you have them.

### Patches on `main` you must not regress

| commit | what |
|---|---|
| `1bc3a78` | `SERVED_NAME` in `serve_gpt_oss.sh` — the client must request the exact advertised name |
| `8713247` | persist per-rollout **scores** in the step dump |
| `321637d` | persist the full generation **text** — line 419 used to strip it |
| `e22db2d` | `scripts/preflight_puct.sh` — run it before every PUCT launch |

`bash scripts/preflight_puct.sh <EXPERIMENT_NAME> <RAY_TEMP_DIR>` checks that the code about to
run has the score and text persistence, that `SERVED_NAME` exists, that you are not behind
`origin/main`, and that the experiment name and Ray temp dir are free.

**Runs started before `321637d` have no `text` field** — scores and code only. That includes
`qwen3-8b-ac1-inference-only` and every gpt-oss run.

A repeated failure worth knowing: committing to a branch and then `git checkout main` silently
reverted the working-tree patch **twice**. After switching branches, verify the patch is still
present before launching.

---

## 4. Results

All records: `~/data/discover-runs` (19 entries). Values below are best-over-history.

### Erdős minimum overlap — minimize, target 0.3808

| Setting | Runs |
|---|---|
| Qwen3-8B trained (VERL) | 0.3809193429, 0.3809267260, 0.3809884112 |
| Qwen3-8B inference-only | 0.3809621418, 0.3809867812, 0.3809915481 |
| gpt-oss-120b inference-only | **0.3808638785**, 0.3808693927 |

A fourth trained run (`erdos-gradclip1-50step`, `grad_clip=1.0`) is excluded by the user's
instruction; the three shown used `grad_clip=1e9`.

### AC1 — **minimize**, target 1.5030

| Setting | Runs |
|---|---|
| Qwen3-8B trained (VERL) | **1.5045237**, 1.5047907, 1.5051116 |
| Qwen3-8B inference-only | 1.5056703, 1.5061002, 1.5073663 |

### Circle packing 26 — maximize, target 2.636

trained 2.6359829903 · inference-only 2.6359830849

### Scoring vs. published numbers

Our Erdős scorer returns the **verifier-recomputed** `verified_c5`; upstream's
`evaluate_erdos_solution()` returns the model-**reported** `c5_bound`. Ours is strictly
stricter, so **our numbers are not comparable to the paper's published ones**. They *are*
comparable to Einstein Arena: EA submissions carry only a `values` array with no reported score
and no tolerance, and running EA's verifier unmodified on three of our constructions
(n=58/400/960) reproduced our values to 0–5.55e-17.

---

## 5. What the results say

Ordered by how much headroom the task has left against target, training's advantage falls in
exactly that order:

| Task | gap to target | training vs inference-only |
|---|---|---|
| AC1 | 1.5e-3 | **decisive** — all 9 pairwise comparisons favour training, ranges disjoint |
| Erdős | 1.1e-4 | marginal — 7 of 9, ranges overlap |
| Circle packing | ~2e-8 (saturated) | none — inference-only marginally ahead |

**Working hypothesis:** RL pays off when the base model is not yet an effective mutation
operator — a threshold to cross, not a ceiling to raise. Once a task saturates, training adds
nothing.

Two things that are *not* conclusions:

- This is **not an equal-compute comparison**. VERL's best is over training rollouts; PUCT's is
  over search. Only "what each reaches in 50 steps" is being compared.
- n=3 per cell. "Disjoint ranges" is descriptive, not a test.

Separately: on Erdős, **gpt-oss-120b inference-only beats every trained Qwen3-8B run.** Model
size is confounded with method there, but it suggests that on some tasks a bigger base model
buys more than test-time training on a smaller one.

Published ledger (updated, all runs final): https://claude.ai/artifact/3jHnuxcS9miLuRzQ58Uj7r

---

## 6. Next experiment (planned, not started)

**Trained checkpoint vs base model, same PUCT inference loop, on AC1.**

AC1 is the task where training clearly wins, so it is the only one where the win can be
attributed. The open question: did the model actually get better, or did training just do more
search and stumble on a better construction? Run the identical inference-only loop with the
trained checkpoint and with the base model. If the trained checkpoint wins at equal search
budget, the model improved; if they tie, AC1's advantage was a by-product of extra search.

The VERL archives under `~/data/discover-runs/*_verl_*` and `ac1-repeat*` carry `latest/` and
`latest_checkpointed_iteration.txt`. **Nobody has verified these checkpoints still load** —
that is step one.

Two further ideas, in rough priority order: more repeats per cell (every setting except Erdős
and AC1 is still a single point), and distilling a frontier model's rollout traces into the
small model via SFT instead of running RL.

---

## 7. Stale docs — do not trust these

- **`CLAUDE.md`** describes the cluster as `ai-innovation-h100-10/11-preserve` at
  `10.241.128.30/16`. That is not the current fleet. Setup instructions and architecture notes
  are still broadly right; the node table is not.
- **`EXPERIMENT_PLAN.md`** is a Qwen3-4B plan with every row marked `Pending`. We ran Qwen3-8B
  and finished. It also calls AC1 a *maximization* task — **AC1 is minimized** (target 1.5030,
  lower is better). Do not take the direction from that file.
- `PERF_INVESTIGATION.md` and `gpu4_thermal_ticket.md` are historical incident write-ups, not
  current state.

---

## 8. Operating notes that cost time to learn

- `pgrep -fc <pat>` prints **two lines** when nothing matches if you wrote `|| echo 0` — both
  fire, and a safety gate reading it silently passes. Use `pgrep -f ... | wc -l`.
- `gh api --ref main` is not a flag. Use `?ref=main` as a query parameter, or you download an
  empty file and md5 it to `d41d8cd9...`.
- `timeout` does not exist on macOS by default; `bc` is not installed on the hosts. Use `awk`.
- ZFS compresses aggressively: `du -sh` (on-disk) can be ~2.6× smaller than `du -sb`
  (apparent). A disk-space test written from `/dev/zero` compresses to nothing and proves
  nothing — use `/dev/urandom`.
- On the lumens, a `nvidia-smi` compute PID with no `/proc` entry is usually a PID-namespace
  mismatch, not a zombie. It clears when the real process exits.
- A vLLM replica answering 200 on `/health` does **not** prove it works — pair it with a GPU
  memory check.
- Full PUCT output is ~0.9 GB per 50-step run with `text` persisted, not the 25–50 GB an
  earlier estimate claimed.

---

## 9. gpt-oss training support — what works, what to watch

Branch `feat/gpt-oss-training` (this repo) plus `feat/mxfp4-dequantize-passthrough` (the verl
submodule) make the **VERL training path** accept `openai/gpt-oss-*`. Debugged on
gpt-oss-20b, node10 GPUs 4-7. Three patches, each for a distinct wall:

1. **MXFP4 cannot be trained.** The checkpoint stores every MoE expert as a `blocks`/`scales`
   pair, and `Mxfp4HfQuantizer.is_trainable` raises outright. verl's FSDP engine now detects
   `quantization_config.quant_method == "mxfp4"` and passes
   `Mxfp4Config(dequantize=True)` to `from_pretrained`, which converts the experts to bf16
   **at load time, in memory** — no pre-materialized bf16 checkpoint on disk is needed.
   Disk stays ~20-27GB for the 20b; memory holds ~42GB.
2. **vLLM must be told to stop expecting MXFP4.** The actor now pushes dequantized bf16
   weights, but vLLM still read `quant_method` from the raw HF config and picked
   `_load_weights_mxfp4`, which wants fused `w13_weight` names and dies with
   `KeyError: 'layers.0.mlp.experts.w13_weight'`. The rollout server overrides the config to
   `{"quant_method": ""}` — **an empty string, not `None` and not a missing key**. vLLM reads
   it twice with different accessors: `ModelConfig` does `.get("quant_method", "").lower()`
   (fatal on `None`) while `gpt_oss.load_weights` subscripts the key directly (fatal if
   absent). Only `""` satisfies both.
3. **The two-phase generation protocol had Qwen hardcoded into it.** See below — this was the
   subtle one.

### The generation protocol is the part that bites

`agent_loop.py` budgets generation in two phases: think under a cap, then force an answer.
That algorithm is model-agnostic, but two strings in it were Qwen3-specific — the marker
saying reasoning is over (`</think>`) and the prefill that ends it. Both now come from the
renderer family.

gpt-oss does not delimit reasoning with tags; it reasons in a harmony **`analysis` channel**
and answers in a separate **`final` channel**. Closing analysis is therefore not enough — the
final channel has to be opened explicitly, so the prefill is
`<|end|><|start|>assistant<|channel|>final<|message|>` and the "reasoning done" marker is
`<|channel|>final<|message|>`. Both tokenize to clean special-token sequences and round-trip
through decode/encode, which `_contains_pattern` relies on.

**Why this matters more than it sounds:** run gpt-oss-20b at `gpt_oss_high_reasoning` with
one plain generate call and it spends the *entire* budget inside `analysis`, never emitting a
code block. Measured: 7 of 8 rollouts scored 0.0 with `eval_error: "no code block extracted"`,
outputs all 110-130K chars — exactly the token cap. The single rollout that reached `final`
scored 2.37. So **an all-zero `critic/score` on gpt-oss is a generation-protocol symptom, not
a scoring or weight-sync failure.** Check `rollouts/N.jsonl` for the `code` field before
suspecting anything else: empty `code` means the model never got out of `analysis`.

### Budget against the response cap, not just the context

`agent_loop.py` used to size its budgets from the model context alone. On gpt-oss the context
(32768) and verl's `max_response_length` (28672) differ, and budgeting on the context
overruns the cap — so the forced final answer gets truncated away by the very limit it was
meant to fit inside. `_budget()` now takes the min of both. On Qwen3 the two caps coincide,
so this only ever lowers a budget that would have been truncated anyway.

### Knobs added to `run_verl.sh`

All default to the previous behaviour, so Qwen3 runs are unaffected:

| Variable | Purpose |
|---|---|
| `LORA_TARGET_MODULES` | default `all-linear` |
| `MODEL_DTYPE` | → `fsdp_config.model_dtype`; `bf16` avoids fp32's 84GB for the 20b |
| `PARAM_OFFLOAD` / `OPTIMIZER_OFFLOAD` | were hardcoded `True`, tuned for a dense 8B |
| `ATTN_IMPL` | → `override_config.attn_implementation` |
| `ROLLOUT_MAX_MODEL_LEN` | **must be set alongside `DISCOVER_MAX_MODEL_LEN`** |
| `DISCOVER_GENERATION_STRATEGY` | `two_phase` (default) or `single_phase` |
| `DISCOVER_PHASE1_MAX_TOKENS` | position cap on phase 1; leave room for the answer |

`DISCOVER_MAX_MODEL_LEN` only affects the agent loop. If `ROLLOUT_MAX_MODEL_LEN` is left
unset, vLLM falls back to `max_position_embeddings` (131072 for gpt-oss), the KV cache does
not fit, and the run dies with `EngineDeadError` — which looks nothing like a context-length
problem.

### Known limitation, and it is a real confound

`GptOssExperts` holds its weights as raw `nn.Parameter`, **not** `nn.Linear`. PEFT's
`all-linear` therefore reaches only `q_proj/k_proj/v_proj/o_proj` — **the experts are never
adapted.** Any gpt-oss-vs-Qwen3 comparison is comparing "LoRA on attention only" against
"LoRA on attention + MLP". Do not present such a comparison as like-for-like without saying
so, or without first teaching PEFT to target the expert parameters.

### Other things that cost time here

- `run_verl.sh:23` dereferenced `$CONDA_DEFAULT_ENV` unguarded, so every non-interactive
  launch died under `set -u`. Fixed, but older copies of the script still have it.
- Host **RAM**, not GPU memory, is the binding constraint when loading the 120b: verl's
  `get_init_weight_context_manager` has rank 0 materialize the *full* model on CPU while
  other ranks use meta tensors. On a 99GB host the 42GB bf16 20b peaked at 51GB anon and
  got killed. TP equal to the GPU count keeps it to one replica and one materialization.
- These hosts are LXC containers: `memory.current` counts page cache, so Ray's OOM monitor
  kills jobs while `free` still reports tens of GB available. `free` is not the number Ray
  is looking at.
- Killing a vLLM run with `pkill` on the launcher leaves **EngineCore workers holding ~18GB
  per card**. The next attempt then dies of CUDA OOM for no visible reason. Kill by PID from
  `nvidia-smi --query-compute-apps`, and filter to your own user — GPU 0 on node10 carries
  another user's job.
