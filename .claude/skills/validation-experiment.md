---
name: validation-experiment
description: |
  Run TTT-Discover experiments with the proper smoke-test-first workflow. Use this skill whenever
  the user wants to run an experiment, validate code changes, do a smoke test, test a task, or
  start training. The key insight is that a 5-minute smoke test catches the same bugs as a 90-minute
  validation run, so always start small after code changes. Also use when the user asks "is the code
  working?", "test this task", "try a quick run", or mentions running any config YAML file.
---

# Experiment Runner

## Workflow

After code changes, escalate gradually — a 5-minute smoke test catches the same code bugs as a 60-minute validation run, saving significant GPU time.

| Config | samples/step | epochs | Runtime | When to use |
|--------|-------------|--------|---------|-------------|
| `config_smoke_test.yaml` | 16 | 1 | 5-10 min | After code changes, first-time setup |
| `config_validate.yaml` | 512 | 1 | 30-90 min | Pre-training verification |
| `config_paper.yaml` | 512 | 50 | 40-60 hrs | Full paper reproduction |

All configs use the same hyperparameters (kl_penalty, learning_rate, etc.) — only scale differs.

## Instructions

### Step 1: Verify vLLM is running

```bash
curl -s http://localhost:8888/v1/models >/dev/null 2>&1 && echo "vLLM: OK" || echo "vLLM: NOT RUNNING"
```

If not running, use the `start-vllm` skill first. No experiment can run without the inference server.

### Step 2: Pick the right config

- **Code was changed** -> start with `config_smoke_test.yaml` (always)
- **No code changes, re-running** -> can skip to `config_validate.yaml`
- **User wants full training** -> confirm validation passed first

Ask the user if unsure whether code has changed since the last successful run.

### Step 3: Run

```bash
cd examples/<task>
bash run.sh <config>.yaml
```

For anything beyond a smoke test, run in background and provide the monitor command:
```bash
tail -f tinker_log/<experiment_name>/train.log
```

### Step 4: Interpret results

Check `train.log` for errors and `metrics.jsonl` for scores. The key metric varies by task:

| Task | Conda env | Key metric | Direction | Non-default params |
|------|-----------|------------|-----------|-------------------|
| circle_packing | discover_math | raw_score/max | higher is better | — |
| ac_inequalities | discover_math | raw_score/max or min | depends on AC1/AC2 | — |
| erdos_min_overlap | discover_math | raw_score/min | lower is better | — |
| denoising | discover_denoising | custom | — | needs openproblems patch |
| gpu_mode | discover_gpumode | raw_score/min | lower is better | kl=0.01 |
| ahc | discover_ale | raw_score/max | higher is better | kl=0.01, lr=2e-5, phase1=22000 |

If smoke test passes, suggest validation. If validation passes, approve full training. Run tasks serially — they share the vLLM server.
