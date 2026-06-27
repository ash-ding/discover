---
name: log-experiment-results
description: After a training experiment completes successfully, extract metrics and append results to docs/experiment_results.md
trigger: Use after any training run finishes (when you see "Training completed successfully")
---

# Log Experiment Results

When a training experiment completes successfully, follow these steps to record the results:

## 1. Identify the experiment

Find the experiment name from:
- The config file's `experiment_name` field
- Or the `tinker_log/<experiment_name>/` directory that was created

## 2. Extract key metrics

Read the following files from `tinker_log/<experiment_name>/`:

### From `metrics.jsonl` (last line = final step):
```bash
tail -1 tinker_log/<experiment_name>/metrics.jsonl | python3 -c "
import json, sys
m = json.load(sys.stdin)
print(f\"Step: {m.get('progress/batch', 'N/A')}\")
print(f\"Best score: {m.get('env/all/raw_score/max', 'N/A')}\")
print(f\"Mean score: {m.get('env/all/raw_score/mean', 'N/A')}\")
print(f\"Min score: {m.get('env/all/raw_score/min', 'N/A')}\")
print(f\"KL penalty: {m.get('kl_policy_base', 'N/A')}\")
print(f\"Total time: {m.get('time/total', 'N/A')}s\")
print(f\"Training time: {m.get('time/train', 'N/A')}s\")
print(f\"Sampling time: {m.get('time/sampling', 'N/A')}s\")
"
```

### Find best score across ALL steps:
```bash
python3 -c "
import json
best = -float('inf')
with open('tinker_log/<experiment_name>/metrics.jsonl') as f:
    for line in f:
        m = json.loads(line)
        score = m.get('env/all/raw_score/max', -float('inf'))
        if score > best:
            best = score
            best_step = m.get('progress/batch', 0)
print(f'Best ever: {best} at step {best_step}')
"
```

### From `config.json`:
```bash
python3 -c "
import json
c = json.load(open('tinker_log/<experiment_name>/config.json'))
print(f\"Model: {c.get('model_name', 'N/A')}\")
print(f\"Epochs: {c.get('num_epochs', 'N/A')}\")
print(f\"Group size: {c.get('group_size', 'N/A')}\")
print(f\"Groups per batch: {c.get('groups_per_batch', 'N/A')}\")
print(f\"Learning rate: {c.get('learning_rate', 'N/A')}\")
print(f\"KL coef: {c.get('kl_penalty_coef', 'N/A')}\")
print(f\"LoRA rank: {c.get('lora_rank', 'N/A')}\")
print(f\"Training GPUs: {c.get('training_gpu_ids', [c.get('training_gpu_id')])}\")
"
```

## 3. Append to results file

Create or append to `docs/experiment_results.md`:

```markdown
## <Task Name> - <Date>

**Experiment**: `<experiment_name>`  
**Model**: Qwen3-8B  
**Config**: <num_epochs> epochs, <group_size>×<groups_per_batch> samples/step, LoRA rank=<lora_rank>  
**Hardware**: TP=4 inference (GPUs 0-3), <N>-GPU training (GPUs <list>)  

### Results

| Metric | Value |
|--------|-------|
| Best Score (ever) | <best_score> @ step <step> |
| Final Score (max) | <final_max> |
| Final Score (mean) | <final_mean> |
| Total Time | <total_time>s (~<hours>h) |
| Training Time | <train_time>s (<percent>%) |
| Sampling Time | <sample_time>s (<percent>%) |
| KL Penalty (final) | <kl> |

### Notes

- <Any observations about convergence, best practices, issues encountered>
- Checkpoint: `tinker_log/local_checkpoints/<experiment_name>/sampler_final/`

---
```

## 4. Format guidelines

- Use consistent headers: `## <Task> - YYYY-MM-DD`
- Keep table format aligned
- Calculate percentages: `train_time / total_time * 100`
- Convert seconds to hours for long runs: `total_time / 3600`
- For minimization tasks (AC1), note that lower is better
- For maximization tasks (Circle Packing, AC2, etc.), higher is better

## 5. Example

```markdown
## Circle Packing (26 circles) - 2026-06-27

**Experiment**: `circle-packing-26-10epoch`  
**Model**: Qwen3-8B  
**Config**: 10 epochs, 64×8 samples/step, LoRA rank=32  
**Hardware**: TP=4 inference (GPUs 0-3), 4-GPU training (GPUs 4-7)  

### Results

| Metric | Value |
|--------|-------|
| Best Score (ever) | 2.547 @ step 7 |
| Final Score (max) | 2.512 |
| Final Score (mean) | 1.834 |
| Total Time | 28,450s (~7.9h) |
| Training Time | 9,120s (32%) |
| Sampling Time | 18,200s (64%) |
| KL Penalty (final) | 0.0015 |

### Notes

- Converged around epoch 6-7, slight degradation in final epochs
- Best checkpoint: step 7 with score 2.547
- Training efficiency: ~3.5× speedup with 4-GPU parallel training
- Checkpoint: `tinker_log/local_checkpoints/circle-packing-26-10epoch/sampler_final/`

---
```

## When NOT to use this skill

- Smoke tests or validation runs (only log full experiments)
- Failed experiments (only successful completions)
- Resume runs (only log once at final completion)
