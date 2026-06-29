---
name: log-experiment-results
description: After a training experiment completes successfully, generate detailed statistics table, training curve plot, and summary in experiment directory
trigger: Use after any training run finishes (when you see "Training completed successfully")
---

# Log Experiment Results

When a training experiment completes successfully, generate comprehensive results in the experiment's `tinker_log/<experiment_name>/` directory.

## Output Files (all saved to `tinker_log/<experiment_name>/`)

1. **`results_table.csv`** - Per-step statistics table
2. **`training_curve.png`** - Training progress visualization
3. **`summary.md`** - Human-readable summary with key findings

## Implementation Steps

### 1. Identify the experiment

```bash
# From the log output or config
EXPERIMENT_NAME="circle-packing-26-10epoch"
LOG_DIR="tinker_log/${EXPERIMENT_NAME}"
```

### 2. Generate per-step statistics table

Create `${LOG_DIR}/results_table.csv` with columns:
- `step`: Training step number
- `score_max`: Maximum score across all rollout samples
- `score_mean`: Average score across all rollout samples
- `score_min`: Minimum score across all rollout samples
- `reward_max`: Maximum reward
- `reward_mean`: Average reward
- `reward_min`: Minimum reward
- `kl_penalty`: KL divergence penalty
- `time_total`: Cumulative time (seconds)
- `time_sampling`: Time spent on sampling (seconds)
- `time_training`: Time spent on training (seconds)

```python
import json
import csv

metrics_file = f"{log_dir}/metrics.jsonl"
output_csv = f"{log_dir}/results_table.csv"

data = []
with open(metrics_file) as f:
    for line in f:
        m = json.loads(line)
        data.append({
            'step': m.get('progress/batch', 0),
            'score_max': m.get('env/all/raw_score/max', 0),
            'score_mean': m.get('env/all/raw_score/mean', 0),
            'score_min': m.get('env/all/raw_score/min', 0),
            'reward_max': m.get('env/all/reward/max', 0),
            'reward_mean': m.get('env/all/reward/mean', 0),
            'reward_min': m.get('env/all/reward/min', 0),
            'kl_penalty': m.get('kl_policy_base', 0),
            'time_total': m.get('time/total', 0),
            'time_sampling': m.get('time/sampling', 0),
            'time_training': m.get('time/train', 0),
        })

with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
```

### 3. Generate training curve plot

Create `${LOG_DIR}/training_curve.png` with:
- **X-axis**: Training step
- **Y-axis**: Score/Reward
- **Shaded region**: `[score_min, score_max]` at each step
- **Dashed line**: `score_mean` (average)
- **Solid line**: `score_max` (best per step)

```python
import matplotlib.pyplot as plt
import pandas as pd

# Read data
df = pd.read_csv(f"{log_dir}/results_table.csv")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot shaded region (min-max range)
ax.fill_between(df['step'], df['score_min'], df['score_max'], 
                alpha=0.3, color='blue', label='Min-Max Range')

# Plot mean line (dashed)
ax.plot(df['step'], df['score_mean'], 'b--', linewidth=2, label='Mean Score')

# Plot max line (solid)
ax.plot(df['step'], df['score_max'], 'b-', linewidth=2, label='Max Score')

# Labels and formatting
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Training Progress: {experiment_name}', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Save
plt.tight_layout()
plt.savefig(f"{log_dir}/training_curve.png", dpi=150, bbox_inches='tight')
plt.close()
```

### 4. Generate summary markdown

Create `${LOG_DIR}/summary.md`:

```markdown
# Experiment Summary: <experiment_name>

**Date**: <date>  
**Task**: <task_name>  
**Model**: Qwen3-8B  

## Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | <num_epochs> |
| Group Size | <group_size> |
| Groups per Batch | <groups_per_batch> |
| Learning Rate | <learning_rate> |
| KL Penalty Coef | <kl_penalty_coef> |
| LoRA Rank | <lora_rank> |
| Training GPUs | <training_gpu_ids> |

## Results

| Metric | Value |
|--------|-------|
| **Best Score (ever)** | **<best_score>** @ step <best_step> |
| Final Score (max) | <final_max> |
| Final Score (mean) | <final_mean> |
| Final Score (min) | <final_min> |
| Total Time | <total_time>s (~<hours>h) |
| Training Time | <train_time>s (<percent>%) |
| Sampling Time | <sample_time>s (<percent>%) |

## Training Curve

![Training Progress](training_curve.png)

## Per-Step Statistics

See [results_table.csv](results_table.csv) for detailed per-step metrics.

## Key Observations

- Best score achieved at step <best_step>: <best_score>
- <Convergence pattern description>
- <Performance notes>

## Checkpoints

- **Latest**: `tinker_log/local_checkpoints/<experiment_name>/latest_sampler/`
- **Best** (if saved): `tinker_log/local_checkpoints/<experiment_name>/sampler_<best_step>/`
- **Resume**: `tinker_log/local_checkpoints/<experiment_name>/state_<last_saved_step>/`
```

### 5. Complete implementation script

Create a standalone script `scripts/analyze_experiment.py`:

```python
#!/usr/bin/env python3
"""
Analyze completed training experiment and generate results.

Usage:
    python scripts/analyze_experiment.py <experiment_name>
    python scripts/analyze_experiment.py circle-packing-26-10epoch
"""

import sys
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def analyze_experiment(experiment_name: str):
    log_dir = Path(f"tinker_log/{experiment_name}")
    
    if not log_dir.exists():
        print(f"Error: {log_dir} does not exist")
        return
    
    # 1. Load metrics
    metrics_file = log_dir / "metrics.jsonl"
    config_file = log_dir / "config.json"
    
    data = []
    with open(metrics_file) as f:
        for line in f:
            m = json.loads(line)
            data.append({
                'step': m.get('progress/batch', 0),
                'score_max': m.get('env/all/raw_score/max', 0),
                'score_mean': m.get('env/all/raw_score/mean', 0),
                'score_min': m.get('env/all/raw_score/min', 0),
                'reward_max': m.get('env/all/reward/max', 0),
                'reward_mean': m.get('env/all/reward/mean', 0),
                'reward_min': m.get('env/all/reward/min', 0),
                'kl_penalty': m.get('kl_policy_base', 0),
                'time_total': m.get('time/total', 0),
                'time_sampling': m.get('time/sampling', 0),
                'time_training': m.get('time/train', 0),
            })
    
    df = pd.DataFrame(data)
    
    # Load config
    with open(config_file) as f:
        config = json.load(f)
    
    # 2. Save CSV table
    csv_path = log_dir / "results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved table: {csv_path}")
    
    # 3. Generate plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(df['step'], df['score_min'], df['score_max'], 
                    alpha=0.3, color='blue', label='Min-Max Range')
    ax.plot(df['step'], df['score_mean'], 'b--', linewidth=2, label='Mean Score')
    ax.plot(df['step'], df['score_max'], 'b-', linewidth=2, label='Max Score')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Training Progress: {experiment_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = log_dir / "training_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {plot_path}")
    
    # 4. Generate summary
    best_idx = df['score_max'].idxmax()
    best_score = df.loc[best_idx, 'score_max']
    best_step = df.loc[best_idx, 'step']
    
    final = data[-1]
    total_time = final['time_total']
    train_time = final['time_training']
    sample_time = final['time_sampling']
    
    summary = f"""# Experiment Summary: {experiment_name}

**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Model**: {config.get('model_name', 'N/A')}  

## Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | {config.get('num_epochs', 'N/A')} |
| Group Size | {config.get('group_size', 'N/A')} |
| Groups per Batch | {config.get('groups_per_batch', 'N/A')} |
| Learning Rate | {config.get('learning_rate', 'N/A')} |
| KL Penalty Coef | {config.get('kl_penalty_coef', 'N/A')} |
| LoRA Rank | {config.get('lora_rank', 'N/A')} |
| Training GPUs | {config.get('training_gpu_ids', [config.get('training_gpu_id')])} |

## Results

| Metric | Value |
|--------|-------|
| **Best Score (ever)** | **{best_score:.4f}** @ step {int(best_step)} |
| Final Score (max) | {final['score_max']:.4f} |
| Final Score (mean) | {final['score_mean']:.4f} |
| Final Score (min) | {final['score_min']:.4f} |
| Total Time | {total_time:.0f}s (~{total_time/3600:.2f}h) |
| Training Time | {train_time:.0f}s ({train_time/total_time*100:.1f}%) |
| Sampling Time | {sample_time:.0f}s ({sample_time/total_time*100:.1f}%) |

## Training Curve

![Training Progress](training_curve.png)

## Per-Step Statistics

See [results_table.csv](results_table.csv) for detailed per-step metrics.

## Key Observations

- Best score achieved at step {int(best_step)}: {best_score:.4f}
- Training completed {len(data)} steps
- Final KL penalty: {final['kl_penalty']:.6f}

## Checkpoints

- **Latest**: `tinker_log/local_checkpoints/{experiment_name}/latest_sampler/`
- **Resume**: `tinker_log/local_checkpoints/{experiment_name}/state_*/`
"""
    
    summary_path = log_dir / "summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"✓ Saved summary: {summary_path}")
    
    print(f"\nBest score: {best_score:.4f} @ step {int(best_step)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_experiment.py <experiment_name>")
        sys.exit(1)
    
    analyze_experiment(sys.argv[1])
```

## Usage

After an experiment completes, run:

```bash
python scripts/analyze_experiment.py <experiment_name>
```

This will generate all three files in `tinker_log/<experiment_name>/`:
- `results_table.csv`
- `training_curve.png`
- `summary.md`

## When NOT to use this skill

- Smoke tests or validation runs (only log full experiments)
- Failed experiments (only successful completions)
- Mid-training (only when fully complete)
