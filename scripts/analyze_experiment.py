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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
                    alpha=0.3, color='lightgreen', label='Min-Max Range')
    ax.plot(df['step'], df['score_mean'], 'b--', linewidth=2, label='Mean Score')
    ax.plot(df['step'], df['score_max'], 'r-', linewidth=2, label='Max Score')

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

    # Generate per-step table for markdown
    step_table_rows = []
    for row in data:
        step_table_rows.append(
            f"| {int(row['step'])} | {row['score_max']:.4f} | {row['score_mean']:.4f} | "
            f"{row['score_min']:.4f} | {row['reward_max']:.4f} | {row['reward_mean']:.4f} | "
            f"{row['kl_penalty']:.6f} | {row['time_total']:.0f}s |"
        )
    step_table = "\n".join(step_table_rows)

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

## Overall Results

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

| Step | Score Max | Score Mean | Score Min | Reward Max | Reward Mean | KL Penalty | Total Time |
|------|-----------|------------|-----------|------------|-------------|------------|------------|
{step_table}

Full CSV data: [results_table.csv](results_table.csv)

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

    print(f"\n{'='*60}")
    print(f"Best score: {best_score:.4f} @ step {int(best_step)}")
    print(f"Total time: {total_time/3600:.2f}h")
    print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_experiment.py <experiment_name>")
        sys.exit(1)

    analyze_experiment(sys.argv[1])
