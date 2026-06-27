---
name: config-validation
description: |
  Validate YAML configuration files for TTT-Discover experiments against DiscoverConfig schema and runtime environment.
  Use this skill whenever the user mentions config files, YAML validation, experiment configuration, parameter checking,
  or is about to start a training run with a custom or modified config. Also use when debugging experiment startup failures
  that might stem from misconfiguration — bad GPU allocation, missing model paths, or parameter mismatches are the most
  common causes of wasted experiment time. Even if the user just says "check this before I run it", this skill applies.
---

# Configuration Validation

Catches config errors in seconds that would otherwise waste hours of GPU time. The most common failures are GPU overlap (training_gpu_id inside the TP range), missing model paths, and unknown YAML keys that get silently ignored.

## Instructions

Run this validation script, replacing `<CONFIG>` with the user's config file path:

```bash
python3 << 'PYEOF' <CONFIG>
import yaml, inspect, os, sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "config_paper.yaml"
with open(filepath) as f:
    config = yaml.safe_load(f)

from ttt_discover.discovery import DiscoverConfig
valid = set(inspect.signature(DiscoverConfig.__init__).parameters.keys()) - {"self"}
extra_keys = {"num_circles", "target_score"}
unknown = set(config.keys()) - valid - extra_keys
errors, warnings = [], []

if unknown:
    # Unknown keys are silently dropped by the inspect-based config loader,
    # so the user may think a param is active when it's actually ignored.
    warnings.append(f"Unknown keys (will be IGNORED by DiscoverConfig): {unknown}")

tp = config.get("inference_tp_size", 1)
tg = config.get("training_gpu_id", 1)
if tg < tp:
    errors.append(f"training_gpu_id ({tg}) overlaps inference GPUs (0-{tp-1}). "
                  f"Set training_gpu_id >= inference_tp_size.")
if tp not in (1, 2, 4, 8):
    warnings.append(f"inference_tp_size ({tp}) is not a power of 2 — vLLM requires this")

mp = config.get("local_model_path", "")
if mp and not os.path.exists(mp):
    warnings.append(f"Model path not found: {mp}")

lr = config.get("learning_rate", 0)
if lr and (lr < 1e-7 or lr > 1e-2):
    warnings.append(f"learning_rate ({lr}) outside typical range [1e-6, 1e-3]")

for e in errors:
    print(f"ERROR: {e}")
for w in warnings:
    print(f"WARNING: {w}")
if not errors:
    print(f"\nConfig summary:")
    print(f"  epochs={config.get('num_epochs')}, group_size={config.get('group_size')}, "
          f"groups_per_batch={config.get('groups_per_batch')}")
    print(f"  lr={config.get('learning_rate')}, kl={config.get('kl_penalty_coef')}, "
          f"tp={tp}, training_gpu={tg}")
    print(f"  training_batch_size={config.get('training_batch_size', 1)}, "
          f"max_train_seq_len={config.get('max_train_seq_len', 32768)}")
    print("Validation passed")
sys.exit(1 if errors else 0)
PYEOF
```

If vLLM is running, also verify the model length matches:
```bash
curl -s http://localhost:8888/v1/models >/dev/null 2>&1 && echo "vLLM: running" || echo "vLLM: not running (skip server compatibility check)"
```

Report all errors, warnings, and the config summary to the user. If there are unknown keys, emphasize that they will be silently ignored — this is the most common source of "I set X but it didn't take effect" bugs.
