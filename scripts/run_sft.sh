#!/usr/bin/env bash
# SFT for the A1/A2 experiments (docs/sft-experiments.md).
#
# Trains a LoRA adapter on a dataset produced by scripts/build_sft_dataset.py,
# with the same adapter geometry as the RL runs (rank 32, alpha 32,
# all-linear), so an A1 checkpoint and an RL checkpoint differ only in how
# they were trained, not in what was trainable.
#
#   TRAIN_FILE=data/sft/erdos_a1_cap4.parquet EXPERIMENT_NAME=erdos-sft-a1 \
#       bash scripts/run_sft.sh
#
# Overridables (defaults in brackets):
#   MODEL_PATH   [~/models/Qwen3-8B]
#   TRAIN_FILE   (required) parquet from build_sft_dataset.py
#   EXPERIMENT_NAME (required)
#   EPOCHS       [2]
#   LR           [2e-5]
#   LORA_RANK    [32]
#   MAX_LENGTH   [30720]   must match build_sft_dataset.py --max-length
#   BATCH_SIZE   [32]      global batch (sequences)
#   MICRO_TOKENS [32768]   max tokens per GPU per micro-batch (dynamic bsz)
#   NGPUS        [8]
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TRAIN_FILE:?set TRAIN_FILE to a build_sft_dataset.py parquet}"
: "${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"
MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen3-8B}
EPOCHS=${EPOCHS:-2}
LR=${LR:-2e-5}
LORA_RANK=${LORA_RANK:-32}
MAX_LENGTH=${MAX_LENGTH:-30720}
BATCH_SIZE=${BATCH_SIZE:-32}
MICRO_TOKENS=${MICRO_TOKENS:-32768}
NGPUS=${NGPUS:-8}

OUT_DIR="checkpoints/sft/${EXPERIMENT_NAME}"
if [ -e "$OUT_DIR" ]; then
    echo "refusing to overwrite existing $OUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

# Qwen3's chat template only keeps an assistant turn's <think> block when that
# turn sits after the last user query of the WHOLE conversation
# (loop.index0 > ns.last_query_index). MultiTurnSFTDataset renders each turn
# SEPARATELY, so that condition never holds and the reasoning is silently
# stripped from every training target. Train with a tokenizer copy whose
# template drops the position gate; serving keeps the stock tokenizer.
SFT_TOKENIZER="${SFT_TOKENIZER:-${MODEL_PATH%/}-sft-tokenizer}"
if [ ! -f "$SFT_TOKENIZER/tokenizer_config.json" ]; then
    mkdir -p "$SFT_TOKENIZER"
    cp "$MODEL_PATH/tokenizer.json" "$SFT_TOKENIZER/"
    cp "$MODEL_PATH/special_tokens_map.json" "$SFT_TOKENIZER/" 2>/dev/null || true
    python3 - "$MODEL_PATH/tokenizer_config.json" "$SFT_TOKENIZER/tokenizer_config.json" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
old = "{%- if loop.index0 > ns.last_query_index %}"
assert old in cfg["chat_template"], "chat template anchor not found"
cfg["chat_template"] = cfg["chat_template"].replace(old, "{%- if true %}")
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
PYEOF
    echo "built think-preserving tokenizer at $SFT_TOKENIZER"
fi

# Snapshot the exact configuration next to the checkpoint, like run_verl.sh.
python3 - "$OUT_DIR/config_snapshot.json" <<PYEOF
import json, subprocess, sys, time
json.dump({
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "git_hash": subprocess.run(["git", "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
    "train_file": "$TRAIN_FILE", "model_path": "$MODEL_PATH",
    "epochs": "$EPOCHS", "lr": "$LR", "lora_rank": "$LORA_RANK",
    "max_length": "$MAX_LENGTH", "batch_size": "$BATCH_SIZE",
    "micro_tokens": "$MICRO_TOKENS", "ngpus": "$NGPUS",
}, open(sys.argv[1], "w"), indent=2)
PYEOF

export WANDB_MODE=${WANDB_MODE:-offline}

torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
    -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_FILE" \
    data.train_batch_size="$BATCH_SIZE" \
    data.max_length="$MAX_LENGTH" \
    data.max_token_len_per_gpu="$MICRO_TOKENS" \
    data.use_dynamic_bsz=True \
    data.truncation=error \
    model.path="$MODEL_PATH" \
    model.tokenizer_path="$SFT_TOKENIZER" \
    engine.model_dtype=bf16 \
    model.lora_rank="$LORA_RANK" \
    model.lora_alpha="$LORA_RANK" \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=True \
    optim.lr="$LR" \
    optim.lr_warmup_steps_ratio=0.03 \
    trainer.project_name=discover-sft \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUT_DIR" \
    trainer.total_epochs="$EPOCHS" \
    trainer.logger='[console]' \
    trainer.save_freq=-1 \
    trainer.seed=1 \
    "$@"
