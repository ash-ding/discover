#!/usr/bin/env python3
"""Export LoRA adapter from VERL FSDP checkpoint.

Merges FSDP sharded checkpoint files, extracts LoRA delta parameters,
and saves as a standard PEFT adapter that can be loaded with any
parallel configuration.

Usage:
    python scripts/export_lora.py checkpoints/ttt-discover/my-run/latest/actor

Output:
    checkpoints/ttt-discover/my-run/latest/actor/exported_lora/
        adapter_model.safetensors
        adapter_config.json

The exported adapter can be:
  - Loaded with PeftModel.from_pretrained(base_model, path)
  - Used as VERL training init via model.lora_adapter_path config
  - Used for cross-parallel-config resume
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import torch


def load_sharded_state_dict(actor_dir: str) -> dict:
    """Load and merge all FSDP sharded model files into a single state dict."""
    fsdp_config_path = os.path.join(actor_dir, "fsdp_config.json")
    if not os.path.exists(fsdp_config_path):
        raise FileNotFoundError(f"No fsdp_config.json found in {actor_dir}")

    with open(fsdp_config_path) as f:
        fsdp_config = json.load(f)
    world_size = fsdp_config.get("world_size", 1)

    print(f"Loading {world_size} shards from {actor_dir}")

    # Load all shards
    shards = []
    for rank in range(world_size):
        path = os.path.join(actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shard not found: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        shards.append(shard)
        print(f"  Loaded rank {rank}: {len(shard)} keys")

    def _to_local(t):
        """Convert DTensor/ShardedTensor to plain torch.Tensor."""
        if hasattr(t, '_local_tensor'):
            return t._local_tensor
        if hasattr(t, 'local_tensor'):
            return t.local_tensor()
        if hasattr(t, 'local_shards'):
            shards = t.local_shards()
            if shards:
                return shards[0].tensor
        return t

    # Merge shards into full state dict
    full_state = {}
    all_keys = set()
    for shard in shards:
        all_keys.update(shard.keys())

    for key in sorted(all_keys):
        tensors = [_to_local(s[key]) for s in shards if key in s]
        if len(tensors) == 1:
            full_state[key] = tensors[0]
        elif len(tensors) == world_size:
            if tensors[0].dim() == 0:
                full_state[key] = tensors[0]
            else:
                full_state[key] = torch.cat(tensors, dim=0)
        else:
            print(f"  Warning: key {key} found in {len(tensors)}/{world_size} shards")
            full_state[key] = tensors[0]

    print(f"Merged state dict: {len(full_state)} keys")
    return full_state


def extract_lora_params(state_dict: dict) -> OrderedDict:
    """Extract LoRA adapter parameters from a full state dict."""
    lora_params = OrderedDict()
    for key, value in state_dict.items():
        if "lora_" in key:
            # Clean up FSDP/PEFT key prefixes
            clean_key = key
            # Remove common prefixes
            for prefix in ["_fsdp_wrapped_module.", "base_model.model.", "model."]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            lora_params[clean_key] = value

    print(f"Extracted {len(lora_params)} LoRA parameters")
    if lora_params:
        total_params = sum(p.numel() for p in lora_params.values())
        total_bytes = sum(p.numel() * p.element_size() for p in lora_params.values())
        print(f"  Total LoRA params: {total_params:,} ({total_bytes / 1024 / 1024:.1f} MB)")
    return lora_params


def save_peft_adapter(lora_params: OrderedDict, lora_meta: dict, output_dir: str):
    """Save LoRA parameters in PEFT-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    # Save weights
    from safetensors.torch import save_file
    save_file(lora_params, os.path.join(output_dir, "adapter_model.safetensors"))

    # Save adapter config
    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": lora_meta.get("base_model", ""),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "lora_alpha": lora_meta.get("lora_alpha", lora_meta.get("lora_rank", 32)),
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_meta.get("lora_rank", 32),
        "revision": None,
        "target_modules": lora_meta.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        "task_type": lora_meta.get("task_type", "CAUSAL_LM"),
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"Saved PEFT adapter to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export LoRA adapter from VERL FSDP checkpoint")
    parser.add_argument("checkpoint_dir", help="Path to actor checkpoint directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: <checkpoint_dir>/exported_lora)")
    args = parser.parse_args()

    actor_dir = args.checkpoint_dir
    output_dir = args.output or os.path.join(actor_dir, "exported_lora")

    # Load LoRA metadata
    lora_meta_path = os.path.join(actor_dir, "lora_train_meta.json")
    if os.path.exists(lora_meta_path):
        with open(lora_meta_path) as f:
            lora_meta = json.load(f)
        print(f"LoRA meta: rank={lora_meta.get('lora_rank')}, alpha={lora_meta.get('lora_alpha')}")
    else:
        print("Warning: No lora_train_meta.json found, using defaults")
        lora_meta = {"lora_rank": 32, "lora_alpha": 32}

    # Load and merge sharded checkpoint
    state_dict = load_sharded_state_dict(actor_dir)

    # Extract LoRA parameters
    lora_params = extract_lora_params(state_dict)
    if not lora_params:
        print("ERROR: No LoRA parameters found in checkpoint. Is this a LoRA-trained model?")
        sys.exit(1)

    # Save as PEFT adapter
    save_peft_adapter(lora_params, lora_meta, output_dir)
    print(f"\nDone! Adapter saved to: {output_dir}")
    print(f"To use: PeftModel.from_pretrained(base_model, '{output_dir}')")


if __name__ == "__main__":
    main()
