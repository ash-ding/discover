"""Entry point for TTT-Discover training with VERL backend.

Usage:
    python -m ttt_discover.verl_integration.main --config config/circle_packing.yaml

Or with overrides:
    python -m ttt_discover.verl_integration.main \
        --config config/circle_packing.yaml \
        --total_epochs 1 --group_size 4 --groups_per_batch 2
"""

import argparse
import logging
import os
import sys

import yaml

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TTT-Discover + VERL training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--total_epochs", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--groups_per_batch", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--mock", action="store_true",
                        help="Use MockWorkerBridge for CPU-only pipeline testing")
    return parser.parse_args()


def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for key, val in overrides.items():
        if val is not None:
            config[key] = val

    return config


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()
    overrides = {
        "total_epochs": args.total_epochs,
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
        "experiment_name": args.experiment_name,
    }
    config = load_config(args.config, overrides)

    logger.info(f"Config: {config}")

    # Disable Ray task event telemetry (saves ~1-2ms per eval)
    os.environ.setdefault("RAY_task_events_report_interval_ms", "0")

    # Initialize Ray
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Set up VERL worker groups
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], trust_remote_code=True
    )

    # Create PUCT data source
    from ttt_discover.verl_integration.puct_data_source import PUCTDataSource
    puct_data_source = PUCTDataSource(
        model_name=config["model_name"],
        env_module=config["env_module"],
        env_class=config["env_class"],
        problem_type=config.get("problem_type", ""),
        groups_per_batch=config.get("groups_per_batch", 8),
        group_size=config.get("group_size", 64),
        max_prompt_length=config.get("max_prompt_length", 4096),
        log_dir=config.get("log_dir", "./tinker_log"),
        puct_file_path=config.get("puct_file_path", "./tinker_log/puct_sampler.json"),
        max_buffer_size=config.get("max_buffer_size", 1000),
        puct_c=config.get("puct_c", 1.0),
        topk_children=config.get("topk_children", 2),
        resume_step=config.get("resume_step"),
        eval_timeout=config.get("eval_timeout", 530),
        num_cpus_per_task=config.get("num_cpus_per_task", 1),
        phase1_max_tokens=config.get("phase1_max_tokens", 26000),
        renderer_name=config.get("renderer_name", "qwen3"),
    )

    # Create reward function
    from ttt_discover.verl_integration.reward_function import compute_score

    # Initialize worker group (VERL or mock)
    actor_rollout_ref_wg = _create_worker_group(config, mock=args.mock)

    # Optional: WandB logger
    wandb_logger = None
    if config.get("wandb_mode", "offline") != "disabled":
        try:
            import wandb
            wandb.init(
                project=config.get("wandb_project", "ttt-discover"),
                name=config.get("experiment_name", "discover-verl"),
                config=config,
                mode=config.get("wandb_mode", "offline"),
            )
            wandb_logger = wandb
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # Create and run trainer
    from ttt_discover.verl_integration.discover_trainer import DiscoverTrainer
    trainer = DiscoverTrainer(
        config=config,
        puct_data_source=puct_data_source,
        actor_rollout_ref_wg=actor_rollout_ref_wg,
        reward_fn=compute_score,
        tokenizer=tokenizer,
        wandb_logger=wandb_logger,
    )

    trainer.fit()

    if wandb_logger:
        wandb_logger.finish()


def _create_worker_group(config: dict, mock: bool = False):
    """Create worker group — mock for testing, VERL for GPU training."""
    import torch

    use_mock = mock or not torch.cuda.is_available()
    if use_mock and not mock:
        logger.warning("No CUDA GPUs detected, falling back to MockWorkerBridge")

    logger.info(f"Worker backend: {'MockWorkerBridge' if use_mock else 'VERLWorkerBridge'}")
    logger.info(f"  Model: {config['model_name']}")
    logger.info(f"  LoRA rank: {config.get('lora_rank', 32)}")
    logger.info(f"  FSDP SP: {config.get('ulysses_sp_size', 4)}")
    logger.info(f"  vLLM TP: {config.get('inference_tp_size', 4)}")
    logger.info(f"  GPUs: {config.get('n_gpus', 8)}")

    if use_mock:
        from ttt_discover.verl_integration.verl_worker_bridge import MockWorkerBridge
        return MockWorkerBridge(config)
    else:
        from ttt_discover.verl_integration.verl_worker_bridge import VERLWorkerBridge
        return VERLWorkerBridge(config)


if __name__ == "__main__":
    main()
