"""
Task-level orchestration for rollout statistics collection.
"""

import asyncio
import gc
import inspect
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch

from ttt_discover.rl.train import Config
from ttt_discover.utils.config_loader import load_config
from ttt_discover.discovery import DiscoverConfig
from ttt_discover.tinker_utils.dataset_builder import DatasetConfig, get_single_problem_dataset_builder

from scripts.rollout_stats.loop import do_rollout_only_loop
from scripts.rollout_stats.utils import extract_length_stats


def _get_valid_discover_config_keys():
    """Return the set of valid parameter names for DiscoverConfig.__init__."""
    return set(inspect.signature(DiscoverConfig.__init__).parameters.keys())


def _build_discover_config_for_task(task_name: str, config_dict: dict) -> DiscoverConfig:
    """
    Build a DiscoverConfig with the correct env_type for the given task.

    Mirrors the logic in each task's discover_*_local() function:
    - Filters config_dict to only valid DiscoverConfig keys
    - Sets env_type to the task-specific environment class
    - Sets problem_type appropriately
    - Handles task-specific env vars (e.g., GPU mode evaluator settings)
    """
    valid_keys = _get_valid_discover_config_keys()

    if task_name == "circle_packing":
        from examples.circle_packing.env import CirclePackingEnv
        cfg = {k: v for k, v in config_dict.items() if k in valid_keys}
        cfg["env_type"] = CirclePackingEnv
        cfg["problem_type"] = str(config_dict.get("num_circles", "26"))
        return DiscoverConfig(**cfg)

    elif task_name == "ahc":
        from examples.ahc.env import AhcEnv
        cfg = {k: v for k, v in config_dict.items() if k in valid_keys}
        cfg["env_type"] = AhcEnv
        cfg["problem_type"] = config_dict.get("problem_type", "ahc039")
        return DiscoverConfig(**cfg)

    elif task_name == "gpu_mode":
        from examples.gpu_mode.env import GpuModeEnv
        # Set environment variables for local evaluator (same as discover_gpu_mode_local)
        if "kernel_eval_gpu" in config_dict:
            os.environ["KERNEL_EVAL_GPU"] = str(config_dict["kernel_eval_gpu"])
        if "kernel_eval_use_container" in config_dict:
            os.environ["KERNEL_EVAL_USE_CONTAINER"] = str(config_dict["kernel_eval_use_container"]).lower()
        elif "kernel_eval_use_docker" in config_dict:
            os.environ["KERNEL_EVAL_USE_DOCKER"] = str(config_dict["kernel_eval_use_docker"]).lower()
        if "kernel_eval_timeout" in config_dict:
            os.environ["KERNEL_EVAL_TIMEOUT"] = str(config_dict["kernel_eval_timeout"])
        if "kernel_eval_retries" in config_dict:
            os.environ["KERNEL_EVAL_RETRIES"] = str(config_dict["kernel_eval_retries"])

        cfg = {k: v for k, v in config_dict.items() if k in valid_keys}
        cfg["env_type"] = GpuModeEnv
        cfg["problem_type"] = config_dict.get("problem_type", "trimul")
        return DiscoverConfig(**cfg)

    elif task_name == "denoising":
        from examples.denoising.env import DenoisingEnv
        cfg = {k: v for k, v in config_dict.items() if k in valid_keys}
        cfg["env_type"] = DenoisingEnv
        cfg["problem_type"] = ""
        return DiscoverConfig(**cfg)

    else:
        raise ValueError(f"Unknown task: {task_name}. "
                         f"Valid tasks: circle_packing, ahc, gpu_mode, denoising")


async def collect_rollout_stats_for_task(
    config_path: str,
    task_name: str
) -> dict:
    """
    Collect rollout length statistics for a single task.

    Args:
        config_path: Path to YAML config file (e.g., "examples/circle_packing/config_paper.yaml")
        task_name: Task name: "circle_packing", "ahc", "gpu_mode", or "denoising"

    Returns:
        Dictionary with structure:
        {
            "task": task_name,
            "config": {...},
            "round1": {...},  # 1 prompt x 50 samples
            "round2": {...}   # 8 prompts x 6 samples (48 total)
        }
    """
    # 1. Load config from YAML
    print(f"Loading configuration from: {config_path}")
    config_dict = load_config(config_path)

    # 2. Build DiscoverConfig with correct env_type for this task
    discover_config = _build_discover_config_for_task(task_name, config_dict)

    # 3. Build DatasetConfig and rl Config (mirrors discovery.py::discover_impl)
    log_path = f"./tinker_log/rollout-stats-{task_name}"
    os.makedirs(log_path, exist_ok=True)

    dataset_config = DatasetConfig(
        env_type=discover_config.env_type,
        problem_type=discover_config.problem_type,
        batch_size=discover_config.groups_per_batch,
        group_size=discover_config.group_size,
        model_name_for_tokenizer=discover_config.model_name,
        renderer_name=discover_config.renderer_name,
        num_cpus_per_task=discover_config.num_cpus_per_task,
        eval_timeout=discover_config.eval_timeout,
        log_path=log_path,
    )
    dataset_builder = get_single_problem_dataset_builder(dataset_config)

    cfg = Config(
        env_type=dataset_config.env_type,
        problem_type=discover_config.problem_type,
        learning_rate=discover_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=discover_config.model_name,
        lora_rank=discover_config.lora_rank,
        temperature=discover_config.temperature,
        wandb_project="rollout-stats",
        wandb_name=f"rollout-stats-{task_name}",
        log_path=log_path,
        load_checkpoint_path=None,
        kl_penalty_coef=discover_config.kl_penalty_coef,
        num_substeps=1,
        save_every=999,
        num_epochs=2,
        loss_fn="importance_sampling",
        adv_estimator="entropic_adaptive_beta",
        adv_estimator_beta=2.0,
        remove_constant_reward_groups=False,
        phase1_max_tokens=discover_config.phase1_max_tokens,
        local_model_path=discover_config.local_model_path,
        use_local_backend=discover_config.use_local_backend,
        inference_gpu_id=discover_config.inference_gpu_id,
        training_gpu_id=discover_config.training_gpu_id,
        inference_tp_size=discover_config.inference_tp_size,
        max_model_len=discover_config.max_model_len,
        training_batch_size=discover_config.training_batch_size,
        max_train_seq_len=discover_config.max_train_seq_len,
    )

    # 4. Initialize service client
    print(f"Initializing service client for {task_name}...")
    if cfg.use_local_backend:
        from ttt_discover.local_backend import LocalServiceClient
        service_client = LocalServiceClient(
            model_name_or_path=cfg.local_model_path or cfg.model_name,
            inference_gpu_id=cfg.inference_gpu_id,
            training_gpu_id=cfg.training_gpu_id,
            inference_tp_size=cfg.inference_tp_size,
            max_model_len=cfg.max_model_len,
            experiment_name=cfg.wandb_name or task_name,
            training_batch_size=cfg.training_batch_size,
            max_train_seq_len=cfg.max_train_seq_len,
        )
    else:
        import tinker
        service_client = tinker.ServiceClient(base_url=None)

    # 5. Create training client (needed to get the sampling client)
    print(f"Creating training client...")
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    # 6. Create dataset
    print(f"Creating dataset...")
    dataset = await cfg.dataset_builder()

    # 7. Get sampling client from service client
    sampling_client = service_client.create_sampling_client()

    try:
        # 8. Run 2 rounds of rollouts (no training)
        print(f"Running 2 rounds of rollouts...")
        all_rounds = await do_rollout_only_loop(
            cfg=cfg,
            service_client=service_client,
            dataset=dataset,
            sampling_client=sampling_client,
            num_rounds=2
        )

        # 9. Extract statistics for each round
        round1_stats = extract_length_stats(all_rounds[0])
        round2_stats = extract_length_stats(all_rounds[1])

        return {
            "task": task_name,
            "config": {
                "model_name": cfg.model_name,
                "phase1_max_tokens": cfg.phase1_max_tokens,
                "max_model_len": cfg.max_model_len,
                "inference_tp_size": cfg.inference_tp_size,
                "temperature": cfg.temperature,
                "round1": {"groups_per_batch": 1, "group_size": 50},
                "round2": {"groups_per_batch": 8, "group_size": 6},
            },
            "round1": round1_stats,
            "round2": round2_stats,
        }

    finally:
        print(f"\nCleaning up resources for {task_name}...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
