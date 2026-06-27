"""
Rollout-only loop for statistics collection (no training).
"""

import asyncio
from typing import List
from tqdm.asyncio import tqdm as async_tqdm

from ttt_discover.rl.train import Config, do_group_rollout_and_filter_constant_reward
from ttt_discover.rl.types import TrajectoryGroup


async def do_rollout_only_loop(
    cfg: Config,
    service_client,
    dataset,
    sampling_client,
    num_rounds: int = 2
) -> List[List[TrajectoryGroup]]:
    """
    Run rollout-only loop for statistics collection.

    Important: Round 1 and Round 2 use DIFFERENT sampling configs:
    - Round 1: groups_per_batch=1, group_size=50 (1 prompt, 50 samples)
    - Round 2: groups_per_batch=8, group_size=6  (8 prompts, 48 samples)

    We temporarily modify dataset's batch_size and group_size between rounds.

    Args:
        cfg: Training config object
        service_client: Service client for model access
        dataset: Dataset object with PUCT sampler
        sampling_client: Sampling client for generation
        num_rounds: Number of rollout rounds (default: 2)

    Returns:
        List of [trajectory_groups_P] for each round
    """
    all_rounds = []

    # Save original config
    original_batch_size = dataset.batch_size
    original_group_size = dataset.group_size

    for round_idx in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1} / {num_rounds}")
        print(f"{'='*60}\n")

        # Configure sampling parameters for this round
        if round_idx == 0:
            # Round 1: 1 prompt, 50 completions
            dataset.batch_size = 1
            dataset.group_size = 50
        else:
            # Round 2: 8 prompts (PUCT top-8), 6 completions each
            dataset.batch_size = 8
            dataset.group_size = 6

        print(f"Config: batch_size={dataset.batch_size}, group_size={dataset.group_size}")
        print(f"Total samples: {dataset.batch_size} × {dataset.group_size} = {dataset.batch_size * dataset.group_size}")

        # Get batch (PUCT sampler will select states based on current statistics)
        env_group_builders_P = dataset.get_batch(index=0)
        print(f"Sampled {len(env_group_builders_P)} prompts from PUCT sampler")

        # Do rollouts (exactly like training loop)
        trajectory_groups_P = await async_tqdm.gather(
            *[
                asyncio.create_task(
                    do_group_rollout_and_filter_constant_reward(
                        sampling_client,
                        builder,
                        temperature=cfg.temperature,
                        do_remove_constant_reward_groups=False,
                        step_idx=round_idx,
                        model_name=cfg.local_model_path or cfg.model_name,
                        phase1_max_tokens=cfg.phase1_max_tokens,
                        context_window=cfg.max_model_len,
                    ),
                    name=f"sample_task_{i}",
                )
                for i, builder in enumerate(env_group_builders_P)
            ],
            desc=f"Rollouts [round {round_idx + 1}]",
            total=len(env_group_builders_P),
        )

        # Filter out None results (constant reward groups)
        trajectory_groups_P = [tg for tg in trajectory_groups_P if tg is not None]

        # Store results
        all_rounds.append(trajectory_groups_P)

        total_samples = sum(len(tg.trajectories_G) for tg in trajectory_groups_P)
        print(f"✓ Round {round_idx + 1} completed: {len(trajectory_groups_P)} trajectory groups, {total_samples} total samples")

        # PUCT sampler has already been updated during rollout
        # Next round's get_batch() will use updated statistics

    # Restore original config
    dataset.batch_size = original_batch_size
    dataset.group_size = original_group_size

    return all_rounds
