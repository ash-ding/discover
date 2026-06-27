"""
Utility functions for computing statistics on rollout length data.
"""

import numpy as np
from typing import List
from ttt_discover.rl.types import TrajectoryGroup


def compute_stats(lengths: List[int]) -> dict:
    """
    Compute comprehensive statistics for a list of lengths.

    Args:
        lengths: List of integer lengths (prompt, generation, or total tokens)

    Returns:
        Dictionary with statistics:
        {
            "min": int,
            "max": int,
            "avg": float,
            "median": float,
            "std": float,
            "p25": float,
            "p50": float,
            "p75": float,
            "p90": float,
            "p95": float,
            "p99": float
        }
    """
    if not lengths:
        return {
            k: 0.0 for k in ["min", "max", "avg", "median", "std",
                              "p25", "p50", "p75", "p90", "p95", "p99"]
        }

    arr = np.array(lengths)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "avg": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def extract_length_stats(trajectory_groups: List[TrajectoryGroup]) -> dict:
    """
    Extract length statistics from rollout data.

    Args:
        trajectory_groups: List of TrajectoryGroup objects from rollouts

    Returns:
        Dictionary with structure:
        {
            "num_samples": int,
            "prompt_lengths": [int, ...],          # Raw data
            "generation_lengths": [int, ...],      # Raw data
            "total_lengths": [int, ...],           # Raw data
            "prompt_stats": {...},                 # Statistics (from compute_stats)
            "generation_stats": {...},             # Statistics (from compute_stats)
            "total_stats": {...}                   # Statistics (from compute_stats)
        }
    """
    prompt_lens = []
    gen_lens = []

    for traj_group in trajectory_groups:
        for traj in traj_group.trajectories_G:
            # Usually single-step for these tasks (one code generation)
            # But iterate all transitions to be safe
            for trans in traj.transitions:
                prompt_len = trans.ob.length
                gen_len = len(trans.ac.tokens)
                prompt_lens.append(prompt_len)
                gen_lens.append(gen_len)

    total_lens = [p + g for p, g in zip(prompt_lens, gen_lens)]

    return {
        "num_samples": len(prompt_lens),
        "prompt_lengths": prompt_lens,           # Keep raw data
        "generation_lengths": gen_lens,          # Keep raw data
        "total_lengths": total_lens,             # Keep raw data
        "prompt_stats": compute_stats(prompt_lens),
        "generation_stats": compute_stats(gen_lens),
        "total_stats": compute_stats(total_lens),
    }
