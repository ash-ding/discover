"""VERL-compatible reward function wrapping TTT-Discover's sandbox evaluator.

VERL expects: compute_score(data_source, solution_str, ground_truth, extra_info) -> float
TTT-Discover uses: SandboxRewardEvaluator.get_reward(code, state) -> dict
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_evaluator_cache: dict[str, Any] = {}


def _get_evaluator(env_module: str, env_class: str, problem_type: str,
                   log_dir: str, eval_timeout: int, num_cpus_per_task: int):
    """Lazily create and cache reward evaluators."""
    cache_key = f"{env_module}.{env_class}:{problem_type}"
    if cache_key not in _evaluator_cache:
        import importlib
        mod = importlib.import_module(env_module)
        env_cls = getattr(mod, env_class)
        reward_fn_cls = env_cls.reward_function
        _evaluator_cache[cache_key] = reward_fn_cls(
            problem_type=problem_type,
            log_dir=log_dir,
            eval_timeout=eval_timeout,
            num_cpus_per_task=num_cpus_per_task,
        )
    return _evaluator_cache[cache_key]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str] = None,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """VERL-compatible reward function.

    Args:
        data_source: dataset identifier (e.g. "circle_packing").
        solution_str: the model's complete response text.
        ground_truth: unused (reward computed from code execution).
        extra_info: dict with keys:
            - env_module: module path (e.g. "examples.circle_packing.env")
            - env_class: class name (e.g. "CirclePackingEnv")
            - problem_type: problem variant (e.g. "26")
            - state: serialized State object for the environment
            - log_dir: path for evaluation logs
            - eval_timeout: seconds before timeout (default 530)
            - num_cpus_per_task: CPUs per evaluation (default 1)

    Returns:
        Numeric reward score.
    """
    if extra_info is None:
        logger.warning("compute_score called without extra_info, returning 0.0")
        return 0.0

    evaluator = _get_evaluator(
        env_module=extra_info["env_module"],
        env_class=extra_info["env_class"],
        problem_type=extra_info.get("problem_type", ""),
        log_dir=extra_info.get("log_dir", "./tinker_log"),
        eval_timeout=extra_info.get("eval_timeout", 530),
        num_cpus_per_task=extra_info.get("num_cpus_per_task", 1),
    )

    try:
        result = evaluator.get_reward(solution_str, state=extra_info.get("state"))
        reward = float(result.get("reward", 0.0))
    except Exception as e:
        logger.warning(f"Reward evaluation failed: {e}")
        reward = 0.0

    return reward
