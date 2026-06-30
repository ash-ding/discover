"""VERL-compatible reward function for circle packing (and other TTT-Discover tasks).

This follows VERL's custom reward function protocol:
    def compute_score(data_source, solution_str, ground_truth, extra_info) -> float
"""

import logging
import re

logger = logging.getLogger(__name__)


def _extract_last_code_block(text: str) -> str:
    """Extract the last Python code block from model output."""
    pattern = re.compile(r'```python\n(.*?)(?:\n```)', re.DOTALL)
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].group(1).rstrip()
    return ""


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None, **kwargs):
    """Evaluate circle packing solution by running code in sandbox.

    Returns sum of radii if valid, 0.0 otherwise.
    """
    code = _extract_last_code_block(solution_str)
    if not code:
        return 0.0

    extra = extra_info or {}
    env_module = extra.get("env_module", "examples.circle_packing.env")
    env_class = extra.get("env_class", "CirclePackingEnv")
    problem_type = extra.get("problem_type", "26")
    eval_timeout = extra.get("eval_timeout", 530)
    num_cpus = extra.get("num_cpus_per_task", 1)

    try:
        import importlib
        mod = importlib.import_module(env_module)
        env_cls = getattr(mod, env_class)
        reward_fn_cls = env_cls.reward_function

        evaluator = reward_fn_cls(
            problem_type=problem_type,
            log_dir=extra.get("log_dir", "./tinker_log"),
            eval_timeout=eval_timeout,
            num_cpus_per_task=num_cpus,
        )
        result = evaluator.get_reward(code)
        return float(result.get("reward", 0.0))
    except Exception as e:
        logger.debug(f"Reward eval failed: {e}")
        return 0.0
