"""VERL-compatible reward function for circle packing (and other TTT-Discover tasks).

This follows VERL's custom reward function protocol:
    def compute_score(data_source, solution_str, ground_truth, extra_info) -> float
"""

import logging
import re
import threading

logger = logging.getLogger(__name__)

_evaluator_cache = {}
_evaluator_lock = threading.Lock()


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
    extra = extra_info or {}
    env_module = extra.get("env_module", "examples.circle_packing.env")
    env_class = extra.get("env_class", "CirclePackingEnv")
    problem_type = extra.get("problem_type", "26")
    eval_timeout = extra.get("eval_timeout", 530)
    num_cpus = extra.get("num_cpus_per_task", 1)

    cache_key = (env_module, env_class, problem_type, eval_timeout)
    try:
        with _evaluator_lock:
            if cache_key not in _evaluator_cache:
                import importlib
                mod = importlib.import_module(env_module)
                env_cls = getattr(mod, env_class)
                reward_fn_cls = env_cls.reward_function
                _evaluator_cache[cache_key] = reward_fn_cls(
                    problem_type=problem_type,
                    log_dir=extra.get("log_dir", "./tinker_log"),
                    eval_timeout=eval_timeout,
                    num_cpus_per_task=num_cpus,
                )
            evaluator = _evaluator_cache[cache_key]

        result = evaluator.get_reward(solution_str, state=extra.get("state"))
        reward = float(result.get("reward", 0.0))
        msg = result.get("msg", "")
        return {"score": reward, "eval_msg": msg}
    except Exception as e:
        logger.warning(f"Reward eval failed: {type(e).__name__}: {e}")
        return {"score": 0.0, "eval_msg": f"{type(e).__name__}: {e}"}
