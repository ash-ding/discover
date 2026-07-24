"""Step-boundary RL training metrics aggregation.

Collects per-completion metrics lazily during rollouts and computes
aggregated statistics at step boundaries. No hot-path logging overhead.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


class MetricsAggregator:
    """Aggregate RL training metrics at step boundaries.

    Usage:
        agg = MetricsAggregator()
        # During rollouts:
        for completion in completions:
            agg.record(completion_reward_extra)
        # At step boundary:
        metrics = agg.compute()  # returns dict, resets internal state
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._rewards: list[float] = []
        self._p1_lens: list[int] = []
        self._p2_lens: list[int] = []
        self._gen_cases: Counter = Counter()
        self._gen_times: list[float] = []
        self._eval_times: list[float] = []
        self._eval_errors: Counter = Counter()
        self._eval_error_types: Counter = Counter()
        self._total_completions: int = 0
        self._successful_evals: int = 0
        self._failed_evals: int = 0
        self._scores_us: list[float] = []
        self._puct_parent_values: list[float] = []

    def record(self, reward: float, reward_extra: dict[str, Any] | None = None):
        """Record metrics from a single completion."""
        self._total_completions += 1
        self._rewards.append(reward)

        if reward_extra is None:
            return

        if "p1_len" in reward_extra:
            self._p1_lens.append(int(reward_extra["p1_len"]))
        if "p2_len" in reward_extra:
            self._p2_lens.append(int(reward_extra["p2_len"]))
        if "gen_case" in reward_extra:
            self._gen_cases[reward_extra["gen_case"]] += 1
        if "gen_time_s" in reward_extra:
            self._gen_times.append(float(reward_extra["gen_time_s"]))
        if "eval_time_s" in reward_extra:
            self._eval_times.append(float(reward_extra["eval_time_s"]))

        if "score_us" in reward_extra:
            self._scores_us.append(float(reward_extra["score_us"]))

        if "puct_parent_value" in reward_extra:
            pv = reward_extra["puct_parent_value"]
            if pv is not None:
                self._puct_parent_values.append(float(pv))

        eval_error = reward_extra.get("eval_error", "")
        if eval_error:
            self._failed_evals += 1
            short_key = _categorize_error(eval_error)
            self._eval_errors[short_key] += 1
            error_type = reward_extra.get("error_type")
            if error_type:
                self._eval_error_types[error_type] += 1
        elif reward > 0:
            self._successful_evals += 1

    def compute(self) -> dict[str, Any]:
        """Compute aggregated metrics for the current step and reset."""
        if self._total_completions == 0:
            self._reset()
            return {}

        metrics: dict[str, Any] = {}
        metrics["rl/total_completions"] = self._total_completions
        metrics["rl/successful_evals"] = self._successful_evals
        metrics["rl/failed_evals"] = self._failed_evals

        if self._rewards:
            metrics.update(_stats(self._rewards, "rl/reward"))
            nonzero = [r for r in self._rewards if r > 0]
            metrics["rl/reward/nonzero_count"] = len(nonzero)
            metrics["rl/reward/nonzero_frac"] = len(nonzero) / len(self._rewards)

        if self._p1_lens:
            metrics.update(_stats(self._p1_lens, "rl/phase1_len"))
        if self._p2_lens:
            p2_nonzero = [l for l in self._p2_lens if l > 0]
            metrics["rl/phase2_count"] = len(p2_nonzero)
            if p2_nonzero:
                metrics.update(_stats(p2_nonzero, "rl/phase2_len"))

        for case, count in self._gen_cases.items():
            metrics[f"rl/gen_case/{case}"] = count

        if self._gen_times:
            metrics.update(_stats(self._gen_times, "rl/gen_time_s"))
        if self._eval_times:
            metrics.update(_stats(self._eval_times, "rl/eval_time_s"))

        if self._scores_us:
            metrics.update(_stats(self._scores_us, "rl/score_us"))

        if self._puct_parent_values:
            metrics.update(_stats(self._puct_parent_values, "rl/puct_parent_value"))

        for err_key, count in self._eval_errors.most_common(10):
            metrics[f"rl/eval_error/{err_key}"] = count

        for etype, count in self._eval_error_types.most_common(5):
            metrics[f"rl/eval_error_type/{etype}"] = count

        self._reset()
        return metrics


def _stats(values: list, prefix: str) -> dict[str, float]:
    """Compute mean/std/min/max for a list of numeric values."""
    if not values:
        return {}
    n = len(values)
    s = sum(values)
    mean = s / n
    sq_sum = sum((v - mean) ** 2 for v in values)
    std = math.sqrt(sq_sum / n) if n > 1 else 0.0
    return {
        f"{prefix}/mean": mean,
        f"{prefix}/std": std,
        f"{prefix}/min": min(values),
        f"{prefix}/max": max(values),
    }


def _categorize_error(error_msg: str) -> str:
    """Categorize eval error into a short key for aggregation."""
    lower = error_msg.lower()
    if "timeout" in lower:
        return "timeout"
    if "no code block" in lower:
        return "no_code"
    if "no @triton" in lower:
        return "no_triton"
    if "http" in lower:
        return "http_error"
    if "oom" in lower or "out of memory" in lower:
        return "oom"
    if "syntax" in lower:
        return "syntax_error"
    if "compilation" in lower or "compile" in lower:
        return "compilation"
    if "import" in lower:
        return "import_error"
    return "other"
