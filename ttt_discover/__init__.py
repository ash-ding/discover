def __getattr__(name):
    if name == "DiscoverConfig":
        from ttt_discover.discovery import DiscoverConfig
        return DiscoverConfig
    if name == "discover":
        from ttt_discover.discovery import discover
        return discover
    if name == "Environment":
        from ttt_discover.tinker_utils.dataset_builder import Environment
        return Environment
    if name == "State":
        from ttt_discover.tinker_utils.state import State
        return State
    if name == "BaseRewardEvaluator":
        from ttt_discover.environments.base_reward_evaluator import BaseRewardEvaluator
        return BaseRewardEvaluator
    if name == "SandboxRewardEvaluator":
        from ttt_discover.environments.sandbox_reward_evaluator import SandboxRewardEvaluator
        return SandboxRewardEvaluator
    raise AttributeError(f"module 'ttt_discover' has no attribute {name!r}")

__all__ = [
    "Environment",
    "DiscoverConfig",
    "discover",
    "State",
    "BaseRewardEvaluator",
    "SandboxRewardEvaluator",
]
