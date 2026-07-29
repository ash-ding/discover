import logging
import numpy as np

from ttt_discover import Environment, SandboxRewardEvaluator, State, DiscoverConfig, discover

logger = logging.getLogger(__name__)


C5_MISMATCH_ATOL = 1e-4


def compute_c5(h: np.ndarray, n_points: int) -> float:
    dx = 2.0 / n_points
    return float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))


class ErdosMinOverlapRewardEvaluator(SandboxRewardEvaluator):
    def get_program_entrypoint(self) -> str:
        return "run"

    def preprocess_generation(self, generation, state) -> str:
        base = "import numpy as np\n\n"

        if state is None:
            raise ValueError(
                "state is required for preprocess_generation. "
                "Use ExperienceSampler to provide initial state with construction."
            )
        if state.construction is not None:
            base += f"initial_h_values = np.array({list(state.construction)!r})\n\n"

        return base + generation

    def get_reward(self, code: str, state: State) -> float:
        output, error_msg = self.execute_code(code, state)
        if error_msg:
            return self._get_failure_entry(error_msg)

        try:
            h_raw, c5_reported, n_points = output
        except (TypeError, ValueError):
            return self._get_failure_entry("Output must be (h_values, c5_bound, n_points) tuple.")

        try:
            h = np.array(h_raw, dtype=np.float64)
        except (ValueError, TypeError) as e:
            return self._get_failure_entry(f"Cannot convert h_values to numpy array: {e}")

        if h.ndim != 1 or h.shape[0] != n_points:
            return self._get_failure_entry(f"Expected 1-D h of length {n_points}, got shape {h.shape}.")
        if not np.all(np.isfinite(h)):
            return self._get_failure_entry("h_values contain NaN or inf values.")
        if np.any(h < 0) or np.any(h > 1):
            return self._get_failure_entry(f"h(x) not in [0, 1]. Range: [{h.min()}, {h.max()}].")

        raw_c5 = compute_c5(h, n_points)
        if not np.isfinite(raw_c5):
            return self._get_failure_entry(f"Computed C5 is not finite: {raw_c5}")
        if abs(raw_c5 - c5_reported) > C5_MISMATCH_ATOL:
            return self._get_failure_entry(
                f"C5 mismatch: reported {c5_reported:.6f}, computed {raw_c5:.6f}."
            )

        target_sum = n_points / 2.0
        actual_sum = float(np.sum(h))
        if actual_sum != target_sum:
            logger.warning(
                "Erdos h normalization triggered: sum(h)=%.6f, target=%.1f, diff=%+.6f. "
                "Model-reported C5=%.10f",
                actual_sum, target_sum, actual_sum - target_sum, c5_reported,
            )
            h = h * (target_sum / actual_sum)
            if np.any(h < 0) or np.any(h > 1):
                return self._get_failure_entry(
                    f"After normalization, h not in [0, 1]. Range: [{h.min()}, {h.max()}]."
                )

        verified_c5 = compute_c5(h, n_points)

        if verified_c5 > c5_reported + 1e-10:
            logger.warning(
                "Erdos C5 degraded after normalization: model-reported=%.10f, "
                "verified=%.10f, gap=%+.10f",
                c5_reported, verified_c5, verified_c5 - c5_reported,
            )

        return {
            "reward": float(1.0 / (1e-8 + verified_c5)),
            "correctness": 1.0,
            "raw_score": verified_c5,
            "msg": f"C5 bound: {verified_c5:.10f} (model reported: {c5_reported:.10f})",
            "result_construction": list(h),
            "stdout": getattr(self, '_last_stdout', ''),
        }


class ErdosMinOverlapEnv(Environment):
    reward_function = ErdosMinOverlapRewardEvaluator
    state_type = State
    max_construction_len = 1000

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        rng = np.random.default_rng()
        n_points = rng.integers(40, 100)
        construction = np.ones(n_points) * 0.5
        perturbation = rng.uniform(-0.4, 0.4, n_points)
        perturbation = perturbation - np.mean(perturbation)
        construction = construction + perturbation
        dx = 2.0 / n_points
        correlation = np.correlate(construction, 1 - construction, mode="full") * dx
        c5_bound = float(np.max(correlation))
        return State(timestep=-1, code="", value=-c5_bound, construction=list(construction))

    def is_maximize(self) -> bool:
        return False # Minimize upper bound

    def get_question(self) -> str:
        state = self.initial_state
        state_ctx = state.to_prompt(0.3808, metric_name="C₅ bound", maximize=False)
        
        # Construct construction section
        construction_section = ""
        if hasattr(state, 'construction') and state.construction is not None and len(state.construction) > 0:
            construction_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""

        # Construct code section
        if state.code and state.code.strip():
            code_section = '''Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.'''
        else:
            code_section = '''Write code to optimize this construction.'''

        # Construct final prompt
        return f'''You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

**Verification**: The verifier independently recomputes C₅ from your returned h.
- If the recomputed C₅ differs from your reported value by more than 1e-4, your solution is **rejected**.
- If `sum(h)` is not exactly `n_points / 2`, the verifier rescales h to enforce the constraint. This rescaling changes C₅ — optimizations that rely on a slightly invalid sum will lose their advantage after correction.

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: {self.eval_timeout}s for your code to run
- **CPUs**: {self.num_cpus_per_task} available

## Rules
- Define `run(seed=42, budget_s=1000, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `initial_h_values` (the current construction, if available) is pre-imported as a numpy array
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

{state_ctx}
{construction_section}
{code_section}
'''


def discover_erdos_min_overlap():
    config = DiscoverConfig(
        env_type=ErdosMinOverlapEnv,
        problem_type="",
        num_cpus_per_task=1,
        eval_timeout=1100,
        experiment_name=f"test-erdos-min-overlap-run",
        wandb_project="erdos-min-overlap",
    )

    discover(config)


def discover_erdos_min_overlap_local():
    config = DiscoverConfig(
        env_type=ErdosMinOverlapEnv,
        problem_type="",
        model_name="Qwen/Qwen3-8B",
        local_model_path="/workspace/home/asherding/models/Qwen3-8B",
        renderer_name="qwen3",
        use_local_backend=True,
        inference_gpu_id=0,
        training_gpu_id=4,
        inference_tp_size=4,
        max_model_len=32768,
        group_size=64,
        groups_per_batch=8,
        num_epochs=1,
        phase1_max_tokens=26000,
        kl_penalty_coef=0.1,
        lora_rank=32,
        learning_rate=4e-5,
        num_cpus_per_task=1,
        eval_timeout=1100,
        experiment_name="erdos-validate",
        wandb_project="erdos-min-overlap",
    )
    discover(config)


if __name__ == "__main__":
    import sys
    import os

    config_path = os.getenv("TTT_CONFIG_PATH")

    if "--local" in sys.argv or config_path:
        if config_path:
            # Load from YAML configuration
            from ttt_discover.utils.config_loader import load_config
            import inspect
            yaml_config = load_config(config_path)

            # Filter valid parameters for DiscoverConfig
            valid_params = inspect.signature(DiscoverConfig.__init__).parameters.keys()
            filtered_config = {k: v for k, v in yaml_config.items() if k in valid_params}

            config = DiscoverConfig(
                env_type=ErdosMinOverlapEnv,
                problem_type="",
                num_cpus_per_task=1,
                eval_timeout=1100,
                **filtered_config
            )
            discover(config)
        else:
            # Fallback to hardcoded version
            discover_erdos_min_overlap_local()
    else:
        discover_erdos_min_overlap()
