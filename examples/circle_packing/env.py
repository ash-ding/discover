import inspect
import numpy as np
import os
from pathlib import Path

from ttt_discover import Environment, SandboxRewardEvaluator, State, DiscoverConfig, discover
from ttt_discover.utils.config_loader import load_config


# Verifier for circle packing
def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True


def check_packing_correctness(centers, radii, num_circles: int) -> bool:
    shape_valid = centers.shape == (num_circles, 2) and radii.shape == (num_circles,)
    if not shape_valid:
        return False

    return validate_packing(centers, radii)


# Task for running circle packing search programs
class CirclePackingReward(SandboxRewardEvaluator):

    def get_program_entrypoint(self) -> str:
        return "run_packing"

    # Just define get reward.
    def get_reward(self, code: str, state: State) -> float:
        output, error_msg = self.execute_code(code, state)
        if error_msg: 
            return self._get_failure_entry(error_msg)

        # Extract output
        centers, radii, _ = output
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        # Check if packing is valid
        if not check_packing_correctness(centers, radii, int(self.problem_type)):
            return self._get_failure_entry("Packing is not valid.")
        
        # Final reward is sum of radii
        sum_of_radii = np.sum(radii)
        return {
            "reward": sum_of_radii,
            "correctness": 1.0,
            "raw_score": sum_of_radii,
            "msg": f"Success; raw_score={sum_of_radii}",
            "result_construction": [], # Do not reuse construction
            "stdout": getattr(self, '_last_stdout', ''),
        }


class CirclePackingEnv(Environment):
    reward_function = CirclePackingReward
    state_type = State

    def get_question(self) -> str:
        """Build prompt from template"""
        assert self.problem_type in {"26", "32"}
        validator_src = inspect.getsource(validate_packing)
        # Assume we do 26
        target = 2.636 if self.problem_type == "26" else 2.940
        state_ctx = self.initial_state.to_prompt(target, metric_name="sum of radii")

        return f"""You are an expert mathematician specializing in circle packing problems and computational geometry.

Your task is to pack {self.problem_type} circles in a unit square [0,1]×[0,1] to maximize the sum of radii.

We will run the below validation function (read-only, do not modify this):
```python
{validator_src}
```

{state_ctx}

Reason about how you could further improve this packing. Consider:
- Are circles placed optimally near boundaries and corners?
- Could a different arrangement (hexagonal, nested, hybrid) yield better results?
- Are there gaps that could be filled with repositioned or resized circles?
- Could optimization parameters or methods be improved?

Rules:
- You must define the run_packing function: def run_packing() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape ({self.problem_type}, 2) and radii has shape ({self.problem_type},).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- Centers must lie within [0,1]^2 and radii must be nonnegative.
- The pair (centers, radii) must satisfy non-overlap and boundary constraints.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- You need to get really creative and think from first principles.

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""


def discover_circle_packing(num_circles: str):
    # Uses default values for most fields
    config = DiscoverConfig(
        env_type=CirclePackingEnv,
        problem_type=num_circles,
        num_cpus_per_task=1,
        eval_timeout=530,
        experiment_name=f"test-circle-packing-{num_circles}-run",
        wandb_project="circle-packing",
    )

    # Run discovery
    discover(config)


def discover_circle_packing_local(num_circles: str, config_path: str = None):
    # Load configuration from YAML file if provided
    if config_path and os.path.exists(config_path):
        import inspect
        print(f"Loading configuration from: {config_path}")
        yaml_config = load_config(config_path)

        # Use num_circles from YAML if present, override problem_type
        if 'num_circles' in yaml_config:
            num_circles = str(yaml_config.pop('num_circles'))
        # Remove non-DiscoverConfig keys
        yaml_config.pop('target_score', None)

        # Filter to valid DiscoverConfig parameters
        valid_params = set(inspect.signature(DiscoverConfig.__init__).parameters.keys())
        filtered_config = {k: v for k, v in yaml_config.items() if k in valid_params}

        config = DiscoverConfig(
            env_type=CirclePackingEnv,
            problem_type=num_circles,
            **filtered_config,
        )
    else:
        # Fallback to hardcoded paper configuration
        print("Using hardcoded paper configuration (Table 9)")
        config = DiscoverConfig(
            env_type=CirclePackingEnv,
            problem_type=num_circles,
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
            num_epochs=50,
            phase1_max_tokens=26000,
            kl_penalty_coef=0.1,
            lora_rank=32,
            learning_rate=4e-5,
            save_every=2,
            num_cpus_per_task=1,
            eval_timeout=530,
            experiment_name=f"circle-packing-{num_circles}",
            wandb_project="circle-packing",
        )
    discover(config)


def discover_circle_packing_validate(num_circles: str):
    # Paper-exact validation configuration (1 epoch only)
    # Using TP=4 for inference (GPUs 0-3) + single GPU training (GPU 4)
    # All parameters match Table 9 from TTT-Discover paper
    config = DiscoverConfig(
        env_type=CirclePackingEnv,
        problem_type=num_circles,
        model_name="Qwen/Qwen3-8B",
        local_model_path="/workspace/home/asherding/models/Qwen3-8B",
        renderer_name="qwen3",
        use_local_backend=True,
        inference_gpu_id=0,
        training_gpu_id=4,
        inference_tp_size=4,
        max_model_len=32768,
        group_size=64,              # Paper value
        groups_per_batch=8,         # Paper value
        num_epochs=1,               # Reduced from 50 for validation only
        phase1_max_tokens=26000,    # Paper value
        kl_penalty_coef=0.1,        # Paper value
        lora_rank=32,               # Paper value
        learning_rate=4e-5,         # Paper value
        save_every=1,
        num_cpus_per_task=1,
        eval_timeout=530,
        experiment_name=f"circle-packing-{num_circles}-paper-validate",
        wandb_project="circle-packing",
    )
    discover(config)


if __name__ == "__main__":
    import sys

    # Check for config file from environment variable
    config_path = os.getenv("TTT_CONFIG_PATH")

    if "--validate" in sys.argv:
        n = sys.argv[sys.argv.index("--validate") + 1] if len(sys.argv) > sys.argv.index("--validate") + 1 else "26"
        discover_circle_packing_validate(n)
    elif "--local" in sys.argv:
        n = sys.argv[sys.argv.index("--local") + 1] if len(sys.argv) > sys.argv.index("--local") + 1 else "26"
        discover_circle_packing_local(n, config_path)
    else:
        num_circles = "26" # or "32"
        discover_circle_packing(num_circles)