from pathlib import Path
import os

from ttt_discover import Environment, BaseRewardEvaluator, State, DiscoverConfig, discover

# Import local evaluator (only local evaluation supported)
from examples.gpu_mode.local_evaluator import LocalKernelEvaluator

# Import prompts
from examples.gpu_mode.prompt import (
    TRIMUL_PROMPT,
    MLA_DECODE_PROMPT,
    MLA_DECODE_PROMPT_END,
)


def get_gpu_mode_error(msg: str) -> dict:
    return {
        "reward": 0.0,
        "msg": msg,
        "correctness": 0.0,
        "raw_score": -1_000_000,
        "result_construction": [],
        "stdout": "",
    }


class GpuModeRewardEvaluator(BaseRewardEvaluator):

    def __init__(self, *args, **kwargs):
        self.problem_type = kwargs.get("problem_type")
        self.log_dir = kwargs.get("log_dir")

        # Only local evaluation supported
        self.eval_mode = "local"

        if self.problem_type == "trimul":
            self.score_scale = 1500
            self.task_name = "trimul"
        elif self.problem_type == "mla_decode_nvidia":
            self.score_scale = 5000
            self.task_name = "mla_decode_nvidia"
        else:
            raise ValueError(f"Unknown problem_type: {self.problem_type}")

        # Initialize local evaluator
        # Priority: 1) kwargs from config, 2) environment variable, 3) default
        gpu_id = kwargs.get("kernel_eval_gpu") or int(os.getenv("KERNEL_EVAL_GPU", "5"))
        timeout = kwargs.get("kernel_eval_timeout") or int(os.getenv("KERNEL_EVAL_TIMEOUT", "1200"))
        max_retries = kwargs.get("kernel_eval_retries") or int(os.getenv("KERNEL_EVAL_RETRIES", "2"))

        # For container isolation, check both old and new parameter names
        # Priority: kernel_eval_use_container > kernel_eval_use_docker > env vars > default
        use_container_config = kwargs.get("kernel_eval_use_container")
        use_docker_config = kwargs.get("kernel_eval_use_docker")  # Backward compatibility

        if use_container_config is not None:
            use_container = use_container_config
        elif use_docker_config is not None:
            use_container = use_docker_config  # Use old parameter
        else:
            # Check environment variables (new > old)
            use_container_env = os.getenv("KERNEL_EVAL_USE_CONTAINER")
            if use_container_env:
                use_container = use_container_env.lower() == "true"
            else:
                use_container = os.getenv("KERNEL_EVAL_USE_DOCKER", "true").lower() == "true"

        self.local_evaluator = LocalKernelEvaluator(
            gpu_id=gpu_id,
            timeout=timeout,
            max_retries=max_retries,
            use_container=use_container,
        )
        print(f"✓ Local evaluator initialized (GPU {gpu_id}, container={'enabled' if use_container else 'disabled'})")

    def get_reward(self, code: str, state: State) -> dict:
        # Prevent no triton kernel code
        if "@triton.jit" not in code:
            return get_gpu_mode_error("Code must contain @triton.jit.")
        # Prevent identity kernel for trimul
        if self.problem_type == "trimul" and "identity" in code:
            return get_gpu_mode_error("Identity kernel is not allowed.")

        # Only local evaluation supported
        return self._evaluate_local(code)

    def _evaluate_local(self, code: str) -> dict:
        """Evaluate using local GPU (never raises exception)."""
        try:
            result = self.local_evaluator.evaluate(
                submission_code=code,
                task_name=self.task_name,
                gpu_type="H100"  # Local evaluation uses H100
            )

            if not result["success"]:
                # Evaluation failed - return penalty
                return {
                    "reward": 0.0,
                    "msg": f"Local evaluation failed: {result['error']}",
                    "correctness": 0.0,
                    "raw_score": result["score_us"],
                    "result_construction": [],
                    "stdout": result.get("stdout", ""),
                }

            # Success
            score_us = result["score_us"]
            reward = self.score_scale / score_us

            return {
                "reward": float(reward),
                "msg": f"Success! Runtime: {score_us:.2f} μs (local evaluation)",
                "correctness": 1.0,
                "raw_score": float(score_us),
                "result_construction": [],
                "stdout": result.get("stdout", ""),
            }

        except Exception as e:
            # Even if evaluator crashes, never raise exception
            import traceback
            print(f"✗ Fatal local evaluation error: {e}")
            traceback.print_exc()

            return {
                "reward": 0.0,
                "msg": f"Fatal local evaluation error: {str(e)}",
                "correctness": 0.0,
                "raw_score": -1_000_000,
                "result_construction": [],
                "stdout": "",
            }



class GpuModeEnv(Environment):
    reward_function = GpuModeRewardEvaluator
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        if problem_type == "mla_decode_nvidia":
            from examples.gpu_mode.prompt import MLA_DECODE_INITIAL_STATE, MLA_DECODE_INITIAL_VALUE
            return State(timestep=-1, code=MLA_DECODE_INITIAL_STATE, value=MLA_DECODE_INITIAL_VALUE, construction=None)
        if problem_type == "trimul":
            return State(timestep=-1, code="", value=-1_000_000, construction=None)
        raise ValueError(f"Unknown problem_type: {problem_type}")

    def _should_keep_code_separators(self) -> bool:
        return False
    
    def is_maximize(self) -> bool:
        return False

    def get_question(self) -> str:
        """Build prompt from template, injecting previous code from state."""
        state = self.initial_state
        target = 1000 if self.problem_type == "trimul" else 1700

        state_ctx = state.to_prompt(target, metric_name="runtime (microseconds)", maximize=False, language="python")

        if self.problem_type == "trimul":
            return f"""{TRIMUL_PROMPT}

{state_ctx}

Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use trition 3.3.1 and these kernels will be run on an H100.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
"""

        if self.problem_type == "mla_decode_nvidia":
            
            return f"""{MLA_DECODE_PROMPT}

{state_ctx}

{MLA_DECODE_PROMPT_END}
"""

        raise ValueError(
            f"Unknown problem_type: {self.problem_type}. "
            "Must be 'trimul' or 'mla_decode_nvidia'"
        )


def discover_gpu_mode(problem_type: str):
    config = DiscoverConfig(
        env_type=GpuModeEnv,
        problem_type=problem_type,
        eval_timeout=530,
        experiment_name=f"test-gpu-mode-{problem_type}-run",
        wandb_project="gpu-mode",
    )
    discover(config)


def discover_gpu_mode_local(problem_type: str):
    # Load config from YAML if TTT_CONFIG_PATH is set
    import os
    config_path = os.getenv("TTT_CONFIG_PATH")
    if config_path:
        from ttt_discover.utils.config_loader import load_config
        import inspect
        cfg = load_config(config_path)

        # Extract GPU evaluation parameters (not in DiscoverConfig)
        gpu_eval_params = {}
        for key in ['gpu_eval_mode', 'kernel_eval_gpu', 'kernel_eval_timeout',
                    'kernel_eval_retries', 'kernel_eval_use_docker', 'kernel_eval_use_container']:
            if key in cfg:
                gpu_eval_params[key] = cfg[key]

        # Set environment variables for evaluator to pick up
        if 'gpu_eval_mode' in gpu_eval_params:
            os.environ['GPU_EVAL_MODE'] = gpu_eval_params['gpu_eval_mode']
        if 'kernel_eval_gpu' in gpu_eval_params:
            os.environ['KERNEL_EVAL_GPU'] = str(gpu_eval_params['kernel_eval_gpu'])
        if 'kernel_eval_use_container' in gpu_eval_params:
            os.environ['KERNEL_EVAL_USE_CONTAINER'] = str(gpu_eval_params['kernel_eval_use_container']).lower()
        elif 'kernel_eval_use_docker' in gpu_eval_params:
            # Backward compatibility
            os.environ['KERNEL_EVAL_USE_DOCKER'] = str(gpu_eval_params['kernel_eval_use_docker']).lower()

        # Filter out keys not in DiscoverConfig
        valid_keys = set(inspect.signature(DiscoverConfig.__init__).parameters.keys())
        cfg_filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        cfg_filtered['env_type'] = GpuModeEnv
        cfg_filtered['problem_type'] = problem_type
        config = DiscoverConfig(**cfg_filtered)
    else:
        print("WARNING: TTT_CONFIG_PATH not set, using fallback config")
        config = DiscoverConfig(
            env_type=GpuModeEnv,
            problem_type=problem_type,
            model_name="Qwen/Qwen3-8B",
            local_model_path="/workspace/home/asherding/models/Qwen3-8B",
            renderer_name="qwen3",
            use_local_backend=True,
            inference_gpu_id=0,
            training_gpu_id=4,
            group_size=64,
            groups_per_batch=8,
            num_epochs=1,
            phase1_max_tokens=26000,
            kl_penalty_coef=0.1,
            lora_rank=32,
            learning_rate=4e-5,
            eval_timeout=530,
            experiment_name=f"gpu-mode-{problem_type}-fallback",
            wandb_project="gpu-mode",
        )
    discover(config)


if __name__ == "__main__":
    import sys
    if "--local" in sys.argv:
        problem = "mla_decode_nvidia" if "--mla" in sys.argv else "trimul"
        discover_gpu_mode_local(problem)
    else:
        discover_gpu_mode("trimul")
        # discover_gpu_mode("mla_decode_nvidia")