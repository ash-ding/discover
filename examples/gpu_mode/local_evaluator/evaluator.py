"""
LocalKernelEvaluator - Container-isolated GPU kernel evaluation with fault tolerance.

Key features:
- Container isolation (Podman/Docker) - GPU crashes don't affect host
- Automatic GPU recovery on failure
- Retry logic with exponential backoff
- Never raises exceptions (returns penalty reward instead)
- Supports both Podman (preferred) and Docker
"""

import os
import sys
import subprocess
import tempfile
import json
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)

ContainerRuntime = Literal["podman", "docker", "none"]


class LocalKernelEvaluator:
    """
    Local GPU kernel evaluator with container isolation (Podman/Docker).

    Runs kernel evaluation in isolated container to prevent GPU crashes
    from affecting training process.

    Container runtime priority: Podman > Docker > subprocess (no isolation)
    """

    def __init__(
        self,
        gpu_id: int = 5,
        timeout: int = 1200,
        max_retries: int = 2,
        penalty_score: float = -1_000_000,
        use_container: bool = True,
        container_runtime: Optional[ContainerRuntime] = None,
        # Backward compatibility (deprecated)
        use_docker: Optional[bool] = None,
    ):
        """
        Initialize evaluator.

        Args:
            gpu_id: GPU device ID to use for evaluation (default: 5)
            timeout: Maximum evaluation time in seconds (default: 1200 = 20min)
            max_retries: Number of retries on GPU crash (default: 2)
            penalty_score: Score returned on failure (default: -1,000,000)
            use_container: Whether to use container isolation (default: True)
            container_runtime: Force specific runtime ("podman"/"docker"/None=auto)
            use_docker: DEPRECATED - use 'use_container' instead
        """
        self.gpu_id = gpu_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.penalty_score = penalty_score

        # Backward compatibility: support old use_docker parameter
        if use_docker is not None:
            logger.warning(
                "Parameter 'use_docker' is deprecated, use 'use_container' instead. "
                "Auto-detection will choose Podman or Docker."
            )
            self.use_container = use_docker
        else:
            self.use_container = use_container

        # Path to worker script
        self.worker_script = Path(__file__).parent / "eval_worker.py"

        # Container image name (same for both Podman and Docker)
        self.container_image = "gpu-kernel-evaluator:latest"

        # Set Podman storage config to use custom location (workspace partition)
        # This must be set before any podman commands
        self._setup_podman_storage()

        # Detect container runtime
        self.container_runtime = self._detect_container_runtime(container_runtime)

        # Verify setup
        self._verify_setup()

    def _setup_podman_storage(self):
        """
        Configure Podman to use custom storage location on workspace partition.

        This avoids filling up the root partition (only 8GB free) by storing
        container images on the workspace partition (400GB+ free).
        """
        import os

        # Path to project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent
        storage_conf = project_root / ".podman_storage" / "storage.conf"

        if storage_conf.exists():
            # Set environment variable for Podman to use custom storage
            os.environ["CONTAINERS_STORAGE_CONF"] = str(storage_conf)
            logger.debug(f"Podman storage config: {storage_conf}")
        else:
            logger.debug(f"Podman storage config not found at {storage_conf}, using default")

    def _detect_container_runtime(
        self,
        forced_runtime: Optional[ContainerRuntime]
    ) -> ContainerRuntime:
        """
        Detect available container runtime.

        Priority: forced > podman > docker > none

        Args:
            forced_runtime: Force specific runtime (for testing/debugging)

        Returns:
            "podman", "docker", or "none"
        """
        if not self.use_container:
            return "none"

        # If user forced a specific runtime, use it
        if forced_runtime in ["podman", "docker"]:
            if shutil.which(forced_runtime):
                logger.info(f"Using forced container runtime: {forced_runtime}")
                return forced_runtime
            else:
                logger.warning(f"Forced runtime '{forced_runtime}' not found, falling back to auto-detect")

        # Auto-detect: prefer Podman (no daemon, more secure)
        for runtime in ["podman", "docker"]:
            if shutil.which(runtime):
                try:
                    result = subprocess.run(
                        [runtime, "info"],
                        capture_output=True,
                        timeout=5,
                        text=True
                    )
                    if result.returncode == 0:
                        logger.info(f"✓ Using container runtime: {runtime}")
                        return runtime
                except Exception as e:
                    logger.debug(f"{runtime} check failed: {e}")
                    continue

        # No container runtime available
        logger.warning("Neither Podman nor Docker available, falling back to subprocess mode (no isolation)")
        return "none"

    def _verify_setup(self):
        """Verify GPU and container runtime are available."""
        # Check GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_id), "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"GPU {self.gpu_id} not available")
            gpu_name = result.stdout.strip()
            logger.info(f"✓ Local evaluator using GPU {self.gpu_id}: {gpu_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to detect GPU {self.gpu_id}: {e}")

        # Check container image if using container
        if self.container_runtime in ["podman", "docker"]:
            try:
                # Check if image exists
                result = subprocess.run(
                    [self.container_runtime, "images", "-q", self.container_image],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if not result.stdout.strip():
                    logger.warning(
                        f"{self.container_runtime.capitalize()} image '{self.container_image}' not found.\n"
                        f"Run 'bash build_container.sh' to build it."
                    )
                    logger.warning("Falling back to subprocess mode (no isolation).")
                    self.container_runtime = "none"
                else:
                    logger.info(f"✓ Container image '{self.container_image}' ready")
            except Exception as e:
                logger.warning(f"Container image check failed: {e}, falling back to subprocess mode")
                self.container_runtime = "none"

    def evaluate(
        self,
        submission_code: str,
        task_name: str = "trimul",
        gpu_type: str = "H100"
    ) -> Dict[str, Any]:
        """
        Evaluate kernel code with fault tolerance.

        Args:
            submission_code: Kernel code to evaluate
            task_name: Task name ("trimul" or "mla_decode_nvidia")
            gpu_type: GPU type for logging purposes

        Returns:
            dict: {
                "success": bool,
                "score_us": float,  # Runtime in microseconds (geometric mean)
                "error": Optional[str],
                "stdout": str,
                "stderr": str
            }
        """
        for attempt in range(1 + self.max_retries):
            try:
                # Run evaluation
                result = self._run_evaluation_worker(
                    submission_code, task_name, gpu_type
                )

                if result["success"]:
                    return result

                # Check if this is a GPU crash
                if not self._is_gpu_crash(result):
                    # Non-crash failure (e.g., correctness test failed)
                    return result

                # GPU crash detected - try recovery
                if attempt < self.max_retries:
                    logger.warning(f"⚠ GPU crash detected, attempting recovery ({attempt + 1}/{self.max_retries})...")
                    self._recover_gpu()
                    time.sleep(2)  # Wait for GPU to stabilize
                    continue
                else:
                    logger.error(f"✗ GPU recovery failed after {self.max_retries} attempts")
                    return self._make_failure_result("GPU crash, recovery failed")

            except subprocess.TimeoutExpired:
                logger.warning(f"⚠ Evaluation timeout ({self.timeout}s)")
                return self._make_failure_result(f"Timeout after {self.timeout}s")

            except Exception as e:
                logger.error(f"⚠ Unexpected error: {e}")
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                return self._make_failure_result(f"Unexpected error: {e}")

        return self._make_failure_result("Max retries exceeded")

    def _run_evaluation_worker(
        self,
        submission_code: str,
        task_name: str,
        gpu_type: str
    ) -> Dict[str, Any]:
        """
        Run evaluation worker in Docker or subprocess.

        Uses temporary directory to pass code and receive results.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write submission code
            submission_file = tmpdir / "submission.py"
            submission_file.write_text(submission_code)

            # Write config (use container-relative paths for container mode)
            if self.container_runtime in ["podman", "docker"]:
                config = {
                    "submission_file": "/workspace/data/submission.py",
                    "task_name": task_name,
                    "gpu_type": gpu_type,
                    "result_file": "/workspace/data/result.json"
                }
            else:
                config = {
                    "submission_file": str(submission_file),
                    "task_name": task_name,
                    "gpu_type": gpu_type,
                    "result_file": str(tmpdir / "result.json")
                }
            config_file = tmpdir / "config.json"
            config_file.write_text(json.dumps(config, indent=2))

            # Run worker
            if self.container_runtime in ["podman", "docker"]:
                stdout, stderr, returncode = self._run_container_worker(tmpdir, config_file)
            else:
                stdout, stderr, returncode = self._run_subprocess_worker(tmpdir, config_file)

            # Read result
            result_file = tmpdir / "result.json"
            if result_file.exists():
                try:
                    result = json.loads(result_file.read_text())
                    result["stdout"] = stdout
                    result["stderr"] = stderr
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse result.json: {e}")

            # Worker crashed without writing result
            return {
                "success": False,
                "score_us": self.penalty_score,
                "error": "Worker process failed to write result",
                "stdout": stdout,
                "stderr": stderr,
                "gpu_crash": True
            }

    def _run_container_worker(
        self,
        tmpdir: Path,
        config_file: Path
    ) -> tuple[str, str, int]:
        """Run worker in container (Podman or Docker)."""
        # Get absolute path to lib directory (contains task definitions)
        lib_dir = Path(__file__).parent.parent / "lib"

        # Build container command
        # Note: --cpus and --memory omitted because cgroup cpu controller
        # is unavailable on the eval nodes (rootless podman limitation)
        container_cmd = [
            self.container_runtime, "run",
            "--rm",  # Remove container after run
            "--network", "none",  # No network access (security)
            "-e", f"EVAL_TIMEOUT={self.timeout}",  # Pass timeout to worker
            "-v", f"{tmpdir}:/workspace/data",  # Mount temp dir
            "-v", f"{lib_dir}:/workspace/lib:ro",  # Mount lib dir (read-only)
        ]

        # GPU access: different syntax for Podman vs Docker
        if self.container_runtime == "podman":
            container_cmd.extend([
                "--device", f"nvidia.com/gpu={self.gpu_id}"  # Podman syntax
            ])
        else:  # docker
            container_cmd.extend([
                "--gpus", f"device={self.gpu_id}"  # Docker syntax
            ])

        # Add image and config path (ENTRYPOINT already runs eval_worker.py)
        container_cmd.extend([
            self.container_image,
            "/workspace/data/config.json"
        ])

        logger.debug(f"Running {self.container_runtime}: {' '.join(container_cmd)}")

        try:
            proc = subprocess.Popen(
                container_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)
            return stdout, stderr, proc.returncode

        except subprocess.TimeoutExpired:
            # Kill container
            proc.kill()
            proc.wait()
            raise

    def _run_subprocess_worker(
        self,
        tmpdir: Path,
        config_file: Path
    ) -> tuple[str, str, int]:
        """Run worker in subprocess (fallback when Docker unavailable)."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["EVAL_TIMEOUT"] = str(self.timeout)

        # Fix CUDA multiprocessing issues
        # eval.py uses multiprocessing.pool which forks processes
        # CUDA context cannot be shared across fork, so we disable CUDA in parent
        # This forces each child process to initialize CUDA fresh
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Critical: Unset any CUDA env vars that might cause issues in multiprocessing
        # The child processes will re-initialize CUDA with CUDA_VISIBLE_DEVICES
        # Remove these to prevent "already initialized" errors
        for key in list(env.keys()):
            if key.startswith("CUDA_") and key != "CUDA_VISIBLE_DEVICES":
                del env[key]

        # Ensure conda python is used for all subprocess calls
        conda_bin = str(Path(sys.executable).parent)
        conda_base = str(Path(sys.executable).parent.parent)

        # Prepend conda bin to PATH (so 'python3' resolves to conda python)
        env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")

        # Set PYTHONPATH to include conda site-packages
        # This ensures child processes (python3 eval.py) can find torch, etc.
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages = str(Path(conda_base) / "lib" / python_version / "site-packages")
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = site_packages + os.pathsep + existing_pythonpath
        else:
            env["PYTHONPATH"] = site_packages

        cmd = [sys.executable, str(self.worker_script), str(config_file)]

        logger.debug(f"Running subprocess: {' '.join(cmd)}")
        logger.debug(f"PATH: {env['PATH'][:200]}...")
        logger.debug(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')[:200]}...")

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(tmpdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)
            return stdout, stderr, proc.returncode

        except subprocess.TimeoutExpired:
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise

    def _is_gpu_crash(self, result: Dict[str, Any]) -> bool:
        """Check if result indicates GPU crash."""
        if result.get("gpu_crash"):
            return True

        # Check stderr for GPU error keywords
        stderr = result.get("stderr", "")
        gpu_error_keywords = [
            "CUDA error",
            "CUDA out of memory",
            "illegal memory access",
            "cudaGetLastError",
            "Segmentation fault",
            "CUDA_ERROR",
        ]
        return any(kw in stderr for kw in gpu_error_keywords)

    def _recover_gpu(self):
        """Attempt to recover GPU after crash."""
        logger.info(f"🔧 Attempting to reset GPU {self.gpu_id}...")

        try:
            # Method 1: Kill processes on GPU
            try:
                subprocess.run(
                    ["fuser", "-k", f"/dev/nvidia{self.gpu_id}"],
                    capture_output=True,
                    timeout=10
                )
                time.sleep(1)
            except:
                pass

            # Method 2: GPU reset (if supported)
            try:
                subprocess.run(
                    ["nvidia-smi", "--gpu-reset", "-i", str(self.gpu_id)],
                    capture_output=True,
                    timeout=10,
                    check=False
                )
            except:
                pass

            time.sleep(2)

            # Verify GPU is responsive
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_id)],
                capture_output=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"✓ GPU {self.gpu_id} recovered")
            else:
                logger.warning(f"✗ GPU {self.gpu_id} recovery verification failed")

        except Exception as e:
            logger.error(f"✗ GPU recovery error: {e}")

    def _make_failure_result(self, error_msg: str) -> Dict[str, Any]:
        """Create failure result."""
        return {
            "success": False,
            "score_us": self.penalty_score,
            "error": error_msg,
            "stdout": "",
            "stderr": error_msg
        }
