import signal
import subprocess
import traceback
from contextlib import contextmanager
from typing import Optional

import modal  # pyright: ignore[reportMissingImports]
import structlog

from libkernelbot.run_eval import FullResult, SystemInfo, run_config

logger = structlog.get_logger(__name__)

class TimeoutException(Exception):
    pass


class ModalRequeueRequest(Exception):
    """Raise to force Modal to retry (requeue) the call."""

_REQUEUE_SENTINEL = "[MODAL_REQUEUE]"

# If we detect any GPU with one of these reported `nvidia-smi` names, we refuse to run.
# For consistency with the backend, we only allow A100 PCIe 80GB and H100 HBM3 PCIe 80GB.
_BANNED_GPU_NAMES = {
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 NVL",
}

_REQUEUE_COUNT_DICT_NAME = "discord-bot-requeue-counts"
_requeue_counts = None


def _get_requeue_counts():
    global _requeue_counts
    if _requeue_counts is None:
        logger.debug("initializing_requeue_counts_dict")
        _requeue_counts = modal.Dict.from_name(_REQUEUE_COUNT_DICT_NAME, create_if_missing=True)
    return _requeue_counts


def _get_request_id(config: dict) -> Optional[str]:
    rid = config.get("_modal_request_id")
    if isinstance(rid, str) and rid.strip():
        return rid
    return None


def _increment_requeue_count(request_id: str) -> int:
    d = _get_requeue_counts()
    try:
        current = int(d[request_id])
    except KeyError:
        current = 0
    current += 1
    d[request_id] = current
    logger.debug("requeue_count_incremented", request_id=request_id, count=current)
    return current


def _pop_requeue_count(request_id: str) -> int:
    d = _get_requeue_counts()
    try:
        value = d.pop(request_id)
    except KeyError:
        value = 0
    logger.debug("requeue_count_popped", request_id=request_id, value=int(value or 0))
    return int(value or 0)


def _detect_nvidia_gpu_names() -> list[str]:
    logger.debug("detecting_nvidia_gpu_names")
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    logger.info("detected_gpu_names", names=names)
    return names


@contextmanager
def timeout(seconds: int):
    """Context manager that raises TimeoutException after specified seconds"""

    def timeout_handler(signum, frame):
        raise TimeoutException(f"Script execution timed out after {seconds} seconds")

    # Set up the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def modal_run_config(  # noqa: C901
    config: dict,
    timeout_seconds: int = 1200,
) -> FullResult:
    request_id = _get_request_id(config)
    logger.info("modal_run_config_start", request_id=request_id, timeout=timeout_seconds)
    try:
        gpu_names = _detect_nvidia_gpu_names()
        if any(name in _BANNED_GPU_NAMES for name in gpu_names):
            logger.warning("banned_gpu_detected", gpu_names=gpu_names)
            attempt = None
            if request_id is not None:
                attempt = _increment_requeue_count(request_id)
            # Use a built-in exception type so the local caller can always deserialize it.
            # (Custom exceptions require the same module to exist locally.)
            attempt_msg = ""
            if attempt is not None:
                # Modal's max_retries counts requeues, not total attempts.
                attempt_msg = f" (attempt={attempt}, requeues_so_far={max(0, attempt - 1)})"
            raise RuntimeError(
                f"{_REQUEUE_SENTINEL} Refusing to run on banned GPU(s) {gpu_names}{attempt_msg}; requeueing request."
            )

        with timeout(timeout_seconds):
            result = run_config(config)
        if request_id is not None:
            result.system.requeues = _pop_requeue_count(request_id)
        return result
    except RuntimeError as e:
        if str(e).startswith(_REQUEUE_SENTINEL):
            logger.info("requeue_sentinel_propagated", request_id=request_id)
            raise
        requeues = _pop_requeue_count(request_id) if request_id is not None else 0
        logger.error("modal_run_runtime_error", request_id=request_id, error=str(e))
        exception = "".join(traceback.format_exception(e))
        return FullResult(
            success=False,
            error=f"Error executing script:\n{exception}",
            runs={},
            system=SystemInfo(requeues=requeues),
        )
    except TimeoutException as e:
        requeues = _pop_requeue_count(request_id) if request_id is not None else 0
        logger.error("modal_run_timeout", request_id=request_id, error=str(e))
        return FullResult(
            success=False,
            error=f"Timeout Error: {str(e)}",
            runs={},
            system=SystemInfo(requeues=requeues),
        )
    except BaseException as e:
        requeues = _pop_requeue_count(request_id) if request_id is not None else 0
        logger.error("modal_run_base_exception", request_id=request_id,
                      error_type=type(e).__name__, error=str(e))
        exception = "".join(traceback.format_exception(e))
        return FullResult(
            success=False,
            error=f"Error executing script:\n{exception}",
            runs={},
            system=SystemInfo(requeues=requeues),
        )
