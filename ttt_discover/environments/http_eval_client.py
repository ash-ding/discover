"""HTTP evaluation client for GPU tasks (trimul, mla_decode_nvidia).

Routes GPU kernel evaluation through an HTTP eval server, providing
consistent error handling, retry logic, and structured diagnostics.
CPU tasks (circle_packing, ac, erdos, denoising, ahc) use
SandboxRewardEvaluator instead.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = frozenset({"trimul", "mla_decode_nvidia"})

SERVER_TASK_NAMES = {
    "mla_decode_nvidia": "mla_decode",
}

_RETRYABLE_ERROR_TYPES = frozenset({"infra_failure"})
_NON_RETRYABLE_ERROR_TYPES = frozenset({"eval_failure", "timeout", "compilation_error"})

REQUIRED_RESPONSE_FIELDS = ("success", "score_us")


@dataclass
class EvalResult:
    success: bool
    score_us: float
    error: Optional[str] = None
    error_type: Optional[str] = None
    logs: dict = field(default_factory=dict)
    test_results: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)

    @property
    def is_retryable(self) -> bool:
        return self.error_type in _RETRYABLE_ERROR_TYPES


class HttpEvalClient:
    """HTTP client for GPU kernel evaluation via POST /eval endpoint.

    Used by GpuModeRewardEvaluator for trimul and mla_decode_nvidia tasks.
    Retries indefinitely on queue-full (HTTP 503) and transient network
    errors with exponential backoff + jitter, until the server returns a
    definitive evaluation result (success, eval_failure, timeout, or
    compilation_error).
    """

    def __init__(
        self,
        server_url: str | None = None,
        timeout: int = 600,
        max_retries: int = 2,
        penalty_score: float = 0.0,
        retry_base_delay: float = 5.0,
        retry_max_delay: float = 60.0,
    ):
        self.server_url = (
            server_url
            or os.environ.get("EVAL_SERVER_URL", "http://localhost:8080")
        ).rstrip("/")
        if not self.server_url.startswith("http"):
            self.server_url = f"http://{self.server_url}"
        self.timeout = timeout
        self.max_retries = max_retries  # kept for backward compat; unused
        self.penalty_score = penalty_score
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    def evaluate(
        self,
        code: str,
        task_name: str,
        timeout: int | None = None,
        extra_params: dict | None = None,
    ) -> EvalResult:
        """Submit code for evaluation, retrying until a definitive result.

        Retries indefinitely on queue-full (HTTP 503) and transient errors.
        Returns immediately when the server produces a real evaluation
        result — whether success, eval_failure, timeout, or compilation_error.
        """
        req_timeout = timeout or self.timeout
        server_task_name = SERVER_TASK_NAMES.get(task_name, task_name)
        payload: dict[str, Any] = {
            "code": code,
            "task_name": server_task_name,
            "timeout": req_timeout,
        }
        if extra_params:
            payload.update(extra_params)

        data = json.dumps(payload).encode()
        url = f"{self.server_url}/eval"

        attempt = 0
        last_error: str | None = None
        t_start = time.monotonic()

        while True:
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                resp_bytes = urllib.request.urlopen(req, timeout=req_timeout).read()
                resp = json.loads(resp_bytes)
                result = self._parse_response(resp)

                if not result.is_retryable:
                    if attempt > 0:
                        logger.info(
                            "Eval succeeded after %d retries (%.0fs): task=%s",
                            attempt, time.monotonic() - t_start, task_name,
                        )
                    return result

                last_error = result.error
                self._log_retry(attempt, t_start, last_error)
                self._backoff_sleep(attempt)
                attempt += 1

            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                last_error = f"{type(e).__name__}: {e}"
                self._log_retry(attempt, t_start, last_error)
                self._backoff_sleep(attempt)
                attempt += 1

            except (json.JSONDecodeError, ValueError) as e:
                return EvalResult(
                    success=False,
                    score_us=self.penalty_score,
                    error=f"Response parse error: {e}",
                    error_type="infra_failure",
                )

    def _backoff_sleep(self, attempt: int) -> None:
        delay = min(self.retry_base_delay * (2 ** min(attempt, 10)), self.retry_max_delay)
        delay *= 0.5 + random.random()
        time.sleep(delay)

    def _log_retry(self, attempt: int, t_start: float, last_error: str | None) -> None:
        elapsed = time.monotonic() - t_start
        if attempt < 3 or attempt % 5 == 0:
            logger.warning(
                "Eval retry attempt %d (%.0fs elapsed): %s",
                attempt + 1, elapsed, last_error,
            )

    def _parse_response(self, resp: dict) -> EvalResult:
        """Parse and validate the eval server response."""
        for field_name in REQUIRED_RESPONSE_FIELDS:
            if field_name not in resp:
                return EvalResult(
                    success=False,
                    score_us=self.penalty_score,
                    error=f"Missing required field: {field_name}",
                    error_type="infra_failure",
                    raw_response=resp,
                )

        success = bool(resp["success"])
        score_us = float(resp.get("score_us", 0.0))

        if success and score_us <= 0:
            return EvalResult(
                success=False,
                score_us=self.penalty_score,
                error=f"Success=true but score_us={score_us} <= 0",
                error_type="eval_failure",
                raw_response=resp,
            )

        if success and not math.isfinite(score_us):
            return EvalResult(
                success=False,
                score_us=self.penalty_score,
                error=f"Non-finite score_us={score_us}",
                error_type="eval_failure",
                raw_response=resp,
            )

        return EvalResult(
            success=success,
            score_us=score_us,
            error=resp.get("error"),
            error_type=resp.get("error_type"),
            logs=resp.get("logs", {}),
            test_results=resp.get("test_results", {}),
            timing=resp.get("timing", {}),
            raw_response=resp,
        )

    def is_available(self) -> bool:
        """Check if the eval server is reachable."""
        try:
            req = urllib.request.Request(
                f"{self.server_url}/health",
                method="GET",
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False
