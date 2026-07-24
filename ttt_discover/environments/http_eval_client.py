"""Unified HTTP evaluation client for all TTT-Discover tasks.

Routes code evaluation through a centralized HTTP eval server, providing
consistent error handling, retry logic, and structured diagnostics across
all 8 tasks.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = frozenset({
    "circle_packing", "cp32", "ac1", "ac2",
    "erdos", "denoising", "gpu_mode", "ahc039",
})

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
    """HTTP client for the centralized eval server.

    Supports all 8 TTT-Discover tasks via POST /eval endpoint.
    Handles retries on infrastructure failures, validates responses,
    and returns penalty scores when the server is unavailable.
    """

    def __init__(
        self,
        server_url: str | None = None,
        timeout: int = 3600,
        max_retries: int = 2,
        penalty_score: float = 0.0,
    ):
        self.server_url = (
            server_url
            or os.environ.get("EVAL_SERVER_URL", "http://localhost:8080")
        ).rstrip("/")
        if not self.server_url.startswith("http"):
            self.server_url = f"http://{self.server_url}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.penalty_score = penalty_score

    def evaluate(
        self,
        code: str,
        task_name: str,
        timeout: int | None = None,
        extra_params: dict | None = None,
    ) -> EvalResult:
        """Submit code for evaluation and return structured result.

        Args:
            code: The code to evaluate.
            task_name: One of the supported task names.
            timeout: Per-request timeout override (seconds).
            extra_params: Additional parameters to include in the request payload.

        Returns:
            EvalResult with success/failure info, score, logs, and timing.
        """
        req_timeout = timeout or self.timeout
        payload: dict[str, Any] = {
            "code": code,
            "task_name": task_name,
            "timeout": req_timeout,
        }
        if extra_params:
            payload.update(extra_params)

        data = json.dumps(payload).encode()
        url = f"{self.server_url}/eval"

        last_error: str | None = None
        for attempt in range(1 + self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                resp_bytes = urllib.request.urlopen(req, timeout=req_timeout).read()
                resp = json.loads(resp_bytes)
                result = self._parse_response(resp)

                if not result.success and result.is_retryable and attempt < self.max_retries:
                    logger.warning(
                        "Eval infra_failure on attempt %d/%d: %s",
                        attempt + 1, self.max_retries + 1, result.error,
                    )
                    time.sleep(2 * (attempt + 1))
                    continue

                return result

            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "HTTP eval attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries + 1, last_error,
                )
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
            except (json.JSONDecodeError, ValueError) as e:
                last_error = f"Response parse error: {e}"
                logger.error("Eval response parse failed: %s", last_error)
                break

        return EvalResult(
            success=False,
            score_us=self.penalty_score,
            error=f"Server unavailable after {self.max_retries + 1} attempts: {last_error}",
            error_type="infra_failure",
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
