"""
Local GPU kernel evaluator with container isolation (Podman/Docker).

Provides fault-tolerant local evaluation for GPU kernels without Modal dependency.
Supports both Podman (preferred) and Docker for container isolation.
"""

from .evaluator import LocalKernelEvaluator, PooledKernelEvaluator

__all__ = ["LocalKernelEvaluator", "PooledKernelEvaluator"]
