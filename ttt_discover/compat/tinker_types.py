"""Lightweight type shim preserving the tinker SDK interface used by non-backend code.

Only data types needed by rl/, tinker_utils/, and environments/ are kept here.
Training-specific types (Datum, ForwardBackwardOutput, etc.) are intentionally
excluded — VERL handles those concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch

T = TypeVar("T")

LossFnType = str


# Stub types for legacy code that references training/inference types.
# These are no longer functional — VERL handles training and inference.
class _Stub:
    """Placeholder for removed types. Allows type annotations to resolve."""
    pass

SamplingClient = _Stub
TrainingClient = _Stub
ServiceClient = _Stub


class APIFuture(Generic[T]):
    """Stub for legacy APIFuture type annotations."""
    pass


class Datum:
    """Stub for legacy Datum. See data_processing.py for the real conversion logic."""
    def __init__(self, model_input=None, loss_fn_inputs=None):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs or {}


class ForwardBackwardOutput:
    """Stub."""
    pass


class OptimStepResponse:
    """Stub."""
    pass


class ModelInputChunk:
    @property
    def length(self) -> int:
        raise NotImplementedError


@dataclass
class EncodedTextChunk(ModelInputChunk):
    tokens: list[int]

    def __post_init__(self):
        self.tokens = list(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class ModelInput:
    chunks: list[ModelInputChunk] = field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(c.length for c in self.chunks)

    @classmethod
    def empty(cls) -> ModelInput:
        return cls(chunks=[])

    def append_int(self, token_id: int) -> ModelInput:
        new_chunks = list(self.chunks)
        if new_chunks and isinstance(new_chunks[-1], EncodedTextChunk):
            last = new_chunks[-1]
            new_chunks[-1] = EncodedTextChunk(tokens=last.tokens + [token_id])
        else:
            new_chunks.append(EncodedTextChunk(tokens=[token_id]))
        return ModelInput(chunks=new_chunks)


class TensorData:
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @property
    def data(self) -> list:
        return self._tensor.tolist()

    @classmethod
    def from_torch(cls, t: torch.Tensor) -> TensorData:
        return cls(t)

    def to_torch(self) -> torch.Tensor:
        return self._tensor


@dataclass
class SamplingParams:
    stop: list[str] | list[int] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 1.0


@dataclass
class SampleSequence:
    tokens: list[int]
    logprobs: list[float] | None


@dataclass
class SampleResult:
    sequences: list[SampleSequence]


# Expose a `types` sub-attribute so `tinker.types.EncodedTextChunk` works
class _Types:
    EncodedTextChunk = EncodedTextChunk
    ModelInputChunk = ModelInputChunk
    ModelInput = ModelInput

types = _Types()
