from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, Union

import torch

T = TypeVar("T")

LossFnType = str


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
class Datum:
    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]


@dataclass
class SamplingParams:
    stop: list[str] | list[int] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 1.0


@dataclass
class AdamParams:
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class SampleSequence:
    tokens: list[int]
    logprobs: list[float] | None


@dataclass
class SampleResult:
    sequences: list[SampleSequence]


@dataclass
class ForwardBackwardOutput:
    loss_fn_outputs: list[dict[str, TensorData]]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimStepResponse:
    path: str = ""


@dataclass
class SaveStateResponse:
    path: str = ""


class APIFuture(Generic[T]):
    def __init__(self, value: T):
        self._value = value

    async def result_async(self) -> T:
        return self._value

    @classmethod
    def from_value(cls, value: T) -> APIFuture[T]:
        return cls(value)
