from typing import Generic, TypeVar

T = TypeVar("T")


class LocalFuture(Generic[T]):
    """Drop-in replacement for tinker.APIFuture that wraps a synchronous result."""

    def __init__(self, value: T):
        self._value = value

    async def result_async(self) -> T:
        return self._value

    @classmethod
    def from_value(cls, value: T) -> "LocalFuture[T]":
        return cls(value)
