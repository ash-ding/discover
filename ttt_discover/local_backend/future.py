class LocalFuture:
    """Drop-in replacement for tinker.APIFuture that wraps a synchronous result."""

    def __init__(self, value):
        self._value = value

    async def result_async(self):
        return self._value
