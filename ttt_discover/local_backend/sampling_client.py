import logging
import os
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=3600, sock_read=3600)


def _model_input_to_token_ids(model_input) -> list[int]:
    if hasattr(model_input, "to_ints"):
        return model_input.to_ints()
    tokens: list[int] = []
    for chunk in model_input.chunks:
        if hasattr(chunk, "tokens"):
            tokens.extend(chunk.tokens)
    return tokens


@dataclass
class SampledSequence:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str = "stop"


@dataclass
class SampleResponse:
    sequences: list[SampledSequence]


class LocalSamplingClient:
    """HTTP-based sampling client that talks to a standalone vLLM server."""

    _shared_session: aiohttp.ClientSession | None = None

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        lora_name: str | None = None,
        tokenizer: Any = None,
        **kwargs,
    ):
        self.base_url = base_url or os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000"
        )
        self.model_name = model_name or "default"
        self.lora_name = lora_name
        self._tokenizer = tokenizer
        self._lora_counter = 0

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True, trust_remote_code=True
            )
        return self._tokenizer

    async def _get_session(self) -> aiohttp.ClientSession:
        # Create a new session for each request to avoid connection sharing issues
        # when many concurrent requests hit connection errors simultaneously
        connector = aiohttp.TCPConnector(limit=256, limit_per_host=256)
        return aiohttp.ClientSession(timeout=_TIMEOUT, connector=connector)

    async def update_lora(self, adapter_path: str):
        adapter_path = os.path.abspath(adapter_path)
        old_name = self.lora_name
        self._lora_counter += 1
        self.lora_name = f"lora_v{self._lora_counter}"

        session = await self._get_session()
        try:
            # Unload old adapter first (V1 max_loras=1 requires this)
            if old_name:
                try:
                    url = f"{self.base_url}/v1/unload_lora_adapter"
                    async with session.post(
                        url, json={"lora_name": old_name}
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f"Unloaded old LoRA adapter '{old_name}'")
                except Exception:
                    pass

            url = f"{self.base_url}/v1/load_lora_adapter"
            async with session.post(
                url, json={"lora_name": self.lora_name, "lora_path": adapter_path}
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    if "already been loaded" in text:
                        logger.info(f"LoRA adapter '{self.lora_name}' already loaded, skipping")
                    else:
                        raise RuntimeError(
                            f"Failed to load LoRA adapter ({resp.status}): {text}"
                        )
                else:
                    logger.info(f"Loaded LoRA adapter '{self.lora_name}' from {adapter_path}")
        finally:
            await session.close()

    async def sample_async(
        self,
        prompt,
        num_samples: int,
        sampling_params,
    ) -> SampleResponse:
        token_ids = _model_input_to_token_ids(prompt)

        stop: list[str] | None = None
        stop_token_ids: list[int] | None = None
        if sampling_params.stop:
            if isinstance(sampling_params.stop[0], int):
                stop_token_ids = list(sampling_params.stop)
            else:
                stop = list(sampling_params.stop)

        payload: dict[str, Any] = {
            "prompt": token_ids,
            "n": num_samples,
            "max_tokens": sampling_params.max_tokens,
            "temperature": max(sampling_params.temperature, 0.01),
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
        }
        if stop:
            payload["stop"] = stop
        if stop_token_ids:
            payload["stop_token_ids"] = stop_token_ids
        if self.lora_name:
            payload["model"] = self.lora_name

        url = f"{self.base_url}/v1/completions"

        # Retry logic for transient network errors
        import asyncio
        import sys
        max_retries = 3
        retry_delay = 2.0

        last_exception = None
        session = None
        try:
            for attempt in range(max_retries):
                try:
                    # Get fresh session for each attempt
                    if session:
                        await session.close()
                    session = await self._get_session()

                    async with session.post(url, json=payload) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(
                                f"vLLM sampling failed ({resp.status}): {text}"
                            )
                        data = await resp.json()
                        break  # Success, exit retry loop
                except (aiohttp.client_exceptions.ServerDisconnectedError,
                        aiohttp.client_exceptions.ClientConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        print(f"[sampling_client] Connection error (attempt {attempt+1}/{max_retries}): {type(e).__name__}, retrying in {retry_delay}s...", file=sys.stderr)
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print(f"[sampling_client] Connection failed after {max_retries} attempts", file=sys.stderr)
                        raise last_exception
                except Exception as e:
                    # Log detailed error info for debugging (non-retryable errors)
                    print(f"[sampling_client] Request failed: {type(e).__name__}: {e}", file=sys.stderr)
                    prompt_len = prompt.length if hasattr(prompt, 'length') else len(prompt)
                    print(f"[sampling_client] Prompt length: {prompt_len} tokens", file=sys.stderr)
                    print(f"[sampling_client] max_tokens: {sampling_params.max_tokens}", file=sys.stderr)
                    raise
        finally:
            # Always close the session when done
            if session:
                await session.close()

        sequences: list[SampledSequence] = []
        for choice in data["choices"]:
            logprobs_data = choice.get("logprobs")
            if logprobs_data and "token_logprobs" in logprobs_data:
                token_logprobs = logprobs_data["token_logprobs"]
                raw_tokens = logprobs_data.get("tokens", [])
                if raw_tokens and isinstance(raw_tokens[0], str) and raw_tokens[0].startswith("token_id:"):
                    gen_token_ids = [int(t.split(":")[1]) for t in raw_tokens]
                else:
                    gen_token_ids = logprobs_data.get("token_ids")
                    if gen_token_ids is None:
                        tokenizer = self._get_tokenizer()
                        gen_text = choice.get("text", "")
                        gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
                        if len(gen_token_ids) != len(token_logprobs):
                            gen_token_ids = gen_token_ids[: len(token_logprobs)]
                sequences.append(
                    SampledSequence(
                        tokens=list(gen_token_ids),
                        logprobs=[
                            lp if lp is not None else 0.0 for lp in token_logprobs
                        ],
                        stop_reason=(
                            "stop"
                            if choice.get("finish_reason") == "stop"
                            else "length"
                        ),
                    )
                )
            else:
                raise RuntimeError("vLLM response missing logprobs data")

        return SampleResponse(sequences=sequences)

    async def compute_logprobs_async(self, sequence_input) -> list[float]:
        token_ids = _model_input_to_token_ids(sequence_input)
        if len(token_ids) < 2:
            return [0.0] * len(token_ids)

        tokenizer = self._get_tokenizer()
        prompt_text = tokenizer.decode(token_ids, skip_special_tokens=False)

        payload: dict[str, Any] = {
            "prompt": prompt_text,
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,
        }
        if self.lora_name:
            payload["model"] = self.lora_name

        session = await self._get_session()
        url = f"{self.base_url}/v1/completions"
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"vLLM logprobs failed ({resp.status}): {text}"
                )
            data = await resp.json()

        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])

        result = [lp if lp is not None else 0.0 for lp in token_logprobs]
        if len(result) > len(token_ids):
            result = result[: len(token_ids)]
        return result

    def shutdown(self):
        pass
