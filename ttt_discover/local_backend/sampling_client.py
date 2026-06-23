import logging
import os
from typing import Any

import tinker

logger = logging.getLogger(__name__)


class LocalSamplingClient:
    """Drop-in replacement for tinker.SamplingClient using vLLM for local inference."""

    def __init__(
        self,
        model_name_or_path: str,
        gpu_id: int = 0,
        lora_adapter_path: str | None = None,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
    ):
        from vllm import LLM

        self.model_name_or_path = model_name_or_path
        self.gpu_id = gpu_id
        self.lora_adapter_path = lora_adapter_path

        logger.info(
            f"Initializing vLLM engine on GPU {gpu_id} with model {model_name_or_path} (TP={tensor_parallel_size})"
        )
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enable_lora=True,
            max_lora_rank=64,
            disable_custom_all_reduce=True,
        )
        self._lora_id_counter = 1
        logger.info("vLLM engine initialized")

    def _get_lora_request(self):
        if self.lora_adapter_path is None:
            return None
        from vllm.lora.request import LoRARequest
        return LoRARequest("lora_adapter", self._lora_id_counter, self.lora_adapter_path)

    def update_lora(self, adapter_path: str):
        self.lora_adapter_path = adapter_path
        self._lora_id_counter += 1

    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
    ) -> tinker.types.SampleResponse:
        from vllm import SamplingParams as VllmSamplingParams

        token_ids = prompt.to_ints()

        stop_token_ids = []
        for s in (sampling_params.stop or []):
            if isinstance(s, int):
                stop_token_ids.append(s)

        vllm_params = VllmSamplingParams(
            n=num_samples,
            max_tokens=sampling_params.max_tokens,
            temperature=max(sampling_params.temperature, 0.01),
            stop_token_ids=stop_token_ids if stop_token_ids else None,
            logprobs=1,
        )

        outputs = self.llm.generate(
            prompts=None,
            prompt_token_ids=[token_ids],
            sampling_params=vllm_params,
            lora_request=self._get_lora_request(),
        )

        sequences = []
        for output in outputs[0].outputs:
            logprobs_list = []
            if output.logprobs:
                for i, lp_dict in enumerate(output.logprobs):
                    tok = output.token_ids[i]
                    if lp_dict and tok in lp_dict:
                        logprobs_list.append(lp_dict[tok].logprob)
                    else:
                        logprobs_list.append(0.0)
            else:
                logprobs_list = [0.0] * len(output.token_ids)

            sequences.append(
                tinker.types.SampledSequence(
                    tokens=list(output.token_ids),
                    logprobs=logprobs_list,
                    stop_reason=(
                        "stop" if output.finish_reason == "stop" else "length"
                    ),
                )
            )

        return tinker.types.SampleResponse(sequences=sequences)

    async def compute_logprobs_async(
        self, sequence: tinker.ModelInput
    ) -> list[float]:
        from vllm import SamplingParams as VllmSamplingParams

        token_ids = sequence.to_ints()

        vllm_params = VllmSamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=1.0,
        )

        outputs = self.llm.generate(
            prompts=None,
            prompt_token_ids=[token_ids],
            sampling_params=vllm_params,
            lora_request=self._get_lora_request(),
        )

        prompt_lps = outputs[0].prompt_logprobs
        result = []
        for i in range(len(token_ids)):
            if prompt_lps is None or i >= len(prompt_lps) or prompt_lps[i] is None:
                result.append(0.0)
            else:
                lp_dict = prompt_lps[i]
                tok = token_ids[i]
                if tok in lp_dict:
                    result.append(lp_dict[tok].logprob)
                else:
                    result.append(0.0)
        return result

    def shutdown(self):
        if hasattr(self, "llm"):
            del self.llm
            import torch
            torch.cuda.empty_cache()
