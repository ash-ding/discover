"""Bridge between DiscoverTrainer and VERL's colocated worker infrastructure.

This module provides two implementations:
- MockWorkerBridge: CPU-only mock for testing the pipeline without GPUs
- VERLWorkerBridge: Real VERL integration for GPU training

Use mock_mode=True (default when no GPUs available) for pipeline validation.
"""

import logging
import os
import random
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MockWorkerBridge:
    """Mock worker bridge for CPU-only pipeline testing.

    Simulates generate_sequences, compute_ref_log_prob, and update_actor
    with random data. Validates that the full PUCT→rollout→reward→advantage→train
    pipeline works end-to-end without requiring GPUs or model weights.
    """

    def __init__(self, config: dict):
        self.config = config
        self._step = 0
        logger.info("Using MockWorkerBridge (CPU-only pipeline test)")

    def generate_sequences(
        self,
        prompt_ids: list[int],
        n: int = 1,
        max_tokens: int = 26000,
        temperature: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> list[dict]:
        """Generate mock completions with random tokens and logprobs."""
        outputs = []
        for _ in range(n):
            resp_len = random.randint(10, min(50, max_tokens))
            tokens = [random.randint(100, 150000) for _ in range(resp_len)]
            logprobs = [random.uniform(-5.0, -0.1) for _ in range(resp_len)]
            outputs.append({"tokens": tokens, "logprobs": logprobs})
        return outputs

    def compute_ref_log_prob(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
    ) -> list[list[float]]:
        """Return mock reference log-probs."""
        return [
            [random.uniform(-5.0, -0.1) for _ in r]
            for r in response_ids_list
        ]

    def update_actor(self, gen_batch: dict) -> dict:
        """Simulate actor update, return mock metrics."""
        self._step += 1
        n = len(gen_batch.get("response_ids", []))
        adv = gen_batch.get("advantages")
        metrics = {"actor/loss": random.uniform(0.1, 1.0)}
        if adv is not None:
            metrics["actor/adv_mean"] = float(adv.mean())
            metrics["actor/adv_std"] = float(adv.std())
        logger.info(f"MockWorkerBridge.update_actor: step={self._step}, n_seqs={n}")
        return metrics

    def save_checkpoint(self, path: str):
        """Mock checkpoint save."""
        os.makedirs(path, exist_ok=True)
        logger.info(f"MockWorkerBridge: checkpoint saved to {path}")


class VERLWorkerBridge:
    """Real VERL worker bridge for GPU training.

    Wraps VERL's ActorRolloutRefWorker group to provide the interface
    expected by DiscoverTrainer. Uses colocated mode with FSDP + vLLM.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model_name"]
        self.lora_rank = config.get("lora_rank", 32)
        self.n_gpus = config.get("n_gpus", 8)
        self.tp_size = config.get("inference_tp_size", 4)
        self.sp_size = config.get("ulysses_sp_size", 4)
        self.max_model_len = config.get("max_model_len", 32768)
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.4)

        self._initialized = False
        self._wg = None

    def _lazy_init(self):
        if self._initialized:
            return
        self._init_verl_workers()
        self._initialized = True

    def _init_verl_workers(self):
        """Set up VERL's colocated actor/rollout/ref worker group.

        This requires GPU hardware and all VERL dependencies.
        """
        import ray
        from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
        from verl.workers.engine_workers import ActorRolloutRefWorker

        resource_pool = RayResourcePool(
            process_on_nodes=[self.n_gpus],
            use_gpu=True,
        )

        worker_dict_cls = ActorRolloutRefWorker.default_worker_cls(
            actor_strategy="fsdp",
            rollout_name="vllm",
        )

        self._wg = RayWorkerGroup(
            resource_pool=resource_pool,
            worker_dict_cls=worker_dict_cls,
            worker_config={
                "model": {
                    "path": self.model_name,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": float(self.lora_rank),
                    "target_modules": "all-linear",
                    "enable_gradient_checkpointing": True,
                    "trust_remote_code": True,
                },
                "actor": {
                    "strategy": "fsdp",
                    "fsdp_config": {
                        "ulysses_sequence_parallel_size": self.sp_size,
                        "param_offload": False,
                    },
                    "lr": self.config.get("learning_rate", 4e-5),
                    "grad_clip": 1.0,
                    "ppo_epochs": 1,
                    "clip_ratio": self.config.get("clip_ratio", 1000.0),
                    "use_dynamic_bsz": True,
                    "ppo_max_token_len_per_gpu": self.max_model_len,
                },
                "rollout": {
                    "name": "vllm",
                    "tensor_model_parallel_size": self.tp_size,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "max_model_len": self.max_model_len,
                    "free_cache_engine": True,
                    "enforce_eager": False,
                },
                "ref": {
                    "fsdp_config": {
                        "ulysses_sequence_parallel_size": self.sp_size,
                        "param_offload": True,
                    },
                },
            },
        )
        logger.info("VERL worker group initialized successfully")

    def generate_sequences(
        self,
        prompt_ids: list[int],
        n: int = 1,
        max_tokens: int = 26000,
        temperature: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> list[dict]:
        self._lazy_init()

        from verl.protocol import DataProto

        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.arange(len(prompt_ids)).unsqueeze(0),
        })

        gen_config = {
            "n": n,
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
        }
        if stop:
            gen_config["stop"] = stop

        result = self._wg.generate_sequences(batch, **gen_config)

        outputs = []
        for i in range(n):
            resp_ids = result.batch["responses"][i].tolist()
            resp_logprobs = result.batch["old_log_probs"][i].tolist()
            resp_len = int(result.batch["response_mask"][i].sum().item())
            outputs.append({
                "tokens": resp_ids[:resp_len],
                "logprobs": resp_logprobs[:resp_len],
            })
        return outputs

    def compute_ref_log_prob(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
    ) -> list[list[float]]:
        self._lazy_init()

        from verl.protocol import DataProto

        max_prompt = max(len(p) for p in prompt_ids_list)
        max_resp = max(len(r) for r in response_ids_list)
        bsz = len(prompt_ids_list)

        input_ids = torch.zeros(bsz, max_prompt + max_resp, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        response_mask = torch.zeros(bsz, max_resp)

        for i, (p, r) in enumerate(zip(prompt_ids_list, response_ids_list)):
            full = p + r
            input_ids[i, :len(full)] = torch.tensor(full)
            attention_mask[i, :len(full)] = 1
            response_mask[i, :len(r)] = 1

        batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        })

        ref_output = self._wg.compute_ref_log_prob(batch)
        ref_log_probs = ref_output.batch["ref_log_prob"]

        result = []
        for i in range(bsz):
            resp_len = len(response_ids_list[i])
            result.append(ref_log_probs[i, :resp_len].tolist())
        return result

    def update_actor(self, gen_batch: dict) -> dict:
        self._lazy_init()

        from verl.protocol import DataProto

        num_seq = len(gen_batch["response_ids"])
        max_resp = max(len(r) for r in gen_batch["response_ids"])
        max_prompt = max(len(p) for p in gen_batch["prompt_ids"])

        input_ids = torch.zeros(num_seq, max_prompt + max_resp, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        old_log_probs = torch.zeros(num_seq, max_resp)
        response_mask = gen_batch["response_mask"]
        advantages = gen_batch["advantages"]

        fine_grained_mask = torch.ones(num_seq, max_resp)
        for i in range(num_seq):
            p = gen_batch["prompt_ids"][i]
            r = gen_batch["response_ids"][i]
            full = p + r
            input_ids[i, :len(full)] = torch.tensor(full)
            attention_mask[i, :len(full)] = 1

            lp = gen_batch["response_logprobs"][i]
            old_log_probs[i, :len(lp)] = torch.tensor(lp)

            fg = gen_batch["fine_grained_mask"][i]
            fine_grained_mask[i, :len(fg)] = torch.tensor(fg)

        batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_log_probs": old_log_probs,
            "response_mask": response_mask,
            "advantages": advantages,
            "fine_grained_mask": fine_grained_mask,
        })

        metrics = self._wg.update_actor(batch)
        return metrics if isinstance(metrics, dict) else {}

    def save_checkpoint(self, path: str):
        self._lazy_init()
        self._wg.save_checkpoint(path)
        logger.info(f"VERL checkpoint saved to {path}")
