"""Multi-GPU parallel training client using ThreadPoolExecutor.

Each GPU holds a full model+LoRA replica. Data is split across GPUs,
forward/backward runs in parallel (CUDA releases the GIL), then LoRA
gradients are summed to the primary replica for the optimizer step.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import tinker

from ttt_discover.local_backend.future import LocalFuture
from ttt_discover.local_backend.loss import importance_sampling_loss, ppo_clip_loss

logger = logging.getLogger(__name__)


def _load_model_on_gpu(model_name_or_path, lora_rank, gpu_id, from_checkpoint=None):
    """Load base model + LoRA on a specific GPU. Returns (model, device)."""
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, PeftModel

    device = torch.device(f"cuda:{gpu_id}")
    attn_impl = "flash_attention_2"
    try:
        import flash_attn
    except ImportError:
        attn_impl = "sdpa"

    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if from_checkpoint:
        model = PeftModel.from_pretrained(base, from_checkpoint)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    else:
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_config)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    return model, device


def _process_datum_on_device(model, device, prompt_ids, target_tokens, old_logprobs,
                             advantages, mask, loss_fn, max_train_seq_len):
    """Run forward + backward for a single datum. Returns (new_logprobs_cpu, loss_value)."""
    target_tokens = target_tokens.to(device)
    old_logprobs = old_logprobs.to(device)
    advantages = advantages.to(device)
    if mask is not None:
        mask = mask.to(device)

    target_len = len(target_tokens)
    max_prompt_len = max_train_seq_len - target_len
    if max_prompt_len < 1:
        max_prompt_len = 1
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    full_ids = prompt_ids + target_tokens.long().tolist()
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    prompt_len = len(prompt_ids)
    outputs = model(input_ids)
    logits = outputs.logits[0]
    target_logits = logits[prompt_len - 1: prompt_len + target_len - 1]

    log_probs = torch.log_softmax(target_logits.float(), dim=-1)
    new_logprobs = log_probs.gather(1, target_tokens.unsqueeze(1).long()).squeeze(1)

    if mask is None:
        mask = torch.ones_like(new_logprobs)

    if loss_fn == "importance_sampling":
        loss = importance_sampling_loss(new_logprobs, old_logprobs, advantages, mask)
    elif loss_fn == "ppo":
        loss = ppo_clip_loss(new_logprobs, old_logprobs, advantages, mask)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    loss.backward()
    return new_logprobs.detach().cpu(), loss.item()


class DistributedTrainingClient:
    """Multi-GPU parallel training using ThreadPoolExecutor for concurrent forward/backward."""

    def __init__(
        self,
        model_name_or_path: str,
        lora_rank: int = 32,
        gpu_ids: list[int] = None,
        checkpoint_dir: str = "./tinker_log/local_checkpoints",
        max_train_seq_len: int = 32768,
        from_checkpoint: str | None = None,
        load_optimizer: bool = False,
    ):
        from transformers import AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.lora_rank = lora_rank
        self.gpu_ids = gpu_ids or [0, 1]
        self.num_gpus = len(self.gpu_ids)
        self.checkpoint_dir = checkpoint_dir
        self.max_train_seq_len = max_train_seq_len
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"Loading {self.num_gpus} model replicas on GPUs {self.gpu_ids}")
        self.replicas = []
        for gpu_id in self.gpu_ids:
            model, device = _load_model_on_gpu(
                model_name_or_path, lora_rank, gpu_id, from_checkpoint
            )
            self.replicas.append({"model": model, "device": device, "gpu_id": gpu_id})

        trainable = sum(p.numel() for p in self.replicas[0]["model"].parameters() if p.requires_grad)
        logger.info(f"LoRA trainable params per replica: {trainable:,}")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.replicas[0]["model"].parameters() if p.requires_grad],
            lr=4e-5, betas=(0.9, 0.95), eps=1e-8,
        )
        if load_optimizer and from_checkpoint:
            opt_path = os.path.join(from_checkpoint, "optimizer.pt")
            if os.path.exists(opt_path):
                self.optimizer.load_state_dict(
                    torch.load(opt_path, map_location=self.replicas[0]["device"], weights_only=True)
                )
                logger.info("Optimizer state loaded")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=True
        )
        self._executor = ThreadPoolExecutor(max_workers=self.num_gpus)

    def get_tokenizer(self):
        return self._tokenizer

    def _process_shard(self, rank, data_shard, loss_fn):
        """Process a shard of data on one GPU replica. Called in a thread."""
        model = self.replicas[rank]["model"]
        device = self.replicas[rank]["device"]
        outputs = []
        total_loss = 0.0

        for datum in data_shard:
            prompt_ids = datum.model_input.to_ints()
            target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
            old_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            mask_data = datum.loss_fn_inputs.get("mask")
            mask = mask_data.to_torch() if mask_data is not None else None

            new_lp_cpu, loss_val = _process_datum_on_device(
                model, device, prompt_ids, target_tokens, old_logprobs,
                advantages, mask, loss_fn, self.max_train_seq_len,
            )
            total_loss += loss_val
            outputs.append({"logprobs": tinker.TensorData.from_torch(new_lp_cpu)})

        return outputs, total_loss

    def _sync_gradients_to_primary(self):
        """Sum LoRA gradients from all replicas into the primary (rank 0)."""
        primary_model = self.replicas[0]["model"]
        primary_device = self.replicas[0]["device"]

        primary_params = dict(primary_model.named_parameters())
        for rank in range(1, self.num_gpus):
            replica_params = dict(self.replicas[rank]["model"].named_parameters())
            for name, param in primary_params.items():
                if param.requires_grad and param.grad is not None:
                    other_grad = replica_params[name].grad
                    if other_grad is not None:
                        param.grad.add_(other_grad.to(primary_device))

    def _zero_replica_grads(self):
        """Zero gradients on non-primary replicas."""
        for rank in range(1, self.num_gpus):
            for param in self.replicas[rank]["model"].parameters():
                if param.grad is not None:
                    param.grad = None

    def _sync_weights_to_replicas(self):
        """Copy updated LoRA weights from primary to all replicas."""
        primary_state = {}
        for name, param in self.replicas[0]["model"].named_parameters():
            if param.requires_grad:
                primary_state[name] = param.data

        for rank in range(1, self.num_gpus):
            replica_device = self.replicas[rank]["device"]
            for name, param in self.replicas[rank]["model"].named_parameters():
                if name in primary_state:
                    param.data.copy_(primary_state[name].to(replica_device))

    async def forward_backward_async(
        self, data: list[tinker.Datum], loss_fn: str = "importance_sampling",
    ) -> LocalFuture:
        chunk_size = (len(data) + self.num_gpus - 1) // self.num_gpus
        shards = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(self.num_gpus)]

        futures = []
        for rank in range(self.num_gpus):
            if shards[rank]:
                futures.append(
                    self._executor.submit(self._process_shard, rank, shards[rank], loss_fn)
                )

        all_outputs = []
        total_loss = 0.0
        for f in futures:
            outputs, loss = f.result()
            all_outputs.extend(outputs)
            total_loss += loss

        self._sync_gradients_to_primary()
        self._zero_replica_grads()

        result = tinker.types.ForwardBackwardOutput(
            loss_fn_output_type=loss_fn,
            loss_fn_outputs=all_outputs,
            metrics={"loss": total_loss / max(len(data), 1)},
        )
        return LocalFuture(result)

    async def optim_step_async(self, adam_params: tinker.AdamParams) -> LocalFuture:
        for pg in self.optimizer.param_groups:
            pg["lr"] = adam_params.learning_rate
            pg["betas"] = (adam_params.beta1, adam_params.beta2)
            pg["eps"] = adam_params.eps

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.replicas[0]["model"].parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._sync_weights_to_replicas()

        return LocalFuture(tinker.types.OptimStepResponse(metrics={}))

    async def save_state_async(self, name: str) -> LocalFuture:
        path = os.path.join(self.checkpoint_dir, f"state_{name}")
        os.makedirs(path, exist_ok=True)
        self.replicas[0]["model"].save_pretrained(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        logger.info(f"State saved to {path}")

        class _Result:
            def __init__(self, p): self.path = p
        return LocalFuture(_Result(path))

    async def save_weights_for_sampler_async(self, name: str) -> LocalFuture:
        path = os.path.join(self.checkpoint_dir, f"sampler_{name}")
        os.makedirs(path, exist_ok=True)
        self.replicas[0]["model"].save_pretrained(path)
        logger.info(f"Sampler weights saved to {path}")

        class _Result:
            def __init__(self, p): self.path = p
        return LocalFuture(_Result(path))

    def set_shared_sampling_client(self, client):
        self._shared_sampling_client = client

    async def save_weights_and_get_sampling_client_async(self):
        path = os.path.join(self.checkpoint_dir, "latest_sampler")
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
        self.replicas[0]["model"].save_pretrained(path)

        adapter_config = os.path.join(path, "adapter_config.json")
        adapter_weights = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(adapter_config) or not os.path.exists(adapter_weights):
            raise FileNotFoundError(
                f"LoRA adapter files incomplete: config={os.path.exists(adapter_config)}, "
                f"weights={os.path.exists(adapter_weights)}"
            )

        if hasattr(self, "_shared_sampling_client") and self._shared_sampling_client is not None:
            await self._shared_sampling_client.update_lora(path)
            return self._shared_sampling_client
        raise RuntimeError("No shared sampling client set")

    async def create_sampling_client(self, path: str):
        path = os.path.abspath(path)
        if hasattr(self, "_shared_sampling_client") and self._shared_sampling_client is not None:
            await self._shared_sampling_client.update_lora(path)
            return self._shared_sampling_client
        raise RuntimeError("No shared sampling client set")
