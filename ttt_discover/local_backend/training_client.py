import logging
import os
from typing import Any, Literal

import torch
import tinker

from ttt_discover.local_backend.future import LocalFuture
from ttt_discover.local_backend.loss import importance_sampling_loss, ppo_clip_loss

logger = logging.getLogger(__name__)


class LocalTrainingClient:
    """Drop-in replacement for tinker.TrainingClient using HuggingFace + PEFT."""

    def __init__(
        self,
        model_name_or_path: str,
        lora_rank: int = 32,
        gpu_id: int = 1,
        checkpoint_dir: str = "./tinker_log/local_checkpoints",
        training_batch_size: int = 1,
        max_train_seq_len: int = 32768,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig

        self.model_name_or_path = model_name_or_path
        self.lora_rank = lora_rank
        self.device = torch.device(f"cuda:{gpu_id}")
        self.checkpoint_dir = checkpoint_dir
        self.training_batch_size = training_batch_size
        self.max_train_seq_len = max_train_seq_len
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"Loading model {model_name_or_path} on GPU {gpu_id}")
        attn_impl = "flash_attention_2"
        try:
            import flash_attn
        except ImportError:
            attn_impl = "sdpa"
            logger.info("flash-attn not available, using SDPA")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.train()
        logger.info(
            f"LoRA applied: trainable={sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} params"
        )

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=4e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=True
        )

    @classmethod
    def from_checkpoint(
        cls,
        model_name_or_path: str,
        checkpoint_path: str,
        lora_rank: int = 32,
        gpu_id: int = 1,
        load_optimizer: bool = False,
        checkpoint_dir: str | None = None,
        training_batch_size: int = 1,
        max_train_seq_len: int = 32768,
    ) -> "LocalTrainingClient":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        instance = object.__new__(cls)
        instance.model_name_or_path = model_name_or_path
        instance.lora_rank = lora_rank
        instance.device = torch.device(f"cuda:{gpu_id}")
        instance.checkpoint_dir = checkpoint_dir or os.path.dirname(checkpoint_path)
        instance.training_batch_size = training_batch_size
        instance.max_train_seq_len = max_train_seq_len

        logger.info(f"Loading model from checkpoint {checkpoint_path}")
        attn_impl = "flash_attention_2"
        try:
            import flash_attn
        except ImportError:
            attn_impl = "sdpa"
        base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": instance.device},
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        instance.base_model = base
        instance.model = PeftModel.from_pretrained(base, checkpoint_path)
        instance.model.enable_input_require_grads()
        instance.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        instance.model.train()

        instance.optimizer = torch.optim.AdamW(
            [p for p in instance.model.parameters() if p.requires_grad],
            lr=4e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        if load_optimizer:
            opt_path = os.path.join(checkpoint_path, "optimizer.pt")
            if os.path.exists(opt_path):
                instance.optimizer.load_state_dict(
                    torch.load(opt_path, map_location=instance.device, weights_only=True)
                )
                logger.info("Optimizer state loaded")

        instance._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=True
        )
        return instance

    def get_tokenizer(self):
        return self._tokenizer

    async def forward_backward_async(
        self,
        data: list[tinker.Datum],
        loss_fn: str = "importance_sampling",
    ) -> LocalFuture:
        loss_fn_outputs = []
        total_loss = 0.0

        max_train_seq_len = self.max_train_seq_len
        batch_size = self.training_batch_size

        if batch_size == 1:
            # Serial processing (original implementation)
            for datum in data:
                prompt_ids = datum.model_input.to_ints()
                target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch().to(self.device)
                old_logprobs = datum.loss_fn_inputs["logprobs"].to_torch().to(self.device)
                advantages = datum.loss_fn_inputs["advantages"].to_torch().to(self.device)

                target_len = len(target_tokens)
                max_prompt_len = max_train_seq_len - target_len
                if max_prompt_len < 1:
                    max_prompt_len = 1
                if len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[-max_prompt_len:]

                full_ids = prompt_ids + target_tokens.long().tolist()
                input_ids = torch.tensor(
                    [full_ids], dtype=torch.long, device=self.device
                )

                prompt_len = len(prompt_ids)
                target_len = len(target_tokens)

                outputs = self.model(input_ids)
                logits = outputs.logits[0]

                target_logits = logits[prompt_len - 1 : prompt_len + target_len - 1]

                log_probs = torch.log_softmax(target_logits.float(), dim=-1)
                new_logprobs = log_probs.gather(
                    1, target_tokens.unsqueeze(1).long()
                ).squeeze(1)

                mask_data = datum.loss_fn_inputs.get("mask")
                if mask_data is not None:
                    mask = mask_data.to_torch().to(self.device)
                else:
                    mask = torch.ones_like(new_logprobs)

                if loss_fn == "importance_sampling":
                    loss = importance_sampling_loss(new_logprobs, old_logprobs, advantages, mask)
                elif loss_fn == "ppo":
                    loss = ppo_clip_loss(new_logprobs, old_logprobs, advantages, mask)
                else:
                    raise ValueError(f"Unknown loss function: {loss_fn}")

                loss.backward()
                total_loss += loss.item()

                loss_fn_outputs.append({
                    "logprobs": tinker.TensorData.from_torch(new_logprobs.detach().cpu())
                })

        else:
            # Batch processing
            pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0

            for batch_idx in range(0, len(data), batch_size):
                batch_data = data[batch_idx:batch_idx + batch_size]
                actual_bs = len(batch_data)

                # Prepare batch data
                sequences = []
                target_tokens_list = []
                old_logprobs_list = []
                advantages_list = []
                masks_list = []
                prompt_lens = []

                for datum in batch_data:
                    prompt_ids = datum.model_input.to_ints()
                    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
                    old_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
                    advantages = datum.loss_fn_inputs["advantages"].to_torch()

                    # Truncate prompt if necessary
                    target_len = len(target_tokens)
                    max_prompt_len = max_train_seq_len - target_len
                    if max_prompt_len < 1:
                        max_prompt_len = 1
                    if len(prompt_ids) > max_prompt_len:
                        prompt_ids = prompt_ids[-max_prompt_len:]

                    full_ids = prompt_ids + target_tokens.tolist()
                    sequences.append(full_ids)
                    prompt_lens.append(len(prompt_ids))

                    target_tokens_list.append(target_tokens.to(self.device))
                    old_logprobs_list.append(old_logprobs.to(self.device))
                    advantages_list.append(advantages.to(self.device))

                    mask_data = datum.loss_fn_inputs.get("mask")
                    if mask_data is not None:
                        masks_list.append(mask_data.to_torch().to(self.device))
                    else:
                        masks_list.append(None)

                # Padding to batch max length
                max_len = max(len(seq) for seq in sequences)
                padded_seqs = []
                attention_masks = []

                for seq in sequences:
                    padding_len = max_len - len(seq)
                    padded_seq = seq + [pad_token_id] * padding_len
                    mask = [1] * len(seq) + [0] * padding_len
                    padded_seqs.append(padded_seq)
                    attention_masks.append(mask)

                # Convert to tensors
                input_ids = torch.tensor(padded_seqs, dtype=torch.long, device=self.device)
                attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)

                # Batch forward
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                # Compute loss for each sample in batch
                batch_loss = 0.0
                for i in range(actual_bs):
                    prompt_len = prompt_lens[i]
                    target_tokens = target_tokens_list[i]
                    old_logprobs = old_logprobs_list[i]
                    advantages = advantages_list[i]
                    target_len = len(target_tokens)

                    # Extract target logits for this sample
                    start_idx = prompt_len - 1
                    end_idx = start_idx + target_len
                    sample_logits = logits[i, start_idx:end_idx]

                    # Compute log probs
                    log_probs = torch.log_softmax(sample_logits.float(), dim=-1)
                    new_logprobs = log_probs.gather(1, target_tokens.unsqueeze(1).long()).squeeze(1)

                    # Get mask
                    if masks_list[i] is not None:
                        mask = masks_list[i]
                    else:
                        mask = torch.ones_like(new_logprobs)

                    # Compute loss
                    if loss_fn == "importance_sampling":
                        loss = importance_sampling_loss(new_logprobs, old_logprobs, advantages, mask)
                    elif loss_fn == "ppo":
                        loss = ppo_clip_loss(new_logprobs, old_logprobs, advantages, mask)
                    else:
                        raise ValueError(f"Unknown loss function: {loss_fn}")

                    batch_loss += loss

                    # Store outputs
                    loss_fn_outputs.append({
                        "logprobs": tinker.TensorData.from_torch(new_logprobs.detach().cpu())
                    })

                    # Release intermediate tensors
                    del sample_logits, log_probs, new_logprobs

                # Release batch logits
                del logits, outputs

                # Backward on batch average loss
                avg_loss = batch_loss / actual_bs
                avg_loss.backward()
                total_loss += avg_loss.item()

        result = tinker.types.ForwardBackwardOutput(
            loss_fn_output_type=loss_fn,
            loss_fn_outputs=loss_fn_outputs,
            metrics={"loss": total_loss / max(len(data) // batch_size if batch_size > 1 else len(data), 1)},
        )
        return LocalFuture(result)

    async def optim_step_async(self, adam_params: tinker.AdamParams) -> LocalFuture:
        for pg in self.optimizer.param_groups:
            pg["lr"] = adam_params.learning_rate
            pg["betas"] = (adam_params.beta1, adam_params.beta2)
            pg["eps"] = adam_params.eps

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        return LocalFuture(tinker.types.OptimStepResponse(metrics={}))

    async def save_state_async(self, name: str) -> LocalFuture:
        path = os.path.join(self.checkpoint_dir, f"state_{name}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        logger.info(f"State saved to {path}")

        class _Result:
            def __init__(self, p):
                self.path = p
        return LocalFuture(_Result(path))

    async def save_weights_for_sampler_async(self, name: str) -> LocalFuture:
        path = os.path.join(self.checkpoint_dir, f"sampler_{name}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        logger.info(f"Sampler weights saved to {path}")

        class _Result:
            def __init__(self, p):
                self.path = p
        return LocalFuture(_Result(path))

    def set_shared_sampling_client(self, client):
        """Set a shared sampling client to avoid re-creating vLLM engines."""
        self._shared_sampling_client = client

    async def save_weights_and_get_sampling_client_async(self):
        path = os.path.join(self.checkpoint_dir, "latest_sampler")
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)

        # Verify LoRA files are fully written before loading into vLLM
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
