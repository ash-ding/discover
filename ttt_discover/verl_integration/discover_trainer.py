"""Custom VERL trainer for TTT-Discover.

Subclasses VERL's RayPPOTrainer to inject:
- PUCT state reuse (dynamic prompt generation per step)
- Two-phase token completion (think → answer)
- Entropic adaptive beta advantage computation
- Sandboxed reward evaluation
"""

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DiscoverTrainer:
    """Orchestrates the TTT-Discover RL loop using VERL infrastructure.

    Rather than subclassing RayPPOTrainer (which tightly couples data loading,
    rollout, and training), this class composes VERL's worker groups and drives
    the training loop explicitly — matching TTT-Discover's synchronous on-policy
    pipeline while leveraging VERL's colocated actor/rollout/ref worker.
    """

    def __init__(
        self,
        config: dict,
        puct_data_source: "PUCTDataSource",
        actor_rollout_ref_wg: Any,
        reward_fn: Any,
        tokenizer: Any,
        wandb_logger: Optional[Any] = None,
    ):
        self.config = config
        self.puct_data_source = puct_data_source
        self.actor_rollout_ref_wg = actor_rollout_ref_wg
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.wandb_logger = wandb_logger

        self.total_epochs = config.get("total_epochs", 50)
        self.save_freq = config.get("save_freq", 2)
        self.group_size = config.get("group_size", 64)
        self.phase1_max_tokens = config.get("phase1_max_tokens", 26000)
        self.context_window = config.get("max_model_len", 32768)
        self.kl_coef = config.get("kl_coef", 0.1)
        self.log_dir = config.get("log_dir", "./tinker_log")
        self.experiment_name = config.get("experiment_name", "discover-verl")
        self.checkpoint_dir = os.path.join(
            self.log_dir, "local_checkpoints", self.experiment_name
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.phase2_prefill = "\n\n... I need to give my final answer now.\n</think>\n"
        self.phase2_prefill_ids = self.tokenizer.encode(
            self.phase2_prefill, add_special_tokens=False
        )

    def fit(self):
        """Main training loop: PUCT → rollout → reward → advantage → train."""
        start_step = self.puct_data_source.get_step()

        for step in range(start_step, self.total_epochs):
            step_start = time.time()
            logger.info(f"Step {step}/{self.total_epochs}")

            # 1. Get batch from PUCT data source
            t0 = time.time()
            batch = self.puct_data_source.get_batch()
            t_puct = time.time() - t0

            # 2. Generate rollouts (phase 1 + conditional phase 2)
            t0 = time.time()
            gen_batch = self._generate_with_two_phase(batch)
            t_rollout = time.time() - t0

            # 3. Compute rewards
            t0 = time.time()
            rewards, new_states = self._compute_rewards(gen_batch, batch["meta"])
            t_reward = time.time() - t0

            # 4. Update PUCT state
            parent_states = [m["state"] for m in batch["meta"]]
            expanded_parents = []
            for parent in parent_states:
                expanded_parents.extend([parent] * self.group_size)
            self.puct_data_source.update_puct(
                new_states, expanded_parents, step=step
            )

            # 5. Compute log-probs and advantages
            t0 = time.time()
            gen_batch = self._compute_logprobs_and_advantages(gen_batch, rewards)
            t_advantage = time.time() - t0

            # 6. Update actor
            t0 = time.time()
            train_metrics = self._update_actor(gen_batch)
            t_train = time.time() - t0

            # 7. Log and checkpoint
            step_time = time.time() - step_start
            metrics = {
                "progress/step": step,
                "progress/done_frac": (step + 1) / self.total_epochs,
                "time/puct": t_puct,
                "time/rollout": t_rollout,
                "time/reward": t_reward,
                "time/advantage": t_advantage,
                "time/train": t_train,
                "time/total": step_time,
                "reward/mean": np.mean(rewards),
                "reward/max": np.max(rewards),
                "reward/min": np.min(rewards),
                "reward/std": np.std(rewards),
            }
            if train_metrics:
                metrics.update(train_metrics)

            self._log_metrics(metrics, step)

            if (step + 1) % self.save_freq == 0:
                self._save_checkpoint(step)

            logger.info(
                f"Step {step} done in {step_time:.1f}s — "
                f"reward mean={np.mean(rewards):.4f} max={np.max(rewards):.4f}"
            )

        logger.info("Training complete!")

    def _generate_with_two_phase(self, batch: dict) -> dict:
        """Generate completions with two-phase token completion.

        Phase 1: generate up to phase1_max_tokens.
        Phase 2: if thinking tokens exhausted without </think>,
                 inject prefill and continue generating.

        Returns dict with:
            response_ids: list of list[int] (per-sequence generated tokens)
            response_logprobs: list of list[float] (per-token logprobs)
            fine_grained_mask: list of list[float] (1.0=train, 0.0=skip)
            prompt_ids: list of list[int] (original prompt tokens)
        """
        all_responses = []
        all_logprobs = []
        all_masks = []
        all_prompt_ids = []

        for i, meta in enumerate(batch["meta"]):
            prompt_ids = batch["input_ids"][i][batch["attention_mask"][i].bool()].tolist()
            prompt_len = len(prompt_ids)
            phase1_budget = self.phase1_max_tokens - prompt_len

            if phase1_budget <= 0:
                logger.warning(f"Prompt {i} too long ({prompt_len}), skipping")
                for _ in range(self.group_size):
                    all_responses.append([])
                    all_logprobs.append([])
                    all_masks.append([])
                    all_prompt_ids.append(prompt_ids)
                continue

            # Phase 1: generate n=group_size completions
            phase1_results = self.actor_rollout_ref_wg.generate_sequences(
                prompt_ids=prompt_ids,
                n=self.group_size,
                max_tokens=phase1_budget,
                temperature=self.config.get("temperature", 1.0),
            )

            for j in range(self.group_size):
                p1_tokens = phase1_results[j]["tokens"]
                p1_logprobs = phase1_results[j]["logprobs"]

                needs_phase2 = (
                    len(p1_tokens) >= phase1_budget
                    and not self._contains_pattern(p1_tokens, "</think>")
                )

                if not needs_phase2:
                    all_responses.append(p1_tokens)
                    all_logprobs.append(p1_logprobs)
                    all_masks.append([1.0] * len(p1_tokens))
                    all_prompt_ids.append(prompt_ids)
                    continue

                # Phase 2: inject prefill and continue
                phase2_prompt = prompt_ids + p1_tokens + self.phase2_prefill_ids
                phase2_budget = (
                    self.context_window - len(phase2_prompt) - 50
                )

                if phase2_budget <= 0:
                    all_responses.append(p1_tokens)
                    all_logprobs.append(p1_logprobs)
                    all_masks.append([1.0] * len(p1_tokens))
                    all_prompt_ids.append(prompt_ids)
                    continue

                phase2_results = self.actor_rollout_ref_wg.generate_sequences(
                    prompt_ids=phase2_prompt,
                    n=1,
                    max_tokens=phase2_budget,
                    temperature=self.config.get("temperature", 1.0),
                )

                p2_tokens = phase2_results[0]["tokens"]
                p2_logprobs = phase2_results[0]["logprobs"]

                combined_tokens = p1_tokens + self.phase2_prefill_ids + p2_tokens
                combined_logprobs = (
                    p1_logprobs
                    + [0.0] * len(self.phase2_prefill_ids)
                    + p2_logprobs
                )
                combined_mask = (
                    [1.0] * len(p1_tokens)
                    + [0.0] * len(self.phase2_prefill_ids)
                    + [1.0] * len(p2_tokens)
                )

                all_responses.append(combined_tokens)
                all_logprobs.append(combined_logprobs)
                all_masks.append(combined_mask)
                all_prompt_ids.append(prompt_ids)

        return {
            "response_ids": all_responses,
            "response_logprobs": all_logprobs,
            "fine_grained_mask": all_masks,
            "prompt_ids": all_prompt_ids,
            "meta": batch["meta"],
        }

    def _contains_pattern(self, tokens: list[int], pattern: str) -> bool:
        """Check if token sequence contains a text pattern."""
        pattern_ids = self.tokenizer.encode(pattern, add_special_tokens=False)
        if len(pattern_ids) > len(tokens):
            return False
        for i in range(len(tokens) - len(pattern_ids) + 1):
            if tokens[i:i + len(pattern_ids)] == pattern_ids:
                return True
        return False

    def _compute_rewards(
        self, gen_batch: dict, metas: list[dict]
    ) -> tuple[list[float], list]:
        """Compute rewards by decoding responses and running sandbox evaluation.

        Returns (rewards, new_states) where rewards is flat list across all
        prompts × group_size.
        """
        rewards = []
        new_states = []

        for idx, response_ids in enumerate(gen_batch["response_ids"]):
            prompt_idx = idx // self.group_size
            meta = metas[prompt_idx]

            response_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            )

            score = self.reward_fn(
                data_source=meta.get("data_source", ""),
                solution_str=response_text,
                ground_truth=None,
                extra_info=meta,
            )
            rewards.append(float(score))

            from ttt_discover.tinker_utils.state import State
            new_state = State(
                timestep=meta["state"].timestep + 1,
                construction=None,
                code=response_text,
                value=float(score),
            )
            new_states.append(new_state)

        return rewards, new_states

    def _compute_logprobs_and_advantages(
        self, gen_batch: dict, rewards: list[float]
    ) -> dict:
        """Compute reference log-probs, KL penalty, and entropic adaptive beta advantages.

        This is the core algorithmic step that must match the original implementation.
        """
        from verl.trainer.ppo.adv_estimators.entropic_adaptive_beta import (
            compute_entropic_adaptive_beta_advantage,
        )

        num_sequences = len(gen_batch["response_ids"])
        groups_per_batch = num_sequences // self.group_size

        # Build index array: which sequences share the same prompt
        index = np.repeat(np.arange(groups_per_batch), self.group_size)

        # Compute ref log-probs for KL penalty
        if self.kl_coef > 0:
            ref_logprobs = self.actor_rollout_ref_wg.compute_ref_log_prob(
                gen_batch["prompt_ids"], gen_batch["response_ids"]
            )
        else:
            ref_logprobs = None

        # Pack into token-level tensors for advantage computation
        max_resp_len = max(len(r) for r in gen_batch["response_ids"])
        token_rewards = torch.zeros(num_sequences, max_resp_len)
        response_mask = torch.zeros(num_sequences, max_resp_len)

        for i, (resp_ids, reward) in enumerate(
            zip(gen_batch["response_ids"], rewards)
        ):
            resp_len = len(resp_ids)
            if resp_len > 0:
                token_rewards[i, resp_len - 1] = reward
                response_mask[i, :resp_len] = 1.0

        # Apply KL penalty to rewards
        if ref_logprobs is not None and self.kl_coef > 0:
            for i in range(num_sequences):
                old_lp = gen_batch["response_logprobs"][i]
                ref_lp = ref_logprobs[i]
                resp_len = len(old_lp)
                if resp_len > 0:
                    kl_per_token = torch.tensor(old_lp[:resp_len]) - torch.tensor(ref_lp[:resp_len])
                    avg_kl = kl_per_token.mean()
                    kl_adjustment = self.kl_coef * (avg_kl - kl_per_token)
                    # Adjust the reward at the last token
                    token_rewards[i, resp_len - 1] += kl_adjustment.sum().item()

        advantages, returns = compute_entropic_adaptive_beta_advantage(
            token_level_rewards=token_rewards,
            response_mask=response_mask,
            index=index,
        )

        gen_batch["advantages"] = advantages
        gen_batch["returns"] = returns
        gen_batch["response_mask"] = response_mask
        gen_batch["token_rewards"] = token_rewards
        gen_batch["index"] = index

        return gen_batch

    def _update_actor(self, gen_batch: dict) -> dict:
        """Send training data to VERL's actor worker for a policy update.

        Returns training metrics.
        """
        return self.actor_rollout_ref_wg.update_actor(gen_batch)

    def _log_metrics(self, metrics: dict, step: int):
        """Log metrics to console and optionally WandB."""
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)
        logger.info(f"Step {step}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(metrics.items())
        ))

    def _save_checkpoint(self, step: int):
        """Save LoRA weights, optimizer state, and PUCT sampler."""
        ckpt_dir = os.path.join(self.checkpoint_dir, f"step_{step:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save VERL actor checkpoint (LoRA + optimizer)
        self.actor_rollout_ref_wg.save_checkpoint(ckpt_dir)

        # Save PUCT sampler
        self.puct_data_source.save(step)

        logger.info(f"Checkpoint saved at step {step}: {ckpt_dir}")
