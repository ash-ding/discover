"""Dynamic PUCT-driven data source for VERL.

Replaces VERL's static parquet loading with on-the-fly prompt generation
from PUCTSampler state and task environments. Reuses the existing
SingleProblemDataset / Environment infrastructure for prompt building.
"""

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import structlog
import torch
from transformers import AutoTokenizer

logger = structlog.get_logger(__name__)


class PUCTDataSource:
    """Generates training prompts dynamically from PUCT sampler state.

    Each call to get_batch():
    1. Samples states from PUCTSampler
    2. Creates Environment instances and calls get_question()
    3. Tokenizes and returns a dict suitable for VERL's DataProto
    """

    def __init__(
        self,
        model_name: str,
        env_module: str,
        env_class: str,
        problem_type: str,
        groups_per_batch: int = 8,
        group_size: int = 64,
        max_prompt_length: int = 4096,
        log_dir: str = "./tinker_log",
        puct_file_path: str = "./tinker_log/puct_sampler.json",
        max_buffer_size: int = 1000,
        puct_c: float = 1.0,
        topk_children: int = 2,
        resume_step: Optional[int] = None,
        eval_timeout: int = 530,
        num_cpus_per_task: int = 1,
        phase1_max_tokens: int = 26000,
        renderer_name: str = "qwen3",
    ):
        self.model_name = model_name
        self.env_module = env_module
        self.env_class = env_class
        self.problem_type = problem_type
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.max_prompt_length = max_prompt_length
        self.log_dir = log_dir
        self.eval_timeout = eval_timeout
        self.num_cpus_per_task = num_cpus_per_task
        self.phase1_max_tokens = phase1_max_tokens

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        mod = importlib.import_module(env_module)
        self._env_cls = getattr(mod, env_class)

        # Build a renderer for prompt generation
        from ttt_discover.tinker_utils import renderers
        self._renderer = renderers.get_renderer(renderer_name, tokenizer=self._tokenizer)

        # Create a lightweight config object for Environment.__init__
        self._dataset_config = _DatasetConfig(
            problem_type=problem_type,
            env_type=self._env_cls,
            batch_size=groups_per_batch,
            group_size=group_size,
            num_cpus_per_task=num_cpus_per_task,
            eval_timeout=eval_timeout,
            log_path=os.path.join(log_dir, "discover-verl"),
            timeout=8000.0,
        )

        from ttt_discover.tinker_utils.sampler import PUCTSampler
        os.makedirs(os.path.dirname(puct_file_path) or ".", exist_ok=True)
        self.sampler = PUCTSampler(
            file_path=puct_file_path,
            env_type=self._env_cls,
            problem_type=problem_type,
            max_buffer_size=max_buffer_size,
            batch_size=groups_per_batch,
            resume_step=resume_step,
            puct_c=puct_c,
            topk_children=topk_children,
        )

        self._step = resume_step or 0
        logger.info("puct_data_source_init", env_module=env_module, env_class=env_class,
                     problem_type=problem_type, groups_per_batch=groups_per_batch,
                     group_size=group_size, resume_step=resume_step)

    def get_batch(self) -> dict[str, Any]:
        """Generate a batch of prompts from PUCT-sampled states.

        Returns dict with:
            input_ids: (groups_per_batch, max_prompt_len) int tensor
            attention_mask: (groups_per_batch, max_prompt_len) bool tensor
            prompt_lengths: (groups_per_batch,) int tensor
            meta: list[dict] with per-prompt metadata
        """
        states = self.sampler.sample_states(self.groups_per_batch)
        logger.debug("get_batch_sampled", num_states=len(states))

        prompts = []
        meta = []
        for state in states:
            prompt_text = self._build_prompt(state)
            prompts.append(prompt_text)
            meta.append({
                "state": state,
                "env_module": self.env_module,
                "env_class": self.env_class,
                "problem_type": self.problem_type,
                "log_dir": self.log_dir,
                "eval_timeout": self.eval_timeout,
                "num_cpus_per_task": self.num_cpus_per_task,
                "phase1_max_tokens": self.phase1_max_tokens,
                "data_source": f"{self.env_class}_{self.problem_type}",
            })

        tokenized = self._tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )

        prompt_lengths = tokenized["attention_mask"].sum(dim=-1)
        logger.info("get_batch_complete", num_prompts=len(prompts),
                     prompt_lengths=prompt_lengths.tolist())

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt_lengths": prompt_lengths,
            "meta": meta,
        }

    def _build_prompt(self, state) -> str:
        """Build prompt by instantiating Environment and calling get_question()."""
        env = self._env_cls(
            renderer=self._renderer,
            initial_state=state,
            sampler=self.sampler,
            config=self._dataset_config,
        )
        return env.get_question()

    def update_puct(
        self,
        new_states: list,
        parent_states: list,
        step: Optional[int] = None,
    ):
        if step is not None:
            self._step = step
        logger.debug("update_puct", new_states=len(new_states),
                      parent_states=len(parent_states), step=self._step)
        self.sampler.update_states(
            new_states, parent_states, save=True, step=self._step
        )

    def save(self, step: int):
        self.sampler.flush(step=step)
        self._step = step
        logger.info("puct_state_saved", step=step)

    def get_step(self) -> int:
        return self._step


@dataclass
class _DatasetConfig:
    """Lightweight config matching the fields Environment.__init__ reads."""
    problem_type: str
    env_type: type
    batch_size: int
    group_size: int
    num_cpus_per_task: int = 1
    eval_timeout: int = 530
    log_path: str = "./tinker_log"
    timeout: float = 8000.0
    convo_prefix: Any = None
