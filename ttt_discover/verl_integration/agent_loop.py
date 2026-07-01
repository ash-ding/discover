"""Custom VERL AgentLoop for TTT-Discover.

Replaces default rollout generation with:
- PUCT state sampling (batch-level, with lineage blocking)
- Two-phase token completion (think + answer) with stop sequences
- Fine-grained response masking (prefill tokens excluded from training)
- PUCT state update for ALL attempts (including failures)

Activate via config:
  actor_rollout_ref.rollout.agent.agent_loop_manager_class: \
    "ttt_discover.verl_integration.agent_loop:DiscoverAgentLoopManagerTQ"
"""

import asyncio
import importlib
import logging
import os
import re
import uuid
from typing import Any, Optional

import numpy as np
import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@ray.remote
class PUCTSamplerActor:
    """Ray actor wrapping PUCTSampler for shared state across workers."""

    def __init__(self, file_path, env_module, env_class, problem_type,
                 max_buffer_size=1000, batch_size=8, puct_c=1.0,
                 topk_children=2, resume_step=None):
        mod = importlib.import_module(env_module)
        self.env_cls = getattr(mod, env_class)
        self.problem_type = problem_type

        from ttt_discover.tinker_utils.sampler import PUCTSampler
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        self.sampler = PUCTSampler(
            file_path=file_path,
            env_type=self.env_cls,
            problem_type=problem_type,
            max_buffer_size=max_buffer_size,
            batch_size=batch_size,
            resume_step=resume_step,
            puct_c=puct_c,
            topk_children=topk_children,
        )
        self._step = resume_step or 0

    def sample_states(self, n):
        """Sample n diverse states with lineage blocking."""
        return self.sampler.sample_states(n)

    def update_states(self, states, parent_states, step=None):
        if step is not None:
            self._step = step
        self.sampler.update_states(states, parent_states, save=True, step=self._step)

    def record_failed_rollouts(self, parent_states, step=None):
        """Record failed rollouts to increment visit counts without adding children.

        Falls back to updating with empty children if record_failed_rollout
        is not available on the sampler.
        """
        if step is not None:
            self._step = step
        for parent in parent_states:
            if hasattr(self.sampler, 'record_failed_rollout'):
                self.sampler.record_failed_rollout(parent)
            elif hasattr(self.sampler, '_n'):
                pid = parent.id
                if pid in self.sampler._n:
                    self.sampler._n[pid] += 1
                    self.sampler._T += 1

    def flush(self, step):
        self.sampler.flush(step=step)
        self._step = step

    def get_step(self):
        return self._step


@ray.remote
class DiscoverAgentLoopWorkerTQ(AgentLoopWorker):
    """Agent loop worker with two-phase completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()
        self._tokenizer = None
        self._renderer = None
        self._env_cls = None
        self._puct_actor = None
        self._stop_token_ids = []

    def set_discover_config(self, discover_config: dict, puct_actor):
        """Called after worker creation to inject TTT-Discover config."""
        self._discover_config = discover_config
        self._puct_actor = puct_actor

        from transformers import AutoTokenizer
        model_path = self.config.actor_rollout_ref.model.path
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        from ttt_discover.tinker_utils import renderers
        self._renderer = renderers.get_renderer("qwen3", tokenizer=self._tokenizer)

        mod = importlib.import_module(discover_config["env_module"])
        self._env_cls = getattr(mod, discover_config["env_class"])

        self._phase1_max_tokens = discover_config.get("phase1_max_tokens", 26000)
        self._context_window = discover_config.get("max_model_len", 32768)
        self._context_buffer = 50
        self._phase2_prefill = "\n\n... I need to give my final answer now.\n</think>\n"
        self._phase2_prefill_ids = self._tokenizer.encode(
            self._phase2_prefill, add_special_tokens=False
        )
        self._stop_token_ids = self._renderer.get_stop_sequences()

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Override: use pre-assigned PUCT states + two-phase completion."""
        validate = batch.get("validate", False)
        if isinstance(validate, torch.Tensor):
            validate = bool(validate.item())
        batch.pop("validate", None)

        trajectory_info = await get_trajectory_info(
            batch["global_steps"], batch["index"], validate
        )

        for i in range(len(batch)):
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    pass

            task = asyncio.create_task(
                self._run_prompt_discover(
                    prompt, trajectory=trajectory_info[i], validate=validate
                )
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt_discover(
        self, prompt: dict, trajectory: dict, validate: bool
    ) -> None:
        """Generate n completions with two-phase completion for a pre-assigned PUCT state."""
        logger.info(f"_run_prompt_discover called, keys={list(prompt.keys())}")
        uid = prompt["uid"]
        partition_id = "train" if not validate else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})

        try:
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not validate else config.val_kwargs.n)

            # State is pre-assigned by the manager (batch-level PUCT sampling)
            state = prompt["_puct_state"]

            # Build prompt from environment
            from ttt_discover.verl_integration.puct_data_source import _DatasetConfig
            dataset_config = _DatasetConfig(
                problem_type=self._discover_config["problem_type"],
                env_type=self._env_cls,
                batch_size=1,
                group_size=n,
                num_cpus_per_task=self._discover_config.get("num_cpus_per_task", 1),
                eval_timeout=self._discover_config.get("eval_timeout", 530),
                log_path=self._discover_config.get("log_dir", "./tinker_log"),
            )
            env = self._env_cls(
                renderer=self._renderer,
                initial_state=state,
                sampler=self._puct_actor,
                config=dataset_config,
            )
            prompt_text = env.get_question()

            # Render with Qwen3Renderer (adds chat template + <think>\n)
            messages = [{"role": "user", "content": prompt_text}]
            model_input = self._renderer.build_generation_prompt(messages)
            prompt_ids = []
            for chunk in model_input.chunks:
                if hasattr(chunk, 'tokens'):
                    prompt_ids.extend(chunk.tokens)

            prompt["_prompt_ids"] = prompt_ids
            prompt["raw_prompt"] = messages

            sampling_params = {
                "temperature": float(config.temperature),
                "top_p": float(config.top_p),
                "top_k": int(config.top_k),
                "repetition_penalty": 1.0,
                "logprobs": config.calculate_log_probs,
                "stop_token_ids": self._stop_token_ids,
            }

            # Debug: log prompt tokens to verify format
            decoded_end = self._tokenizer.decode(prompt_ids[-20:])
            logger.info(f"Prompt ends with: {repr(decoded_end)}")

            # Collect all outputs for batch PUCT update
            outputs_and_scores = []

            tasks = []
            for session_id in range(n):
                task = asyncio.create_task(
                    self._generate_two_phase(
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        prompt=prompt,
                        trajectory=trajectory,
                        validate=validate,
                        session_id=session_id,
                    )
                )
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors in generation
            valid_results = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    logger.error(f"Session {i} failed: {type(r).__name__}: {r}")
                else:
                    valid_results.append(r)
            logger.info(f"Generation complete: {len(valid_results)}/{len(results)} succeeded")

            # Batch PUCT update: collect all results for this prompt group
            if not validate and self._puct_actor is not None:
                global_steps = prompt.get("global_steps", 0)
                successful_states = []
                successful_parents = []
                failed_parents = []

                for output, code, score in valid_results:
                    if score > 0 and code:
                        from ttt_discover.tinker_utils.state import State
                        new_state = State(
                            timestep=state.timestep + 1,
                            construction=None,
                            code=code,
                            value=score,
                        )
                        successful_states.append(new_state)
                        successful_parents.append(state)
                    else:
                        failed_parents.append(state)

                if successful_states:
                    ray.get(self._puct_actor.update_states.remote(
                        successful_states, successful_parents, step=global_steps
                    ))
                if failed_parents:
                    ray.get(self._puct_actor.record_failed_rollouts.remote(
                        failed_parents, step=global_steps
                    ))

            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})

        except Exception as e:
            logger.exception(f"Error in _run_prompt_discover: {e}")
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _generate_two_phase(
        self,
        prompt_ids: list[int],
        sampling_params: dict,
        prompt: dict,
        trajectory: dict,
        validate: bool,
        session_id: int,
    ) -> tuple[AgentLoopOutput, str, float]:
        """Two-phase generation: thinking + forced answer.

        Returns (output, extracted_code, reward_score).
        """
        import time
        t0 = time.time()

        prompt_len = len(prompt_ids)
        phase1_budget = self._phase1_max_tokens - prompt_len
        if phase1_budget <= 0:
            logger.warning(f"Prompt too long ({prompt_len}), using minimal budget")
            phase1_budget = 100

        # Phase 1: thinking (with stop sequences)
        logger.info(f"_generate_two_phase: session={session_id}, prompt_len={prompt_len}, phase1_budget={phase1_budget}")
        request_id = uuid.uuid4().hex
        phase1_output = await self.llm_client.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params={**sampling_params, "max_tokens": phase1_budget},
        )

        p1_tokens = phase1_output.token_ids
        p1_logprobs = phase1_output.log_probs or [0.0] * len(p1_tokens)
        logger.info(f"Phase1 result: {len(p1_tokens)} tokens, stop_reason={phase1_output.stop_reason}, first_tokens={p1_tokens[:10]}")

        hit_stop = (
            phase1_output.stop_reason == "stop"
            or self._hit_stop_token(p1_tokens)
        )
        needs_phase2 = (
            not hit_stop
            and len(p1_tokens) >= phase1_budget
            and not self._contains_pattern(p1_tokens, "</think>")
        )

        if not needs_phase2:
            response_ids = p1_tokens
            response_logprobs = p1_logprobs
            response_mask = [1] * len(p1_tokens)
        else:
            # Check if model already produced </think> — continue without prefill
            if self._contains_pattern(p1_tokens, "</think>"):
                phase2_prompt = prompt_ids + p1_tokens
                phase2_budget = self._context_window - len(phase2_prompt) - self._context_buffer
                if phase2_budget <= 0:
                    response_ids = p1_tokens
                    response_logprobs = p1_logprobs
                    response_mask = [1] * len(p1_tokens)
                else:
                    request_id_p2 = uuid.uuid4().hex
                    phase2_output = await self.llm_client.generate(
                        request_id=request_id_p2,
                        prompt_ids=phase2_prompt,
                        sampling_params={**sampling_params, "max_tokens": phase2_budget},
                    )
                    p2_tokens = phase2_output.token_ids
                    p2_logprobs = phase2_output.log_probs or [0.0] * len(p2_tokens)
                    response_ids = p1_tokens + p2_tokens
                    response_logprobs = p1_logprobs + p2_logprobs
                    response_mask = [1] * len(p1_tokens) + [1] * len(p2_tokens)
            else:
                # Inject </think> prefill
                phase2_prompt = prompt_ids + p1_tokens + self._phase2_prefill_ids
                phase2_budget = self._context_window - len(phase2_prompt) - self._context_buffer
                if phase2_budget <= 0:
                    response_ids = p1_tokens + self._phase2_prefill_ids
                    response_logprobs = p1_logprobs + [0.0] * len(self._phase2_prefill_ids)
                    response_mask = [1] * len(p1_tokens) + [0] * len(self._phase2_prefill_ids)
                else:
                    request_id_p2 = uuid.uuid4().hex
                    phase2_output = await self.llm_client.generate(
                        request_id=request_id_p2,
                        prompt_ids=phase2_prompt,
                        sampling_params={**sampling_params, "max_tokens": phase2_budget},
                    )
                    p2_tokens = phase2_output.token_ids
                    p2_logprobs = phase2_output.log_probs or [0.0] * len(p2_tokens)

                    response_ids = p1_tokens + self._phase2_prefill_ids + p2_tokens
                    response_logprobs = (
                        p1_logprobs
                        + [0.0] * len(self._phase2_prefill_ids)
                        + p2_logprobs
                    )
                    response_mask = (
                        [1] * len(p1_tokens)
                        + [0] * len(self._phase2_prefill_ids)
                        + [1] * len(p2_tokens)
                    )

        gen_time = time.time() - t0

        # Construct AgentLoopOutput
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            metrics=AgentLoopMetrics(generate_sequences=gen_time),
            extra_fields={
                "min_global_steps": prompt.get("global_steps", 0),
                "max_global_steps": prompt.get("global_steps", 0),
            },
        )

        # Compute reward
        response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        code = self._extract_last_code_block(response_text)
        score = 0.0

        if code and not validate:
            try:
                extra_info = {
                    "env_module": self._discover_config["env_module"],
                    "env_class": self._discover_config["env_class"],
                    "problem_type": self._discover_config["problem_type"],
                    "log_dir": self._discover_config.get("log_dir", "./tinker_log"),
                    "eval_timeout": self._discover_config.get("eval_timeout", 530),
                    "num_cpus_per_task": self._discover_config.get("num_cpus_per_task", 1),
                    "state": prompt.get("_puct_state"),
                }
                from ttt_discover.verl_integration.verl_reward import compute_score
                score = compute_score(
                    data_source=self._discover_config.get("data_source", "circle_packing"),
                    solution_str=response_text,
                    ground_truth=None,
                    extra_info=extra_info,
                )
            except Exception as e:
                logger.warning(f"Reward eval failed: {type(e).__name__}: {e}")
                score = 0.0

        output.reward_score = score
        output.extra_fields["reward_extra_info"] = {"acc": float(score > 0)}

        # Write to TransferQueue
        await self._write_to_tq(output, prompt, session_id, validate)

        return output, code, score

    async def _write_to_tq(
        self, output: AgentLoopOutput, prompt: dict, session_id: int, validate: bool
    ) -> None:
        """Write output to TransferQueue in VERL's expected format."""
        uid = prompt["uid"]
        partition_id = "train" if not validate else "val"
        key = f"{uid}_{session_id}_0"

        prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
        responses = torch.tensor(output.response_ids, dtype=torch.int64)
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        position_ids = torch.arange(len(input_ids), dtype=torch.int64)

        field = output.as_dict()
        field["uid"] = uid
        field["session_id"] = session_id
        field["global_steps"] = prompt.get("global_steps", 0)
        field["raw_prompt"] = prompt.get("raw_prompt", [])
        field["data_source"] = self._discover_config.get("data_source", "circle_packing_26")
        field["num_turns"] = 1
        field.pop("multi_modal_data", None)
        field["loss_mask"] = field["response_mask"]
        field["input_ids"] = input_ids
        field["position_ids"] = position_ids
        field["attention_mask"] = attention_mask
        field["multi_modal_inputs"] = {}

        prompt_len = prompts.size(0)
        response_len = responses.size(0)
        tag = {
            "status": "success",
            "prompt_len": prompt_len,
            "response_len": response_len,
            "seq_len": prompt_len + response_len,
            "global_steps": prompt.get("global_steps", 0),
            "min_global_steps": output.extra_fields.get("min_global_steps", 0),
            "max_global_steps": output.extra_fields.get("max_global_steps", 0),
        }

        await tq.async_kv_batch_put(
            keys=[key],
            fields=list_of_dict_to_tensordict([field]),
            tags=[tag],
            partition_id=partition_id,
        )

    def _hit_stop_token(self, tokens: list[int]) -> bool:
        if not tokens or not self._stop_token_ids:
            return False
        return tokens[-1] in self._stop_token_ids

    def _contains_pattern(self, tokens: list[int], pattern: str) -> bool:
        pattern_ids = self._tokenizer.encode(pattern, add_special_tokens=False)
        if len(pattern_ids) > len(tokens):
            return False
        for i in range(len(tokens) - len(pattern_ids) + 1):
            if tokens[i:i + len(pattern_ids)] == pattern_ids:
                return True
        return False

    @staticmethod
    def _extract_last_code_block(text: str) -> str:
        """Extract last code block, matching original last_codeblock_postprocess behavior."""
        languages = ['python', 'cpp', 'java', 'cuda']
        languages_pattern = '|'.join(re.escape(lang) for lang in languages)
        codeblock_start = f'```({languages_pattern})'
        pattern = re.compile(codeblock_start + r'\n(?!```)(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(2).rstrip()
        return ""


class DiscoverAgentLoopManagerTQ(AgentLoopManager):
    """AgentLoopManager with batch-level PUCT sampling and two-phase completion."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = DiscoverAgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

        self._discover_config = {
            "env_module": os.environ.get("DISCOVER_ENV_MODULE", "examples.circle_packing.env"),
            "env_class": os.environ.get("DISCOVER_ENV_CLASS", "CirclePackingEnv"),
            "problem_type": os.environ.get("DISCOVER_PROBLEM_TYPE", "26"),
            "phase1_max_tokens": int(os.environ.get("DISCOVER_PHASE1_MAX_TOKENS", "26000")),
            "max_model_len": int(os.environ.get("DISCOVER_MAX_MODEL_LEN", "32768")),
            "eval_timeout": int(os.environ.get("DISCOVER_EVAL_TIMEOUT", "530")),
            "num_cpus_per_task": int(os.environ.get("DISCOVER_NUM_CPUS_PER_TASK", "1")),
            "log_dir": os.environ.get("DISCOVER_LOG_DIR", "./tinker_log"),
            "data_source": os.environ.get("DISCOVER_DATA_SOURCE", "circle_packing_26"),
            "puct_file_path": os.environ.get("DISCOVER_PUCT_FILE_PATH", "./tinker_log/puct_sampler.json"),
            "puct_c": float(os.environ.get("DISCOVER_PUCT_C", "1.0")),
            "topk_children": int(os.environ.get("DISCOVER_TOPK_CHILDREN", "2")),
            "max_buffer_size": int(os.environ.get("DISCOVER_MAX_BUFFER_SIZE", "1000")),
        }

        self._puct_actor = PUCTSamplerActor.remote(
            file_path=self._discover_config["puct_file_path"],
            env_module=self._discover_config["env_module"],
            env_class=self._discover_config["env_class"],
            problem_type=self._discover_config["problem_type"],
            max_buffer_size=self._discover_config["max_buffer_size"],
            batch_size=self.config.data.train_batch_size,
            puct_c=self._discover_config["puct_c"],
            topk_children=self._discover_config["topk_children"],
        )

        self._global_steps = 0

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        ray.get([
            worker.set_discover_config.remote(
                instance._discover_config, instance._puct_actor
            )
            for worker in instance.agent_loop_workers
        ])
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """Sample PUCT states at batch level, assign to prompts, then dispatch."""
        batch_size = len(prompts)
        logger.info(f"DiscoverAgentLoopManagerTQ.generate_sequences: batch_size={batch_size}, keys={list(prompts.keys())}")

        # Extract global_steps for PUCT tracking
        if "global_steps" in prompts:
            gs = prompts["global_steps"]
            if isinstance(gs, torch.Tensor):
                self._global_steps = int(gs[0].item()) if gs.dim() > 0 else int(gs.item())
            elif isinstance(gs, NonTensorData):
                self._global_steps = int(gs.data)

        # Batch-level PUCT sampling: sample all states at once for diversity
        states = ray.get(self._puct_actor.sample_states.remote(batch_size))

        # Assign states to prompts before dispatching to workers
        if "_puct_state" not in prompts.keys():
            state_data = NonTensorStack(*[NonTensorData(s) for s in states])
            prompts["_puct_state"] = state_data

        chunks = prompts.chunk(len(self.agent_loop_workers))
        ray.get([
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
        ])

        # PUCT flush is handled by trainer's _save_latest_checkpoint() after training step
