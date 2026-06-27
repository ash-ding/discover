import logging
import os

from ttt_discover.local_backend.sampling_client import LocalSamplingClient
from ttt_discover.local_backend.training_client import LocalTrainingClient

logger = logging.getLogger(__name__)


class LocalServiceClient:
    """Orchestrates HTTP sampling client + local PEFT training client."""

    def __init__(
        self,
        model_name_or_path: str,
        inference_gpu_id: int = 0,
        training_gpu_id: int = 1,
        inference_tp_size: int = 1,
        max_model_len: int = 32768,
        experiment_name: str = "default",
        training_batch_size: int = 1,
        max_train_seq_len: int = 32768,
        training_gpu_ids: list | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.training_gpu_id = training_gpu_id
        self.experiment_name = experiment_name
        self.training_batch_size = training_batch_size
        self.max_train_seq_len = max_train_seq_len
        self.training_gpu_ids = training_gpu_ids
        self.vllm_base_url = os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000"
        )
        self._tokenizer = None
        self._inference_client = None

    @property
    def _use_distributed(self):
        return self.training_gpu_ids and len(self.training_gpu_ids) > 1

    def _get_inference_client(self) -> LocalSamplingClient:
        if self._inference_client is None:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    use_fast=True,
                    trust_remote_code=True,
                )
            self._inference_client = LocalSamplingClient(
                base_url=self.vllm_base_url,
                model_name=self.model_name_or_path,
                tokenizer=self._tokenizer,
            )
        return self._inference_client

    def _create_distributed_client(self, lora_rank, from_checkpoint=None, load_optimizer=False):
        from ttt_discover.local_backend.distributed_training_client import DistributedTrainingClient
        tc = DistributedTrainingClient(
            model_name_or_path=self.model_name_or_path,
            lora_rank=lora_rank,
            gpu_ids=self.training_gpu_ids,
            checkpoint_dir=f"./tinker_log/local_checkpoints/{self.experiment_name}",
            max_train_seq_len=self.max_train_seq_len,
            from_checkpoint=from_checkpoint,
            load_optimizer=load_optimizer,
        )
        return tc

    async def create_lora_training_client_async(
        self, model_name: str, rank: int = 32
    ):
        if self._use_distributed:
            tc = self._create_distributed_client(lora_rank=rank)
        else:
            tc = LocalTrainingClient(
                model_name_or_path=self.model_name_or_path,
                lora_rank=rank,
                gpu_id=self.training_gpu_id,
                checkpoint_dir=f"./tinker_log/local_checkpoints/{self.experiment_name}",
                training_batch_size=self.training_batch_size,
                max_train_seq_len=self.max_train_seq_len,
            )
        self._tokenizer = tc.get_tokenizer()
        inference_client = self._get_inference_client()
        tc.set_shared_sampling_client(inference_client)
        return tc

    async def create_training_client_from_state_async(
        self, state_path: str
    ):
        if self._use_distributed:
            tc = self._create_distributed_client(
                lora_rank=32, from_checkpoint=state_path, load_optimizer=False,
            )
        else:
            tc = LocalTrainingClient.from_checkpoint(
                model_name_or_path=self.model_name_or_path,
                checkpoint_path=state_path,
                gpu_id=self.training_gpu_id,
                load_optimizer=False,
                checkpoint_dir=f"./tinker_log/local_checkpoints/{self.experiment_name}",
                training_batch_size=self.training_batch_size,
                max_train_seq_len=self.max_train_seq_len,
            )
        self._tokenizer = tc.get_tokenizer()
        inference_client = self._get_inference_client()
        tc.set_shared_sampling_client(inference_client)
        return tc

    async def create_training_client_from_state_with_optimizer_async(
        self, state_path: str
    ):
        if self._use_distributed:
            tc = self._create_distributed_client(
                lora_rank=32, from_checkpoint=state_path, load_optimizer=True,
            )
        else:
            tc = LocalTrainingClient.from_checkpoint(
                model_name_or_path=self.model_name_or_path,
                checkpoint_path=state_path,
                gpu_id=self.training_gpu_id,
                load_optimizer=True,
                checkpoint_dir=f"./tinker_log/local_checkpoints/{self.experiment_name}",
                training_batch_size=self.training_batch_size,
                max_train_seq_len=self.max_train_seq_len,
            )
        self._tokenizer = tc.get_tokenizer()
        inference_client = self._get_inference_client()
        tc.set_shared_sampling_client(inference_client)
        return tc

    def create_sampling_client(
        self, base_model: str | None = None
    ) -> LocalSamplingClient:
        """Return an HTTP client without LoRA — used for base model logprobs (KL penalty)."""
        # Use the provided base_model parameter if given, otherwise fall back to self.model_name_or_path
        model_to_use = base_model if base_model is not None else self.model_name_or_path

        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_to_use,
                use_fast=True,
                trust_remote_code=True,
            )
        return LocalSamplingClient(
            base_url=self.vllm_base_url,
            model_name=model_to_use,
            lora_name=None,
            tokenizer=self._tokenizer,
        )
