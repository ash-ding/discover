import logging

from ttt_discover.local_backend.sampling_client import LocalSamplingClient
from ttt_discover.local_backend.training_client import LocalTrainingClient

logger = logging.getLogger(__name__)


class LocalServiceClient:
    """Drop-in replacement for tinker.ServiceClient for local RL training."""

    def __init__(
        self,
        model_name_or_path: str,
        inference_gpu_id: int = 0,
        training_gpu_id: int = 1,
        inference_tp_size: int = 1,
        max_model_len: int = 32768,
        experiment_name: str = "default",
    ):
        self.model_name_or_path = model_name_or_path
        self.inference_gpu_id = inference_gpu_id
        self.training_gpu_id = training_gpu_id
        self.inference_tp_size = inference_tp_size
        self.max_model_len = max_model_len
        self.experiment_name = experiment_name
        self._inference_client = None

    def _get_inference_client(self) -> LocalSamplingClient:
        if self._inference_client is None:
            self._inference_client = LocalSamplingClient(
                model_name_or_path=self.model_name_or_path,
                gpu_id=self.inference_gpu_id,
                lora_adapter_path=None,
                tensor_parallel_size=self.inference_tp_size,
                max_model_len=self.max_model_len,
            )
        return self._inference_client

    def _attach_shared_client(self, training_client: LocalTrainingClient):
        training_client.set_shared_sampling_client(self._get_inference_client())

    async def create_lora_training_client_async(
        self, model_name: str, rank: int = 32
    ) -> LocalTrainingClient:
        self._get_inference_client()
        tc = LocalTrainingClient(
            model_name_or_path=self.model_name_or_path,
            lora_rank=rank,
            gpu_id=self.training_gpu_id,
            checkpoint_dir=f"./tinker_log/local_checkpoints/{self.experiment_name}",
        )
        tc.set_shared_sampling_client(self._inference_client)
        return tc

    async def create_training_client_from_state_async(
        self, state_path: str
    ) -> LocalTrainingClient:
        self._get_inference_client()
        tc = LocalTrainingClient.from_checkpoint(
            model_name_or_path=self.model_name_or_path,
            checkpoint_path=state_path,
            gpu_id=self.training_gpu_id,
            load_optimizer=False,
            checkpoint_dir=f"./tinker_log/local_checkpoints/{self.experiment_name}",
        )
        tc.set_shared_sampling_client(self._inference_client)
        return tc

    async def create_training_client_from_state_with_optimizer_async(
        self, state_path: str
    ) -> LocalTrainingClient:
        self._get_inference_client()
        tc = LocalTrainingClient.from_checkpoint(
            model_name_or_path=self.model_name_or_path,
            checkpoint_path=state_path,
            gpu_id=self.training_gpu_id,
            load_optimizer=True,
            checkpoint_dir=f"./tinker_log/local_checkpoints/{self.experiment_name}",
        )
        self._attach_shared_client(tc)
        return tc

    def create_sampling_client(self, base_model: str | None = None) -> "BaseModelSamplingProxy":
        return BaseModelSamplingProxy(self._get_inference_client())


class BaseModelSamplingProxy:
    """Proxy that temporarily disables LoRA for base model logprob computation (KL penalty)."""

    def __init__(self, client: LocalSamplingClient):
        self._client = client

    async def compute_logprobs_async(self, sequence) -> list[float]:
        saved = self._client.lora_adapter_path
        self._client.lora_adapter_path = None
        try:
            return await self._client.compute_logprobs_async(sequence)
        finally:
            self._client.lora_adapter_path = saved
