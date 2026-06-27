from ttt_discover.local_backend.types import (
    AdamParams,
    APIFuture as APIFutureType,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    LossFnType,
    ModelInput,
    ModelInputChunk,
    OptimStepResponse,
    SampleResult,
    SampleSequence,
    SamplingParams,
    SaveStateResponse,
    TensorData,
)
from ttt_discover.local_backend.sampling_client import LocalSamplingClient
from ttt_discover.local_backend.training_client import LocalTrainingClient
from ttt_discover.local_backend.distributed_training_client import DistributedTrainingClient
from ttt_discover.local_backend.service_client import LocalServiceClient
from ttt_discover.local_backend.future import LocalFuture

# Expose types as a sub-module so tinker.types.EncodedTextChunk works
from ttt_discover.local_backend import types

# Provide aliases matching the tinker SDK interface
SamplingClient = LocalSamplingClient
TrainingClient = LocalTrainingClient
ServiceClient = LocalServiceClient
APIFuture = LocalFuture

__all__ = [
    "AdamParams",
    "APIFutureType",
    "Datum",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "LossFnType",
    "ModelInput",
    "ModelInputChunk",
    "OptimStepResponse",
    "SampleResult",
    "SampleSequence",
    "SamplingClient",
    "SamplingParams",
    "SaveStateResponse",
    "ServiceClient",
    "TensorData",
    "TrainingClient",
    "types",
    "LocalFuture",
    "LocalSamplingClient",
    "LocalTrainingClient",
    "LocalServiceClient",
    "APIFuture",
]
