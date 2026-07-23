from enum import Enum
from typing import Type

import structlog

from libkernelbot.consts import GPU
from libkernelbot.report import RunProgressReporter

logger = structlog.get_logger(__name__)


class Launcher:
    def __init__(self, name: str, gpus: Type[Enum]):
        self.name = name
        self.gpus = gpus

    async def run_submission(self, config: dict, gpu_type: GPU, status: RunProgressReporter):
        logger.info("run_submission_called", launcher=self.name, gpu_type=gpu_type.name)
        raise NotImplementedError()
