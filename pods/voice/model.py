"""DiffSinger model loader scaffold.

Provides a placeholder loader structure to be replaced with real
DiffSinger + vocoder initialization in later steps.
"""

import os
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    diffsinger_model_path: str | None = None
    vocoder_model_path: str | None = None


class DiffSingerLoader:
    """Scaffold for DiffSinger model and vocoder initialization."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig(
            diffsinger_model_path=os.getenv("DIFFSINGER_MODEL_PATH"),
            vocoder_model_path=os.getenv("VOCODER_MODEL_PATH"),
        )
        self._ready = False

    def initialize(self) -> None:
        """Initialize models if paths are provided. Stub implementation."""
        if self.config.diffsinger_model_path:
            logger.info("DiffSinger model path set: %s", self.config.diffsinger_model_path)
        if self.config.vocoder_model_path:
            logger.info("Vocoder model path set: %s", self.config.vocoder_model_path)
        # In a future step, load actual weights here.
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready
