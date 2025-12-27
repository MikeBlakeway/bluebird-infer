"""Voice pod configuration and initialization."""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration for voice pod."""

    # Service
    SERVICE_NAME = "voice"
    SERVICE_PORT = int(os.getenv("VOICE_PORT", 8004))
    ENV = os.getenv("ENV", "dev")

    # Models
    DIFFSINGER_MODEL_PATH = os.getenv("DIFFSINGER_MODEL_PATH")
    VOCODER_MODEL_PATH = os.getenv("VOCODER_MODEL_PATH")

    # Idempotency
    REQUIRE_IDEMPOTENCY = os.getenv("REQUIRE_IDEMPOTENCY", "true").lower() == "true"

    # Limits
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 5000))
    MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", 600))  # 10 minutes

    # OTEL
    OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO" if ENV == "prod" else "DEBUG")


__all__ = ["Config"]
