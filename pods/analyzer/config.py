"""Analyzer pod configuration and initialization."""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration for analyzer pod."""

    # Service
    SERVICE_NAME = "analyzer"
    SERVICE_PORT = int(os.getenv("ANALYZER_PORT", 8001))
    ENV = os.getenv("ENV", "dev")

    # Limits
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10 MB
    MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", 600))  # 10 minutes

    # OTEL
    OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO" if ENV == "prod" else "DEBUG")


__all__ = ["Config"]
