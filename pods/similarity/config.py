"""Configuration for Similarity Pod."""

import os
from typing import Optional


class Config:
    """Pod configuration from environment."""

    SERVICE_NAME = "similarity"
    SERVICE_VERSION = "0.1.0"
    SERVICE_PORT = int(os.getenv("BB_SIMILARITY_PORT", "8005"))

    # S3 Configuration
    S3_ENDPOINT = os.getenv("BB_S3_ENDPOINT", "http://localhost:9000")
    S3_ACCESS_KEY = os.getenv("BB_S3_ACCESS_KEY", "minioadmin")
    S3_SECRET = os.getenv("BB_S3_SECRET", "minioadmin")
    S3_BUCKET = os.getenv("BB_BUCKET", "bluebird")

    # Logging
    LOG_LEVEL = os.getenv("BB_LOG_LEVEL", "INFO")

    # OTEL (Optional)
    OTEL_ENDPOINT: Optional[str] = os.getenv("BB_OTEL_ENDPOINT")

    # Similarity Thresholds (will be exposed as settings later)
    PASS_THRESHOLD = 0.35           # score < 0.35 = pass
    BORDERLINE_THRESHOLD = 0.48     # 0.35 <= score < 0.48 = borderline
    BLOCK_THRESHOLD = 0.48          # score >= 0.48 = block


config = Config()
