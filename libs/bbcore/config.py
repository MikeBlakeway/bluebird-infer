"""Configuration loading for Bluebird inference pods.

Reads environment variables into a typed settings object using Pydantic v2.

Env variables (see .env.example):
- BB_S3_ENDPOINT
- BB_S3_ACCESS_KEY
- BB_S3_SECRET
- BB_BUCKET (default: bluebird)
- BB_OTEL_ENDPOINT (optional)
- BB_LOG_LEVEL (default: INFO)
- BB_ENV (default: development)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field


class Settings(BaseModel):
    BB_S3_ENDPOINT: str = Field(..., description="MinIO/S3 endpoint URL")
    BB_S3_ACCESS_KEY: str = Field(..., description="S3 access key")
    BB_S3_SECRET: str = Field(..., description="S3 secret key")
    BB_BUCKET: str = Field(default="bluebird", description="S3 bucket name")

    BB_OTEL_ENDPOINT: Optional[str] = Field(
        default=None, description="OTLP HTTP endpoint (e.g., http://localhost:4318)"
    )
    BB_LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    BB_ENV: str = Field(default="development", description="Environment name")

    class Config:
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment and memoize.

    Raises:
        ValueError: if required environment variables are missing.
    """

    env = {
        "BB_S3_ENDPOINT": os.getenv("BB_S3_ENDPOINT"),
        "BB_S3_ACCESS_KEY": os.getenv("BB_S3_ACCESS_KEY"),
        "BB_S3_SECRET": os.getenv("BB_S3_SECRET"),
        "BB_BUCKET": os.getenv("BB_BUCKET", "bluebird"),
        "BB_OTEL_ENDPOINT": os.getenv("BB_OTEL_ENDPOINT"),
        "BB_LOG_LEVEL": os.getenv("BB_LOG_LEVEL", "INFO"),
        "BB_ENV": os.getenv("BB_ENV", "development"),
    }

    # Validate presence of required fields
    missing = [k for k in ("BB_S3_ENDPOINT", "BB_S3_ACCESS_KEY", "BB_S3_SECRET") if not env[k]]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return Settings.model_validate(env)


__all__ = ["Settings", "get_settings"]
