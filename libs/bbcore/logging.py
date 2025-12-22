"""Structured logging and optional OpenTelemetry setup for Bluebird pods."""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional

from .config import get_settings

try:
    # Optional OTEL tracing
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - OTEL is optional
    trace = None  # type: ignore


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        # Attach trace IDs if available
        try:
            span = trace.get_current_span() if trace else None
            if span and span.get_span_context():
                ctx = span.get_span_context()
                payload["trace_id"] = format(ctx.trace_id, "032x")
                payload["span_id"] = format(ctx.span_id, "016x")
        except Exception:
            pass
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: Optional[str] = None) -> None:
    s = get_settings()
    log_level = (level or s.BB_LOG_LEVEL).upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level, logging.INFO))


def setup_tracing(service_name: str = "bluebird-pod") -> None:
    """Initialize OpenTelemetry tracing if endpoint is configured."""
    s = get_settings()
    if not s.BB_OTEL_ENDPOINT or not trace:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=s.BB_OTEL_ENDPOINT))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["setup_logging", "setup_tracing", "get_logger", "JsonFormatter"]
