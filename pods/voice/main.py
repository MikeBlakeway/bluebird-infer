"""Voice Synthesis Pod: FastAPI service scaffold for AI vocals.

Provides health checks and a placeholder /synthesize endpoint to be
implemented in subsequent days with DiffSinger + vocoder integration.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Config


# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


class SynthesizeRequest(BaseModel):
    """Request to synthesize a vocal section (placeholder).

    This schema will be extended to include phoneme alignment inputs,
    F0 curves, and timing once DiffSinger integration lands.
    """

    lyrics: List[str] = Field(default_factory=list, description="List of lyric lines")
    artist: Optional[str] = Field(default=None, description="AI artist/voice preset")
    bpm: Optional[int] = Field(default=None, ge=40, le=300)
    duration: Optional[float] = Field(default=None, ge=0.5, le=120.0)
    seed: int = Field(default=42, ge=0, description="Deterministic seed")


class SynthesizePlaceholderResponse(BaseModel):
    """Placeholder response indicating synthesis is not yet implemented."""

    status: str
    message: str
    service: str
    version: str


# Service metadata
SERVICE_NAME = Config.SERVICE_NAME
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = Config.SERVICE_PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info(f"{SERVICE_NAME} pod starting (v{SERVICE_VERSION})...")
    yield
    logger.info(f"{SERVICE_NAME} pod shutting down...")


app = FastAPI(
    title="Voice Synthesis Pod",
    description="AI vocal synthesis scaffold awaiting DiffSinger integration",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """API info endpoint."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "description": "AI voice synthesis (scaffold)",
        "endpoints": {
            "GET /": "This info",
            "POST /health": "Health check",
            "POST /synthesize": "Placeholder for voice synthesis",
        },
    }


@app.post("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
    }


@app.post("/synthesize", response_model=SynthesizePlaceholderResponse)
async def synthesize(request: SynthesizeRequest):
    """Placeholder endpoint for voice synthesis.

    Returns a 501-like response indicating not implemented yet.
    """
    logger.info(
        "Received synthesize request (seed=%s, artist=%s, lines=%d)",
        request.seed,
        request.artist,
        len(request.lyrics),
    )

    # For now, return a placeholder response. Actual synthesis will be added
    # when DiffSinger + vocoder integration is implemented.
    return {
        "status": "not-implemented",
        "message": "Voice synthesis not yet integrated",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level="info",
    )
