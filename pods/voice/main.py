"""Voice Synthesis Pod: FastAPI service scaffold for AI vocals.

Provides health checks and a minimal /synthesize endpoint returning
generated test audio. Will be replaced with DiffSinger + vocoder
integration in subsequent steps.
"""

import io
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf

from config import Config
from .model import DiffSingerLoader
from .g2p import align_lyrics_to_phonemes


# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


class SynthesizeRequest(BaseModel):
    """Request to synthesize a vocal section (stub).

    This schema will be extended with phoneme alignment inputs,
    F0 curves, and timing once DiffSinger integration lands.
    """

    lyrics: List[str] = Field(default_factory=list, description="List of lyric lines")
    artist: Optional[str] = Field(default=None, description="AI artist/voice preset")
    bpm: int = Field(default=120, ge=40, le=300)
    duration: float = Field(default=4.0, ge=0.5, le=120.0)
    seed: int = Field(default=42, ge=0, description="Deterministic seed")
    f0: Optional[List[float]] = Field(default=None, description="Optional F0 curve (Hz) at ~100 fps")


class SynthesizeResponse(BaseModel):
    """Response with synthesized vocal audio info (stub)."""

    duration: float
    sample_rate: int
    bit_depth: int
    stem_name: str
    audio: str  # base64 encoded WAV
    message: str


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


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """Generate a simple test vocal audio buffer (stub).

    Produces a short sine tone sequence influenced by seed and lyric count,
    encoded as base64 WAV for transport. This validates the endpoint and
    audio pipeline ahead of model integration.
    """
    try:
        logger.info(
            "Received synthesize request (seed=%s, artist=%s, lines=%d, bpm=%s, dur=%.2f)",
            request.seed,
            request.artist,
            len(request.lyrics),
            request.bpm,
            request.duration,
        )

        sample_rate = 48000
        bit_depth = 24

        # Deterministic RNG seeded by request.seed
        rng = np.random.default_rng(request.seed)

        # Alignment stub (unused in audio, placeholder for future integration)
        _alignment = align_lyrics_to_phonemes(request.lyrics, request.bpm)

        sample_count = int(sample_rate * request.duration)

        # If an F0 curve is provided (frames at ~100 Hz), synthesize from it.
        if hasattr(request, "f0") and request.f0 is not None and len(request.f0) > 0:
            frame_times = np.linspace(0, request.duration, len(request.f0), endpoint=False)
            sample_times = np.linspace(0, request.duration, sample_count, endpoint=False)
            f0_track = np.interp(sample_times, frame_times, np.array(request.f0, dtype=np.float32))
            # Angular frequency per sample
            omega = 2 * np.pi * f0_track
            phase = np.cumsum(omega / sample_rate).astype(np.float32)
            sine = np.sin(phase).astype(np.float32)
            # Unvoiced handling: zero-amplitude where f0 <= 0
            voiced_mask = (f0_track > 0.0).astype(np.float32)
        else:
            # Base frequency derived from seed and lyric count
            base_freq = 220.0 + (request.seed % 100)  # 220–319 Hz
            line_mod = max(1, len(request.lyrics))
            freq = base_freq * (1.0 + 0.05 * (line_mod - 1))
            t = np.linspace(0, request.duration, sample_count, endpoint=False)
            sine = np.sin(2 * np.pi * freq * t).astype(np.float32)
            voiced_mask = np.ones_like(sine, dtype=np.float32)

        # Envelope: attack 50ms, decay 200ms, sustain 0.7, release last 200ms
        attack = int(0.050 * sample_rate)
        decay = int(0.200 * sample_rate)
        release = int(0.200 * sample_rate)
        sustain_len = max(0, len(sine) - (attack + decay + release))
        env = np.concatenate([
            np.linspace(0.0, 1.0, attack, endpoint=False),
            np.linspace(1.0, 0.7, decay, endpoint=False),
            np.full(sustain_len, 0.7, dtype=np.float32),
            np.linspace(0.7, 0.0, release, endpoint=False),
        ]).astype(np.float32)
        env = env[: len(sine)]
        audio = (sine * env * voiced_mask).astype(np.float32)

        # Convert to mono WAV bytes (PCM_24)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sample_rate, subtype="PCM_24", format="WAV")
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.getvalue()

        import base64

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        message = "Generated F0-driven vocal tone" if (hasattr(request, "f0") and request.f0) else "Generated test vocal tone"

        return {
            "duration": request.duration,
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
            "stem_name": "vocals",
            "audio": audio_b64,
            "message": message,
        }

    except Exception as e:
        logger.error(f"Voice synthesis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level="info",
    )
