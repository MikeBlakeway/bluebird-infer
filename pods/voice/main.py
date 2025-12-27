"""Voice Synthesis Pod: FastAPI service scaffold for AI vocals.

Provides health checks and a minimal /synthesize endpoint returning
generated test audio. Will be replaced with DiffSinger + vocoder
integration in subsequent steps.
"""

import io
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Header
from opentelemetry import trace
from pydantic import BaseModel, Field

from config import Config
from model import DiffSingerLoader
from g2p import get_aligner


# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)
tracer = trace.get_tracer("bluebird.voice")


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
    speaker_id: Optional[str] = Field(default=None, description="Voice speaker identifier")
    bpm: int = Field(default=120, ge=40, le=300)
    duration: float = Field(default=4.0, ge=0.5, le=120.0)
    seed: int = Field(default=42, ge=0, description="Deterministic seed")
    f0: Optional[List[float]] = Field(default=None, description="Optional F0 curve (Hz) at ~100 fps")
    phonemes: Optional[List[str]] = Field(default=None, description="Aligned phoneme sequence")
    durations: Optional[List[float]] = Field(default=None, description="Durations per phoneme (seconds)")
    noise_scale: float = Field(default=0.667, ge=0.0, le=2.0, description="Synthesis noise scale")
    energy: Optional[List[float]] = Field(default=None, description="Optional energy envelope")


class SynthesizeResponse(BaseModel):
    """Response with synthesized vocal audio info (stub)."""

    duration: float
    sample_rate: int
    bit_depth: int
    stem_name: str
    audio: str  # base64 encoded WAV
    message: str
    speaker_id: Optional[str] = None
    phoneme_count: int
    has_f0: bool


# Service metadata
SERVICE_NAME = Config.SERVICE_NAME
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = Config.SERVICE_PORT


loader = DiffSingerLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info(f"{SERVICE_NAME} pod starting (v{SERVICE_VERSION})...")
    try:
        loader.initialize()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to initialize models: %s", exc)
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
    status = "ok" if loader.is_ready() else "starting"
    if loader.get_error():
        status = "error"
    return {
        "status": status,
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "ready": loader.is_ready(),
    }


def _synthesize_fallback(
    f0_curve: np.ndarray,
    duration: float,
    sample_rate: int = 48000,
    seed: int = 0,
) -> np.ndarray:
    """Fallback F0-driven sine synthesis when models unavailable.

    Used when DiffSinger models are loading or unavailable.
    Provides deterministic output based on seed for preview purposes.
    """
    np.random.seed(seed)
    sample_count = int(sample_rate * duration)

    # Use provided F0 curve or default to constant frequency
    if len(f0_curve) > 0:
        frame_times = np.linspace(0, duration, len(f0_curve), endpoint=False)
        sample_times = np.linspace(0, duration, sample_count, endpoint=False)
        f0_track = np.interp(
            sample_times,
            frame_times,
            f0_curve,
            left=f0_curve[0] if len(f0_curve) > 0 else 200.0,
            right=f0_curve[-1] if len(f0_curve) > 0 else 200.0,
        )
        voiced_mask = (f0_track > 0.0).astype(np.float32)
    else:
        # Default base frequency derived from seed
        base_freq = 220.0 + (seed % 100)
        f0_track = np.full(sample_count, base_freq, dtype=np.float32)
        voiced_mask = np.ones_like(f0_track, dtype=np.float32)

    # Generate phase-based sine
    omega = 2 * np.pi * f0_track.astype(np.float32)
    phase = np.cumsum(omega / sample_rate).astype(np.float32)
    sine = np.sin(phase).astype(np.float32)

    # Simple envelope: attack 50ms, release 200ms
    attack_samples = int(0.050 * sample_rate)
    release_samples = int(0.200 * sample_rate)
    sustain_samples = max(0, sample_count - attack_samples - release_samples)

    envelope = np.concatenate([
        np.linspace(0.0, 1.0, attack_samples, endpoint=False),
        np.full(sustain_samples, 1.0, dtype=np.float32),
        np.linspace(1.0, 0.0, release_samples, endpoint=False),
    ]).astype(np.float32)
    envelope = envelope[:sample_count]

    # Apply envelope and voiced mask
    audio = (sine * envelope * voiced_mask).astype(np.float32)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(
    request: SynthesizeRequest,
    idempotency_key: Optional[str] = Header(default=None, convert_underscores=False, alias="Idempotency-Key"),
):
    """Generate a simple test vocal audio buffer (stub).

    Produces a short sine tone sequence influenced by seed and lyric count,
    encoded as base64 WAV for transport. This validates the endpoint and
    audio pipeline ahead of model integration.
    """
    if Config.REQUIRE_IDEMPOTENCY and not idempotency_key:
        raise HTTPException(status_code=400, detail="Missing Idempotency-Key header")

    if not loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not initialized")

    with tracer.start_as_current_span("voice.synthesize") as span:
        try:
            span.set_attribute("voice.seed", request.seed)
            span.set_attribute("voice.artist", request.artist or "")
            span.set_attribute("voice.bpm", request.bpm)
            span.set_attribute("voice.duration", request.duration)
            span.set_attribute("voice.idempotency_key", idempotency_key or "")
            span.set_attribute("voice.lines", len(request.lyrics))
            span.set_attribute("voice.speaker_id", request.speaker_id or "")
            if request.f0:
                span.set_attribute("voice.f0_frames", len(request.f0))
            if request.phonemes:
                span.set_attribute("voice.phonemes", len(request.phonemes))

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

            # Alignment using real G2P
            aligner = get_aligner()
            alignment = aligner.align_lyrics_to_frames(
                request.lyrics,
                request.bpm,
                request.duration,
                provided_phonemes=request.phonemes,
                provided_durations=request.durations,
            )

            span.set_attribute("voice.alignment_phonemes", len(alignment.get("phonemes", [])))
            span.set_attribute("voice.alignment_frames", alignment.get("frame_count", 0))

            # Validate phoneme/duration pairing if provided
            if request.phonemes is not None:
                if request.durations is None or len(request.phonemes) != len(request.durations):
                    raise HTTPException(status_code=400, detail="phonemes and durations must be provided with equal length")

            if request.durations is not None and sum(request.durations) > request.duration + 1e-3:
                raise HTTPException(status_code=400, detail="Sum of durations exceeds requested duration")

            # Prepare F0 curve for synthesis
            f0_curve = np.array(request.f0, dtype=np.float32) if hasattr(request, "f0") and request.f0 else np.array([], dtype=np.float32)

            # Determine phonemes for synthesis
            if alignment.get("phonemes"):
                synthesis_phonemes = alignment["phonemes"]
                synthesis_durations = alignment.get("durations", [])
            elif request.phonemes:
                synthesis_phonemes = request.phonemes
                synthesis_durations = request.durations or [request.duration / len(request.phonemes)] * len(request.phonemes)
            else:
                synthesis_phonemes = []
                synthesis_durations = []

            span.set_attribute("voice.synthesis_phonemes", len(synthesis_phonemes))

            # Try real synthesis if models loaded, fallback to stub
            try:
                if loader.is_ready():
                    audio = loader.synthesize(
                        phonemes=synthesis_phonemes,
                        durations=synthesis_durations,
                        f0_curve=f0_curve,
                        speaker_id=request.speaker_id or "default",
                        seed=request.seed,
                    )
                    span.set_attribute("voice.synthesis_method", "diffsinger")
                    message = "Generated using DiffSinger synthesis"
                else:
                    logger.warning("Models not ready; using fallback synthesis (seed=%d)", request.seed)
                    audio = _synthesize_fallback(
                        f0_curve=f0_curve,
                        duration=request.duration,
                        sample_rate=sample_rate,
                        seed=request.seed,
                    )
                    span.set_attribute("voice.synthesis_method", "fallback")
                    message = "Generated using fallback F0 synthesis (models loading)"
            except Exception as e:
                logger.error("Real synthesis failed: %s; using fallback", e)
                audio = _synthesize_fallback(
                    f0_curve=f0_curve,
                    duration=request.duration,
                    sample_rate=sample_rate,
                    seed=request.seed,
                )
                span.set_attribute("voice.synthesis_method", "fallback_error")
                message = f"Generated using fallback synthesis (error: {type(e).__name__})"

            # Convert to mono WAV bytes (PCM_24)
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, sample_rate, subtype="PCM_24", format="WAV")
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.getvalue()

            import base64

            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

            return {
                "duration": request.duration,
                "sample_rate": sample_rate,
                "bit_depth": bit_depth,
                "stem_name": "vocals",
                "audio": audio_b64,
                "message": message,
                "speaker_id": request.speaker_id,
                "phoneme_count": len(request.phonemes) if request.phonemes else 0,
                "has_f0": bool(request.f0),
            }
        except HTTPException:
            # Re-raise HTTPException without modification (FastAPI will handle)
            raise
        except Exception as e:
            import traceback
            error_detail = str(e) if str(e) else f"{type(e).__name__}"
            logger.error(f"Synthesis error: {type(e).__name__}: {error_detail}", exc_info=True)
            raise HTTPException(status_code=400, detail=error_detail)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level="info",
    )
