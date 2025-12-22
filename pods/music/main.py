"""Music Synthesis Pod: FastAPI service for procedural music generation.

Generates drums, bass, and guitar tracks with perfect grid alignment.
"""

import asyncio
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from synth import BassSynth, DrumSynth, GuitarSynth, SynthConfig
from grid import GridAligner, GridRenderer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynthesizeRequest(BaseModel):
    """Request to synthesize a music section."""

    bpm: int = Field(default=120, ge=40, le=300)
    duration: float = Field(default=8.0, ge=0.5, le=120.0)
    include_drums: bool = True
    include_bass: bool = True
    include_guitar: bool = False
    seed: int = Field(default=42, ge=0)
    bass_note: int = Field(default=36, ge=0, le=127)
    master_volume: float = Field(default=1.0, ge=0.1, le=2.0)


class SynthesizeResponse(BaseModel):
    """Response with synthesized audio info."""

    duration: float
    sample_rate: int
    bit_depth: int
    stems: dict[str, str]  # stem_name -> base64 encoded WAV
    mixed_audio: str  # base64 encoded WAV
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


# Service configuration
SERVICE_NAME = "music"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8002


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info(f"{SERVICE_NAME} pod starting (v{SERVICE_VERSION})...")
    yield
    logger.info(f"{SERVICE_NAME} pod shutting down...")


app = FastAPI(
    title="Music Synthesis Pod",
    description="Procedural music generation for drums, bass, guitar",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """API info endpoint."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "description": "Procedural music synthesis (drums, bass, guitar)",
        "endpoints": {
            "GET /": "This info",
            "POST /health": "Health check",
            "POST /synthesize": "Generate music section",
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
    """Synthesize a music section with specified instruments.

    Args:
        request: Synthesis parameters (BPM, duration, instruments, seed)

    Returns:
        Audio files as base64-encoded WAV data
    """
    try:
        config = SynthConfig(sample_rate=48000, bit_depth=24)

        # Create synthesizers with seed for determinism
        drum_synth = DrumSynth(sample_rate=config.sample_rate, seed=request.seed)
        bass_synth = BassSynth(sample_rate=config.sample_rate, seed=request.seed)
        guitar_synth = GuitarSynth(sample_rate=config.sample_rate, seed=request.seed)

        # Create grid aligner for perfect timing
        aligner = GridAligner(sample_rate=config.sample_rate)
        renderer = GridRenderer(
            sample_rate=config.sample_rate, bpm=request.bpm
        )

        # Generate stems
        stems = {}

        if request.include_drums:
            drums = drum_synth.generate_pattern(
                request.duration, bpm=request.bpm
            )
            stems["drums"] = drums

        if request.include_bass:
            # Simple bass pattern: alternating between root and fifth
            bass_note = request.bass_note
            bass_fifth = bass_note + 7  # Perfect fifth
            bass_notes = [
                bass_note,
                bass_note,
                bass_fifth,
                bass_note,
            ]  # I-I-V-I pattern
            bass_line = bass_synth.generate_line(
                bass_notes, note_duration=2.0, bpm=request.bpm
            )
            stems["bass"] = bass_line

        if request.include_guitar:
            # Simple guitar chord
            chord_notes = [40, 45, 50, 55]  # Em chord on bass strings
            guitar = guitar_synth.generate_chord(chord_notes, request.duration)
            stems["guitar"] = guitar

        # Render with grid alignment
        mix_levels = {
            "drums": 0.8 * request.master_volume,
            "bass": 0.7 * request.master_volume,
            "guitar": 0.5 * request.master_volume,
        }

        mixed_audio, aligned_stems = renderer.render_section(
            request.duration, stems, mix_levels
        )

        # Convert to WAV bytes
        def audio_to_wav(audio: np.ndarray) -> bytes:
            """Convert audio array to WAV bytes."""
            wav_buffer = io.BytesIO()
            sf.write(
                wav_buffer,
                audio,
                config.sample_rate,
                subtype="PCM_24",
                format="WAV",
            )
            wav_buffer.seek(0)
            return wav_buffer.getvalue()

        # Prepare response
        stem_wav_data = {}
        for stem_name, stem_audio in aligned_stems.items():
            wav_bytes = audio_to_wav(stem_audio)
            # Encode as base64 for JSON transport
            import base64
            stem_wav_data[stem_name] = base64.b64encode(wav_bytes).decode("utf-8")

        mixed_wav = audio_to_wav(mixed_audio)
        import base64
        mixed_b64 = base64.b64encode(mixed_wav).decode("utf-8")

        return {
            "duration": request.duration,
            "sample_rate": config.sample_rate,
            "bit_depth": config.bit_depth,
            "stems": stem_wav_data,
            "mixed_audio": mixed_b64,
            "message": f"Generated {len(stems)} stems at {request.bpm} BPM",
        }

    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level="info",
    )
