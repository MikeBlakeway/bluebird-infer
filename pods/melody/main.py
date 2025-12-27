"""Melody Generation Pod: FastAPI service for MIDI + F0 melodies.

Generates deterministic melodies from syllables and optional contour,
optionally returning an F0 curve for downstream processors.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from bbcore.logging import setup_logging, setup_tracing
from bbmelody.generator import generate_melody
from bbmelody.midi_utils import melody_to_f0_curve, quantize_melody
from bbmelody.chord_utils import COMMON_PROGRESSIONS

logger = logging.getLogger(__name__)

SERVICE_NAME = "melody"
SERVICE_VERSION = "0.1.0"
DEFAULT_PORT = 8003


class MelodyNote(BaseModel):
    """Single melody note."""

    midi: int = Field(..., ge=0, le=127)
    onset: float = Field(..., ge=0.0)
    duration: float = Field(..., gt=0.0)


class F0Curve(BaseModel):
    """Continuous F0 curve."""

    times: List[float]
    values: List[float]


class MelodyMetadata(BaseModel):
    """Metadata describing the generated melody."""

    key: str
    bpm: int
    progression: str
    note_range: Tuple[int, int]
    quantized: bool
    grid_resolution: Optional[float] = None
    sample_rate: int
    hop_length: int


class GenerateMelodyRequest(BaseModel):
    """Request payload for melody generation."""

    syllables: List[str] = Field(..., min_length=1, description="Lyric syllables")
    key: str = Field(default="C", description="Key, e.g., C, Dm, F#")
    bpm: int = Field(default=120, ge=40, le=300)
    progression: str = Field(default="pop1")
    contour: Optional[List[float]] = Field(
        default=None, description="Optional contour guidance [-1,1]"
    )
    note_range: Tuple[int, int] = Field(default=(60, 72), description="(min, max) MIDI")
    seed: int = Field(default=42, ge=0, description="Deterministic seed")
    quantize: bool = Field(default=False, description="Snap timing to grid")
    grid_resolution: float = Field(
        default=0.25,
        gt=0.0,
        description="Grid size in seconds when quantizing",
    )
    emit_f0: bool = Field(default=False, description="Include F0 curve output")
    sample_rate: int = Field(default=16000, ge=8000, le=96000)
    hop_length: int = Field(default=160, ge=1)

    @field_validator("note_range")
    @classmethod
    def validate_note_range(cls, value: Tuple[int, int]) -> Tuple[int, int]:
        low, high = value
        if low < 0 or high > 127:
            raise ValueError("note_range must be within MIDI 0-127")
        if low >= high:
            raise ValueError("note_range min must be less than max")
        return value

    @field_validator("progression")
    @classmethod
    def validate_progression(cls, value: str) -> str:
        if value not in COMMON_PROGRESSIONS:
            raise ValueError(f"Unknown progression: {value}")
        return value

    @field_validator("contour")
    @classmethod
    def validate_contour(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if not value:
            raise ValueError("contour must not be empty when provided")
        return value

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("key must be provided")
        return value.strip()


class GenerateMelodyResponse(BaseModel):
    """Response payload with notes and optional F0."""

    notes: List[MelodyNote]
    seed: int
    metadata: MelodyMetadata
    f0: Optional[F0Curve] = None
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks for logging/tracing."""
    try:
        setup_logging()
    except Exception as exc:  # pragma: no cover - logging fallback
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"Logging fallback (missing env?): {exc}")

    try:
        setup_tracing(service_name=f"{SERVICE_NAME}-pod")
    except Exception as exc:  # pragma: no cover - optional tracing
        logger.info(f"Tracing not configured: {exc}")
    logger.info(f"{SERVICE_NAME} pod starting (v{SERVICE_VERSION})...")
    yield
    logger.info(f"{SERVICE_NAME} pod shutting down...")


app = FastAPI(
    title="Melody Pod",
    description="Deterministic melody generation (MIDI + F0)",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """API info endpoint."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "description": "Procedural melody generation (MIDI + optional F0)",
        "endpoints": {
            "GET /": "This info",
            "POST /health": "Health check",
            "POST /generate": "Generate melody",
        },
    }


@app.post("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.post("/generate", response_model=GenerateMelodyResponse)
async def generate(request: GenerateMelodyRequest):
    """Generate melody notes (and optional F0 curve)."""
    try:
        # Generate melody with deterministic seed
        melody = generate_melody(
            syllables=request.syllables,
            key=request.key,
            bpm=request.bpm,
            progression_name=request.progression,
            contour=request.contour,
            seed=request.seed,
            note_range=request.note_range,
        )

        if request.quantize:
            melody = quantize_melody(melody, grid_resolution=request.grid_resolution)

        # Build note objects
        notes = [
            MelodyNote(midi=int(pitch), onset=float(onset), duration=float(duration))
            for pitch, onset, duration in melody
        ]

        f0_payload = None
        if request.emit_f0:
            times, f0_curve = melody_to_f0_curve(
                melody,
                sample_rate=request.sample_rate,
                hop_length=request.hop_length,
            )
            f0_payload = F0Curve(times=times.tolist(), values=f0_curve.astype(float).tolist())

        metadata = MelodyMetadata(
            key=request.key,
            bpm=request.bpm,
            progression=request.progression,
            note_range=request.note_range,
            quantized=request.quantize,
            grid_resolution=request.grid_resolution if request.quantize else None,
            sample_rate=request.sample_rate,
            hop_length=request.hop_length,
        )

        return GenerateMelodyResponse(
            notes=notes,
            seed=request.seed,
            metadata=metadata,
            f0=f0_payload,
            message=f"Generated {len(notes)} notes at {request.bpm} BPM",
        )

    except ValueError as exc:
        logger.error(f"Validation error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error(f"Generation error: {exc}")
        raise HTTPException(status_code=500, detail="Melody generation failed") from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    port = int(os.getenv("MELODY_PORT", DEFAULT_PORT))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "dev") == "dev",
        log_level="info",
    )
