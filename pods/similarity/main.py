"""Similarity Pod - Melody and rhythm similarity checking service.

Checks if a generated melody is too similar to a reference melody,
providing export gating and recommendations for users.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from config import config
from similarity_checker import create_checker, SimilarityScore

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


# ============================================================================
# Startup/Shutdown
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info(f"{config.SERVICE_NAME} pod starting (v{config.SERVICE_VERSION})...")
    yield
    logger.info(f"{config.SERVICE_NAME} pod shutting down...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Similarity Pod",
    description="Melody and rhythm similarity checking service",
    version=config.SERVICE_VERSION,
    lifespan=lifespan,
)


# ============================================================================
# Request/Response Models
# ============================================================================


class CheckRequest(BaseModel):
    """Request to check melody similarity."""

    reference_melody: List[int] = Field(
        ..., description="MIDI note sequence from reference audio (0-127)"
    )
    generated_melody: List[int] = Field(
        ..., description="MIDI note sequence from generated audio (0-127)"
    )
    reference_onsets: Optional[List[float]] = Field(
        default=None,
        description="Optional onset times (seconds) for reference melody",
    )
    generated_onsets: Optional[List[float]] = Field(
        default=None,
        description="Optional onset times (seconds) for generated melody",
    )


class CheckResponse(BaseModel):
    """Response with similarity check result."""

    melody_score: float = Field(..., ge=0.0, le=1.0, description="Melody similarity (0-1)")
    rhythm_score: float = Field(..., ge=0.0, le=1.0, description="Rhythm similarity (0-1)")
    combined_score: float = Field(..., ge=0.0, le=1.0, description="Weighted score (0-1)")
    verdict: str = Field(..., description="'pass', 'borderline', or 'block'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in verdict")
    recommendations: List[str] = Field(
        default_factory=list, description="Actionable recommendations for user"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/")
async def root():
    """API info endpoint."""
    return {
        "service": config.SERVICE_NAME,
        "version": config.SERVICE_VERSION,
        "description": "Melody and rhythm similarity checking",
        "endpoints": {
            "GET /": "This info",
            "POST /health": "Health check",
            "POST /check": "Check melody similarity",
        },
    }


@app.post("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "service": config.SERVICE_NAME,
        "version": config.SERVICE_VERSION,
    }


@app.post("/check", response_model=CheckResponse)
async def check_similarity(request: CheckRequest = Body(...)):
    """Check similarity between reference and generated melodies.

    Args:
        request: Contains reference_melody and generated_melody as MIDI sequences

    Returns:
        CheckResponse with scores, verdict, and recommendations
    """
    try:
        logger.info(
            f"Checking similarity: ref_len={len(request.reference_melody)} "
            f"gen_len={len(request.generated_melody)}"
        )

        # Validate input
        if not request.reference_melody:
            raise ValueError("reference_melody cannot be empty")
        if not request.generated_melody:
            raise ValueError("generated_melody cannot be empty")

        if len(request.reference_melody) > 1000:
            raise ValueError("reference_melody too long (max 1000 notes)")
        if len(request.generated_melody) > 1000:
            raise ValueError("generated_melody too long (max 1000 notes)")

        # Check MIDI note ranges
        for midi in request.reference_melody:
            if not (0 <= midi <= 127):
                raise ValueError(f"Invalid MIDI note in reference: {midi}")
        for midi in request.generated_melody:
            if not (0 <= midi <= 127):
                raise ValueError(f"Invalid MIDI note in generated: {midi}")

        # Run similarity check
        checker = create_checker()
        result: SimilarityScore = checker.check(
            request.reference_melody,
            request.generated_melody,
            request.reference_onsets,
            request.generated_onsets,
        )

        logger.info(
            f"Similarity check complete: verdict={result.verdict} "
            f"score={result.combined_score:.3f}"
        )

        return CheckResponse(
            melody_score=result.melody_score,
            rhythm_score=result.rhythm_score,
            combined_score=result.combined_score,
            verdict=result.verdict,
            confidence=result.confidence,
            recommendations=result.recommendations,
        )

    except ValueError as e:
        logger.warning(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {config.SERVICE_NAME} on port {config.SERVICE_PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.SERVICE_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )
