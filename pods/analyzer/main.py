"""Analyzer Pod - Feature extraction service for Bluebird.

Exposes audio feature extraction capabilities (key, BPM, features) via HTTP.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

from bbcore.audio import read_wav_bytes, DEFAULT_SAMPLE_RATE
from bbcore.logging import setup_logging
from bbfeatures.key_detection import detect_key
from bbfeatures.bpm_detection import detect_bpm_and_onsets
from bbfeatures.ngram import intervals_from_pitches, ngrams, jaccard

logger = logging.getLogger(__name__)


# ============================================================================
# Startup/Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Startup
    logger.info("Analyzer pod starting up")
    yield
    # Shutdown
    logger.info("Analyzer pod shutting down")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Analyzer Pod",
    description="Feature extraction service for Bluebird",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalyzeAudioRequest:
    """Request to analyze audio file."""
    # Using file upload, not JSON body
    pass


class KeyDetectionResponse:
    """Response from key detection endpoint."""
    key: str
    confidence: float


class BPMDetectionResponse:
    """Response from BPM detection endpoint."""
    bpm: float
    confidence: float


class AnalysisResponse:
    """Complete audio analysis response."""
    key: str
    key_confidence: float
    bpm: float
    bpm_confidence: float


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "analyzer"}


@app.post("/analyze/key")
async def analyze_key(file: UploadFile = File(...)):
    """Detect musical key from audio file.

    Args:
        file: Audio file (WAV, MP3, etc.)

    Returns:
        JSON with detected key and confidence
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio, sr = read_wav_bytes(audio_bytes)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Detect key
        key, confidence = detect_key(audio, sr)

        return {
            "key": key,
            "confidence": float(confidence),
        }

    except Exception as e:
        logger.error(f"Key detection error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to detect key: {str(e)}"
        )


@app.post("/analyze/bpm")
async def analyze_bpm(file: UploadFile = File(...)):
    """Detect tempo (BPM) from audio file.

    Args:
        file: Audio file (WAV, MP3, etc.)

    Returns:
        JSON with detected BPM and onsets
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio, sr = read_wav_bytes(audio_bytes)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Detect BPM and onsets
        result = detect_bpm_and_onsets(audio, sr)
        bpm = result["bpm"]
        confidence = 0.8  # Default confidence (could be improved with better estimation)

        return {
            "bpm": float(bpm),
            "confidence": float(confidence),
        }

    except Exception as e:
        logger.error(f"BPM detection error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to detect BPM: {str(e)}"
        )


@app.post("/analyze")
async def analyze_full(file: UploadFile = File(...)):
    """Full audio analysis (key, BPM, features).

    Args:
        file: Audio file (WAV, MP3, etc.)

    Returns:
        JSON with all extracted features
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio, sr = read_wav_bytes(audio_bytes)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio[:, 0]

        result = detect_bpm_and_onsets(audio, sr)
        bpm = result["bpm"]
        bpm_conf = 0.8  # Default confidence
        key, key_conf = detect_key(audio, sr)
        bpm, bpm_conf = detect_bpm(audio, sr)

        return {
            "key": key,
            "key_confidence": float(key_conf),
            "bpm": float(bpm),
            "bpm_confidence": float(bpm_conf),
        }

    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/similarity/ngram")
async def similarity_check(
    reference_file: UploadFile = File(...),
    generated_file: UploadFile = File(...),
):
    """Check melody similarity using n-gram Jaccard distance.

    Args:
        reference_file: Reference melody F0 or MIDI file
        generated_file: Generated melody F0 or MIDI file

    Returns:
        JSON with Jaccard similarity scores
    """
    try:
        # For now, return placeholder
        # Will be implemented when similarity pod is ready
        return {
            "method": "ngram_jaccard",
            "n_values": [3, 4, 5],
            "scores": {
                "3": 0.0,
                "4": 0.0,
                "5": 0.0,
            },
            "verdict": "not_implemented",
        }

    except Exception as e:
        logger.error(f"Similarity check error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Similarity check failed: {str(e)}"
        )


# ============================================================================
# Root
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "analyzer",
        "version": "0.1.0",
        "endpoints": {
            "health": "POST /health",
            "analyze_key": "POST /analyze/key",
            "analyze_bpm": "POST /analyze/bpm",
            "analyze_full": "POST /analyze",
            "similarity": "POST /similarity/ngram",
        }
    }


# ============================================================================
# Logging Setup
# ============================================================================

if __name__ == "__main__":
    setup_logging()
    import uvicorn

    port = int(os.getenv("ANALYZER_PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "dev") == "dev",
    )
