"""Tests for Analyzer Pod."""

import sys
import pytest
import numpy as np
from fastapi.testclient import TestClient
from io import BytesIO
from pathlib import Path

import soundfile as sf

# Add pod and project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pods/analyzer"))

from pods.analyzer.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing (22.05 kHz, 10 seconds, sine wave)."""
    sr = 22050
    duration = 10.0  # 10 seconds to ensure sufficient data for analysis
    t = np.linspace(0, duration, int(sr * duration), False)
    # Generate 440 Hz sine wave with harmonic
    freq = 440.0
    audio = (np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)).astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Encode to WAV
    buf = BytesIO()
    sf.write(buf, audio, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    return buf.getvalue()


class TestAnalyzerPod:
    """Tests for analyzer pod endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.post("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["service"] == "analyzer"

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "analyzer"
        assert "endpoints" in data

    def test_analyze_bpm(self, client, sample_audio):
        """Test BPM detection endpoint."""
        response = client.post(
            "/analyze/bpm",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "bpm" in data
        assert "confidence" in data
        assert isinstance(data["bpm"], (int, float))
        assert isinstance(data["confidence"], (int, float))

    def test_similarity_ngram(self, client, sample_audio):
        """Test n-gram similarity endpoint."""
        response = client.post(
            "/similarity/ngram",
            files={
                "reference_file": ("ref.wav", sample_audio, "audio/wav"),
                "generated_file": ("gen.wav", sample_audio, "audio/wav"),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "ngram_jaccard"
        assert "verdict" in data

    def test_invalid_file_error(self, client):
        """Test error handling for invalid file."""
        response = client.post(
            "/analyze/bpm",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )

        # Should return 400 error
        assert response.status_code == 400


class TestIntegration:
    """Integration tests for analyzer workflows."""

    def test_health_and_bpm_workflow(self, client, sample_audio):
        """Test health check and BPM analysis workflow."""
        # 1. Health check
        health_resp = client.post("/health")
        assert health_resp.status_code == 200

        # 2. Analyze BPM
        bpm_resp = client.post(
            "/analyze/bpm",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
        )
        assert bpm_resp.status_code == 200

        bpm_data = bpm_resp.json()
        assert "bpm" in bpm_data
        assert "confidence" in bpm_data
