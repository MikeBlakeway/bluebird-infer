"""Tests for Melody Pod."""

import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add pod and project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pods/melody"))

from pods.melody.main import app  # noqa: E402


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_request():
    return {
        "syllables": ["sing", "a", "mel", "o", "dy"],
        "key": "C",
        "bpm": 120,
        "progression": "pop1",
        "seed": 42,
    }


class TestMelodyPod:
    """Pod-level API tests."""

    def test_health(self, client):
        response = client.post("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "melody"

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "melody"
        assert "endpoints" in data

    def test_generate_basic(self, client, sample_request):
        response = client.post("/generate", json=sample_request)
        assert response.status_code == 200
        data = response.json()

        assert len(data["notes"]) == len(sample_request["syllables"])
        assert data["metadata"]["bpm"] == sample_request["bpm"]
        assert data["metadata"]["key"] == sample_request["key"]
        assert data["seed"] == sample_request["seed"]

    def test_generate_deterministic(self, client, sample_request):
        response1 = client.post("/generate", json=sample_request)
        response2 = client.post("/generate", json=sample_request)

        assert response1.status_code == 200
        assert response2.status_code == 200

        notes1 = response1.json()["notes"]
        notes2 = response2.json()["notes"]
        assert notes1 == notes2

    def test_generate_different_seeds(self, client, sample_request):
        req_a = {**sample_request, "seed": 1}
        req_b = {**sample_request, "seed": 999}

        notes_a = client.post("/generate", json=req_a).json()["notes"]
        notes_b = client.post("/generate", json=req_b).json()["notes"]

        assert notes_a != notes_b

    def test_generate_quantized(self, client, sample_request):
        request = {**sample_request, "quantize": True, "grid_resolution": 0.25}
        response = client.post("/generate", json=request)
        assert response.status_code == 200

        data = response.json()
        grid = request["grid_resolution"]

        for note in data["notes"]:
            assert math.isclose(note["onset"] / grid, round(note["onset"] / grid), rel_tol=1e-6, abs_tol=1e-6)
            assert math.isclose(note["duration"] / grid, round(note["duration"] / grid), rel_tol=1e-6, abs_tol=1e-6)

        assert data["metadata"]["quantized"] is True
        assert data["metadata"]["grid_resolution"] == grid

    def test_generate_with_f0(self, client, sample_request):
        request = {**sample_request, "emit_f0": True, "sample_rate": 16000, "hop_length": 160}
        response = client.post("/generate", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["f0"] is not None
        assert len(data["f0"]["times"]) == len(data["f0"]["values"])
        assert len(data["f0"]["values"]) > 0

    def test_invalid_note_range(self, client, sample_request):
        request = {**sample_request, "note_range": [80, 60]}
        response = client.post("/generate", json=request)
        # Pydantic validation should fail
        assert response.status_code in (400, 422)


class TestIntegration:
    """Integration smoke test for melody pod."""

    def test_end_to_end(self, client, sample_request):
        request = {**sample_request, "emit_f0": True, "quantize": True, "grid_resolution": 0.25}
        response = client.post("/generate", json=request)
        assert response.status_code == 200

        payload = response.json()
        assert len(payload["notes"]) == len(sample_request["syllables"])
        if payload.get("f0"):
            assert len(payload["f0"]["times"]) == len(payload["f0"]["values"])
