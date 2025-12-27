from fastapi.testclient import TestClient

from pods.voice.main import app


def test_voice_health():
    client = TestClient(app)
    resp = client.post("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "voice"


def test_voice_synthesize_stub():
    client = TestClient(app)
    payload = {
        "lyrics": ["Hello world", "Test line"],
        "bpm": 120,
        "duration": 1.0,
        "seed": 123,
    }
    resp = client.post("/synthesize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["stem_name"] == "vocals"
    assert isinstance(data["audio"], str)
    assert len(data["audio"]) > 1000  # base64 length for ~1s should be non-trivial
