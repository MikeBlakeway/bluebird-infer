from fastapi.testclient import TestClient

from pods.voice.main import app


def test_voice_health():
    client = TestClient(app)
    resp = client.post("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in {"ok", "starting"}
    assert data["service"] == "voice"
    assert "ready" in data


def test_voice_synthesize_stub():
    client = TestClient(app)
    payload = {
        "lyrics": ["Hello world", "Test line"],
        "bpm": 120,
        "duration": 1.0,
        "seed": 123,
        "speaker_id": "spk-test",
    }
    resp = client.post(
        "/synthesize",
        json=payload,
        headers={"Idempotency-Key": "test-stub"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["stem_name"] == "vocals"
    assert data["speaker_id"] == "spk-test"
    assert isinstance(data["audio"], str)
    assert len(data["audio"]) > 1000  # base64 length for ~1s should be non-trivial
    assert data["phoneme_count"] == 0
    assert data["has_f0"] is False


def test_voice_synthesize_f0_track():
    client = TestClient(app)
    # 1 second duration at 100 fps -> 100 frames of 220 Hz
    f0 = [220.0] * 100
    payload = {
        "lyrics": ["A"],
        "bpm": 90,
        "duration": 1.0,
        "seed": 1,
        "f0": f0,
        "phonemes": ["AH", "B"],
        "durations": [0.4, 0.6],
    }
    resp = client.post(
        "/synthesize",
        json=payload,
        headers={"Idempotency-Key": "test-f0"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "F0" in data["message"]
    assert data["sample_rate"] == 48000
    assert len(data["audio"]) > 1000
    assert data["phoneme_count"] == 2
    assert data["has_f0"] is True


def test_voice_synthesize_phoneme_duration_mismatch():
    client = TestClient(app)
    payload = {
        "lyrics": ["Hello"],
        "bpm": 100,
        "duration": 1.0,
        "seed": 2,
        "phonemes": ["AH", "B"],
        "durations": [0.5],
    }
    resp = client.post(
        "/synthesize",
        json=payload,
        headers={"Idempotency-Key": "test-mismatch"},
    )
    assert resp.status_code == 400
    assert "phonemes and durations" in resp.json()["detail"]
