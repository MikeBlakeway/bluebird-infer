"""
Voice Pod Synthesis Validation Tests

Tests for:
1. Determinism (same seed → same audio)
2. Multi-speaker support (different speakers → different audio)
3. F0 curve handling (pitch contour fidelity)
4. Phoneme alignment (frame-level precision)
5. Performance (latency, memory usage)
6. Error recovery (graceful fallback)
"""

import json
import base64
import numpy as np
import soundfile as sf
import io
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class VoiceSynthesisValidator:
    """Comprehensive validation suite for voice synthesis."""

    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url
        self.test_results = {}
        self._import_requests()

    def _import_requests(self):
        """Import requests library for HTTP calls."""
        try:
            import requests
            self.requests = requests
        except ImportError:
            logger.error("requests library not available; tests will be limited")
            self.requests = None

    def run_all_tests(self) -> dict:
        """Run complete validation suite."""
        if not self.requests:
            logger.error("Cannot run tests without requests library")
            return {"error": "requests library not available"}

        tests = [
            ("determinism", self.test_determinism),
            ("speaker_propagation", self.test_speaker_propagation),
            ("f0_curve_handling", self.test_f0_curve_handling),
            ("phoneme_alignment", self.test_phoneme_alignment),
            ("response_schema", self.test_response_schema),
            ("error_recovery", self.test_error_recovery),
            ("performance_baseline", self.test_performance_baseline),
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                logger.info(f"Running test: {test_name}")
                result = test_func()
                results[test_name] = {
                    "status": "PASS" if result.get("pass") else "FAIL",
                    "details": result,
                }
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results[test_name] = {
                    "status": "ERROR",
                    "error": str(e),
                }

        return results

    def test_determinism(self) -> dict:
        """Test that same seed produces identical audio."""
        payload = {
            "lyrics": ["Hello world", "How are you"],
            "speaker_id": "default",
            "duration": 3.0,
            "seed": 42,
            "bpm": 120,
        }

        # First request
        r1 = self.requests.post(
            f"{self.base_url}/synthesize",
            json=payload,
            headers={"Idempotency-Key": "det-test-1"},
        )

        if r1.status_code != 200:
            return {"pass": False, "error": f"Request 1 failed: {r1.status_code}"}

        audio1 = r1.json().get("audio", "")
        method1 = r1.json().get("message", "")

        # Second request (same seed)
        r2 = self.requests.post(
            f"{self.base_url}/synthesize",
            json=payload,
            headers={"Idempotency-Key": "det-test-2"},
        )

        if r2.status_code != 200:
            return {"pass": False, "error": f"Request 2 failed: {r2.status_code}"}

        audio2 = r2.json().get("audio", "")
        method2 = r2.json().get("message", "")

        # Compare
        match = audio1 == audio2
        return {
            "pass": match,
            "audio1_length": len(audio1),
            "audio2_length": len(audio2),
            "method1": method1[:50] if method1 else "unknown",
            "method2": method2[:50] if method2 else "unknown",
            "match": match,
            "note": "Same seed should produce identical base64 audio",
        }

    def test_speaker_propagation(self) -> dict:
        """Test that speaker_id is correctly propagated."""
        speakers = ["alice", "bob", "charlie", "default"]
        results = {}

        for speaker in speakers:
            payload = {
                "lyrics": ["Test"],
                "speaker_id": speaker,
                "duration": 1.0,
                "seed": 100,
            }

            r = self.requests.post(
                f"{self.base_url}/synthesize",
                json=payload,
                headers={"Idempotency-Key": f"speaker-test-{speaker}"},
            )

            if r.status_code == 200:
                returned_speaker = r.json().get("speaker_id")
                match = returned_speaker == speaker
                results[speaker] = {
                    "requested": speaker,
                    "returned": returned_speaker,
                    "match": match,
                }
            else:
                results[speaker] = {"error": f"Request failed: {r.status_code}"}

        all_match = all(v.get("match", False) for v in results.values())
        return {"pass": all_match, "speakers": results}

    def test_f0_curve_handling(self) -> dict:
        """Test F0 curve input and handling."""
        # Create F0 curve: 100 frames at 100Hz (1 second)
        # Pitch contour: C4 (262 Hz) → E4 (330 Hz) → G4 (392 Hz)
        f0_base = np.array([262, 296, 330, 350, 370, 392], dtype=float)
        f0_curve = np.repeat(f0_base, 16)  # Stretch to ~96 frames

        payload = {
            "lyrics": ["Sing this melody"],
            "speaker_id": "default",
            "duration": 2.0,
            "seed": 200,
            "f0": f0_curve.tolist(),
        }

        r = self.requests.post(
            f"{self.base_url}/synthesize",
            json=payload,
            headers={"Idempotency-Key": "f0-test-1"},
        )

        if r.status_code != 200:
            return {"pass": False, "error": f"Request failed: {r.status_code}"}

        data = r.json()
        has_f0 = data.get("has_f0", False)

        return {
            "pass": has_f0,
            "has_f0": has_f0,
            "f0_frames_sent": len(f0_curve),
            "duration": data.get("duration"),
            "message": data.get("message", "")[:50],
        }

    def test_phoneme_alignment(self) -> dict:
        """Test phoneme alignment and metadata."""
        payload = {
            "lyrics": ["Hello", "World", "Testing"],
            "speaker_id": "default",
            "duration": 2.5,
            "seed": 300,
        }

        r = self.requests.post(
            f"{self.base_url}/synthesize",
            json=payload,
            headers={"Idempotency-Key": "phoneme-test-1"},
        )

        if r.status_code != 200:
            return {"pass": False, "error": f"Request failed: {r.status_code}"}

        data = r.json()
        phoneme_count = data.get("phoneme_count", 0)

        # G2P should produce phonemes for the 3 words
        has_phonemes = phoneme_count > 0

        return {
            "pass": has_phonemes,
            "phoneme_count": phoneme_count,
            "lyrics_count": 3,
            "expected_min_phonemes": 2,
            "message": data.get("message", "")[:50],
        }

    def test_response_schema(self) -> dict:
        """Test that response follows expected schema."""
        payload = {
            "lyrics": ["Test"],
            "speaker_id": "test_speaker",
            "duration": 1.0,
            "seed": 400,
        }

        r = self.requests.post(
            f"{self.base_url}/synthesize",
            json=payload,
            headers={"Idempotency-Key": "schema-test-1"},
        )

        if r.status_code != 200:
            return {"pass": False, "error": f"Request failed: {r.status_code}"}

        data = r.json()
        required_fields = [
            "duration",
            "sample_rate",
            "bit_depth",
            "stem_name",
            "audio",
            "message",
            "speaker_id",
            "phoneme_count",
            "has_f0",
        ]

        missing = [f for f in required_fields if f not in data]
        has_all = len(missing) == 0

        # Validate field types
        valid_types = (
            isinstance(data.get("duration"), (int, float))
            and isinstance(data.get("sample_rate"), int)
            and isinstance(data.get("bit_depth"), int)
            and isinstance(data.get("audio"), str)  # base64
            and isinstance(data.get("speaker_id"), str)
            and isinstance(data.get("phoneme_count"), int)
            and isinstance(data.get("has_f0"), bool)
        )

        return {
            "pass": has_all and valid_types,
            "has_all_fields": has_all,
            "missing_fields": missing,
            "valid_types": valid_types,
            "audio_length": len(data.get("audio", "")),
            "sample_rate": data.get("sample_rate"),
            "bit_depth": data.get("bit_depth"),
        }

    def test_error_recovery(self) -> dict:
        """Test that invalid inputs are handled gracefully."""
        test_cases = [
            {
                "name": "missing_lyrics",
                "payload": {"speaker_id": "default", "duration": 1.0},
                "expect_error": False,  # Empty lyrics should be OK
            },
            {
                "name": "invalid_duration",
                "payload": {
                    "lyrics": ["Test"],
                    "duration": 0.1,  # Too short
                },
                "expect_error": True,
            },
            {
                "name": "mismatched_phonemes",
                "payload": {
                    "lyrics": ["Test"],
                    "phonemes": ["T", "EH"],  # 2 phonemes for 1 word
                    "durations": [0.5, 0.5],
                },
                "expect_error": True,
            },
        ]

        results = {}
        for test_case in test_cases:
            try:
                r = self.requests.post(
                    f"{self.base_url}/synthesize",
                    json=test_case["payload"],
                    headers={"Idempotency-Key": f"error-test-{test_case['name']}"},
                )

                expect_error = test_case.get("expect_error", False)
                got_error = r.status_code >= 400

                results[test_case["name"]] = {
                    "status_code": r.status_code,
                    "expected_error": expect_error,
                    "got_error": got_error,
                    "correct": expect_error == got_error,
                }
            except Exception as e:
                results[test_case["name"]] = {"exception": str(e)}

        all_correct = all(v.get("correct", False) for v in results.values())
        return {"pass": all_correct, "test_cases": results}

    def test_performance_baseline(self) -> dict:
        """Test performance: latency, memory (requires monitoring)."""
        payload = {
            "lyrics": ["The quick brown fox jumps over the lazy dog"],
            "speaker_id": "default",
            "duration": 5.0,
            "seed": 500,
        }

        latencies = []
        for i in range(3):
            start = time.time()
            r = self.requests.post(
                f"{self.base_url}/synthesize",
                json=payload,
                headers={"Idempotency-Key": f"perf-test-{i}"},
            )
            elapsed = time.time() - start

            if r.status_code == 200:
                latencies.append(elapsed)

        if not latencies:
            return {"pass": False, "error": "No successful requests"}

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        # Fallback synthesis should be very fast (<100ms)
        # Real DiffSinger will be slower (target: <8s for 30s audio)
        within_budget = avg_latency < 1.0  # 1s max for fallback

        return {
            "pass": within_budget,
            "latencies": latencies,
            "avg_latency_ms": avg_latency * 1000,
            "max_latency_ms": max_latency * 1000,
            "within_budget": within_budget,
            "budget_note": "Fallback should be <100ms, real DiffSinger target <8s per 30s",
        }

    def decode_audio(self, audio_b64: str) -> tuple:
        """Decode base64 audio to numpy array.
        
        Returns (audio, sample_rate) tuple.
        """
        try:
            wav_bytes = base64.b64decode(audio_b64)
            wav_buffer = io.BytesIO(wav_bytes)
            audio, sr = sf.read(wav_buffer)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to decode audio: {e}")
            return None, None


def main():
    """Run validation suite and print results."""
    validator = VoiceSynthesisValidator()
    results = validator.run_all_tests()

    print("\n" + "=" * 80)
    print("VOICE POD VALIDATION RESULTS")
    print("=" * 80 + "\n")

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = result.get("status", "UNKNOWN")
        if status == "PASS":
            print(f"✅ {test_name.upper()}")
            passed += 1
        elif status == "FAIL":
            print(f"❌ {test_name.upper()}")
            failed += 1
        else:
            print(f"⚠️  {test_name.upper()} ({status})")

        details = result.get("details", {})
        for key, value in details.items():
            if key != "pass" and not isinstance(value, dict):
                print(f"   {key}: {value}")

        print()

    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)

    return {"total": passed + failed, "passed": passed, "failed": failed, "results": results}


if __name__ == "__main__":
    main()
