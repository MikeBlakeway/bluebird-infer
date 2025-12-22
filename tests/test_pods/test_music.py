"""Tests for Music Synthesis Pod."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

# Import from music pod
music_pod_path = Path(__file__).parent.parent.parent / "pods" / "music"
sys.path.insert(0, str(music_pod_path))

from synth import BassSynth, DrumSynth, GuitarSynth, SynthConfig
from grid import GridAligner, GridRenderer
from main import app


class TestDrumSynth:
    """Test drum synthesis."""

    def test_kick_generation(self):
        """Test kick drum generation."""
        drum_synth = DrumSynth(sample_rate=48000, seed=42)
        kick = drum_synth.generate_kick(duration=0.3, bpm=120)

        assert len(kick) == int(0.3 * 48000)
        assert kick.dtype == np.float32
        assert np.max(np.abs(kick)) <= 1.0
        assert np.min(kick) >= -1.0

    def test_snare_generation(self):
        """Test snare drum generation."""
        drum_synth = DrumSynth(sample_rate=48000, seed=42)
        snare = drum_synth.generate_snare(duration=0.2)

        assert len(snare) == int(0.2 * 48000)
        assert snare.dtype == np.float32
        assert np.max(np.abs(snare)) <= 1.0

    def test_hihat_generation(self):
        """Test hi-hat generation."""
        drum_synth = DrumSynth(sample_rate=48000, seed=42)
        hihat = drum_synth.generate_hihat(duration=0.1)

        assert len(hihat) == int(0.1 * 48000)
        assert np.max(np.abs(hihat)) <= 1.0

    def test_drum_pattern_generation(self):
        """Test full drum pattern generation."""
        drum_synth = DrumSynth(sample_rate=48000, seed=42)
        pattern = drum_synth.generate_pattern(duration=8.0, bpm=120)

        assert len(pattern) == int(8.0 * 48000)
        assert pattern.dtype == np.float32
        assert np.max(np.abs(pattern)) <= 1.0

    def test_deterministic_generation(self):
        """Test that same seed produces same output."""
        drum_synth1 = DrumSynth(sample_rate=48000, seed=42)
        pattern1 = drum_synth1.generate_pattern(duration=4.0, bpm=120)

        drum_synth2 = DrumSynth(sample_rate=48000, seed=42)
        pattern2 = drum_synth2.generate_pattern(duration=4.0, bpm=120)

        np.testing.assert_array_almost_equal(pattern1, pattern2)

    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different patterns."""
        drum_synth1 = DrumSynth(sample_rate=48000, seed=42)
        pattern1 = drum_synth1.generate_pattern(duration=2.0, bpm=120)

        drum_synth2 = DrumSynth(sample_rate=48000, seed=99)
        pattern2 = drum_synth2.generate_pattern(duration=2.0, bpm=120)

        # They should be different
        assert not np.allclose(pattern1, pattern2)


class TestBassSynth:
    """Test bass synthesis."""

    def test_midi_to_freq_conversion(self):
        """Test MIDI to frequency conversion."""
        bass_synth = BassSynth(sample_rate=48000)

        # A4 = 69 should be 440 Hz
        freq = bass_synth.midi_to_freq(69)
        assert abs(freq - 440.0) < 1.0

        # C1 = 36 should be lower
        freq_c1 = bass_synth.midi_to_freq(36)
        assert freq_c1 < 100  # Very low note

    def test_note_generation(self):
        """Test single note generation."""
        bass_synth = BassSynth(sample_rate=48000, seed=42)
        note = bass_synth.generate_note(midi_note=36, duration=1.0, envelope="sine")

        assert len(note) == 48000
        assert note.dtype == np.float32
        assert np.max(np.abs(note)) <= 1.0

    def test_bass_line_generation(self):
        """Test bass line generation."""
        bass_synth = BassSynth(sample_rate=48000, seed=42)
        bass_notes = [36, 36, 43, 36]  # I-I-V-I pattern
        bass_line = bass_synth.generate_line(bass_notes, note_duration=2.0, bpm=120)

        # 4 notes * 2 beats * (60/120) seconds = 4 seconds
        expected_samples = 4 * 48000
        assert len(bass_line) == expected_samples

    def test_waveform_types(self):
        """Test different waveform types."""
        bass_synth = BassSynth(sample_rate=48000, seed=42)

        for waveform in ["sine", "saw", "square", "triangle"]:
            note = bass_synth.generate_note(
                midi_note=36, duration=0.5, envelope=waveform
            )
            assert len(note) == int(0.5 * 48000)
            assert np.max(np.abs(note)) <= 1.0


class TestGuitarSynth:
    """Test guitar synthesis."""

    def test_pluck_generation(self):
        """Test single pluck generation."""
        guitar_synth = GuitarSynth(sample_rate=48000, seed=42)
        pluck = guitar_synth.generate_pluck(midi_note=40, duration=2.0)

        assert len(pluck) == int(2.0 * 48000)
        assert np.max(np.abs(pluck)) <= 1.0

    def test_chord_generation(self):
        """Test guitar chord generation."""
        guitar_synth = GuitarSynth(sample_rate=48000, seed=42)
        chord = guitar_synth.generate_chord([40, 45, 50, 55], duration=2.0)

        assert len(chord) == int(2.0 * 48000)
        assert np.max(np.abs(chord)) <= 1.0


class TestGridAligner:
    """Test grid alignment."""

    def test_beat_to_samples(self):
        """Test beat position to sample conversion."""
        aligner = GridAligner(sample_rate=48000)

        # At 120 BPM, each beat = 0.5 seconds = 24000 samples
        samples = aligner.beat_to_samples(beat=1.0, bpm=120)
        assert samples == 24000

        samples = aligner.beat_to_samples(beat=4.0, bpm=120)
        assert samples == 96000

    def test_samples_to_beat(self):
        """Test sample index to beat conversion."""
        aligner = GridAligner(sample_rate=48000)

        beat = aligner.samples_to_beat(sample_idx=24000, bpm=120)
        assert abs(beat - 1.0) < 0.01

        beat = aligner.samples_to_beat(sample_idx=96000, bpm=120)
        assert abs(beat - 4.0) < 0.01

    def test_section_boundaries(self):
        """Test section boundary calculation."""
        aligner = GridAligner(sample_rate=48000)

        # 16-beat section at 120 BPM
        start, end = aligner.get_section_boundaries(8.0, bpm=120, beats_per_section=16)

        # 16 beats * 0.5 sec/beat = 8 seconds
        assert end - start == int(8.0 * 48000)

    def test_round_to_beat_boundary(self):
        """Test rounding to beat boundary."""
        aligner = GridAligner(sample_rate=48000)

        # 23000 samples should round to 24000 (beat boundary at 120 BPM)
        rounded = aligner.round_to_beat_boundary(samples=23000, bpm=120)
        assert rounded == 24000


class TestGridRenderer:
    """Test grid rendering."""

    def test_render_section(self):
        """Test section rendering with stems."""
        renderer = GridRenderer(sample_rate=48000, bpm=120)

        # Create simple test stems
        drums = np.sin(2 * np.pi * 100 * np.arange(48000) / 48000).astype(np.float32)
        bass = np.sin(2 * np.pi * 50 * np.arange(48000) / 48000).astype(np.float32) * 0.5

        stems = {"drums": drums, "bass": bass}
        mix_levels = {"drums": 0.8, "bass": 0.7}

        mixed, aligned = renderer.render_section(1.0, stems, mix_levels)

        assert len(mixed) > 0
        assert np.max(np.abs(mixed)) <= 1.0
        assert "drums" in aligned
        assert "bass" in aligned


class TestMusicPodAPI:
    """Test Music Synthesis Pod API."""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.post("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "music"

    def test_root_endpoint(self):
        """Test root/info endpoint."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data

    def test_synthesize_basic(self):
        """Test basic synthesis."""
        client = TestClient(app)

        request_data = {
            "bpm": 120,
            "duration": 2.0,
            "include_drums": True,
            "include_bass": True,
            "include_guitar": False,
            "seed": 42,
        }

        response = client.post("/synthesize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["duration"] == 2.0
        assert data["sample_rate"] == 48000
        assert data["bit_depth"] == 24
        assert "mixed_audio" in data
        assert "stems" in data

    def test_synthesize_with_guitar(self):
        """Test synthesis with guitar."""
        client = TestClient(app)

        request_data = {
            "bpm": 100,
            "duration": 1.0,
            "include_drums": True,
            "include_bass": False,
            "include_guitar": True,
            "seed": 99,
        }

        response = client.post("/synthesize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "guitar" in data["stems"]
        assert "drums" in data["stems"]

    def test_synthesize_deterministic(self):
        """Test that same seed produces same output."""
        client = TestClient(app)

        request_data = {
            "bpm": 120,
            "duration": 1.0,
            "include_drums": True,
            "include_bass": True,
            "seed": 42,
        }

        response1 = client.post("/synthesize", json=request_data)
        data1 = response1.json()

        response2 = client.post("/synthesize", json=request_data)
        data2 = response2.json()

        # Same seed should produce identical output
        assert data1["mixed_audio"] == data2["mixed_audio"]

    def test_synthesize_different_seeds(self):
        """Test that different seeds produce different output."""
        client = TestClient(app)

        request_data1 = {
            "bpm": 120,
            "duration": 1.0,
            "include_drums": True,
            "include_bass": True,
            "seed": 42,
        }

        request_data2 = {
            "bpm": 120,
            "duration": 1.0,
            "include_drums": True,
            "include_bass": True,
            "seed": 99,
        }

        response1 = client.post("/synthesize", json=request_data1)
        response2 = client.post("/synthesize", json=request_data2)

        data1 = response1.json()
        data2 = response2.json()

        # Different seeds should produce different output
        assert data1["mixed_audio"] != data2["mixed_audio"]

    def test_synthesize_performance(self):
        """Test synthesis performance."""
        import time

        client = TestClient(app)

        request_data = {
            "bpm": 120,
            "duration": 8.0,
            "include_drums": True,
            "include_bass": True,
            "seed": 42,
        }

        start_time = time.time()
        response = client.post("/synthesize", json=request_data)
        end_time = time.time()

        assert response.status_code == 200
        elapsed = end_time - start_time

        # Should complete in under 2 seconds (target: <2s per section)
        assert elapsed < 2.0, f"Synthesis took {elapsed:.2f}s (target: <2.0s)"

    def test_synthesize_invalid_bpm(self):
        """Test synthesis with invalid BPM."""
        client = TestClient(app)

        request_data = {
            "bpm": 999,  # Too high
            "duration": 1.0,
            "include_drums": True,
            "seed": 42,
        }

        response = client.post("/synthesize", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_synthesize_master_volume(self):
        """Test synthesis with different master volume levels."""
        client = TestClient(app)

        request_data = {
            "bpm": 120,
            "duration": 1.0,
            "include_drums": True,
            "include_bass": True,
            "seed": 42,
            "master_volume": 0.5,
        }

        response = client.post("/synthesize", json=request_data)
        assert response.status_code == 200


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test full synthesis pipeline."""
        # Create synths
        drum_synth = DrumSynth(sample_rate=48000, seed=42)
        bass_synth = BassSynth(sample_rate=48000, seed=42)

        # Generate stems
        drums = drum_synth.generate_pattern(8.0, bpm=120)
        bass = bass_synth.generate_line([36, 36, 43, 36], note_duration=2.0, bpm=120)

        # Render
        renderer = GridRenderer(sample_rate=48000, bpm=120)
        stems = {"drums": drums[:len(bass)], "bass": bass}
        mixed, aligned = renderer.render_section(
            duration=8.0, stems=stems, mix_levels={"drums": 0.8, "bass": 0.7}
        )

        assert len(mixed) > 0
        assert np.max(np.abs(mixed)) <= 1.0
