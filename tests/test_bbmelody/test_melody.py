"""Tests for bbmelody library."""

import pytest
import numpy as np

from bbmelody import (
    ChordType,
    key_to_midi_root,
    get_scale_notes,
    get_chord_notes,
    progression_to_chords,
    MelodyGenerator,
    generate_melody,
    midi_to_freq,
    freq_to_midi,
    melody_to_f0_curve,
    f0_curve_to_midi,
    quantize_melody,
)


class TestChordUtils:
    """Tests for chord_utils module."""

    def test_key_to_midi_root(self):
        """Test key name to MIDI conversion."""
        assert key_to_midi_root("C") == 60
        assert key_to_midi_root("D") == 62
        assert key_to_midi_root("F#") == 66
        assert key_to_midi_root("Bb") == 70
        assert key_to_midi_root("Cm") == 60  # Minor keys return root

    def test_get_scale_notes(self):
        """Test scale generation."""
        # C major scale
        c_major = get_scale_notes(60, "major")
        assert 60 in c_major  # C
        assert 62 in c_major  # D
        assert 64 in c_major  # E
        assert 65 in c_major  # F
        assert 67 in c_major  # G
        assert 69 in c_major  # A
        assert 71 in c_major  # B

        # A minor scale
        a_minor = get_scale_notes(69, "minor")
        assert 69 in a_minor  # A
        assert 71 in a_minor  # B
        assert 72 in a_minor  # C

    def test_get_chord_notes(self):
        """Test chord generation."""
        # C major chord
        c_maj = get_chord_notes(60, ChordType.MAJOR)
        assert c_maj == [60, 64, 67]  # C, E, G

        # D minor chord
        d_min = get_chord_notes(62, ChordType.MINOR)
        assert d_min == [62, 65, 69]  # D, F, A

        # G7 chord
        g7 = get_chord_notes(67, ChordType.DOMINANT)
        assert g7 == [67, 71, 74, 77]  # G, B, D, F

    def test_progression_to_chords(self):
        """Test chord progression generation."""
        chords = progression_to_chords("pop1", "C", "major")

        assert len(chords) == 4  # I-V-vi-IV

        # First chord (I, C major)
        root, chord_type, notes = chords[0]
        assert root == 60
        assert chord_type == ChordType.MAJOR
        assert 60 in notes  # C


class TestMelodyGenerator:
    """Tests for melody generator."""

    def test_generate_from_lyrics_basic(self):
        """Test basic melody generation."""
        syllables = ["hel", "lo", "world", "to", "day"]
        generator = MelodyGenerator(key="C", bpm=120, seed=42)

        melody = generator.generate_from_lyrics(
            syllables=syllables,
            progression_name="pop1",
        )

        # Should have one note per syllable
        assert len(melody) == len(syllables)

        # Each note should have (pitch, onset, duration)
        for pitch, onset, duration in melody:
            assert 0 <= pitch <= 127  # Valid MIDI range
            assert onset >= 0.0
            assert duration > 0.0

    def test_generate_melody_deterministic(self):
        """Test that same seed produces same melody."""
        syllables = ["test", "mel", "o", "dy"]

        melody1 = generate_melody(
            syllables=syllables,
            key="D",
            bpm=100,
            seed=123,
        )

        melody2 = generate_melody(
            syllables=syllables,
            key="D",
            bpm=100,
            seed=123,
        )

        # Should be identical
        assert len(melody1) == len(melody2)
        for note1, note2 in zip(melody1, melody2):
            assert note1 == note2

    def test_generate_with_contour(self):
        """Test melody generation with contour guidance."""
        syllables = ["low", "high", "mid", "low"]
        contour = [-1.0, 1.0, 0.0, -0.5]  # Low -> high -> mid -> low

        melody = generate_melody(
            syllables=syllables,
            key="C",
            bpm=120,
            contour=contour,
            seed=42,
        )

        pitches = [pitch for pitch, _, _ in melody]

        # Check general contour trend (not exact due to chord constraints)
        assert pitches[1] >= pitches[0]  # High note higher than low
        assert pitches[3] <= pitches[1]  # Final low note lower than high

    def test_generate_respects_note_range(self):
        """Test that generated notes stay within range."""
        syllables = ["a"] * 10
        note_range = (64, 72)  # E4 to C5

        melody = generate_melody(
            syllables=syllables,
            key="C",
            bpm=120,
            note_range=note_range,
            seed=42,
        )

        for pitch, _, _ in melody:
            assert note_range[0] <= pitch <= note_range[1]


class TestMidiUtils:
    """Tests for MIDI and F0 utilities."""

    def test_midi_to_freq(self):
        """Test MIDI to frequency conversion."""
        # A4 = 440 Hz
        assert abs(midi_to_freq(69) - 440.0) < 0.01

        # C4 (middle C) ≈ 261.63 Hz
        assert abs(midi_to_freq(60) - 261.63) < 0.01

        # A5 = 880 Hz (octave above A4)
        assert abs(midi_to_freq(81) - 880.0) < 0.01

    def test_freq_to_midi(self):
        """Test frequency to MIDI conversion."""
        assert abs(freq_to_midi(440.0) - 69.0) < 0.01
        assert abs(freq_to_midi(261.63) - 60.0) < 0.1

        # Test error on invalid frequency
        with pytest.raises(ValueError):
            freq_to_midi(0.0)
        with pytest.raises(ValueError):
            freq_to_midi(-100.0)

    def test_midi_freq_roundtrip(self):
        """Test MIDI <-> frequency conversion is reversible."""
        for midi_pitch in [60, 69, 72, 48, 84]:
            freq = midi_to_freq(midi_pitch)
            recovered_midi = freq_to_midi(freq)
            assert abs(recovered_midi - midi_pitch) < 0.001

    def test_melody_to_f0_curve(self):
        """Test melody to F0 curve conversion."""
        melody = [
            (60, 0.0, 0.5),   # C4 for 0.5s
            (64, 0.5, 0.5),   # E4 for 0.5s
            (67, 1.0, 0.5),   # G4 for 0.5s
        ]

        times, f0_curve = melody_to_f0_curve(melody, sample_rate=16000, hop_length=160)

        # Check shapes
        assert len(times) == len(f0_curve)
        assert len(f0_curve) > 0

        # Check F0 values are reasonable (not zero during notes)
        assert np.max(f0_curve) > 200  # Should have some pitched content

        # Check frequencies match MIDI pitches approximately
        # (within voiced regions)
        expected_freqs = [midi_to_freq(60), midi_to_freq(64), midi_to_freq(67)]
        for expected_freq in expected_freqs:
            assert np.any(np.abs(f0_curve - expected_freq) < 5.0)  # Within 5 Hz

    def test_f0_curve_to_midi(self):
        """Test F0 curve to MIDI conversion."""
        # Create simple F0 curve: C4 (261.63 Hz) for 1 second
        sample_rate = 16000
        hop_length = 160
        duration = 1.0
        num_frames = int(duration * sample_rate / hop_length)

        times = np.arange(num_frames) * (hop_length / sample_rate)
        f0_curve = np.ones(num_frames) * 261.63  # C4

        notes = f0_curve_to_midi(f0_curve, times, voicing_threshold=50.0)

        # Should detect one note
        assert len(notes) == 1

        pitch, onset, dur = notes[0]
        # Should be close to MIDI 60 (C4)
        assert abs(pitch - 60.0) < 1.0
        assert onset < 0.1  # Starts near beginning
        assert dur > 0.8  # Lasts most of duration

    def test_quantize_melody(self):
        """Test melody timing quantization."""
        melody = [
            (60, 0.03, 0.48),   # Slightly off-grid
            (64, 0.52, 0.47),   # Slightly off-grid
        ]

        quantized = quantize_melody(melody, grid_resolution=0.25)

        # Should snap to 0.25-second grid
        assert quantized[0][1] == 0.0    # Onset quantized to 0.0
        assert quantized[0][2] == 0.5    # Duration quantized to 0.5
        assert quantized[1][1] == 0.5    # Onset quantized to 0.5
        assert quantized[1][2] == 0.5    # Duration quantized to 0.5

    def test_melody_f0_roundtrip(self):
        """Test melody -> F0 -> melody conversion preserves structure."""
        # Use notes with gaps (rests) between them so they're detected separately
        original_melody = [
            (60, 0.0, 0.4),   # C4, with gap after
            (64, 0.6, 0.4),   # E4, with gap after
            (67, 1.2, 0.4),   # G4
        ]

        # Convert to F0 curve
        times, f0_curve = melody_to_f0_curve(
            original_melody,
            sample_rate=16000,
            hop_length=160,
        )

        # Convert back to melody
        recovered_melody = f0_curve_to_midi(f0_curve, times, voicing_threshold=50.0)

        # Should have same number of notes
        assert len(recovered_melody) == len(original_melody)

        # Pitches should be close (within 2 semitones - F0 detection is approximate)
        for orig, recovered in zip(original_melody, recovered_melody):
            orig_pitch, _, _ = orig
            rec_pitch, _, _ = recovered
            assert abs(rec_pitch - orig_pitch) < 2.0


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_melody_generation_pipeline(self):
        """Test complete melody generation workflow."""
        # 1. Generate melody from lyrics
        syllables = ["sing", "a", "song", "of", "joy"]
        melody = generate_melody(
            syllables=syllables,
            key="G",
            bpm=120,
            progression_name="pop1",
            seed=42,
        )

        assert len(melody) == len(syllables)

        # 2. Convert to F0 curve
        times, f0_curve = melody_to_f0_curve(melody)
        assert len(f0_curve) > 0

        # 3. Quantize timing
        quantized = quantize_melody(melody, grid_resolution=0.25)
        assert len(quantized) == len(melody)

        # All steps complete successfully
        assert True
