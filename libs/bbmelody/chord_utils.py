"""Chord progression utilities for melody generation.

Provides chord structures, progressions, and scale helpers for procedural melody.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from enum import Enum


class ChordType(str, Enum):
    """Common chord types."""
    MAJOR = "major"
    MINOR = "minor"
    DOMINANT = "dom7"
    MINOR_7 = "m7"
    MAJOR_7 = "maj7"


# Chord intervals from root (semitones)
CHORD_INTERVALS: Dict[ChordType, Tuple[int, ...]] = {
    ChordType.MAJOR: (0, 4, 7),
    ChordType.MINOR: (0, 3, 7),
    ChordType.DOMINANT: (0, 4, 7, 10),
    ChordType.MINOR_7: (0, 3, 7, 10),
    ChordType.MAJOR_7: (0, 4, 7, 11),
}


# Scale patterns (semitones from root)
SCALE_PATTERNS: Dict[str, Tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
}


# Common progressions (scale degrees, 1-indexed)
COMMON_PROGRESSIONS: Dict[str, List[Tuple[int, ChordType]]] = {
    "pop1": [(1, ChordType.MAJOR), (5, ChordType.MAJOR), (6, ChordType.MINOR), (4, ChordType.MAJOR)],  # I-V-vi-IV
    "pop2": [(1, ChordType.MAJOR), (4, ChordType.MAJOR), (5, ChordType.MAJOR), (1, ChordType.MAJOR)],  # I-IV-V-I
    "blues": [(1, ChordType.DOMINANT), (4, ChordType.DOMINANT), (1, ChordType.DOMINANT), (5, ChordType.DOMINANT)],  # I7-IV7-I7-V7
    "jazz": [(2, ChordType.MINOR_7), (5, ChordType.DOMINANT), (1, ChordType.MAJOR_7), (1, ChordType.MAJOR_7)],  # ii7-V7-Imaj7
}


def key_to_midi_root(key: str) -> int:
    """Convert key name to MIDI root pitch (e.g., 'C' -> 60, 'D' -> 62).

    Supports major and minor keys (e.g., 'C', 'Cm', 'F#', 'Bb').
    Returns MIDI pitch of root in octave 4 (middle octave).
    """
    note_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    key = key.strip()
    # Handle minor keys (e.g., "Cm" -> "C")
    if key.endswith("m"):
        key = key[:-1]

    # Parse accidentals
    base_note = key[0].upper()
    accidental = key[1:] if len(key) > 1 else ""

    if base_note not in note_map:
        raise ValueError(f"Invalid key: {key}")

    midi_note = 60 + note_map[base_note]  # Middle C = 60

    if "#" in accidental:
        midi_note += accidental.count("#")
    if "b" in accidental:
        midi_note -= accidental.count("b")

    return midi_note


def get_scale_notes(root_midi: int, scale_type: str = "major") -> List[int]:
    """Get MIDI pitches for a scale starting from root.

    Returns 2 octaves worth of scale notes.
    """
    if scale_type not in SCALE_PATTERNS:
        raise ValueError(f"Unknown scale type: {scale_type}")

    pattern = SCALE_PATTERNS[scale_type]
    notes = []
    for octave in range(3):  # 3 octaves for range
        for interval in pattern:
            notes.append(root_midi + octave * 12 + interval)
    return notes


def get_chord_notes(root_midi: int, chord_type: ChordType, octave_offset: int = 0) -> List[int]:
    """Get MIDI pitches for a chord.

    Args:
        root_midi: Root note MIDI pitch
        chord_type: Type of chord
        octave_offset: Octave shift from root (-1, 0, 1, etc.)

    Returns:
        List of MIDI pitches for chord tones
    """
    if chord_type not in CHORD_INTERVALS:
        raise ValueError(f"Unknown chord type: {chord_type}")

    intervals = CHORD_INTERVALS[chord_type]
    base = root_midi + (octave_offset * 12)
    return [base + interval for interval in intervals]


def progression_to_chords(
    progression_name: str,
    key: str,
    scale_type: str = "major",
) -> List[Tuple[int, ChordType, List[int]]]:
    """Convert progression name to chord sequence with MIDI pitches.

    Returns:
        List of (root_midi, chord_type, chord_notes)
    """
    if progression_name not in COMMON_PROGRESSIONS:
        raise ValueError(f"Unknown progression: {progression_name}")

    root_midi = key_to_midi_root(key)
    scale_notes = get_scale_notes(root_midi, scale_type)
    progression = COMMON_PROGRESSIONS[progression_name]

    result = []
    for degree, chord_type in progression:
        # Scale degree to MIDI pitch (1-indexed -> 0-indexed)
        chord_root = scale_notes[degree - 1]
        chord_notes = get_chord_notes(chord_root, chord_type)
        result.append((chord_root, chord_type, chord_notes))

    return result


__all__ = [
    "ChordType",
    "CHORD_INTERVALS",
    "SCALE_PATTERNS",
    "COMMON_PROGRESSIONS",
    "key_to_midi_root",
    "get_scale_notes",
    "get_chord_notes",
    "progression_to_chords",
]
