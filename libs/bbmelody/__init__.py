"""Bluebird Melody Generation

Procedural melody composition and MIDI/F0 conversion utilities.
"""

__version__ = "0.1.0"

from .chord_utils import (
    ChordType,
    CHORD_INTERVALS,
    SCALE_PATTERNS,
    COMMON_PROGRESSIONS,
    key_to_midi_root,
    get_scale_notes,
    get_chord_notes,
    progression_to_chords,
)
from .generator import MelodyGenerator, generate_melody
from .midi_utils import (
    midi_to_freq,
    freq_to_midi,
    melody_to_f0_curve,
    f0_curve_to_midi,
    quantize_melody,
)

__all__ = [
    # Chord utilities
    "ChordType",
    "CHORD_INTERVALS",
    "SCALE_PATTERNS",
    "COMMON_PROGRESSIONS",
    "key_to_midi_root",
    "get_scale_notes",
    "get_chord_notes",
    "progression_to_chords",
    # Melody generation
    "MelodyGenerator",
    "generate_melody",
    # MIDI/F0 utilities
    "midi_to_freq",
    "freq_to_midi",
    "melody_to_f0_curve",
    "f0_curve_to_midi",
    "quantize_melody",
]
