"""Procedural melody generation from lyrics and chord progressions.

Rule-based melody composition with contour guidance and seed control.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from .chord_utils import (
    ChordType,
    get_scale_notes,
    key_to_midi_root,
    progression_to_chords,
)


class MelodyGenerator:
    """Procedural melody generator with deterministic seed control.

    Generates melodies that:
    - Follow chord progressions (use chord tones + passing notes)
    - Respect contour guidance (rise/fall/stable patterns)
    - Align with lyric syllables
    - Are reproducible (same seed = same melody)
    """

    def __init__(
        self,
        key: str,
        bpm: int,
        scale_type: str = "major",
        seed: Optional[int] = None,
    ):
        """Initialize melody generator.

        Args:
            key: Musical key (e.g., 'C', 'Dm', 'F#')
            bpm: Tempo in beats per minute
            scale_type: Scale to use ('major', 'minor', 'dorian', etc.)
            seed: Random seed for reproducibility
        """
        self.key = key
        self.bpm = bpm
        self.scale_type = scale_type
        self.root_midi = key_to_midi_root(key)
        self.scale_notes = get_scale_notes(self.root_midi, scale_type)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_from_lyrics(
        self,
        syllables: List[str],
        progression_name: str = "pop1",
        contour: Optional[List[float]] = None,
        note_range: Tuple[int, int] = (60, 72),  # C4 to C5
    ) -> List[Tuple[int, float, float]]:
        """Generate melody aligned to lyric syllables.

        Args:
            syllables: List of syllable strings
            progression_name: Chord progression to follow
            contour: Optional contour guidance (normalized -1 to 1)
            note_range: (min_midi, max_midi) pitch range

        Returns:
            List of (midi_pitch, onset_time, duration) tuples in seconds
        """
        if not syllables:
            raise ValueError("No syllables provided")

        # Get chord progression
        chords = progression_to_chords(progression_name, self.key, self.scale_type)

        # Calculate timing (simple quarter notes for now)
        beats_per_syllable = 1.0  # One beat per syllable
        seconds_per_beat = 60.0 / self.bpm

        # Generate contour if not provided
        if contour is None:
            contour = self._generate_default_contour(len(syllables))
        elif len(contour) != len(syllables):
            # Interpolate contour to match syllable count
            contour = self._interpolate_contour(contour, len(syllables))

        # Generate melody notes
        melody = []
        current_time = 0.0

        # Filter scale notes to range
        scale_in_range = [n for n in self.scale_notes if note_range[0] <= n <= note_range[1]]
        if not scale_in_range:
            raise ValueError(f"No scale notes in range {note_range}")

        for i, syllable in enumerate(syllables):
            # Select chord for this position (cycle through progression)
            chord_idx = (i * len(chords) // len(syllables)) % len(chords)
            _, chord_type, chord_notes = chords[chord_idx]

            # Select pitch based on contour and chord
            pitch = self._select_pitch(
                contour[i],
                chord_notes,
                scale_in_range,
                note_range,
                previous_pitch=melody[-1][0] if melody else None,
            )

            duration = seconds_per_beat * beats_per_syllable
            melody.append((pitch, current_time, duration))
            current_time += duration

        return melody

    def _generate_default_contour(self, length: int) -> List[float]:
        """Generate a natural-sounding default contour (arch shape)."""
        if length <= 1:
            return [0.0] * length

        # Arch contour: rise then fall
        x = np.linspace(0, 1, length)
        contour = np.sin(x * np.pi)  # 0 -> 1 -> 0
        # Normalize to [-0.5, 0.5] for subtler movement
        return (contour * 0.5).tolist()

    def _interpolate_contour(self, contour: List[float], target_length: int) -> List[float]:
        """Interpolate contour to match target length."""
        if len(contour) == target_length:
            return contour

        x_old = np.linspace(0, 1, len(contour))
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, contour).tolist()

    def _select_pitch(
        self,
        contour_value: float,
        chord_notes: List[int],
        scale_notes: List[int],
        note_range: Tuple[int, int],
        previous_pitch: Optional[int] = None,
    ) -> int:
        """Select MIDI pitch based on contour and chord.

        Favors chord tones (70% probability) and smooth voice leading.
        """
        # Determine target range based on contour (-1 to 1)
        range_span = note_range[1] - note_range[0]
        # Map contour [-1, 1] to [0, 1] for range positioning
        normalized_position = (contour_value + 1.0) / 2.0
        target_pitch = note_range[0] + int(normalized_position * range_span)

        # Prefer chord tones (70% of the time)
        use_chord_tone = random.random() < 0.7

        if use_chord_tone:
            # Find chord tones in range
            chord_in_range = [n for n in chord_notes if note_range[0] <= n <= note_range[1]]
            if chord_in_range:
                # Pick closest chord tone to target
                pitch = min(chord_in_range, key=lambda n: abs(n - target_pitch))
            else:
                # Fallback to scale notes
                pitch = min(scale_notes, key=lambda n: abs(n - target_pitch))
        else:
            # Use passing tones from scale
            pitch = min(scale_notes, key=lambda n: abs(n - target_pitch))

        # Smooth voice leading: avoid large jumps
        if previous_pitch is not None:
            max_interval = 7  # Perfect fifth
            if abs(pitch - previous_pitch) > max_interval:
                # Find closer alternative
                candidates = [n for n in scale_notes if abs(n - previous_pitch) <= max_interval]
                if candidates:
                    # Pick closest to original target
                    pitch = min(candidates, key=lambda n: abs(n - target_pitch))

        return pitch


def generate_melody(
    syllables: List[str],
    key: str,
    bpm: int,
    progression_name: str = "pop1",
    contour: Optional[List[float]] = None,
    seed: Optional[int] = None,
    note_range: Tuple[int, int] = (60, 72),
) -> List[Tuple[int, float, float]]:
    """Convenience function to generate melody.

    Returns:
        List of (midi_pitch, onset_time, duration) tuples
    """
    generator = MelodyGenerator(key=key, bpm=bpm, seed=seed)
    return generator.generate_from_lyrics(
        syllables=syllables,
        progression_name=progression_name,
        contour=contour,
        note_range=note_range,
    )


__all__ = ["MelodyGenerator", "generate_melody"]
