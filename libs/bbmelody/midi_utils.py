"""MIDI and F0 conversion utilities.

Converts between MIDI pitch values and fundamental frequency (F0) curves.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def midi_to_freq(midi_pitch: float) -> float:
    """Convert MIDI pitch to frequency in Hz.

    Uses standard MIDI tuning: A4 (MIDI 69) = 440 Hz.
    """
    return 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))


def freq_to_midi(freq: float) -> float:
    """Convert frequency in Hz to MIDI pitch (float)."""
    if freq <= 0:
        raise ValueError(f"Frequency must be positive: {freq}")
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def melody_to_f0_curve(
    melody: List[Tuple[int, float, float]],
    sample_rate: int = 16000,
    hop_length: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert melody notes to continuous F0 curve.

    Args:
        melody: List of (midi_pitch, onset_time, duration) tuples
        sample_rate: Audio sample rate in Hz
        hop_length: Hop length for F0 curve (samples per frame)

    Returns:
        (times, f0_curve) where:
            times: Time in seconds for each frame
            f0_curve: F0 values in Hz (0 for unvoiced/rest)
    """
    if not melody:
        return np.array([]), np.array([])

    # Determine total duration
    max_time = max(onset + dur for _, onset, dur in melody)

    # Calculate number of frames
    hop_seconds = hop_length / sample_rate
    num_frames = int(np.ceil(max_time / hop_seconds)) + 1

    times = np.arange(num_frames) * hop_seconds
    f0_curve = np.zeros(num_frames, dtype=np.float32)

    # Fill F0 values for each note
    for midi_pitch, onset, duration in melody:
        freq = midi_to_freq(float(midi_pitch))

        # Find frame indices for this note
        start_frame = int(onset / hop_seconds)
        end_frame = int((onset + duration) / hop_seconds)

        # Ensure within bounds
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame + 1)

        # Set F0 for this note's duration
        f0_curve[start_frame:end_frame] = freq

    return times, f0_curve


def f0_curve_to_midi(
    f0_curve: np.ndarray,
    times: np.ndarray,
    voicing_threshold: float = 50.0,
) -> List[Tuple[float, float, float]]:
    """Convert F0 curve to discrete MIDI notes.

    Groups consecutive voiced frames into notes.

    Args:
        f0_curve: F0 values in Hz
        times: Time in seconds for each frame
        voicing_threshold: Minimum F0 to consider voiced (Hz)

    Returns:
        List of (midi_pitch, onset_time, duration) tuples
    """
    if len(f0_curve) == 0:
        return []

    # Identify voiced frames
    voiced = f0_curve > voicing_threshold

    notes = []
    in_note = False
    note_start_idx = 0
    note_pitches = []

    for i, is_voiced in enumerate(voiced):
        if is_voiced and not in_note:
            # Note onset
            in_note = True
            note_start_idx = i
            note_pitches = [f0_curve[i]]
        elif is_voiced and in_note:
            # Continue note
            note_pitches.append(f0_curve[i])
        elif not is_voiced and in_note:
            # Note offset
            in_note = False

            # Calculate average pitch for note
            avg_freq = np.mean(note_pitches)
            midi_pitch = freq_to_midi(avg_freq)

            # Note timing
            onset_time = times[note_start_idx]
            duration = times[i - 1] - times[note_start_idx]

            notes.append((float(midi_pitch), float(onset_time), float(duration)))
            note_pitches = []

    # Handle final note if still in progress
    if in_note and note_pitches:
        avg_freq = np.mean(note_pitches)
        midi_pitch = freq_to_midi(avg_freq)
        onset_time = times[note_start_idx]
        duration = times[-1] - times[note_start_idx]
        notes.append((float(midi_pitch), float(onset_time), float(duration)))

    return notes


def quantize_melody(
    melody: List[Tuple[int, float, float]],
    grid_resolution: float = 0.25,
) -> List[Tuple[int, float, float]]:
    """Quantize melody timing to grid (e.g., 16th notes).

    Args:
        melody: List of (midi_pitch, onset_time, duration)
        grid_resolution: Grid size in seconds (0.25 = 16th note at 120 BPM)

    Returns:
        Quantized melody
    """
    quantized = []

    for pitch, onset, duration in melody:
        # Quantize onset to nearest grid point
        quantized_onset = round(onset / grid_resolution) * grid_resolution

        # Quantize duration to nearest grid point (minimum 1 grid unit)
        quantized_duration = max(
            grid_resolution,
            round(duration / grid_resolution) * grid_resolution,
        )

        quantized.append((pitch, quantized_onset, quantized_duration))

    return quantized


__all__ = [
    "midi_to_freq",
    "freq_to_midi",
    "melody_to_f0_curve",
    "f0_curve_to_midi",
    "quantize_melody",
]
