"""Procedural music synthesis for drums, bass, and guitar.

Deterministic synthesis using seed-driven randomization for reproducibility.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SynthConfig:
    """Configuration for procedural synthesis."""

    sample_rate: int = 48000
    bit_depth: int = 24
    drum_tempo: int = 120  # BPM (will be overridden)
    bass_note: int = 36  # MIDI note (C1)
    guitar_tuning: Tuple[int, int, int, int, int, int] = (40, 45, 50, 55, 59, 64)  # EADGBE


class DrumSynth:
    """Synthesize drum tracks with kick, snare, hi-hat."""

    def __init__(self, sample_rate: int = 48000, seed: int = 0):
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed)

    def generate_kick(self, duration: float, bpm: int = 120) -> np.ndarray:
        """Generate kick drum using sine sweep.

        Args:
            duration: Duration in seconds
            bpm: Tempo in beats per minute

        Returns:
            Kick drum audio as float32 array (normalized -1 to 1)
        """
        samples = int(duration * self.sample_rate)
        t = np.arange(samples) / self.sample_rate

        # Sine sweep from 150 Hz to 50 Hz over 200ms
        freq_start = 150.0
        freq_end = 50.0
        decay_time = 0.2
        mask = t < decay_time

        freq_t = np.where(
            mask, freq_start + (freq_end - freq_start) * (t / decay_time), freq_end
        )
        phase = 2 * np.pi * np.cumsum(freq_t) / self.sample_rate
        kick = np.sin(phase) * np.exp(-3 * t)

        # Normalize and cast to float32 for consistency
        kick = kick / np.max(np.abs(kick))
        return kick.astype(np.float32)

    def generate_snare(self, duration: float) -> np.ndarray:
        """Generate snare drum using filtered noise.

        Args:
            duration: Duration in seconds

        Returns:
            Snare drum audio as float32 array
        """
        samples = int(duration * self.sample_rate)
        t = np.arange(samples) / self.sample_rate

        # Filtered white noise with decay
        noise = self.rng.normal(0, 1, samples)

        # High-pass filter envelope (snare is bright and short)
        decay = np.exp(-8 * t)  # Decay to silence over ~375ms
        snare = noise * decay

        snare = snare / np.max(np.abs(snare))
        return snare.astype(np.float32)

    def generate_hihat(self, duration: float) -> np.ndarray:
        """Generate closed hi-hat using filtered noise.

        Args:
            duration: Duration in seconds

        Returns:
            Hi-hat audio as float32 array
        """
        samples = int(duration * self.sample_rate)
        t = np.arange(samples) / self.sample_rate

        # Band-limited noise (hi-hat is bright but not too bright)
        noise = self.rng.normal(0, 1, samples)

        # Quick decay envelope (hi-hat is very short)
        decay = np.exp(-15 * t)
        hihat = noise * decay

        hihat = hihat / np.max(np.abs(hihat))
        return hihat.astype(np.float32)

    def generate_pattern(self, duration: float, bpm: int = 120) -> np.ndarray:
        """Generate a drum pattern for specified duration.

        Creates a 4/4 drum pattern with kick, snare, and hi-hat.

        Args:
            duration: Duration in seconds
            bpm: Tempo in beats per minute

        Returns:
            Combined drum track as float32 array
        """
        samples = int(duration * self.sample_rate)
        drums = np.zeros(samples)

        # Calculate beat duration in seconds
        beat_duration = 60.0 / bpm

        # Build pattern: kick on 1, 1.5, 3; snare on 2, 4; hi-hat on every eighth note
        beat_times = []

        # Kick pattern
        for beat in [0, 0.5, 2]:
            beat_times.append((beat * beat_duration, "kick"))

        # Snare pattern
        for beat in [1, 3]:
            beat_times.append((beat * beat_duration, "snare"))

        # Hi-hat on eighths
        for eighth in np.arange(0, 4, 0.5):
            beat_times.append((eighth * beat_duration, "hihat"))

        # Sort by time
        beat_times.sort()

        # Render each hit
        for beat_time, drum_type in beat_times:
            start_idx = int(beat_time * self.sample_rate)
            # Skip hits that would start beyond requested duration
            if start_idx >= samples:
                continue
            if drum_type == "kick":
                sound = self.generate_kick(0.3, bpm)
            elif drum_type == "snare":
                sound = self.generate_snare(0.2)
            else:  # hihat
                sound = self.generate_hihat(0.1)

            # Add to track
            end_idx = min(start_idx + len(sound), samples)
            drum_len = end_idx - start_idx
            drums[start_idx:end_idx] += sound[:drum_len] * 0.8

        # Normalize to -1...1
        if np.max(np.abs(drums)) > 0:
            drums = drums / np.max(np.abs(drums))

        return drums.astype(np.float32)


class BassSynth:
    """Synthesize bass tracks."""

    def __init__(self, sample_rate: int = 48000, seed: int = 0):
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed)

    def midi_to_freq(self, midi_note: int) -> float:
        """Convert MIDI note to frequency in Hz."""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def generate_note(
        self, midi_note: int, duration: float, envelope: str = "saw"
    ) -> np.ndarray:
        """Generate a bass note with specified waveform.

        Args:
            midi_note: MIDI note number (e.g., 36 = C1)
            duration: Duration in seconds
            envelope: Waveform type: 'sine', 'saw', 'square', 'triangle'

        Returns:
            Bass note as float32 array
        """
        samples = int(duration * self.sample_rate)
        t = np.arange(samples) / self.sample_rate
        freq = self.midi_to_freq(midi_note)

        # Generate waveform
        phase = 2 * np.pi * freq * t

        if envelope == "sine":
            wave = np.sin(phase)
        elif envelope == "saw":
            # Sawtooth: ramp -1 to 1
            wave = 2 * (phase / (2 * np.pi) % 1.0) - 1
        elif envelope == "square":
            wave = np.sign(np.sin(phase))
        elif envelope == "triangle":
            # Triangle: ramp up then down
            sawtooth = 2 * (phase / (2 * np.pi) % 1.0) - 1
            wave = 2 * np.abs(sawtooth) - 1
        else:
            wave = np.sin(phase)

        # ADSR envelope
        attack, decay, sustain_level, release = 0.01, 0.05, 0.7, 0.1

        envelope_vals = np.ones_like(t)

        # Attack
        attack_samples = int(attack * self.sample_rate)
        envelope_vals[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay
        decay_start = attack_samples
        decay_end = attack_samples + int(decay * self.sample_rate)
        decay_len = decay_end - decay_start
        envelope_vals[decay_start:decay_end] = np.linspace(
            1, sustain_level, decay_len
        )

        # Sustain
        sustain_end = len(envelope_vals) - int(release * self.sample_rate)
        envelope_vals[decay_end:sustain_end] = sustain_level

        # Release
        release_start = sustain_end
        release_len = len(envelope_vals) - release_start
        envelope_vals[release_start:] = np.linspace(sustain_level, 0, release_len)

        bass = wave * envelope_vals

        # Normalize
        if np.max(np.abs(bass)) > 0:
            bass = bass / np.max(np.abs(bass))

        return bass.astype(np.float32)

    def generate_line(
        self, midi_notes: list, note_duration: float, bpm: int = 120
    ) -> np.ndarray:
        """Generate a bass line from MIDI notes.

        Args:
            midi_notes: List of MIDI note numbers
            note_duration: Duration of each note in beats
            bpm: Tempo in beats per minute

        Returns:
            Bass line as float32 array
        """
        beat_duration = 60.0 / bpm
        note_seconds = note_duration * beat_duration

        bass_line = []
        for midi_note in midi_notes:
            note = self.generate_note(midi_note, note_seconds, envelope="saw")
            bass_line.append(note)

        # Concatenate all notes
        full_line = np.concatenate(bass_line)
        return full_line.astype(np.float32)


class GuitarSynth:
    """Synthesize guitar tracks."""

    def __init__(self, sample_rate: int = 48000, seed: int = 0):
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed)
        # Standard EADGBE tuning
        self.tuning = [40, 45, 50, 55, 59, 64]  # MIDI notes

    def midi_to_freq(self, midi_note: int) -> float:
        """Convert MIDI note to frequency in Hz."""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def generate_pluck(self, midi_note: int, duration: float) -> np.ndarray:
        """Generate a single guitar pluck (Karplus-Strong-like).

        Args:
            midi_note: MIDI note number
            duration: Duration in seconds

        Returns:
            Guitar pluck as float32 array
        """
        samples = int(duration * self.sample_rate)
        t = np.arange(samples) / self.sample_rate

        # Excitation: initial burst of harmonics
        freq = self.midi_to_freq(midi_note)

        # Plucked string oscillation with harmonics
        phase = 2 * np.pi * freq * t
        fundamental = np.sin(phase) * 0.6
        harmonic2 = np.sin(2 * phase) * 0.3
        harmonic3 = np.sin(3 * phase) * 0.1

        wave = fundamental + harmonic2 + harmonic3

        # Exponential decay envelope (natural string decay)
        decay_time = 2.0  # 2 seconds for full decay
        envelope = np.exp(-(t / decay_time) ** 1.5)

        pluck = wave * envelope

        # Normalize
        if np.max(np.abs(pluck)) > 0:
            pluck = pluck / np.max(np.abs(pluck))

        return pluck.astype(np.float32)

    def generate_chord(self, midi_notes: list, duration: float) -> np.ndarray:
        """Generate a guitar chord (strum simulation).

        Args:
            midi_notes: List of MIDI note numbers
            duration: Duration in seconds

        Returns:
            Guitar chord as float32 array
        """
        samples = int(duration * self.sample_rate)
        chord = np.zeros(samples)

        # Strum: each note starts with slight offset
        strum_time = 0.05  # 50ms strum time
        strum_offset = strum_time / len(midi_notes)

        for i, midi_note in enumerate(midi_notes):
            note_start = int(i * strum_offset * self.sample_rate)
            pluck = self.generate_pluck(midi_note, duration)

            # Add to chord
            end_idx = min(note_start + len(pluck), samples)
            pluck_len = end_idx - note_start
            chord[note_start:end_idx] += pluck[:pluck_len] * 0.5

        # Normalize
        if np.max(np.abs(chord)) > 0:
            chord = chord / np.max(np.abs(chord))

        return chord.astype(np.float32)
