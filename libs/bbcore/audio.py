"""Audio I/O helpers for Bluebird.

Standardizes WAV reading/writing with validation for 48 kHz / 24-bit PCM.
Uses soundfile for robust I/O. Waveforms are returned as float32 arrays
in range [-1.0, 1.0].
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import soundfile as sf


DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_SUBTYPE = "PCM_24"  # 24-bit


def validate_format(sample_rate: int, channels: int, subtype: str = DEFAULT_SUBTYPE) -> None:
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(f"Invalid sample rate: {sample_rate} (expected {DEFAULT_SAMPLE_RATE})")
    if channels not in (1, 2):
        raise ValueError(f"Invalid channel count: {channels} (expected 1 or 2)")
    if subtype != DEFAULT_SUBTYPE:
        raise ValueError(f"Invalid subtype: {subtype} (expected {DEFAULT_SUBTYPE})")


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV file and return (audio, sample_rate).

    Audio is returned as float32 np.ndarray with shape (samples, channels).
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio, sr


def read_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Read WAV from bytes and return (audio, sample_rate)."""
    with io.BytesIO(data) as buf:
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)
    return audio, sr


def write_wav(path: str, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE, subtype: str = DEFAULT_SUBTYPE) -> None:
    """Write audio to WAV with the specified format.

    Accepts audio as shape (samples,) or (samples, channels). Values should be
    float32 in [-1, 1].
    """
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    channels = audio.shape[1]
    validate_format(sample_rate, channels, subtype)
    sf.write(path, audio, sample_rate, subtype=subtype)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo to mono by averaging channels. If mono, return as-is."""
    if audio.ndim == 1:
        return audio
    if audio.shape[1] == 1:
        return audio[:, 0]
    return audio.mean(axis=1)


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SUBTYPE",
    "validate_format",
    "read_wav",
    "read_wav_bytes",
    "write_wav",
    "to_mono",
]
