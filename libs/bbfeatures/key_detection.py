"""Musical key detection using chroma features.

Returns a best key guess among 24 keys (12 pitch classes × {major, minor})
and a confidence score in [0, 1].
"""

from __future__ import annotations

import numpy as np
import librosa
from typing import Dict, Tuple


MAJOR_PROFILE = np.array([6, 2, 3, 2, 6, 2, 3, 2, 5, 2, 3, 2], dtype=float)
MINOR_PROFILE = np.array([6, 2, 3, 6, 2, 2, 3, 2, 6, 2, 2, 3], dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = v.sum()
    return v / s if s > 0 else v


def chroma_from_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute mean chroma vector from audio."""
    # Ensure mono for chroma stability
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    return chroma.mean(axis=1)


def detect_key(audio: np.ndarray, sr: int) -> Dict[str, object]:
    """Detect musical key from audio.

    Returns dict: {"key": "C major", "confidence": 0.0..1.0}
    """
    chroma = _normalize(chroma_from_audio(audio, sr))

    scores = []
    for shift in range(12):
        rot = np.roll(chroma, -shift)
        major_score = float(np.dot(rot, MAJOR_PROFILE))
        minor_score = float(np.dot(rot, MINOR_PROFILE))
        scores.append((shift, "major", major_score))
        scores.append((shift, "minor", minor_score))

    best = max(scores, key=lambda x: x[2])
    total = sum(s for _, _, s in scores)
    confidence = (best[2] / total) if total > 0 else 0.0

    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_name = f"{pitch_classes[best[0]]} {best[1]}"
    return {"key": key_name, "confidence": confidence}


__all__ = ["detect_key", "chroma_from_audio"]
