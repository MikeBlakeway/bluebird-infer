"""Tempo (BPM) and onset detection using librosa."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import librosa


def detect_bpm_and_onsets(audio: np.ndarray, sr: int) -> Dict[str, object]:
    """Estimate BPM and return onset times in seconds.

    Returns: {"bpm": float, "onsets": [float, ...]}
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Onset strength and detection
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")

    # Tempo estimation using beat tracking
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    bpm = float(tempo[0]) if tempo.size else 0.0

    return {"bpm": bpm, "onsets": onsets.tolist()}


__all__ = ["detect_bpm_and_onsets"]
