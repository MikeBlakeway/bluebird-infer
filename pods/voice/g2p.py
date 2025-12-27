"""G2P (Grapheme-to-Phoneme) alignment module.

Converts lyrics to phoneme sequences and estimates frame-level durations
for use in DiffSinger synthesis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    from g2p_en import G2p
except ImportError:
    G2p = None

logger = logging.getLogger(__name__)


class PhonemeAligner:
    """G2P-based phoneme alignment for lyrics."""

    def __init__(self):
        """Initialize G2P model."""
        if G2p is None:
            logger.warning("g2p_en not available; using fallback stub.")
            self.g2p = None
        else:
            try:
                self.g2p = G2p()
            except Exception as exc:
                logger.error("Failed to load G2P model: %s", exc)
                self.g2p = None

    def lyrics_to_phonemes(self, lyrics: List[str]) -> Tuple[List[str], List[str]]:
        """Convert lyrics (words) to flattened phoneme sequence."""
        if not lyrics:
            return [], []

        if self.g2p is None:
            # Fallback: use first 2 chars of each word or silence
            phonemes, names = [], []
            for word in lyrics:
                clean = "".join(c for c in word.lower() if c.isalpha())
                if clean:
                    phonemes.extend(["SP"])  # silence
                    phonemes.extend([f"P{i}" for i in range(min(2, len(clean)))])
                    names.extend([word] * (2 + min(2, len(clean))))
            return phonemes, names

        try:
            phonemes, names = [], []
            for word in lyrics:
                g2p_out = self.g2p(word)
                word_phonemes = [p for p in g2p_out if p not in [" ", word]]
                if word_phonemes:
                    phonemes.extend(word_phonemes)
                    names.extend([word] * len(word_phonemes))
            return phonemes, names
        except Exception as exc:
            logger.error("G2P conversion failed: %s; using fallback", exc)
            phonemes, names = [], []
            for word in lyrics:
                clean = "".join(c for c in word.lower() if c.isalpha())
                if clean:
                    phonemes.extend(["SP"])
                    phonemes.extend([f"P{i}" for i in range(min(2, len(clean)))])
                    names.extend([word] * (2 + min(2, len(clean))))
            return phonemes, names

    def align_lyrics_to_frames(
        self,
        lyrics: List[str],
        bpm: int,
        duration: float,
        provided_phonemes: Optional[List[str]] = None,
        provided_durations: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Align phonemes to frame-level timing."""
        if not lyrics:
            return {"phonemes": [], "durations": [], "total": 0.0, "frame_count": 0}

        if provided_phonemes is not None:
            phonemes = provided_phonemes
            durations = provided_durations or [duration / len(phonemes)] * len(phonemes)
        else:
            phonemes, _ = self.lyrics_to_phonemes(lyrics)
            durations = [duration / len(phonemes)] * len(phonemes) if phonemes else []

        frame_rate = 100.0
        frame_count = int(duration * frame_rate)

        return {
            "phonemes": phonemes,
            "durations": durations,
            "total": duration,
            "frame_count": frame_count,
            "frame_rate": frame_rate,
        }


_aligner: Optional[PhonemeAligner] = None


def get_aligner() -> PhonemeAligner:
    """Get or create global aligner."""
    global _aligner
    if _aligner is None:
        _aligner = PhonemeAligner()
    return _aligner


def align_lyrics_to_phonemes(lyrics: List[str], bpm: int) -> Dict[str, Any]:
    """Legacy stub interface for compatibility."""
    aligner = get_aligner()
    duration = (60.0 / max(1, bpm)) * 4 * len(lyrics)
    return aligner.align_lyrics_to_frames(lyrics, bpm, duration)
