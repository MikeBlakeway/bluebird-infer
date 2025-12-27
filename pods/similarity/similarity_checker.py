"""Similarity checking logic for melody and rhythm.

Implements:
- Melody similarity via interval n-gram Jaccard (n = 3..5, weighted)
- Rhythm similarity via DTW on IOI (inter-onset intervals) when onsets provided
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from bbfeatures.ngram import intervals_from_pitches, ngrams, jaccard

try:
    from fastdtw import fastdtw  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    fastdtw = None  # type: ignore


@dataclass
class SimilarityScore:
    """Similarity score result."""

    melody_score: float  # 0.0-1.0, based on n-gram Jaccard
    rhythm_score: float  # 0.0-1.0, based on DTW
    combined_score: float  # weighted: default 0.6*melody + 0.4*rhythm
    verdict: str  # "pass", "borderline", or "block"
    confidence: float  # 0.0-1.0
    recommendations: List[str]  # actionable user recommendations


class SimilarityChecker:
    """Check melody and rhythm similarity between two sequences."""

    def __init__(
        self,
        pass_threshold: float = 0.35,
        borderline_threshold: float = 0.48,
        melody_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        combined_weights: Tuple[float, float] = (0.6, 0.4),
    ):
        """Initialize similarity checker and thresholds.

        melody_weights correspond to n-gram sizes (3, 4, 5)
        combined_weights are (melody, rhythm)
        """
        self.pass_threshold = pass_threshold
        self.borderline_threshold = borderline_threshold
        self.melody_weights = melody_weights
        self.combined_weights = combined_weights

    # ------------------------- Melody Similarity ------------------------- #
    def _melody_similarity(self, ref_midi: List[int], gen_midi: List[int]) -> float:
        """Compute weighted Jaccard over interval n-grams (n=3..5)."""
        if len(ref_midi) < 2 or len(gen_midi) < 2:
            return 0.0
        ref_ints = intervals_from_pitches(ref_midi)
        gen_ints = intervals_from_pitches(gen_midi)

        sims: List[float] = []
        weights = list(self.melody_weights)
        ns = [3, 4, 5]
        for w, n in zip(weights, ns):
            if n <= 0:
                continue
            ref_grams = ngrams(ref_ints, n)
            gen_grams = ngrams(gen_ints, n)
            s = jaccard(ref_grams, gen_grams)
            sims.append(w * s)
        denom = sum(weights)
        return sum(sims) / denom if denom > 0 else 0.0

    # ------------------------- Rhythm Similarity ------------------------ #
    def _normalize_ioi(self, onsets: List[float]) -> List[float]:
        """Compute normalized IOI sequence from onset times.

        Normalization: divide by mean IOI to make average = 1.0
        """
        if len(onsets) < 2:
            return []
        ioi = [max(1e-6, onsets[i + 1] - onsets[i]) for i in range(len(onsets) - 1)]
        mean = sum(ioi) / len(ioi)
        if mean <= 0:
            return ioi
        return [x / mean for x in ioi]

    def _rhythm_similarity(self, ref_onsets: List[float], gen_onsets: List[float]) -> Optional[float]:
        """Compute rhythm similarity via DTW on normalized IOI sequences.

        Returns None if inputs are insufficient or fastdtw is unavailable.
        """
        if fastdtw is None:
            return None
        ref_ioi = self._normalize_ioi(ref_onsets)
        gen_ioi = self._normalize_ioi(gen_onsets)
        if not ref_ioi or not gen_ioi:
            return None
        # DTW distance with absolute difference cost
        dist, path = fastdtw(ref_ioi, gen_ioi, dist=lambda a, b: abs(a - b))
        # Average per-step cost
        steps = max(1, len(path))
        avg_cost = dist / steps
        # Convert to similarity in [0,1], 1.0 when identical (avg_cost=0)
        sim = 1.0 / (1.0 + avg_cost)
        # Clamp due to numerical issues
        return max(0.0, min(1.0, sim))

    # --------------------------- Verdict ------------------------------- #
    def _verdict(self, score: float) -> str:
        if score >= self.borderline_threshold:
            return "block"
        if score >= self.pass_threshold:
            return "borderline"
        return "pass"

    def _recommendations(self, melody_score: float, rhythm_score: Optional[float], verdict: str) -> List[str]:
        recs: List[str] = []
        if verdict == "block":
            if melody_score >= self.borderline_threshold:
                recs.append("Regenerate chorus topline")
                recs.append("Shift melody key by +1 semitone")
            if rhythm_score is not None and rhythm_score >= self.borderline_threshold:
                recs.append("Alter rhythm in verse or chorus")
                recs.append("Increase syncopation or change note grouping")
        elif verdict == "borderline":
            recs.append("Lower similarity budget or change seed")
            recs.append("Vary intervals in the main hook")
        return recs

    # ---------------------------- Public ------------------------------- #
    def check(
        self,
        reference_melody: List[int],
        generated_melody: List[int],
        reference_onsets: Optional[List[float]] = None,
        generated_onsets: Optional[List[float]] = None,
    ) -> SimilarityScore:
        """Check similarity between two melodies (and optional rhythms).

        Melody inputs are MIDI note sequences.
        Rhythm comparison uses onset times (seconds) if both provided.
        """
        # Basic validation mirroring API-level checks
        if not reference_melody or not generated_melody:
            raise ValueError("reference_melody and generated_melody must be non-empty")
        if len(reference_melody) > 1000 or len(generated_melody) > 1000:
            raise ValueError("melody length too long (max 1000 notes)")

        melody_score = self._melody_similarity(reference_melody, generated_melody)

        rhythm_score: Optional[float] = None
        if reference_onsets is not None and generated_onsets is not None:
            rhythm_score = self._rhythm_similarity(reference_onsets, generated_onsets)

        # Combine scores
        m_w, r_w = self.combined_weights
        if rhythm_score is None:
            combined = melody_score  # fallback: melody only
            confidence = 0.75  # slightly reduced without rhythm evidence
        else:
            combined = (m_w * melody_score) + (r_w * rhythm_score)
            confidence = 0.9

        verdict = self._verdict(combined)
        recs = self._recommendations(melody_score, rhythm_score, verdict)

        return SimilarityScore(
            melody_score=melody_score,
            rhythm_score=0.0 if rhythm_score is None else rhythm_score,
            combined_score=combined,
            verdict=verdict,
            confidence=confidence,
            recommendations=recs,
        )


def create_checker() -> SimilarityChecker:
    """Factory function to create configured checker."""
    try:
        from pods.similarity.config import config
    except ImportError:
        # Fallback for test environment
        from .config import config

    return SimilarityChecker(
        pass_threshold=config.PASS_THRESHOLD,
        borderline_threshold=config.BORDERLINE_THRESHOLD,
    )
