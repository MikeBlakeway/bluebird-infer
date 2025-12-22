"""Interval n-gram generation for melody similarity."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple, Dict


def intervals_from_pitches(pitches: Iterable[int]) -> List[int]:
    """Compute semitone intervals between successive MIDI pitches.

    Rests or non-integers should be filtered before calling.
    """
    seq = list(int(p) for p in pitches)
    return [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]


def ngrams(intervals: Iterable[int], n: int) -> Counter[Tuple[int, ...]]:
    """Return Counter of interval n-grams of length n."""
    arr = list(int(x) for x in intervals)
    grams = [tuple(arr[i : i + n]) for i in range(0, len(arr) - n + 1)]
    return Counter(grams)


def jaccard(counter_a: Counter[Tuple[int, ...]], counter_b: Counter[Tuple[int, ...]]) -> float:
    """Jaccard similarity between two n-gram Counters (set-based).

    Note: Only presence/absence is considered here (not frequency).
    """
    set_a = set(counter_a.keys())
    set_b = set(counter_b.keys())
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


__all__ = ["intervals_from_pitches", "ngrams", "jaccard"]
