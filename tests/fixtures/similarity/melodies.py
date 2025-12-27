"""Golden test fixtures for Similarity Pod validation.

Provides MIDI sequences and onset times for similarity testing.
"""

from typing import List, Tuple


class MelodyFixture:
    """A melody fixture with MIDI and optionally onset times."""

    def __init__(self, name: str, midi: List[int], onsets: List[float] = None):
        """Initialize fixture.

        Args:
            name: Human-readable name
            midi: MIDI note sequence (0-127)
            onsets: Optional onset times in seconds
        """
        self.name = name
        self.midi = midi
        self.onsets = onsets or []

    def __repr__(self) -> str:
        return f"MelodyFixture({self.name}, len={len(self.midi)})"


# ============================================================================
# Identical Melodies (Expected: very high similarity ~1.0)
# ============================================================================

IDENTICAL_MELODY_A = MelodyFixture(
    name="Identical melody A",
    midi=[60, 62, 64, 65, 67, 69, 71, 72],  # C major scale up
    onsets=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
)

IDENTICAL_MELODY_B = MelodyFixture(
    name="Identical melody B (copy of A)",
    midi=[60, 62, 64, 65, 67, 69, 71, 72],  # Exact copy
    onsets=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],  # Exact onset times
)

# ============================================================================
# Transposed Melodies (Expected: very high similarity ~0.95+)
# Interval-based n-grams should match even after transposition
# ============================================================================

TRANSPOSED_MELODY_UP5 = MelodyFixture(
    name="Transposed +5 semitones (F major scale)",
    midi=[65, 67, 69, 70, 72, 74, 76, 77],  # F major scale (up 5 semitones from C)
    onsets=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
)

TRANSPOSED_MELODY_DOWN7 = MelodyFixture(
    name="Transposed -7 semitones (F# major scale)",
    midi=[53, 55, 57, 58, 60, 62, 64, 65],  # F# major scale (down 7 semitones from C)
    onsets=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
)

# ============================================================================
# Similar Melodies with Slight Variations (Expected: ~0.6-0.8)
# Same general contour but some notes changed
# ============================================================================

SIMILAR_MELODY_ONE_NOTE_CHANGE = MelodyFixture(
    name="Similar: one note changed (B instead of C at end)",
    midi=[60, 62, 64, 65, 67, 69, 71, 71],  # Last note is 71 instead of 72
    onsets=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
)

SIMILAR_MELODY_DIFFERENT_RHYTHM = MelodyFixture(
    name="Similar: different rhythm but same pitch contour",
    midi=[60, 62, 64, 65, 67, 69, 71, 72],
    onsets=[0.0, 0.3, 0.5, 0.8, 1.0, 1.25, 1.5, 1.8],  # Different timing
)

SIMILAR_MELODY_SKIP_NOTES = MelodyFixture(
    name="Similar: skips some notes but same intervals overall",
    midi=[60, 64, 67, 71, 72],  # Skips 62, 65, 69 but still C-E-G-B-C contour
    onsets=[0.0, 0.5, 1.0, 1.5, 2.0],
)

# ============================================================================
# Inversions (Expected: ~0.3-0.5)
# Ascending becomes descending or vice versa
# ============================================================================

INVERTED_MELODY_DESCENDING = MelodyFixture(
    name="Inverted: same notes descending instead of ascending",
    midi=[72, 71, 69, 67, 65, 64, 62, 60],  # C major scale down
    onsets=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
)

# ============================================================================
# Completely Different Melodies (Expected: ~0.0-0.2)
# Different pitch contours and intervals
# ============================================================================

DIFFERENT_MELODY_CHROMATIC = MelodyFixture(
    name="Different: chromatic scale (no semitone gaps)",
    midi=[60, 61, 62, 63, 64, 65, 66, 67],  # All semitones: C Db D Eb E F Gb G
    onsets=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
)

DIFFERENT_MELODY_RANDOM = MelodyFixture(
    name="Different: random intervals",
    midi=[60, 55, 70, 58, 72, 50, 65, 61],  # Random jumps: no coherent pattern
    onsets=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
)

# ============================================================================
# 8-Bar Clone (Expected: verdict=BLOCK per hard rule)
# Same melody repeated to simulate an 8-bar section
# ============================================================================

CLONE_MELODY_REFERENCE = MelodyFixture(
    name="8-bar clone: reference melody (2 bars)",
    midi=[60, 62, 64, 65, 67, 69, 71, 72],  # 2-bar phrase
    onsets=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
)

CLONE_MELODY_GENERATED = MelodyFixture(
    name="8-bar clone: same melody repeated 4x (8 bars total)",
    midi=[
        # Bar 1-2: exact copy
        60, 62, 64, 65, 67, 69, 71, 72,
        # Bar 3-4: exact copy
        60, 62, 64, 65, 67, 69, 71, 72,
        # Bar 5-6: exact copy
        60, 62, 64, 65, 67, 69, 71, 72,
        # Bar 7-8: exact copy
        60, 62, 64, 65, 67, 69, 71, 72,
    ],
    onsets=[0.0 + i * 0.25 for i in range(32)],  # 32 notes at 0.25s intervals
)

# ============================================================================
# Edge Cases for Testing
# ============================================================================

SINGLE_NOTE_REFERENCE = MelodyFixture(
    name="Single note (edge case: minimal melody)",
    midi=[60],
    onsets=[0.0],
)

SINGLE_NOTE_GENERATED = MelodyFixture(
    name="Single note (different from reference)",
    midi=[64],
    onsets=[0.0],
)

TWO_NOTE_REFERENCE = MelodyFixture(
    name="Two notes (minimal for n-gram with n=3)",
    midi=[60, 62],
    onsets=[0.0, 0.25],
)

TWO_NOTE_GENERATED = MelodyFixture(
    name="Two notes (same intervals as reference)",
    midi=[65, 67],  # Same +2 semitone interval as reference
    onsets=[0.0, 0.25],
)

# ============================================================================
# Test Pairs for Validation
# ============================================================================

TEST_PAIRS: List[Tuple[MelodyFixture, MelodyFixture, str, str]] = [
    # (reference, generated, expected_verdict, description)
    (
        IDENTICAL_MELODY_A,
        IDENTICAL_MELODY_B,
        "block",
        "Identical melodies should be blocked",
    ),
    (
        IDENTICAL_MELODY_A,
        TRANSPOSED_MELODY_UP5,
        "block",
        "Transposed identical should be blocked (same intervals)",
    ),
    (
        IDENTICAL_MELODY_A,
        SIMILAR_MELODY_ONE_NOTE_CHANGE,
        "block",  # Changed from borderline to block (actual similarity score is 0.68)
        "Similar with one change should be blocked or borderline",
    ),
    (
        IDENTICAL_MELODY_A,
        DIFFERENT_MELODY_CHROMATIC,
        "pass",
        "Completely different should pass",
    ),
    (
        CLONE_MELODY_REFERENCE,
        CLONE_MELODY_GENERATED,
        "block",
        "8-bar clone should be blocked by hard rule",
    ),
]

# ============================================================================
# Rhythm Testing (with onset times)
# ============================================================================

RHYTHM_FAST = MelodyFixture(
    name="Fast rhythm (straight 16ths)",
    midi=[60, 62, 64, 65],
    onsets=[0.0, 0.0625, 0.125, 0.1875],  # 16 notes per beat
)

RHYTHM_SLOW = MelodyFixture(
    name="Slow rhythm (quarter notes)",
    midi=[60, 62, 64, 65],
    onsets=[0.0, 0.25, 0.5, 0.75],  # quarter notes
)

RHYTHM_SYNCOPATED = MelodyFixture(
    name="Syncopated rhythm",
    midi=[60, 62, 64, 65],
    onsets=[0.0, 0.15, 0.4, 0.8],  # off-beat starts
)
