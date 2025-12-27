import math
import pytest

from pods.similarity.similarity_checker import SimilarityChecker


def test_melody_similarity_identical_sequence_passes_thresholds():
    checker = SimilarityChecker()
    ref = [60, 62, 64, 65, 67, 69, 71, 72]
    gen = [60, 62, 64, 65, 67, 69, 71, 72]
    result = checker.check(ref, gen)
    assert 0.0 <= result.melody_score <= 1.0
    assert math.isclose(result.melody_score, 1.0, rel_tol=1e-6)
    assert result.rhythm_score == 0.0
    assert result.combined_score == result.melody_score
    # With identical melody only, combined >= 0.48 should block
    assert result.verdict == ("block" if result.combined_score >= checker.borderline_threshold else "borderline")


def test_melody_similarity_different_sequences_is_low():
    checker = SimilarityChecker()
    ref = [60, 62, 64, 65, 67, 69, 71, 72]
    gen = [60, 62, 63, 65, 67, 70, 71, 72]
    result = checker.check(ref, gen)
    assert 0.0 <= result.melody_score <= 1.0
    assert result.melody_score < 0.5
    assert result.verdict in {"pass", "borderline", "block"}


@pytest.mark.parametrize(
    "ref_onsets, gen_onsets, expected_relation",
    [
        ([0.0, 0.5, 1.0, 1.5], [0.0, 0.5, 1.0, 1.5], "high"),
        ([0.0, 0.5, 1.0, 1.5], [0.0, 0.6, 1.3, 1.9], "medium"),
    ],
)
def test_rhythm_similarity_with_onsets_affects_combined(ref_onsets, gen_onsets, expected_relation):
    checker = SimilarityChecker()
    ref = [60, 62, 64, 65]
    gen = [60, 62, 64, 65]
    result = checker.check(ref, gen, ref_onsets, gen_onsets)
    assert 0.0 <= result.rhythm_score <= 1.0
    if expected_relation == "high":
        assert result.rhythm_score > 0.9
        assert result.combined_score >= result.melody_score
    else:
        assert result.rhythm_score < 0.95


def test_empty_inputs_raise_validation_like_behavior():
    checker = SimilarityChecker()
    with pytest.raises(Exception):
        checker.check([], [60])
    with pytest.raises(Exception):
        checker.check([60], [])
