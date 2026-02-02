"""
Tests for confidence scoring utilities.
"""

import pytest
from app.utils.confidence import (
    combine_confidence_scores,
    penalize_confidence,
    boost_confidence,
    confidence_variance,
    confidence_level_name,
)


def test_combine_confidence_mean():
    """Test mean combination."""
    scores = [0.8, 0.9, 0.7]
    result = combine_confidence_scores(scores, method="mean")
    assert abs(result - 0.8) < 0.01


def test_combine_confidence_weighted():
    """Test weighted combination."""
    scores = [0.9, 0.7]
    weights = [0.7, 0.3]
    result = combine_confidence_scores(scores, weights=weights, method="weighted_mean")
    expected = 0.9 * 0.7 + 0.7 * 0.3
    assert abs(result - expected) < 0.01


def test_combine_confidence_min():
    """Test min combination."""
    scores = [0.8, 0.5, 0.9]
    result = combine_confidence_scores(scores, method="min")
    assert result == 0.5


def test_combine_confidence_max():
    """Test max combination."""
    scores = [0.8, 0.5, 0.9]
    result = combine_confidence_scores(scores, method="max")
    assert result == 0.9


def test_penalize_confidence():
    """Test confidence penalty."""
    base = 0.9
    penalized = penalize_confidence(base, 0.8)
    assert penalized < base
    assert abs(penalized - 0.72) < 0.001


def test_boost_confidence():
    """Test confidence boost."""
    base = 0.8
    boosted = boost_confidence(base, 1.1)
    assert boosted > base
    assert abs(boosted - 0.88) < 0.01


def test_boost_confidence_caps_at_max():
    """Test that boost doesn't exceed maximum."""
    base = 0.95
    boosted = boost_confidence(base, 1.2, max_confidence=0.99)
    assert boosted <= 0.99


def test_confidence_variance():
    """Test variance calculation."""
    scores = [0.8, 0.8, 0.8]
    variance = confidence_variance(scores)
    assert variance == 0.0  # No variance when all same


def test_confidence_level_name_very_high():
    """Test confidence level naming."""
    assert confidence_level_name(0.97) == "VERY_HIGH"
    assert confidence_level_name(0.90) == "HIGH"
    assert confidence_level_name(0.75) == "ACCEPTABLE"
    assert confidence_level_name(0.60) == "LOW"
    assert confidence_level_name(0.30) == "VERY_LOW"


def test_confidence_scores_clamped_to_range():
    """Test that confidence scores are clamped to 0-1."""
    # Negative values should be clamped to 0
    result = combine_confidence_scores([-0.5, 0.8], method="mean")
    assert 0.0 <= result <= 1.0
    
    # Values > 1 should be clamped to 1
    result = combine_confidence_scores([1.5, 0.8], method="mean")
    assert 0.0 <= result <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
