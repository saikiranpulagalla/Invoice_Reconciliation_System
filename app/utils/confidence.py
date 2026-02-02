"""
Confidence scoring utilities.
Methods to calculate and combine confidence scores.
"""

from typing import List, Dict, Tuple
from statistics import mean, stdev


def combine_confidence_scores(
    scores: List[float],
    weights: List[float] = None,
    method: str = "weighted_mean"
) -> float:
    """
    Combine multiple confidence scores.
    
    Args:
        scores: List of confidence scores (0-1)
        weights: Optional weights for weighted mean
        method: "mean", "min", "max", or "weighted_mean"
    
    Returns:
        Combined confidence score (0-1)
    """
    if not scores:
        return 0.0
    
    # Validate input scores and log warnings for out-of-range values
    validated_scores = []
    for i, score in enumerate(scores):
        if score < 0.0 or score > 1.0:
            from app.utils.logging import setup_logging
            logger = setup_logging(__name__)
            logger.warning(f"Confidence score {i} out of range: {score}. Clamping to [0,1]")
        validated_scores.append(max(0.0, min(1.0, score)))
    
    scores = validated_scores
    
    if method == "mean":
        return mean(scores)
    elif method == "min":
        return min(scores)
    elif method == "max":
        return max(scores)
    elif method == "weighted_mean":
        if weights is None:
            weights = [1.0] * len(scores)
        if len(weights) != len(scores):
            raise ValueError(f"Weights length ({len(weights)}) must match scores length ({len(scores)})")
        weights = [w / sum(weights) for w in weights]  # Normalize weights
        return sum(s * w for s, w in zip(scores, weights))
    else:
        raise ValueError(f"Unknown confidence combination method: {method}")


def penalize_confidence(
    base_confidence: float,
    penalty_factor: float = 0.9
) -> float:
    """
    Apply a penalty to confidence (e.g., for uncertainty).
    
    Args:
        base_confidence: Original confidence score
        penalty_factor: Factor to multiply by (0-1)
    
    Returns:
        Penalized confidence score
    """
    return max(0.0, min(1.0, base_confidence * penalty_factor))


def boost_confidence(
    base_confidence: float,
    boost_factor: float = 1.1,
    max_confidence: float = 0.99
) -> float:
    """
    Boost confidence (e.g., when multiple sources agree).
    
    Args:
        base_confidence: Original confidence score
        boost_factor: Factor to multiply by (>1)
        max_confidence: Maximum allowed confidence
    
    Returns:
        Boosted confidence score
    """
    return min(max_confidence, base_confidence * boost_factor)


def confidence_variance(scores: List[float]) -> float:
    """
    Calculate variance in confidence scores.
    High variance suggests disagreement between methods.
    
    Args:
        scores: List of confidence scores
    
    Returns:
        Variance (0-1 range approximately)
    """
    if len(scores) < 2:
        return 0.0
    
    return stdev(scores)


def calculate_uncertainty_score(
    confidence: float,
    variance: float = None
) -> float:
    """
    Calculate uncertainty score (inverse of confidence).
    Accounts for variance in scoring.
    
    Args:
        confidence: Confidence score (0-1)
        variance: Optional variance score
    
    Returns:
        Uncertainty score (0-1)
    """
    base_uncertainty = 1.0 - confidence
    
    if variance is not None:
        # Higher variance increases uncertainty
        base_uncertainty = (base_uncertainty + variance) / 2
    
    return min(1.0, max(0.0, base_uncertainty))


def confidence_level_name(confidence: float) -> str:
    """Convert confidence score to readable level name."""
    if confidence >= 0.95:
        return "VERY_HIGH"
    elif confidence >= 0.85:
        return "HIGH"
    elif confidence >= 0.70:
        return "ACCEPTABLE"
    elif confidence >= 0.50:
        return "LOW"
    else:
        return "VERY_LOW"


def interpret_confidence(confidence: float) -> Tuple[str, str]:
    """
    Get human-readable interpretation of confidence score.
    
    Returns:
        (level_name, description)
    """
    levels = {
        "VERY_HIGH": "Very high confidence in this determination",
        "HIGH": "High confidence in this determination",
        "ACCEPTABLE": "Acceptable confidence, manual review recommended",
        "LOW": "Low confidence, manual review strongly recommended",
        "VERY_LOW": "Very low confidence, escalation required",
    }
    
    level = confidence_level_name(confidence)
    return level, levels.get(level, "Unknown confidence level")


def risk_level_name(risk_score: float) -> str:
    """Convert risk score to readable level name."""
    if risk_score >= 0.8:
        return "VERY_HIGH_RISK"
    elif risk_score >= 0.6:
        return "HIGH_RISK"
    elif risk_score >= 0.4:
        return "MEDIUM_RISK"
    elif risk_score >= 0.2:
        return "LOW_RISK"
    else:
        return "VERY_LOW_RISK"


def interpret_risk(risk_score: float) -> Tuple[str, str]:
    """
    Get human-readable interpretation of risk score.
    
    Args:
        risk_score: Risk score (0-1, where higher = more risky)
    
    Returns:
        (level_name, description)
    """
    levels = {
        "VERY_HIGH_RISK": "Very high risk, immediate escalation required",
        "HIGH_RISK": "High risk, human review required",
        "MEDIUM_RISK": "Medium risk, manual review recommended",
        "LOW_RISK": "Low risk, acceptable for processing",
        "VERY_LOW_RISK": "Very low risk, safe for auto-processing",
    }
    
    level = risk_level_name(risk_score)
    return level, levels.get(level, "Unknown risk level")
