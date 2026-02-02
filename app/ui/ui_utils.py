"""
UI Utility Functions

Helper functions for Streamlit visualization.
These are purely for presentation - no business logic.
"""

from typing import Dict, List, Any
from enum import Enum


class SeverityLevel(str, Enum):
    """Discrepancy severity levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionType(str, Enum):
    """Final recommendation action types."""
    AUTO_APPROVE = "auto_approve"
    FLAG_FOR_REVIEW = "flag_for_review"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class ConfidenceLevel(Enum):
    """Confidence level classification."""
    VERY_HIGH = (0.95, "Very High", "🟢")
    HIGH = (0.85, "High", "🟢")
    ACCEPTABLE = (0.70, "Acceptable", "🟡")
    LOW = (0.50, "Low", "🟠")
    VERY_LOW = (0.0, "Very Low", "🔴")
    
    def __init__(self, threshold: float, label: str, emoji: str):
        self.threshold = threshold
        self.label = label
        self.emoji = emoji


def get_confidence_level(score: float) -> ConfidenceLevel:
    """
    Classify a confidence score into a level.
    
    Args:
        score: Confidence score between 0 and 1
    
    Returns:
        ConfidenceLevel enum
    """
    if score >= ConfidenceLevel.VERY_HIGH.threshold:
        return ConfidenceLevel.VERY_HIGH
    elif score >= ConfidenceLevel.HIGH.threshold:
        return ConfidenceLevel.HIGH
    elif score >= ConfidenceLevel.ACCEPTABLE.threshold:
        return ConfidenceLevel.ACCEPTABLE
    elif score >= ConfidenceLevel.LOW.threshold:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.VERY_LOW


def get_action_emoji(action: str) -> str:
    """Get emoji for action type."""
    emoji_map = {
        ActionType.AUTO_APPROVE.value: "🟢",
        ActionType.FLAG_FOR_REVIEW.value: "🟡",
        ActionType.ESCALATE_TO_HUMAN.value: "🔴"
    }
    return emoji_map.get(action, "⚪")


def get_severity_emoji(severity: str) -> str:
    """Get emoji for discrepancy severity."""
    emoji_map = {
        SeverityLevel.HIGH.value: "🔴",
        SeverityLevel.MEDIUM.value: "🟡",
        SeverityLevel.LOW.value: "🟢"
    }
    return emoji_map.get(severity, "⚪")


def format_confidence_display(score: float) -> str:
    """Format confidence score for display."""
    level = get_confidence_level(score)
    return f"{level.emoji} {level.label} ({score:.0%})"


def format_agent_reasoning_lines(reasoning_text: str, agent_name: str) -> List[str]:
    """
    Extract reasoning lines for a specific agent.
    
    Args:
        reasoning_text: Full reasoning text
        agent_name: Name of agent (e.g., "DocumentIntelligenceAgent")
    
    Returns:
        List of reasoning lines for this agent
    """
    if not reasoning_text:
        return []
    
    lines = reasoning_text.split("\n")
    agent_lines = [
        line.strip() 
        for line in lines 
        if agent_name in line and line.strip()
    ]
    return agent_lines


def format_discrepancy_for_display(discrepancy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format discrepancy data for cleaner display.
    
    Args:
        discrepancy: Raw discrepancy dict
    
    Returns:
        Formatted discrepancy dict
    """
    return {
        "type": discrepancy.get("type", "unknown").replace("_", " ").title(),
        "severity": discrepancy.get("severity", "unknown").upper(),
        "confidence": f"{discrepancy.get('confidence', 0):.0%}",
        "explanation": discrepancy.get("explanation", "No explanation"),
        "invoice_value": discrepancy.get("invoice_value"),
        "po_value": discrepancy.get("po_value")
    }


def format_matching_result_for_display(matching: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format matching result for cleaner display.
    
    Args:
        matching: Raw matching result
    
    Returns:
        Formatted matching result
    """
    return {
        "matched_po": matching.get("matched_po") or "Not Matched",
        "confidence": f"{matching.get('match_confidence', 0):.0%}",
        "match_type": matching.get("match_type", "unknown").replace("_", " ").title(),
        "explanation": matching.get("explanation", "No explanation")
    }
