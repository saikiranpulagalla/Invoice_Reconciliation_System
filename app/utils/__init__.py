"""
Shared utilities and helpers.
"""

import json
from typing import Any, Dict
from datetime import datetime


def serialize_for_json(obj: Any) -> Any:
    """Serialize objects that aren't JSON-serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'dict'):  # Pydantic model
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


def dict_to_json_string(data: Dict) -> str:
    """Convert dict to JSON string, handling non-serializable types."""
    return json.dumps(data, default=serialize_for_json, indent=2)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value."""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_percentage_variance(actual: float, expected: float) -> float:
    """Calculate percentage variance between actual and expected."""
    if expected == 0:
        return 0.0
    return abs(actual - expected) / abs(expected)
