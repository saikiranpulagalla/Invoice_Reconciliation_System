"""
Structured logging for the reconciliation system.
"""

import logging
import json
from datetime import datetime
from typing import Any, Optional
from app.config import get_config


config = get_config()


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        
        return json.dumps(log_obj)


def setup_logging(name: str = __name__) -> logging.Logger:
    """Setup and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_formatter = logging.Formatter(config.LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with structured JSON
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)
    
    return logger


def log_agent_action(
    logger: logging.Logger,
    agent_name: str,
    action: str,
    details: Optional[dict] = None,
    confidence: Optional[float] = None,
) -> None:
    """Log an agent action with context."""
    extra = {
        "agent": agent_name,
        "action": action,
    }
    if confidence is not None:
        extra["confidence"] = confidence
    if details:
        extra.update(details)
    
    logger.info(
        f"[{agent_name}] {action}",
        extra={"extra": extra}
    )


def log_discrepancy(
    logger: logging.Logger,
    discrepancy_type: str,
    severity: str,
    explanation: str,
    confidence: float,
) -> None:
    """Log a detected discrepancy."""
    extra = {
        "type": "discrepancy",
        "discrepancy_type": discrepancy_type,
        "severity": severity,
        "explanation": explanation,
        "confidence": confidence,
    }
    logger.warning(
        f"Discrepancy detected: {discrepancy_type} ({severity})",
        extra={"extra": extra}
    )
