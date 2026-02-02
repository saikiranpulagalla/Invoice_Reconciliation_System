"""
Output schemas for the reconciliation results.
Defines the strict JSON output format.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DiscrepancyDetail(BaseModel):
    """Details of a single discrepancy."""
    type: str  # price_mismatch, quantity_mismatch, missing_po, total_variance, extra_item
    severity: str  # low, medium, high
    confidence: float = Field(ge=0.0, le=1.0)
    invoice_value: Any
    po_value: Optional[Any] = None
    explanation: str


class MatchingDetail(BaseModel):
    """Details of the matching result."""
    matched_po: Optional[str] = None
    match_confidence: float = Field(ge=0.0, le=1.0)
    match_type: str = "not_matched"  # exact_reference, supplier_fuzzy, product_fuzzy, not_matched
    alternative_matches: List[Dict[str, Any]] = Field(default_factory=list)
    explanation: str


class ProcessingResults(BaseModel):
    """Processing results for the reconciliation."""
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    document_quality: str  # good, acceptable, poor
    extracted_data: Dict[str, Any]
    matching_results: MatchingDetail
    discrepancies: List[DiscrepancyDetail] = Field(default_factory=list)
    recommended_action: str  # auto_approve, flag_for_review, escalate_to_human
    action_reasoning: str
    agent_reasoning: str


class ReconciliationOutput(BaseModel):
    """Final reconciliation output in strict JSON schema."""
    invoice_id: str
    processing_timestamp: datetime
    processing_results: ProcessingResults
    
    class Config:
        json_schema_extra = {
            "example": {
                "invoice_id": "INV-2026-001",
                "processing_timestamp": "2026-01-30T10:30:00Z",
                "processing_results": {
                    "extraction_confidence": 0.92,
                    "document_quality": "good",
                    "extracted_data": {},
                    "matching_results": {},
                    "discrepancies": [],
                    "recommended_action": "auto_approve",
                    "action_reasoning": "...",
                    "agent_reasoning": "..."
                }
            }
        }
