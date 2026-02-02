"""
Shared state object for the multi-agent system.
All agents read/write from this state to coordinate their work.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.schemas.invoice import ExtractedInvoice
from app.schemas.po import PurchaseOrder, POMatch
from app.schemas.output import DiscrepancyDetail, MatchingDetail


class ReasoningLogEntry(BaseModel):
    """A single entry in the agent reasoning log."""
    timestamp: datetime
    agent_name: str
    message: str
    confidence: Optional[float] = None
    action: Optional[str] = None


class ReconciliationState(BaseModel):
    """
    Shared mutable state object for the reconciliation workflow.
    
    This state is passed between agents. Each agent:
    1. Reads relevant state
    2. Performs its task
    3. Updates state with results
    4. Adds reasoning log entry
    5. Passes state to next agent
    """
    
    # Workflow identification
    invoice_id: str
    processing_timestamp: datetime
    document_path: str  # path to the invoice document
    
    # Extraction phase
    extracted_invoice: Optional[ExtractedInvoice] = None
    extraction_error: Optional[str] = None
    
    # Matching phase
    available_pos: List[PurchaseOrder] = Field(default_factory=list)
    po_match: Optional[POMatch] = None
    matched_po: Optional[PurchaseOrder] = None
    match_error: Optional[str] = None
    
    # Discrepancy detection phase
    discrepancies: List[DiscrepancyDetail] = Field(default_factory=list)
    discrepancy_error: Optional[str] = None
    
    # Resolution phase
    recommended_action: str = "flag_for_review"  # auto_approve, flag_for_review, escalate_to_human
    action_confidence: float = 0.5
    action_reasoning: str = ""
    
    # Human review phase (optional)
    human_review_requested: bool = False
    human_feedback: Optional[str] = None
    human_decision: Optional[str] = None
    
    # Reasoning and audit trail
    reasoning_log: List[ReasoningLogEntry] = Field(default_factory=list)
    
    def add_reasoning(
        self,
        agent_name: str,
        message: str,
        confidence: Optional[float] = None,
        action: Optional[str] = None
    ) -> None:
        """Add an entry to the reasoning log."""
        self.reasoning_log.append(
            ReasoningLogEntry(
                timestamp=datetime.utcnow(),
                agent_name=agent_name,
                message=message,
                confidence=confidence,
                action=action,
            )
        )
    
    def get_agent_reasoning(self) -> str:
        """Get a human-readable summary of the agent reasoning."""
        if not self.reasoning_log:
            return "No reasoning available."
        
        lines = []
        for entry in self.reasoning_log:
            conf_str = f" (confidence: {entry.confidence:.2f})" if entry.confidence else ""
            lines.append(f"[{entry.agent_name}] {entry.message}{conf_str}")
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "invoice_id": self.invoice_id,
            "extraction_status": "completed" if self.extracted_invoice else "pending",
            "matching_status": "completed" if self.po_match else "pending",
            "discrepancies_found": len(self.discrepancies),
            "recommended_action": self.recommended_action,
            "total_agents_participated": len(set(log.agent_name for log in self.reasoning_log)),
        }
