"""
Human Reviewer Agent (Optional)
Placeholder for human review feedback integration.
"""

from app.state import ReconciliationState
from app.utils.logging import setup_logging


logger = setup_logging(__name__)


async def human_reviewer_agent(state: ReconciliationState) -> ReconciliationState:
    """
    Human Reviewer Agent node.
    
    This is a placeholder for integrating human review into the workflow.
    In a real system, this would:
    1. Queue the invoice for human review
    2. Wait for feedback
    3. Update state with human decision
    
    For now, it just logs that human review was requested.
    
    Updates state:
    - human_feedback (if available)
    - human_decision (if available)
    
    Adds reasoning log entry.
    """
    logger.info(f"[HumanReviewerAgent] Human review requested for invoice {state.invoice_id}")
    
    if not state.human_review_requested:
        logger.debug("[HumanReviewerAgent] Human review not needed, passing through")
        return state
    
    logger.warning(f"[HumanReviewerAgent] Invoice {state.invoice_id} queued for human review")
    logger.warning(f"  Recommended action: {state.recommended_action}")
    logger.warning(f"  Action confidence: {state.action_confidence:.2f}")
    
    state.add_reasoning(
        agent_name="HumanReviewerAgent",
        message="Invoice queued for human review. Awaiting human decision.",
        confidence=0.0,
    )
    
    return state
