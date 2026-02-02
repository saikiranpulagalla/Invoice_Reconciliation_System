"""
LangGraph orchestration for the multi-agent reconciliation workflow.
Defines the graph structure and node routing logic.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from app.state import ReconciliationState
from app.agents.document_intelligence import document_intelligence_agent
from app.agents.matching import matching_agent
from app.agents.discrepancy import discrepancy_detection_agent
from app.agents.resolution import resolution_recommendation_agent
from app.agents.human_reviewer import human_reviewer_agent
from app.config import get_config


config = get_config()


def route_after_extraction(state: ReconciliationState) -> Literal["matching_agent", "end"]:
    """Route after document extraction."""
    if state.extraction_error or not state.extracted_invoice:
        # Extraction failed - may escalate, but continue for now
        return "matching_agent"
    return "matching_agent"


def route_after_matching(state: ReconciliationState) -> Literal["discrepancy_agent", "end"]:
    """Route after matching."""
    # Continue to discrepancy detection even if no match found
    return "discrepancy_agent"


def route_after_discrepancy(state: ReconciliationState) -> Literal["resolution_agent", "end"]:
    """Route after discrepancy detection."""
    return "resolution_agent"


def route_after_resolution(state: ReconciliationState) -> Literal["human_reviewer_agent", "end"]:
    """Route after resolution recommendation."""
    if config.ENABLE_HUMAN_REVIEW and state.human_review_requested:
        return "human_reviewer_agent"
    return "end"


def build_reconciliation_graph() -> StateGraph:
    """
    Build the LangGraph workflow for invoice reconciliation.
    
    Flow:
    1. Document Intelligence Agent - Extract invoice data
    2. Matching Agent - Find matching PO
    3. Discrepancy Detection Agent - Identify differences
    4. Resolution Recommendation Agent - Decide action
    5. (Optional) Human Reviewer Agent - Manual review
    """
    
    graph = StateGraph(ReconciliationState)
    
    # Add agent nodes
    graph.add_node("extraction_agent", document_intelligence_agent)
    graph.add_node("matching_agent", matching_agent)
    graph.add_node("discrepancy_agent", discrepancy_detection_agent)
    graph.add_node("resolution_agent", resolution_recommendation_agent)
    
    if config.ENABLE_HUMAN_REVIEW:
        graph.add_node("human_reviewer_agent", human_reviewer_agent)
    
    # Set the entry point
    graph.set_entry_point("extraction_agent")
    
    # Add edges with routing logic
    graph.add_conditional_edges(
        "extraction_agent",
        route_after_extraction,
        {
            "matching_agent": "matching_agent",
            "end": END,
        }
    )
    
    graph.add_conditional_edges(
        "matching_agent",
        route_after_matching,
        {
            "discrepancy_agent": "discrepancy_agent",
            "end": END,
        }
    )
    
    graph.add_conditional_edges(
        "discrepancy_agent",
        route_after_discrepancy,
        {
            "resolution_agent": "resolution_agent",
            "end": END,
        }
    )
    
    if config.ENABLE_HUMAN_REVIEW:
        graph.add_conditional_edges(
            "resolution_agent",
            route_after_resolution,
            {
                "human_reviewer_agent": "human_reviewer_agent",
                "end": END,
            }
        )
        graph.add_edge("human_reviewer_agent", END)
    else:
        graph.add_edge("resolution_agent", END)
    
    return graph.compile()


# Global compiled graph (singleton)
_reconciliation_graph = None


def get_reconciliation_graph():
    """Get or create the compiled reconciliation graph."""
    global _reconciliation_graph
    if _reconciliation_graph is None:
        _reconciliation_graph = build_reconciliation_graph()
    return _reconciliation_graph
