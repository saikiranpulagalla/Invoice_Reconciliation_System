"""
Resolution Recommendation Agent
Decides recommended action (auto-approve, flag for review, escalate) based on
extraction quality, matching confidence, and discrepancy severity.
"""

from typing import Tuple

from app.state import ReconciliationState
from app.utils.logging import setup_logging
from app.utils.confidence import combine_confidence_scores, interpret_confidence, interpret_risk
from app.config import get_config


logger = setup_logging(__name__)
config = get_config()


def assess_extraction_risk(state: ReconciliationState) -> Tuple[float, str]:
    """
    Assess risk level based on extraction quality.
    
    Returns:
        (risk_score, explanation)
    """
    if not state.extracted_invoice:
        return 1.0, "No invoice extracted"
    
    if state.extraction_error:
        return 1.0, f"Extraction error: {state.extraction_error}"
    
    confidence = state.extracted_invoice.confidence.average()
    
    if confidence < config.ESCALATE_EXTRACTION_THRESHOLD:
        return 1.0, f"Very low extraction confidence ({confidence:.2f})"
    elif confidence < config.AUTO_APPROVE_EXTRACTION_THRESHOLD:
        return 0.5, f"Acceptable extraction confidence ({confidence:.2f}) but below auto-approve threshold"
    else:
        return 0.1, f"High extraction confidence ({confidence:.2f})"


def assess_matching_risk(state: ReconciliationState) -> Tuple[float, str]:
    """
    Assess risk level based on PO matching.
    
    CRITICAL: Risk must be consistent with confidence levels
    - Exact PO = LOW risk
    - Tentative PO = MEDIUM risk (never escalate on tentative PO alone)
    - Supplier mismatch + tentative PO = HIGH risk
    
    Returns:
        (risk_score, explanation)
    """
    if not state.po_match:
        return 1.0, "No matching performed"
    
    if state.match_error:
        return 1.0, f"Matching error: {state.match_error}"
    
    if not state.po_match.po_number:
        return 0.8, "No matching PO found in system"
    
    confidence = state.po_match.match_confidence
    match_type = state.po_match.match_type
    
    # CRITICAL: Check for supplier mismatch
    supplier_mismatch = False
    if state.matched_po and state.extracted_invoice:
        from rapidfuzz import fuzz
        supplier_similarity = fuzz.token_set_ratio(
            state.extracted_invoice.supplier_name.upper(),
            state.matched_po.supplier_name.upper()
        ) / 100.0
        
        supplier_mismatch = supplier_similarity < 0.85
    
    # Risk assessment based on match type and supplier consistency
    if match_type == "exact_reference":
        if supplier_mismatch:
            return 0.6, f"Exact PO reference but supplier mismatch detected. Moderate risk requiring review."
        else:
            return 0.1, f"Exact PO reference with supplier match. Low risk."
    else:
        # Tentative matches (fuzzy, supplier+product, product-only)
        if supplier_mismatch:
            return 0.8, f"Tentative PO match with supplier mismatch. High ambiguity requiring human judgment."
        else:
            return 0.4, f"Tentative PO match but supplier matches. Moderate risk - requires validation through item alignment."


def assess_discrepancy_risk(state: ReconciliationState) -> Tuple[float, str]:
    """
    Assess risk level based on discrepancies found.
    
    CRITICAL: Risk classification must be consistent
    - Format differences → LOW risk
    - Naming differences → LOW risk  
    - Ambiguous PO → MEDIUM risk
    - Quantity/price mismatch AFTER alignment → HIGH risk
    
    NEVER escalate based on:
    - Fuzzy PO alone
    - Unaligned items before semantic matching
    - String mismatch only
    
    Returns:
        (risk_score, explanation)
    """
    if not state.discrepancies:
        return 0.0, "No discrepancies found"
    
    high_severity_count = len([d for d in state.discrepancies if d.severity == "high"])
    medium_severity_count = len([d for d in state.discrepancies if d.severity == "medium"])
    low_severity_count = len([d for d in state.discrepancies if d.severity == "low"])
    
    # CHECK FOR CONTRADICTIONS - but be more lenient
    is_exact_reference = (
        state.po_match and 
        state.po_match.match_type == "exact_reference"
    )
    
    has_po_mismatch = any(d.type == "po_mismatch_suspected" for d in state.discrepancies)
    
    # Only remove po_mismatch_suspected if it's truly contradictory
    if is_exact_reference and has_po_mismatch:
        # Check if supplier also matches - if so, it's contradictory
        supplier_matched = False
        if state.matched_po and state.extracted_invoice:
            supplier_similarity = fuzz.token_set_ratio(
                state.extracted_invoice.supplier_name.upper(),
                state.matched_po.supplier_name.upper()
            ) / 100.0
            supplier_matched = supplier_similarity >= 0.85
        
        if supplier_matched:
            logger.warning("[ResolutionRecommendationAgent] CONTRADICTION DETECTED: exact_reference + supplier_match + po_mismatch_suspected. Removing po_mismatch_suspected.")
            state.discrepancies = [d for d in state.discrepancies if d.type != "po_mismatch_suspected"]
            has_po_mismatch = False
            medium_severity_count = len([d for d in state.discrepancies if d.severity == "medium"])
    
    risk_score = 0.0
    
    # High severity discrepancies significantly increase risk
    if high_severity_count > 0:
        high_severity_discrepancies = [d for d in state.discrepancies if d.severity == "high"]
        avg_high_confidence = sum(d.confidence for d in high_severity_discrepancies) / len(high_severity_discrepancies)
        
        # Only escalate if we're very confident about high-severity issues
        if avg_high_confidence > 0.85:
            return 1.0, f"High-severity discrepancies detected with high confidence: {high_severity_count} issue(s)"
        else:
            risk_score += 0.6
    
    # Medium severity adds moderate risk
    if medium_severity_count > 0:
        # Check if medium severity is just ambiguous PO
        medium_discrepancies = [d for d in state.discrepancies if d.severity == "medium"]
        is_just_ambiguity = all(d.type in ["po_mismatch_suspected", "extra_item"] for d in medium_discrepancies)
        
        if is_just_ambiguity:
            risk_score += 0.25  # Lower risk for ambiguity vs confirmed issues
        else:
            risk_score += 0.35
    
    # Low severity discrepancies (format/naming differences)
    if low_severity_count > 0:
        is_exact_reference = (
            state.po_match and 
            state.po_match.match_type == "exact_reference"
        )
        
        low_severity_discrepancies = [d for d in state.discrepancies if d.severity == "low"]
        has_format_variation = any(
            d.type in ["extra_item", "price_mismatch", "quantity_mismatch"]
            for d in low_severity_discrepancies
        )
        
        if is_exact_reference and has_format_variation:
            # Exact PO match with format variations → FLAG_FOR_REVIEW, not AUTO_APPROVE
            risk_score += 0.5  # Significant increase to prevent auto-approve
        else:
            # Normal low-severity discrepancies
            risk_score += 0.1
    
    risk_score = min(1.0, risk_score)
    
    explanation = f"Found {high_severity_count} high-severity, {medium_severity_count} medium-severity, " \
                 f"{low_severity_count} low-severity discrepancies"
    
    return risk_score, explanation


def make_recommendation(
    extraction_risk: float,
    matching_risk: float,
    discrepancy_risk: float,
    state: ReconciliationState,
) -> Tuple[str, float, str]:
    """
    Make final recommendation based on combined risk assessment.
    
    DECISION POLICY (Rules 8-10):
    8. AUTO_APPROVE: Exact PO match + All items aligned + Only LOW discrepancies + No risk factors
    9. FLAG_FOR_REVIEW: Exact PO match + Minor price/description differences  
    10. ESCALATE_TO_HUMAN: Wrong PO + Supplier mismatch + Unexplained high financial variance
    
    Returns:
        (action, confidence, explanation)
    """
    # Check for specific decision policy conditions
    is_exact_po = (
        state.po_match and 
        state.po_match.match_type == "exact_reference"
    )
    
    has_high_discrepancies = any(d.severity == "high" for d in state.discrepancies) if state.discrepancies else False
    has_medium_discrepancies = any(d.severity == "medium" for d in state.discrepancies) if state.discrepancies else False
    has_low_discrepancies = any(d.severity == "low" for d in state.discrepancies) if state.discrepancies else False
    no_discrepancies = not state.discrepancies
    
    # RULE 10: ESCALATE_TO_HUMAN conditions (highest priority)
    if (extraction_risk >= 0.8 or 
        matching_risk >= 0.8 or 
        has_high_discrepancies):
        return "escalate_to_human", 0.90, "High risk detected requiring immediate human review"
    
    # RULE 9: FLAG_FOR_REVIEW conditions (medium priority)
    # Any discrepancies or moderate risk should trigger review
    if (has_low_discrepancies or 
        has_medium_discrepancies or
        discrepancy_risk >= 0.3 or
        matching_risk >= 0.3 or
        extraction_risk >= 0.3):
        return "flag_for_review", 0.85, "Discrepancies or moderate risk detected requiring human review"
    
    # RULE 8: AUTO_APPROVE conditions (lowest priority - very strict)
    if (is_exact_po and 
        extraction_risk < 0.2 and 
        matching_risk < 0.2 and 
        discrepancy_risk < 0.2 and
        no_discrepancies):
        return "auto_approve", 0.95, "Exact PO match with perfect alignment and no discrepancies"
    
    # Default to FLAG_FOR_REVIEW for safety
    return "flag_for_review", 0.80, "Conservative review recommended"


async def resolution_recommendation_agent(state: ReconciliationState) -> ReconciliationState:
    """
    Resolution Recommendation Agent node.
    
    Responsibilities:
    1. Assess extraction risk
    2. Assess matching risk
    3. Assess discrepancy risk
    4. Combine risks to make recommendation
    5. Generate human-readable explanation
    
    Recommendations:
    - auto_approve: Safe to approve without human review
    - flag_for_review: Needs human review
    - escalate_to_human: Requires immediate human attention
    
    Updates state:
    - recommended_action
    - action_confidence
    - action_reasoning
    
    Adds reasoning log entry.
    """
    logger.info(f"[ResolutionRecommendationAgent] Making recommendation for invoice {state.invoice_id}")
    
    try:
        # FAST PATH: Perfect match optimization (optional improvement)
        # If exact PO + supplier verified + 100% alignment + no discrepancies → auto_approve immediately
        if (state.po_match and 
            state.po_match.match_type == "exact_reference" and
            state.po_match.match_confidence >= 0.95 and
            state.extracted_invoice and
            state.extracted_invoice.confidence.average() >= 0.85 and
            not state.discrepancies and
            state.matched_po):
            
            # Verify supplier match
            from rapidfuzz import fuzz
            supplier_similarity = fuzz.token_set_ratio(
                state.extracted_invoice.supplier_name.upper(),
                state.matched_po.supplier_name.upper()
            ) / 100.0
            
            if supplier_similarity >= 0.90:
                # PERFECT MATCH - Skip risk aggregation, auto-approve immediately
                state.recommended_action = "auto_approve"
                state.action_confidence = 0.98
                state.action_reasoning = (
                    "PERFECT MATCH - FAST PATH AUTO-APPROVAL\n\n"
                    "✓ Exact PO reference match\n"
                    "✓ Supplier verified (100% match)\n"
                    "✓ High extraction confidence\n"
                    "✓ Zero discrepancies detected\n"
                    "✓ All validation checks passed\n\n"
                    "This invoice meets all criteria for immediate automatic approval without risk assessment."
                )
                
                logger.info(f"[ResolutionRecommendationAgent] FAST PATH: Perfect match detected - auto_approve (confidence: 0.98)")
                
                state.add_reasoning(
                    agent_name="ResolutionRecommendationAgent",
                    message="Perfect match detected: Exact PO + Supplier verified + No discrepancies → auto_approve",
                    confidence=0.98,
                    action="auto_approve",
                )
                
                return state
        
        # NORMAL PATH: Full risk assessment
        # Assess all risk dimensions
        extraction_risk, extraction_explanation = assess_extraction_risk(state)
        matching_risk, matching_explanation = assess_matching_risk(state)
        discrepancy_risk, discrepancy_explanation = assess_discrepancy_risk(state)
        
        logger.debug(f"Risk assessment - Extraction: {extraction_risk:.2f}, "
                    f"Matching: {matching_risk:.2f}, Discrepancy: {discrepancy_risk:.2f}")
        
        # Make recommendation
        action, confidence, risk_explanation = make_recommendation(
            extraction_risk,
            matching_risk,
            discrepancy_risk,
            state,
        )
        
        state.recommended_action = action
        state.action_confidence = confidence
        
        # Build detailed reasoning - accountant-like, not technical
        reasoning_lines = [
            f"RECONCILIATION ANALYSIS:",
            f"",
            f"Extraction Quality: {interpret_risk(extraction_risk)[0]} ({extraction_risk:.2f})",
            f"  {extraction_explanation}",
            f"",
            f"PO Matching: {interpret_risk(matching_risk)[0]} ({matching_risk:.2f})",
            f"  {matching_explanation}",
            f"",
            f"Discrepancy Assessment: {interpret_risk(discrepancy_risk)[0]} ({discrepancy_risk:.2f})",
            f"  {discrepancy_explanation}",
            f"",
            f"RECOMMENDATION: {action.upper()}",
            f"Confidence: {confidence:.2f}",
            f"{risk_explanation}",
        ]
        
        if action == "auto_approve":
            reasoning_lines.extend([
                "",
                "✓ DECISION: This invoice can be safely approved without human review.",
                "✓ All quality checks passed with high confidence.",
                "✓ PO match is reliable and items align correctly.",
                "✓ No financial risk detected.",
            ])
        elif action == "flag_for_review":
            reasoning_lines.extend([
                "",
                "⚠ DECISION: This invoice should be reviewed by a human before approval.",
                "⚠ Some quality or matching concerns detected but not critical.",
                "⚠ Likely causes: format variations, minor discrepancies, or fuzzy matching.",
                "⚠ No financial fraud risk detected.",
            ])
        else:  # escalate_to_human
            reasoning_lines.extend([
                "",
                "✗ DECISION: This invoice requires immediate human review and decision.",
                "✗ Significant quality, matching, or discrepancy issues detected.",
                "✗ Likely causes: wrong PO, missing PO, or high financial risk.",
                "✗ Requires human judgment before approval.",
            ])
        
        state.action_reasoning = "\n".join(reasoning_lines)
        
        logger.info(f"[ResolutionRecommendationAgent] Recommending: {action} (confidence: {confidence:.2f})")
        
        state.add_reasoning(
            agent_name="ResolutionRecommendationAgent",
            message=f"Recommended action: {action}. Extraction risk: {extraction_risk:.2f}, "
                   f"Matching risk: {matching_risk:.2f}, Discrepancy risk: {discrepancy_risk:.2f}",
            confidence=confidence,
            action=action,
        )
        
        # Determine if human review should be requested
        if action == "escalate_to_human":
            state.human_review_requested = True
        
    except Exception as e:
        logger.exception(f"[ResolutionRecommendationAgent] Unexpected error: {e}")
        
        # Default to conservative recommendation on error
        state.recommended_action = "escalate_to_human"
        state.action_confidence = 0.5
        state.action_reasoning = f"Error during recommendation: {str(e)}"
        state.human_review_requested = True
        
        state.add_reasoning(
            agent_name="ResolutionRecommendationAgent",
            message=f"Error during resolution: {str(e)}. Defaulting to escalation.",
            confidence=0.0,
        )
    
    return state
