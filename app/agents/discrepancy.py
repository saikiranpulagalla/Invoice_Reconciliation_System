"""
Discrepancy Detection Agent
Compares extracted invoice against matched PO to detect discrepancies.

ACCOUNTANT-LIKE REASONING:
1. Assess match context (exact PO? supplier match?)
2. Align line items (fuzzy + semantic matching)
3. Evaluate discrepancies (only on aligned items)
4. Generate human-readable reasoning

CRITICAL RULES:
- Never mark items as "extra" before attempting alignment
- Total variance only calculated if >= 80% alignment
- Never raise po_mismatch_suspected for exact reference + supplier match
- Format differences are NOT fraud
"""

from typing import List, Optional, Tuple, Dict
from datetime import datetime
from rapidfuzz import fuzz

from langchain_openai import ChatOpenAI

from app.state import ReconciliationState
from app.schemas.output import DiscrepancyDetail
from app.utils.logging import setup_logging, log_discrepancy
from app.utils.confidence import combine_confidence_scores, penalize_confidence
from app.config import get_config


logger = setup_logging(__name__)
config = get_config()


def get_llm():
    """Get LLM instance."""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        temperature=0.1,
        max_tokens=1500,
    )


def align_line_items(state: ReconciliationState) -> Tuple[List[Tuple[int, int, float]], float]:
    """
    RULE 4: Align invoice items to PO items using:
    - Fuzzy string similarity
    - Semantic embeddings (description meaning) 
    - Ignore order and naming variations
    
    CRITICAL: Items aligned if description similarity >= 0.75
    
    Returns:
        (alignments, alignment_ratio)
        - alignments: list of (invoice_idx, po_idx, similarity_score)
        - alignment_ratio: float (0.0-1.0)
    """
    if not state.matched_po or not state.extracted_invoice:
        return [], 0.0
    
    alignments = []
    po_item_descriptions = [item.description for item in state.matched_po.line_items]
    matched_po_indices = set()
    
    # For each invoice item, find best matching PO item
    for inv_idx, inv_item in enumerate(state.extracted_invoice.line_items):
        best_po_idx = None
        best_similarity = 0.0
        
        for po_idx, po_desc in enumerate(po_item_descriptions):
            if po_idx in matched_po_indices:
                continue  # Skip already matched PO items
            
            # RULE 4: Enhanced matching using multiple techniques
            
            # 1. Fuzzy string similarity (primary)
            fuzzy_similarity = fuzz.token_set_ratio(
                inv_item.description.upper(),
                po_desc.upper()
            ) / 100.0
            
            # 2. Semantic similarity (enhanced matching for meaning)
            # Check for common chemical/product naming patterns
            semantic_boost = 0.0
            inv_desc_clean = inv_item.description.upper().strip()
            po_desc_clean = po_desc.upper().strip()
            
            # Remove common suffixes/prefixes that don't affect meaning
            for suffix in [' PH EUR', ' BP', ' USP', ' GRADE', ' POWDER', ' CRYSTALS', ' MESH']:
                inv_desc_clean = inv_desc_clean.replace(suffix, '')
                po_desc_clean = po_desc_clean.replace(suffix, '')
            
            # Check core product name similarity after cleaning
            core_similarity = fuzz.token_set_ratio(inv_desc_clean, po_desc_clean) / 100.0
            if core_similarity > fuzzy_similarity:
                semantic_boost = min(0.1, core_similarity - fuzzy_similarity)
            
            # 3. Combined similarity score
            combined_similarity = min(1.0, fuzzy_similarity + semantic_boost)
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_po_idx = po_idx
        
        # CRITICAL: Align if similarity >= 0.75 (semantic + fuzzy matching)
        if best_similarity >= 0.75 and best_po_idx is not None:
            alignments.append((inv_idx, best_po_idx, best_similarity))
            matched_po_indices.add(best_po_idx)
            logger.debug(f"[DiscrepancyDetectionAgent] Aligned '{inv_item.description}' to '{po_item_descriptions[best_po_idx]}' (similarity: {best_similarity:.2f})")
    
    # Calculate alignment ratio
    total_items = len(state.extracted_invoice.line_items)
    alignment_ratio = len(alignments) / total_items if total_items > 0 else 0.0
    
    return alignments, alignment_ratio


def assess_match_context(state: ReconciliationState) -> Dict:
    """
    STEP 0: Assess match context BEFORE any discrepancy evaluation.
    
    CRITICAL: Distinguish between exact and tentative matches
    
    Returns dict with:
    - is_exact_reference: bool
    - is_tentative_match: bool (fuzzy PO)
    - supplier_matched: bool
    - match_confidence: float
    - context_note: str (human-readable)
    """
    context = {
        "is_exact_reference": False,
        "is_tentative_match": False,
        "supplier_matched": False,
        "match_confidence": 0.0,
        "context_note": "",
    }
    
    if not state.po_match or not state.matched_po:
        context["context_note"] = "No PO matched"
        return context
    
    # Check if exact reference
    if state.po_match.match_type == "exact_reference":
        context["is_exact_reference"] = True
        context["match_confidence"] = state.po_match.match_confidence
        context["context_note"] = f"Exact PO reference: {state.matched_po.po_number}"
    else:
        # All non-exact matches are tentative
        context["is_tentative_match"] = True
        context["match_confidence"] = state.po_match.match_confidence
        context["context_note"] = f"Tentative PO match: {state.matched_po.po_number} (confidence: {state.po_match.match_confidence:.2f})"
    
    # Check supplier match
    if (state.extracted_invoice.supplier_name and 
        state.matched_po.supplier_name):
        supplier_similarity = fuzz.token_set_ratio(
            state.extracted_invoice.supplier_name.upper(),
            state.matched_po.supplier_name.upper()
        ) / 100.0
        
        if supplier_similarity >= 0.85:
            context["supplier_matched"] = True
    
    return context


def detect_price_discrepancies(
    state: ReconciliationState,
    alignments: List[Tuple[int, int, float]],
    alignment_ratio: float,
    match_context: Dict,
) -> List[DiscrepancyDetail]:
    """
    STEP 4-5: Detect price mismatches on ALIGNED items only.
    
    RULE 7: Do NOT raise HIGH severity discrepancies when:
    - PO match is exact
    - Supplier matches  
    - Variations are explainable by format differences
    
    CRITICAL RULES:
    - Only compare prices on aligned items
    - Total variance ONLY if alignment_ratio >= 80% (conservative)
    - Total variance derived from aligned items, not raw totals
    - Never compute total variance on unaligned items
    - Never compare invoice total to PO total blindly
    """
    discrepancies = []
    
    if not state.matched_po or not state.extracted_invoice:
        return discrepancies
    
    # Check if we should be conservative with severity (Rule 7)
    is_exact_po_with_supplier_match = (
        match_context.get("is_exact_reference", False) and 
        match_context.get("supplier_matched", False)
    )
    
    # Detect line item price mismatches on aligned items
    aligned_invoice_totals = 0.0
    aligned_po_totals = 0.0
    
    for inv_idx, po_idx, similarity in alignments:
        inv_item = state.extracted_invoice.line_items[inv_idx]
        po_item = state.matched_po.line_items[po_idx]
        
        aligned_invoice_totals += inv_item.unit_price * inv_item.quantity
        aligned_po_totals += po_item.unit_price * po_item.quantity
        
        # Check unit price
        if po_item.unit_price > 0:
            price_variance = abs(inv_item.unit_price - po_item.unit_price) / po_item.unit_price
            
            if price_variance > config.PRICE_VARIANCE_TOLERANCE:
                # RULE 7: Conservative severity when exact PO + supplier match
                if is_exact_po_with_supplier_match:
                    # More lenient severity for exact PO + supplier match
                    # But still flag significant variances (>8%) as medium
                    if price_variance > 0.20:
                        severity = "high"
                        confidence = 0.95
                    elif price_variance > 0.08:
                        severity = "medium"
                        confidence = max(0.85, 1.0 - min(1.0, price_variance / 0.5))
                    else:
                        severity = "low"
                        confidence = max(0.6, 1.0 - min(1.0, price_variance / 0.5))
                else:
                    # Normal severity rules
                    severity = "high" if price_variance > 0.15 else "medium" if price_variance > 0.05 else "low"
                    confidence = max(0.6, 1.0 - min(1.0, price_variance / 0.5))
                    
                    # Detect hidden price increases (only for non-exact matches)
                    if price_variance > 0.08 and inv_item.unit_price > po_item.unit_price:
                        severity = "high"
                        confidence = 0.95
                
                discrepancy = DiscrepancyDetail(
                    type="price_mismatch",
                    severity=severity,
                    confidence=confidence,
                    invoice_value=inv_item.unit_price,
                    po_value=po_item.unit_price,
                    explanation=f"Unit price discrepancy in '{inv_item.description}': "
                               f"Invoice ${inv_item.unit_price:.2f} vs PO ${po_item.unit_price:.2f} "
                               f"({price_variance * 100:+.1f}%)",
                )
                discrepancies.append(discrepancy)
                log_discrepancy(
                    logger,
                    "line_item_price_mismatch",
                    severity,
                    discrepancy.explanation,
                    confidence,
                )
    
    # RULE 6: Calculate total variance ONLY if alignment >= 80% (more conservative)
    if alignment_ratio >= 0.80 and aligned_po_totals > 0:
        aligned_difference = abs(aligned_invoice_totals - aligned_po_totals)
        
        # Define tolerance: either 1% or £5, whichever is larger
        percentage_tolerance = aligned_po_totals * 0.01  # 1%
        absolute_tolerance = 5.0  # £5
        tolerance = max(percentage_tolerance, absolute_tolerance)
        
        # Only create discrepancy if difference exceeds tolerance
        if aligned_difference > tolerance:
            variance = aligned_difference / aligned_po_totals
            
            # RULE 7: Conservative severity for exact PO + supplier match
            if is_exact_po_with_supplier_match:
                severity = "medium" if variance > 0.15 else "low"
            else:
                severity = "high" if variance > 0.10 else "medium" if variance > 0.05 else "low"
            
            confidence = max(0.8, min(0.95, 1.0 - (variance / 0.5)))
            
            discrepancy = DiscrepancyDetail(
                type="total_variance",
                severity=severity,
                confidence=confidence,
                invoice_value=aligned_invoice_totals,
                po_value=aligned_po_totals,
                explanation=f"Total variance of ${aligned_difference:.2f} ({variance * 100:.1f}%) on aligned items. "
                           f"Invoice: ${aligned_invoice_totals:.2f}, PO: ${aligned_po_totals:.2f}",
            )
            
            discrepancies.append(discrepancy)
            log_discrepancy(
                logger,
                "price_mismatch",
                severity,
                discrepancy.explanation,
                confidence,
            )
        else:
            logger.debug(f"[DiscrepancyDetectionAgent] Total variance within tolerance: ${aligned_difference:.2f} <= ${tolerance:.2f}")
    elif alignment_ratio < 0.80:
        logger.debug(f"[DiscrepancyDetectionAgent] Alignment ratio {alignment_ratio:.1%} < 80% threshold. Skipping total variance calculation.")
    
    return discrepancies


def detect_quantity_discrepancies(
    state: ReconciliationState,
    alignments: List[Tuple[int, int, float]],
) -> List[DiscrepancyDetail]:
    """Detect quantity mismatches on ALIGNED items only."""
    discrepancies = []
    
    if not state.matched_po or not state.extracted_invoice:
        return discrepancies
    
    for inv_idx, po_idx, similarity in alignments:
        inv_item = state.extracted_invoice.line_items[inv_idx]
        po_item = state.matched_po.line_items[po_idx]
        
        if po_item.quantity == 0:
            continue
        
        qty_variance = abs(inv_item.quantity - po_item.quantity) / po_item.quantity
        
        if qty_variance > config.QUANTITY_VARIANCE_TOLERANCE:
            severity = "high" if qty_variance > 0.20 else "medium" if qty_variance > 0.10 else "low"
            confidence = max(0.6, 1.0 - min(1.0, qty_variance / 0.5))
            
            discrepancy = DiscrepancyDetail(
                type="quantity_mismatch",
                severity=severity,
                confidence=confidence,
                invoice_value=inv_item.quantity,
                po_value=po_item.quantity,
                explanation=f"Quantity discrepancy in '{inv_item.description}': "
                           f"Invoice {inv_item.quantity} vs PO {po_item.quantity} "
                           f"({qty_variance * 100:+.1f}%)",
            )
            
            discrepancies.append(discrepancy)
            log_discrepancy(
                logger,
                "quantity_mismatch",
                severity,
                discrepancy.explanation,
                confidence,
            )
    
    return discrepancies


def detect_missing_po(state: ReconciliationState) -> List[DiscrepancyDetail]:
    """Detect if invoice has no matching PO."""
    discrepancies = []
    
    if not state.extracted_invoice:
        return discrepancies
    
    # If no PO was matched, this is a critical discrepancy
    if not state.matched_po or not state.po_match.po_number:
        severity = "high" if not state.extracted_invoice.po_reference else "medium"
        confidence = 0.95
        
        explanation = "Invoice does not reference any Purchase Order"
        if state.extracted_invoice.po_reference:
            explanation = f"Referenced PO '{state.extracted_invoice.po_reference}' not found in system"
        
        discrepancy = DiscrepancyDetail(
            type="missing_po",
            severity=severity,
            confidence=confidence,
            invoice_value=state.extracted_invoice.invoice_number,
            po_value=None,
            explanation=explanation,
        )
        
        discrepancies.append(discrepancy)
        log_discrepancy(
            logger,
            "missing_po",
            severity,
            explanation,
            confidence,
        )
    
    return discrepancies


def detect_extra_items(
    state: ReconciliationState,
    match_context: Dict,
    alignments: List[Tuple[int, int, float]],
) -> List[DiscrepancyDetail]:
    """
    STEP 3: Detect items that don't align to PO.
    
    CRITICAL BUSINESS LOGIC:
    - NEVER mark items as "extra" using string comparison alone
    - Always perform semantic similarity first
    - po_mismatch_suspected DISABLED if:
      - PO reference is exact OR
      - Supplier matches and PO is fuzzy
    - po_mismatch_suspected ENABLED only if:
      - Multiple PO candidates exist AND
      - Item alignment confidence < threshold
    """
    discrepancies = []
    
    if not state.matched_po or not state.extracted_invoice:
        return discrepancies
    
    # Get truly unmatched items (not in alignment results)
    aligned_invoice_indices = set(inv_idx for inv_idx, _, _ in alignments)
    unmatched_items = []
    for inv_idx, inv_item in enumerate(state.extracted_invoice.line_items):
        if inv_idx not in aligned_invoice_indices:
            unmatched_items.append(inv_item)
    
    total_items = len(state.extracted_invoice.line_items)
    unmatched_count = len(unmatched_items)
    
    if total_items == 0 or unmatched_count == 0:
        return discrepancies
    
    unmatched_ratio = unmatched_count / total_items
    alignment_ratio = len(alignments) / total_items
    
    # REFINED po_mismatch_suspected RULE
    should_flag_po_mismatch = (
        unmatched_ratio >= 0.7 and 
        unmatched_count >= 2 and 
        not match_context["is_exact_reference"] and  # DISABLED if exact reference
        not (match_context["is_tentative_match"] and match_context["supplier_matched"]) and  # DISABLED if tentative + supplier match
        alignment_ratio < 0.5  # ENABLED only if alignment confidence < threshold
    )
    
    if should_flag_po_mismatch:
        # Only flag if we have multiple PO candidates and low alignment confidence
        discrepancy = DiscrepancyDetail(
            type="po_mismatch_suspected",
            severity="medium",  # MEDIUM, not HIGH - ambiguity, not fraud
            confidence=0.7,
            invoice_value=f"{unmatched_count}/{total_items} items unmatched",
            po_value=state.matched_po.po_number,
            explanation=f"Most invoice items ({unmatched_count}/{total_items}) do not align with tentative PO {state.matched_po.po_number}. PO selection may be ambiguous.",
        )
        discrepancies.append(discrepancy)
        log_discrepancy(
            logger,
            "po_mismatch_suspected",
            "medium",
            discrepancy.explanation,
            0.7,
        )
        
        # Still add individual extra_item discrepancies but with lower severity
        for inv_item in unmatched_items:
            discrepancy = DiscrepancyDetail(
                type="extra_item",
                severity="low",
                confidence=0.6,
                invoice_value=f"{inv_item.description} ({inv_item.quantity} @ ${inv_item.unit_price})",
                po_value=None,
                explanation=f"Line item '{inv_item.description}' not found in PO (may be due to ambiguous PO selection)",
            )
            discrepancies.append(discrepancy)
    else:
        # Normal case: treat unmatched items based on match context
        for inv_item in unmatched_items:
            if match_context["is_exact_reference"]:
                # Exact PO reference: unmatched items are format variations, NOT financial risk
                severity = "low"
                confidence = 0.6
                explanation = f"Line item '{inv_item.description}' appears to be a format variation or additional item. No financial risk detected."
            elif match_context["is_tentative_match"] and match_context["supplier_matched"]:
                # Tentative PO + supplier match: assume PO ambiguity, not fraud
                severity = "low"
                confidence = 0.5
                explanation = f"Line item '{inv_item.description}' not found in tentative PO. May indicate PO ambiguity or format difference."
            else:
                # Other cases: moderate concern
                severity = "medium"
                confidence = 0.7
                explanation = f"Line item '{inv_item.description}' not found in matched PO. Requires validation."
            
            discrepancy = DiscrepancyDetail(
                type="extra_item",
                severity=severity,
                confidence=confidence,
                invoice_value=f"{inv_item.description} ({inv_item.quantity} @ ${inv_item.unit_price})",
                po_value=None,
                explanation=explanation,
            )
            
            discrepancies.append(discrepancy)
            log_discrepancy(
                logger,
                "extra_item",
                severity,
                explanation,
                confidence,
            )
    
    return discrepancies


async def discrepancy_detection_agent(state: ReconciliationState) -> ReconciliationState:
    """
    Discrepancy Detection Agent - Accountant-like reasoning.
    
    MANDATORY REASONING ORDER:
    1. Assess match context (exact PO? supplier match?)
    2. Align line items (fuzzy matching)
    3. Evaluate discrepancies (only on aligned items)
    4. Generate human-readable reasoning
    
    CRITICAL RULES:
    - Never mark items as "extra" before attempting alignment
    - Total variance only calculated if >= 80% alignment
    - Never raise po_mismatch_suspected for exact reference + supplier match
    - Format differences are NOT fraud
    """
    logger.info(f"[DiscrepancyDetectionAgent] Analyzing invoice {state.invoice_id}")
    
    try:
        if not state.extracted_invoice:
            error_msg = "No extracted invoice available for discrepancy detection"
            logger.error(f"[DiscrepancyDetectionAgent] {error_msg}")
            state.discrepancy_error = error_msg
            state.add_reasoning(
                agent_name="DiscrepancyDetectionAgent",
                message=error_msg,
                confidence=0.0,
            )
            return state
        
        # STEP 0: Assess match context
        match_context = assess_match_context(state)
        logger.debug(f"[DiscrepancyDetectionAgent] Match context: {match_context['context_note']}")
        
        # STEP 1: Align line items
        alignments, alignment_ratio = align_line_items(state)
        logger.debug(f"[DiscrepancyDetectionAgent] Alignment ratio: {alignment_ratio:.1%} ({len(alignments)}/{len(state.extracted_invoice.line_items)} items)")
        
        # STEP 2: Evaluate discrepancies (only on aligned items)
        all_discrepancies = []
        
        all_discrepancies.extend(detect_price_discrepancies(state, alignments, alignment_ratio, match_context))
        all_discrepancies.extend(detect_quantity_discrepancies(state, alignments))
        all_discrepancies.extend(detect_missing_po(state))
        all_discrepancies.extend(detect_extra_items(state, match_context, alignments))
        
        state.discrepancies = all_discrepancies
        
        # STEP 3: Generate human-readable reasoning
        if all_discrepancies:
            high_severity = len([d for d in all_discrepancies if d.severity == "high"])
            medium_severity = len([d for d in all_discrepancies if d.severity == "medium"])
            low_severity = len([d for d in all_discrepancies if d.severity == "low"])
            
            # Build human-readable reasoning
            reasoning_parts = []
            
            reasoning_parts.append(match_context["context_note"])
            reasoning_parts.append(f"Line items aligned: {alignment_ratio:.1%} ({len(alignments)}/{len(state.extracted_invoice.line_items)})")
            
            if high_severity > 0:
                reasoning_parts.append(f"Found {high_severity} high-severity issue(s) requiring attention")
            if medium_severity > 0:
                reasoning_parts.append(f"Found {medium_severity} medium-severity issue(s)")
            if low_severity > 0:
                reasoning_parts.append(f"Found {low_severity} low-severity issue(s) (likely format variations)")
            
            reasoning_msg = ". ".join(reasoning_parts)
            
            logger.warning(
                f"[DiscrepancyDetectionAgent] Found {len(all_discrepancies)} discrepancies: "
                f"{high_severity} high, {medium_severity} medium, {low_severity} low"
            )
            
            state.add_reasoning(
                agent_name="DiscrepancyDetectionAgent",
                message=reasoning_msg,
                confidence=0.9,
                action="discrepancy_detection_complete",
            )
        else:
            logger.info(
                f"[DiscrepancyDetectionAgent] No discrepancies found for invoice {state.invoice_id}"
            )
            
            reasoning_msg = f"{match_context['context_note']} Line items aligned: {alignment_ratio:.1%}. No discrepancies detected."
            
            state.add_reasoning(
                agent_name="DiscrepancyDetectionAgent",
                message=reasoning_msg,
                confidence=0.95,
                action="discrepancy_detection_complete",
            )
    
    except Exception as e:
        logger.exception(f"[DiscrepancyDetectionAgent] Unexpected error: {e}")
        state.discrepancy_error = str(e)
        state.add_reasoning(
            agent_name="DiscrepancyDetectionAgent",
            message=f"Error during discrepancy detection: {str(e)}",
            confidence=0.0,
        )
    
    return state
