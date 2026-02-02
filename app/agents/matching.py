"""
Matching Agent
Matches extracted invoice to Purchase Orders using fuzzy matching and semantic search.
"""

import json
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
from rapidfuzz import fuzz

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from app.state import ReconciliationState
from app.schemas.po import PurchaseOrder, POMatch
from app.utils.logging import setup_logging
from app.utils.confidence import combine_confidence_scores, boost_confidence
from app.config import get_config


logger = setup_logging(__name__)
config = get_config()


def get_llm():
    """Get LLM instance based on provider."""
    if config.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.1,
            max_output_tokens=500,
        )
    else:
        return ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_BASE,
            temperature=0.1,
            max_tokens=500,
        )


def match_by_po_reference(
    extracted_po_ref: Optional[str],
    available_pos: List[PurchaseOrder],
) -> Optional[Tuple[PurchaseOrder, float, str, str]]:
    """
    Try to match by exact PO reference FIRST, then fuzzy if no exact match.
    
    CRITICAL: Fuzzy matching only allowed if similarity >= 0.90 AND supplier matches
    """
    if not extracted_po_ref:
        return None
    
    # STEP 1: Attempt EXACT MATCH first
    for po in available_pos:
        if po.po_number.upper().strip() == extracted_po_ref.upper().strip():
            explanation = f"Exact match found: Invoice PO reference '{extracted_po_ref}' matches PO {po.po_number}"
            return po, 0.99, explanation, "exact_reference"
    
    # STEP 2: Only use FUZZY MATCHING if similarity >= 0.90
    for po in available_pos:
        similarity = fuzz.token_set_ratio(
            extracted_po_ref.upper(),
            po.po_number.upper()
        ) / 100.0
        
        # STRICT: >= 0.90 similarity required for fuzzy match
        if similarity >= 0.90:
            explanation = f"Fuzzy match: Invoice PO reference '{extracted_po_ref}' similar to {po.po_number} (similarity: {similarity:.2f})"
            return po, similarity, explanation, "fuzzy_reference"
    
    return None


def match_by_supplier_and_products(
    extracted_supplier: str,
    extracted_items_descriptions: List[str],
    available_pos: List[PurchaseOrder],
) -> List[Tuple[PurchaseOrder, float, str]]:
    """Match by supplier name and product descriptions."""
    matches = []
    
    for po in available_pos:
        supplier_similarity = fuzz.token_set_ratio(
            extracted_supplier.upper(),
            po.supplier_name.upper()
        ) / 100.0
        
        if supplier_similarity < 0.7:
            continue
        
        po_item_descriptions = [item.description for item in po.line_items]
        
        best_product_similarity = 0.0
        for extracted_desc in extracted_items_descriptions:
            for po_desc in po_item_descriptions:
                similarity = fuzz.token_set_ratio(
                    extracted_desc.upper(),
                    po_desc.upper()
                ) / 100.0
                best_product_similarity = max(best_product_similarity, similarity)
        
        if best_product_similarity < 0.7:
            continue
        
        combined_confidence = combine_confidence_scores(
            [supplier_similarity, best_product_similarity],
            weights=[0.3, 0.7],
            method="weighted_mean"
        )
        
        explanation = (
            f"Supplier match: {supplier_similarity:.2f}, "
            f"Product match: {best_product_similarity:.2f}. "
            f"Combined confidence: {combined_confidence:.2f}"
        )
        
        matches.append((po, combined_confidence, explanation))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)


def match_by_products_only(
    extracted_items_descriptions: List[str],
    available_pos: List[PurchaseOrder],
) -> List[Tuple[PurchaseOrder, float, str]]:
    """Match by product descriptions only (broadest matching)."""
    matches = []
    
    for po in available_pos:
        po_item_descriptions = [item.description for item in po.line_items]
        
        matches_found = 0
        total_similarity = 0.0
        
        for extracted_desc in extracted_items_descriptions:
            for po_desc in po_item_descriptions:
                similarity = fuzz.token_set_ratio(
                    extracted_desc.upper(),
                    po_desc.upper()
                ) / 100.0
                
                if similarity > config.FUZZY_MATCHING_THRESHOLD:
                    matches_found += 1
                    total_similarity += similarity
        
        if matches_found == 0:
            continue
        
        avg_similarity = total_similarity / matches_found
        
        explanation = (
            f"Product-only match: {matches_found} items matched "
            f"with average similarity {avg_similarity:.2f}"
        )
        
        matches.append((po, avg_similarity, explanation))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)


async def semantic_match_with_llm(
    state: ReconciliationState,
    candidate_pos: List[PurchaseOrder],
    top_k: int = 3,
) -> List[POMatch]:
    """Use LLM for semantic matching as a tie-breaker or for complex cases."""
    if not candidate_pos or not state.extracted_invoice:
        return []
    
    if len(candidate_pos) > 10:
        candidate_pos = candidate_pos[:10]
    
    po_summaries = []
    for po in candidate_pos:
        items_str = ", ".join([f"{item.description} ({item.quantity} @ {item.unit_price})" 
                               for item in po.line_items[:3]])
        po_summaries.append(f"PO #{po.po_number}: {items_str}")
    
    prompt = f"""Match invoice to PO:

Invoice: {state.extracted_invoice.invoice_number}, {state.extracted_invoice.supplier_name}, ${state.extracted_invoice.total}, PO:{state.extracted_invoice.po_reference or "None"}

POs: {'; '.join(po_summaries)}

JSON: {{"best_match_po_number": "PO-XXX or null", "confidence": 0.9, "reasoning": "brief"}}"""
    
    try:
        llm = get_llm()
        response = await llm.ainvoke(prompt)
        result = json.loads(response.content)
        
        if result.get("best_match_po_number"):
            for po in candidate_pos:
                if po.po_number == result["best_match_po_number"]:
                    return [POMatch(
                        po_number=po.po_number,
                        match_confidence=result.get("confidence", 0.75),
                        match_type="semantic_match",
                        explanation=result.get("reasoning", ""),
                    )]
    except Exception as e:
        logger.warning(f"LLM semantic matching failed: {e}")
    
    return []


def validate_supplier_match(invoice_supplier: str, po_supplier: str) -> Tuple[bool, float]:
    """
    Validate supplier name match using normalized string comparison.
    
    Returns:
        (is_matched, similarity_score)
    
    Rules:
    - Ignore casing, punctuation, suffixes (Ltd, Pvt, Inc)
    - >= 0.90 similarity → supplier is considered matched
    - Supplier mismatch = immediate escalation
    """
    if not invoice_supplier or not po_supplier:
        return False, 0.0
    
    # Normalize: remove common suffixes and punctuation
    def normalize(name: str) -> str:
        name = name.upper()
        # Remove common suffixes
        for suffix in [' LTD', ' LIMITED', ' PVT', ' PRIVATE', ' INC', ' INCORPORATED', ' LLC', ' CORP', ' CORPORATION']:
            name = name.replace(suffix, '')
        # Remove punctuation
        name = ''.join(c for c in name if c.isalnum() or c.isspace())
        return name.strip()
    
    normalized_invoice = normalize(invoice_supplier)
    normalized_po = normalize(po_supplier)
    
    similarity = fuzz.token_set_ratio(
        normalized_invoice,
        normalized_po
    ) / 100.0
    
    # STRICT: >= 0.90 similarity required
    is_matched = similarity >= 0.90
    return is_matched, similarity


async def matching_agent(state: ReconciliationState) -> ReconciliationState:
    """
    Matching Agent - Accountant-like reasoning.
    
    CRITICAL PO MATCHING RULES (HARD CONSTRAINTS):
    1. If invoice contains clearly readable PO reference → treat as authoritative
    2. Perform EXACT match first, do NOT fuzzy-match to different PO unless exact PO does not exist
    3. Fuzzy PO matching ONLY allowed when: no PO reference OR PO reference unreadable/missing
    4. If PO match type is EXACT → disable po_mismatch_suspected, proceed to semantic alignment
    """
    logger.info(f"[MatchingAgent] Matching invoice {state.invoice_id}")
    
    try:
        # Basic validation
        if not state.extracted_invoice:
            error_msg = "No extracted invoice available for matching"
            logger.error(f"[MatchingAgent] {error_msg}")
            state.match_error = error_msg
            state.add_reasoning(
                agent_name="MatchingAgent",
                message=error_msg,
                confidence=0.0,
            )
            return state
        
        if not state.available_pos:
            logger.warning("[MatchingAgent] No POs provided in state.")
            
            # Set empty po_match to indicate no POs available
            state.po_match = POMatch(
                po_number="",
                match_confidence=0.0,
                match_type="no_pos_available",
                explanation="No purchase orders available in system database"
            )
            
            state.add_reasoning(
                agent_name="MatchingAgent",
                message="No purchase orders available in system",
                confidence=0.5,
            )
            return state
        
        # RULE 1 & 2: HARD GATE — EXACT PO REFERENCE (HIGHEST PRIORITY)
        # If invoice contains clearly readable PO reference, treat as authoritative
        po_ref = state.extracted_invoice.po_reference
        
        if po_ref and po_ref.strip():  # Check for clearly readable PO reference
            logger.info(f"[MatchingAgent] Invoice contains PO reference: '{po_ref}' - treating as authoritative")
            
            # Check if exact PO exists in database
            exact_match_found = False
            for po in state.available_pos:
                po_num_normalized = po.po_number.upper().strip()
                po_ref_normalized = po_ref.upper().strip()
                
                if po_num_normalized == po_ref_normalized:
                    # EXACT MATCH FOUND
                    exact_match_found = True
                    logger.info(f"[MatchingAgent] Exact PO reference match found: {po.po_number}")
                    
                    # VALIDATE SUPPLIER (if exact reference)
                    supplier_matched, supplier_similarity = validate_supplier_match(
                        state.extracted_invoice.supplier_name,
                        po.supplier_name
                    )
                    
                    if supplier_matched:
                        supplier_note = f"Supplier verified ({supplier_similarity:.2f} similarity)"
                        confidence = 0.98
                    else:
                        # CRITICAL: Supplier mismatch with exact PO = escalation risk
                        supplier_note = f"⚠ Supplier mismatch detected ({supplier_similarity:.2f} similarity). Exact PO reference overrides but requires human review."
                        confidence = 0.85  # Lower confidence due to supplier mismatch
                    
                    logger.info(f"[MatchingAgent] {supplier_note}")
                    
                    # Set match with appropriate confidence
                    state.po_match = POMatch(
                        po_number=po.po_number,
                        match_confidence=confidence,
                        match_type="exact_reference",
                        explanation=f"Exact PO reference match. {supplier_note}",
                        alternative_matches=[],
                    )
                    state.matched_po = po
                    
                    # Human-readable reasoning
                    reasoning_msg = f"PO reference matched exactly: {po.po_number}. {supplier_note}"
                    if not supplier_matched:
                        reasoning_msg += " Supplier mismatch requires human review."
                    
                    state.add_reasoning(
                        agent_name="MatchingAgent",
                        message=reasoning_msg,
                        confidence=confidence,
                        action="exact_reference_match",
                    )
                    
                    return state  # HARD RETURN - NOTHING ELSE RUNS
            
            # RULE 2: If PO reference exists but no exact match found, this is a critical issue
            if not exact_match_found:
                logger.warning(f"[MatchingAgent] PO reference '{po_ref}' not found in database - this is a critical discrepancy")
                
                state.po_match = POMatch(
                    po_number="",
                    match_confidence=0.0,
                    match_type="missing_po_reference",
                    explanation=f"Referenced PO '{po_ref}' not found in system database",
                )
                
                state.add_reasoning(
                    agent_name="MatchingAgent",
                    message=f"Critical: Referenced PO '{po_ref}' not found in system. This requires immediate human review.",
                    confidence=0.0,
                    action="missing_po_reference",
                )
                
                return state  # HARD RETURN - Do not attempt fuzzy matching when PO reference exists but not found
        
        # RULE 3: FUZZY MATCHING only allowed when no PO reference OR PO reference unreadable
        logger.info(f"[MatchingAgent] No PO reference provided - proceeding to fuzzy matching strategies")
        
        # Strategy 1: Supplier + Products match
        all_matches: List[Tuple[PurchaseOrder, float, str, str]] = []
        best_match = None
        best_confidence = 0.0
        best_explanation = ""
        best_match_type = "not_matched"
        
        supplier_matches = match_by_supplier_and_products(
            state.extracted_invoice.supplier_name,
            [item.description for item in state.extracted_invoice.line_items],
            state.available_pos,
        )
        
        if supplier_matches:
            best_match, best_confidence, best_explanation = supplier_matches[0]
            best_match_type = "supplier_product_match"
            # CRITICAL: Supplier+product matches are TENTATIVE
            best_confidence = min(0.70, best_confidence)
            best_explanation = f"TENTATIVE: {best_explanation}. Requires validation through item alignment."
            
            supplier_matches_formatted = [(m[0], min(0.70, m[1]), f"TENTATIVE: {m[2]}", "supplier_product_match") for m in supplier_matches]
            all_matches.extend(supplier_matches_formatted[:3])
            logger.debug(f"[MatchingAgent] Supplier+product matches found: {len(supplier_matches)}")
        
        # Strategy 2: Products only match (broadest matching)
        if not best_match or best_confidence < 0.60:
            product_matches = match_by_products_only(
                [item.description for item in state.extracted_invoice.line_items],
                state.available_pos,
            )
            
            if product_matches:
                if not best_match:
                    best_match, best_confidence, best_explanation = product_matches[0]
                    best_match_type = "product_only_match"
                    # CRITICAL: Product-only matches are LOW confidence
                    best_confidence = min(0.60, best_confidence)
                    best_explanation = f"LOW CONFIDENCE: {best_explanation}. High ambiguity - requires careful validation."
                
                product_matches_formatted = [(m[0], min(0.60, m[1]), f"LOW CONFIDENCE: {m[2]}", "product_only_match") for m in product_matches]
                all_matches.extend(product_matches_formatted[:2])
                logger.debug(f"[MatchingAgent] Product-only matches found: {len(product_matches)}")
        
        # Create POMatch result (for fuzzy matches only - exact matches already returned above)
        if best_match:
            alternative_matches = [
                {"po_number": m[0].po_number, "confidence": m[1], "reason": m[2]}
                for m in all_matches[1:3] if len(m) >= 3
            ]
            
            state.po_match = POMatch(
                po_number=best_match.po_number,
                match_confidence=best_confidence,  # Keep tentative confidence
                match_type=best_match_type,
                explanation=best_explanation,
                alternative_matches=alternative_matches,
            )
            state.matched_po = best_match
            
            logger.info(
                f"[MatchingAgent] Tentative match to PO {best_match.po_number} "
                f"with confidence {best_confidence:.2f} (type: {best_match_type})"
            )
            
            state.add_reasoning(
                agent_name="MatchingAgent",
                message=f"Tentative match to PO {best_match.po_number}. {best_explanation}",
                confidence=best_confidence,
                action="tentative_matching_complete",
            )
        else:
            logger.warning(f"[MatchingAgent] No matching PO found for invoice {state.invoice_id}")
            
            state.po_match = POMatch(
                po_number="",
                match_confidence=0.0,
                match_type="not_matched",
                explanation="No matching PO found in system",
            )
            
            state.add_reasoning(
                agent_name="MatchingAgent",
                message="No matching PO found. Invoice may be for untracked purchase or supplier.",
                confidence=0.0,
            )
    
    except Exception as e:
        logger.exception(f"[MatchingAgent] Unexpected error: {e}")
        state.match_error = str(e)
        state.add_reasoning(
            agent_name="MatchingAgent",
            message=f"Error during matching: {str(e)}",
            confidence=0.0,
        )
    
    return state
