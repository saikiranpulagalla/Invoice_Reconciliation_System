"""
CRITICAL TEST CASES FOR INTERNSHIP ASSESSMENT

These tests verify the two most critical requirements:
1. Invoice 4: Price discrepancy detection (10% variance)
2. Invoice 5: Missing PO reference with fuzzy matching

PASSING THESE TESTS IS MANDATORY FOR INTERNSHIP ASSESSMENT.
"""

import pytest
from datetime import datetime
from app.state import ReconciliationState
from app.schemas.invoice import ExtractedInvoice, LineItem, InvoiceConfidence, LineItemConfidence
from app.schemas.po import PurchaseOrder, POLineItem, POMatch
from app.agents.discrepancy import (
    discrepancy_detection_agent,
    detect_price_discrepancies,
    align_line_items,
    assess_match_context,
)
from app.agents.matching import matching_agent
from app.agents.resolution import resolution_recommendation_agent


# ============================================================================
# CRITICAL TEST 1: Invoice 4 - Price Discrepancy Detection
# ============================================================================

@pytest.fixture
def po_2024_004():
    """PO-2024-004 with Ibuprofen at £80.00"""
    return PurchaseOrder(
        po_number="PO-2024-004",
        supplier_name="PharmaChem Supplies Ltd",
        po_date=datetime(2024, 4, 15),
        expected_delivery=datetime(2024, 5, 15),
        subtotal=800.00,
        tax=160.00,
        total=960.00,
        line_items=[
            POLineItem(
                line_number="1",
                description="Ibuprofen BP 200mg",
                quantity=10.0,
                unit_price=80.00,  # £80.00 per unit
                total=800.00
            )
        ],
        po_status="open"
    )


@pytest.fixture
def invoice_4_with_price_increase(po_2024_004):
    """
    Invoice 4: Contains 10% price increase on Ibuprofen
    - PO price: £80.00
    - Invoice price: £88.00
    - Variance: 10% (exceeds 5% tolerance)
    
    EXPECTED RESULT:
    - Discrepancy detected: price_mismatch
    - Severity: medium
    - Confidence: 0.85-0.95
    - Action: flag_for_review
    """
    state = ReconciliationState(
        invoice_id="INV-2024-004",
        processing_timestamp=datetime.utcnow(),
        document_path="/test/invoice_4.pdf",
        available_pos=[po_2024_004]
    )
    
    # Extracted invoice with 10% price increase
    state.extracted_invoice = ExtractedInvoice(
        invoice_number="INV-2024-004",
        invoice_date=datetime(2024, 5, 20),
        supplier_name="PharmaChem Supplies Ltd",
        po_reference="PO-2024-004",  # Exact PO reference
        subtotal=880.00,
        tax=176.00,
        total=1056.00,
        line_items=[
            LineItem(
                description="Ibuprofen BP 200mg",
                quantity=10.0,
                unit_price=88.00,  # £88.00 - 10% higher than PO
                total=880.00,
                confidence=LineItemConfidence(
                    description=0.95,
                    quantity=0.95,
                    unit_price=0.95,
                    total=0.95
                )
            )
        ],
        confidence=InvoiceConfidence(
            invoice_number=0.95,
            invoice_date=0.90,
            supplier_name=0.95,
            po_reference=0.95,
            subtotal=0.95,
            tax=0.90,
            total=0.95,
            line_items=0.95
        ),
        document_quality="good",
        extraction_warnings=[]
    )
    
    # Set matched PO (simulating matching agent result)
    state.matched_po = po_2024_004
    state.po_match = POMatch(
        po_number="PO-2024-004",
        match_confidence=0.98,
        match_type="exact_reference",
        explanation="Exact PO reference match"
    )
    
    return state


@pytest.mark.asyncio
async def test_invoice_4_price_discrepancy_detection(invoice_4_with_price_increase):
    """
    CRITICAL TEST: Invoice 4 - Detect 10% price increase on Ibuprofen
    
    This test MUST pass for internship assessment.
    
    Requirements:
    1. Detect price variance of 10% (£88 vs £80)
    2. Severity: medium (exact PO + supplier match = conservative)
    3. Confidence: 0.85-0.95
    4. Type: price_mismatch
    5. Final action: flag_for_review
    """
    state = invoice_4_with_price_increase
    
    # Run discrepancy detection
    state = await discrepancy_detection_agent(state)
    
    # ASSERTION 1: Discrepancies detected
    assert state.discrepancies is not None, "No discrepancies detected"
    assert len(state.discrepancies) > 0, "Expected at least one discrepancy"
    
    # ASSERTION 2: Price mismatch detected
    price_mismatches = [d for d in state.discrepancies if d.type == "price_mismatch"]
    assert len(price_mismatches) > 0, "Price mismatch not detected"
    
    # ASSERTION 3: Severity is medium (conservative due to exact PO + supplier match)
    price_discrepancy = price_mismatches[0]
    assert price_discrepancy.severity in ["medium", "high"], \
        f"Expected severity 'medium' or 'high', got '{price_discrepancy.severity}'"
    
    # ASSERTION 4: Confidence is high (0.85-0.95)
    assert 0.85 <= price_discrepancy.confidence <= 0.95, \
        f"Expected confidence 0.85-0.95, got {price_discrepancy.confidence}"
    
    # ASSERTION 5: Correct values captured
    assert price_discrepancy.invoice_value == 88.00, \
        f"Expected invoice value £88.00, got £{price_discrepancy.invoice_value}"
    assert price_discrepancy.po_value == 80.00, \
        f"Expected PO value £80.00, got £{price_discrepancy.po_value}"
    
    # ASSERTION 6: Explanation mentions the variance
    assert "Ibuprofen" in price_discrepancy.explanation, \
        "Explanation should mention Ibuprofen"
    assert "88" in str(price_discrepancy.explanation) or "88.00" in str(price_discrepancy.explanation), \
        "Explanation should mention invoice price £88"
    assert "80" in str(price_discrepancy.explanation) or "80.00" in str(price_discrepancy.explanation), \
        "Explanation should mention PO price £80"
    
    # Run resolution agent
    state = await resolution_recommendation_agent(state)
    
    # ASSERTION 7: Final action is flag_for_review
    assert state.recommended_action == "flag_for_review", \
        f"Expected action 'flag_for_review', got '{state.recommended_action}'"
    
    print("✅ CRITICAL TEST PASSED: Invoice 4 - Price discrepancy detected correctly")
    print(f"   - Discrepancy type: {price_discrepancy.type}")
    print(f"   - Severity: {price_discrepancy.severity}")
    print(f"   - Confidence: {price_discrepancy.confidence:.2f}")
    print(f"   - Invoice value: £{price_discrepancy.invoice_value:.2f}")
    print(f"   - PO value: £{price_discrepancy.po_value:.2f}")
    print(f"   - Final action: {state.recommended_action}")


@pytest.mark.asyncio
async def test_invoice_4_line_item_alignment(invoice_4_with_price_increase):
    """
    Test that line items align correctly despite price difference.
    
    Ibuprofen should align to Ibuprofen with high similarity (>0.75).
    """
    state = invoice_4_with_price_increase
    
    # Test alignment
    alignments, alignment_ratio = align_line_items(state)
    
    # ASSERTION: Items should align
    assert len(alignments) == 1, "Expected 1 alignment"
    assert alignment_ratio == 1.0, f"Expected 100% alignment, got {alignment_ratio:.1%}"
    
    # ASSERTION: High similarity score
    inv_idx, po_idx, similarity = alignments[0]
    assert similarity >= 0.75, f"Expected similarity >= 0.75, got {similarity:.2f}"
    
    print(f"✅ Line item alignment: {similarity:.2f} similarity")


# ============================================================================
# CRITICAL TEST 2: Invoice 5 - Missing PO Reference (Fuzzy Matching)
# ============================================================================

@pytest.fixture
def po_2024_005():
    """PO-2024-005 for fuzzy matching test"""
    return PurchaseOrder(
        po_number="PO-2024-005",
        supplier_name="BioTech Chemicals Ltd",
        po_date=datetime(2024, 5, 1),
        expected_delivery=datetime(2024, 6, 1),
        subtotal=1500.00,
        tax=300.00,
        total=1800.00,
        line_items=[
            POLineItem(
                line_number="1",
                description="Paracetamol BP 500mg",
                quantity=20.0,
                unit_price=50.00,
                total=1000.00
            ),
            POLineItem(
                line_number="2",
                description="Aspirin BP 300mg",
                quantity=10.0,
                unit_price=50.00,
                total=500.00
            )
        ],
        po_status="open"
    )


@pytest.fixture
def invoice_5_no_po_reference(po_2024_005):
    """
    Invoice 5: NO PO reference - must use fuzzy matching
    
    EXPECTED RESULT:
    - Match via supplier + product strategy
    - Match type: supplier_product_match
    - Confidence: ≤0.70 (TENTATIVE)
    - Action: flag_for_review
    """
    state = ReconciliationState(
        invoice_id="INV-2024-005",
        processing_timestamp=datetime.utcnow(),
        document_path="/test/invoice_5.pdf",
        available_pos=[po_2024_005]
    )
    
    # Extracted invoice with NO PO reference
    state.extracted_invoice = ExtractedInvoice(
        invoice_number="INV-2024-005",
        invoice_date=datetime(2024, 6, 5),
        supplier_name="BioTech Chemicals Ltd",
        po_reference=None,  # NO PO REFERENCE - critical requirement
        subtotal=1500.00,
        tax=300.00,
        total=1800.00,
        line_items=[
            LineItem(
                description="Paracetamol BP 500mg",
                quantity=20.0,
                unit_price=50.00,
                total=1000.00,
                confidence=LineItemConfidence(
                    description=0.90,
                    quantity=0.90,
                    unit_price=0.90,
                    total=0.90
                )
            ),
            LineItem(
                description="Aspirin BP 300mg",
                quantity=10.0,
                unit_price=50.00,
                total=500.00,
                confidence=LineItemConfidence(
                    description=0.90,
                    quantity=0.90,
                    unit_price=0.90,
                    total=0.90
                )
            )
        ],
        confidence=InvoiceConfidence(
            invoice_number=0.90,
            invoice_date=0.85,
            supplier_name=0.90,
            po_reference=0.0,  # No PO reference
            subtotal=0.90,
            tax=0.85,
            total=0.90,
            line_items=0.90
        ),
        document_quality="good",
        extraction_warnings=["No PO reference found on invoice"]
    )
    
    return state


@pytest.mark.asyncio
async def test_invoice_5_fuzzy_matching_no_po_reference(invoice_5_no_po_reference):
    """
    CRITICAL TEST: Invoice 5 - Fuzzy match when no PO reference exists
    
    This test MUST pass for internship assessment.
    
    Requirements:
    1. No PO reference on invoice
    2. Fuzzy match to PO-2024-005 via supplier + product
    3. Match type: supplier_product_match
    4. Confidence: ≤0.70 (TENTATIVE)
    5. Final action: flag_for_review
    """
    state = invoice_5_no_po_reference
    
    # ASSERTION 1: Invoice has no PO reference
    assert state.extracted_invoice.po_reference is None, \
        "Invoice should have no PO reference"
    
    # Run matching agent
    state = await matching_agent(state)
    
    # ASSERTION 2: PO matched via fuzzy matching
    assert state.matched_po is not None, "No PO matched"
    assert state.matched_po.po_number == "PO-2024-005", \
        f"Expected match to PO-2024-005, got {state.matched_po.po_number}"
    
    # ASSERTION 3: Match type is supplier_product_match or product_only_match
    assert state.po_match is not None, "No PO match result"
    assert state.po_match.match_type in ["supplier_product_match", "product_only_match"], \
        f"Expected fuzzy match type, got '{state.po_match.match_type}'"
    
    # ASSERTION 4: Confidence is TENTATIVE (≤0.70)
    assert state.po_match.match_confidence <= 0.70, \
        f"Expected tentative confidence ≤0.70, got {state.po_match.match_confidence:.2f}"
    
    # ASSERTION 5: Explanation mentions TENTATIVE
    assert "TENTATIVE" in state.po_match.explanation or "tentative" in state.po_match.explanation.lower(), \
        "Explanation should indicate tentative match"
    
    # Run discrepancy detection
    state = await discrepancy_detection_agent(state)
    
    # Run resolution agent
    state = await resolution_recommendation_agent(state)
    
    # ASSERTION 6: Final action is flag_for_review (due to tentative match)
    assert state.recommended_action == "flag_for_review", \
        f"Expected action 'flag_for_review', got '{state.recommended_action}'"
    
    print("✅ CRITICAL TEST PASSED: Invoice 5 - Fuzzy matching without PO reference")
    print(f"   - Matched PO: {state.matched_po.po_number}")
    print(f"   - Match type: {state.po_match.match_type}")
    print(f"   - Match confidence: {state.po_match.match_confidence:.2f}")
    print(f"   - Final action: {state.recommended_action}")


@pytest.mark.asyncio
async def test_invoice_5_supplier_product_alignment(invoice_5_no_po_reference):
    """
    Test that supplier and products align correctly for fuzzy matching.
    """
    state = invoice_5_no_po_reference
    
    # Run matching
    state = await matching_agent(state)
    
    # ASSERTION: Supplier should match
    assert state.matched_po is not None
    assert state.matched_po.supplier_name == "BioTech Chemicals Ltd"
    
    # ASSERTION: Products should align
    state = await discrepancy_detection_agent(state)
    alignments, alignment_ratio = align_line_items(state)
    
    assert alignment_ratio >= 0.80, \
        f"Expected alignment ratio >= 80%, got {alignment_ratio:.1%}"
    
    print(f"✅ Supplier and product alignment: {alignment_ratio:.1%}")


# ============================================================================
# ADDITIONAL CRITICAL TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_exact_po_reference_overrides_fuzzy_matching():
    """
    Test that exact PO reference ALWAYS overrides fuzzy matching.
    
    CRITICAL RULE: If invoice has PO reference, use it (don't fuzzy match).
    """
    po_correct = PurchaseOrder(
        po_number="PO-CORRECT",
        supplier_name="Supplier A",
        po_date=datetime(2024, 1, 1),
        subtotal=1000.0,
        total=1100.0,
        line_items=[
            POLineItem(line_number="1", description="Product X", quantity=10, unit_price=100.0, total=1000.0)
        ]
    )
    
    po_wrong = PurchaseOrder(
        po_number="PO-WRONG",
        supplier_name="Supplier A",
        po_date=datetime(2024, 1, 1),
        subtotal=1000.0,
        total=1100.0,
        line_items=[
            POLineItem(line_number="1", description="Product X", quantity=10, unit_price=100.0, total=1000.0)
        ]
    )
    
    state = ReconciliationState(
        invoice_id="TEST-EXACT",
        processing_timestamp=datetime.utcnow(),
        document_path="/test/exact.pdf",
        available_pos=[po_correct, po_wrong]
    )
    
    # Invoice references PO-CORRECT explicitly
    state.extracted_invoice = ExtractedInvoice(
        invoice_number="INV-TEST",
        invoice_date=datetime(2024, 1, 15),
        supplier_name="Supplier A",
        po_reference="PO-CORRECT",  # Explicit reference
        subtotal=1000.0,
        tax=100.0,
        total=1100.0,
        line_items=[
            LineItem(description="Product X", quantity=10, unit_price=100.0, total=1000.0)
        ],
        confidence=InvoiceConfidence(),
        document_quality="good"
    )
    
    # Run matching
    state = await matching_agent(state)
    
    # ASSERTION: Must match to PO-CORRECT (exact reference)
    assert state.matched_po is not None
    assert state.matched_po.po_number == "PO-CORRECT", \
        f"Expected exact match to PO-CORRECT, got {state.matched_po.po_number}"
    assert state.po_match.match_type == "exact_reference", \
        f"Expected match type 'exact_reference', got '{state.po_match.match_type}'"
    assert state.po_match.match_confidence >= 0.95, \
        f"Expected high confidence for exact match, got {state.po_match.match_confidence:.2f}"
    
    print("✅ Exact PO reference correctly overrides fuzzy matching")


@pytest.mark.asyncio
async def test_missing_po_reference_critical_error():
    """
    Test that referenced PO not found in database is a critical error.
    
    CRITICAL RULE: If PO reference exists but not found → critical error (no fuzzy fallback)
    """
    # Create a PO database with some POs, but not the one referenced
    other_po = PurchaseOrder(
        po_number="PO-OTHER",
        supplier_name="Other Supplier",
        po_date=datetime(2024, 1, 1),
        subtotal=500.0,
        total=550.0,
        line_items=[
            POLineItem(line_number="1", description="Other Product", quantity=5, unit_price=100.0, total=500.0)
        ]
    )
    
    state = ReconciliationState(
        invoice_id="TEST-MISSING",
        processing_timestamp=datetime.utcnow(),
        document_path="/test/missing.pdf",
        available_pos=[other_po]  # PO database has POs, but not the referenced one
    )
    
    # Invoice references non-existent PO
    state.extracted_invoice = ExtractedInvoice(
        invoice_number="INV-TEST",
        invoice_date=datetime(2024, 1, 15),
        supplier_name="Supplier A",
        po_reference="PO-NONEXISTENT",  # PO doesn't exist in database
        subtotal=1000.0,
        tax=100.0,
        total=1100.0,
        line_items=[
            LineItem(description="Product X", quantity=10, unit_price=100.0, total=1000.0)
        ],
        confidence=InvoiceConfidence(),
        document_quality="good"
    )
    
    # Run matching
    state = await matching_agent(state)
    
    # ASSERTION: No match found (critical error)
    assert state.matched_po is None, "Should not match any PO"
    assert state.po_match.match_confidence == 0.0, \
        f"Expected confidence 0.0 for missing PO, got {state.po_match.match_confidence}"
    assert state.po_match.match_type == "missing_po_reference", \
        f"Expected match type 'missing_po_reference', got '{state.po_match.match_type}'"
    
    # Run resolution
    state = await resolution_recommendation_agent(state)
    
    # ASSERTION: Must escalate
    assert state.recommended_action == "escalate_to_human", \
        f"Expected action 'escalate_to_human', got '{state.recommended_action}'"
    
    print("✅ Missing PO reference correctly flagged as critical error")


# ============================================================================
# RUN ALL CRITICAL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING CRITICAL INTERNSHIP ASSESSMENT TESTS")
    print("="*80 + "\n")
    
    pytest.main([__file__, "-v", "-s", "--tb=short"])
