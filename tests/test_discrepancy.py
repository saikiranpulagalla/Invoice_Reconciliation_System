"""
Tests for discrepancy detection.
"""

import pytest
from datetime import datetime
from app.state import ReconciliationState
from app.schemas.invoice import ExtractedInvoice, LineItem, InvoiceConfidence, LineItemConfidence
from app.schemas.po import PurchaseOrder, POLineItem, POMatch
from app.agents.discrepancy import detect_price_discrepancies, detect_quantity_discrepancies


@pytest.fixture
def sample_po():
    """Create a sample PO."""
    return PurchaseOrder(
        po_number="PO-001",
        supplier_name="Acme Corp",
        po_date=datetime(2026, 1, 15),
        subtotal=1000.0,
        total=1100.0,
        line_items=[
            POLineItem(line_number="1", description="Widget A", quantity=100, unit_price=8.0, total=800.0),
            POLineItem(line_number="2", description="Widget B", quantity=50, unit_price=4.0, total=200.0),
        ]
    )


@pytest.fixture
def sample_state(sample_po):
    """Create a sample state."""
    state = ReconciliationState(
        invoice_id="INV-001",
        processing_timestamp=datetime.utcnow(),
        document_path="/tmp/test.pdf",
    )
    state.matched_po = sample_po
    return state


def test_detect_10_percent_price_increase(sample_state):
    """Test detection of 10% hidden price increase."""
    # Create invoice with 10% price increase on Widget A
    sample_state.extracted_invoice = ExtractedInvoice(
        invoice_number="INV-001",
        invoice_date=datetime(2026, 1, 28),
        supplier_name="Acme Corp",
        po_reference="PO-001",
        subtotal=1080.0,
        tax=108.0,
        total=1188.0,
        line_items=[
            LineItem(
                description="Widget A",
                quantity=100,
                unit_price=8.80,  # 10% higher than PO
                total=880.0,
                confidence=LineItemConfidence(unit_price=0.98),
            ),
            LineItem(
                description="Widget B",
                quantity=50,
                unit_price=4.0,
                total=200.0,
            ),
        ],
    )
    
    # Set PO match for context
    sample_state.po_match = POMatch(
        po_number="PO-001",
        match_confidence=0.98,
        match_type="exact_reference",
        explanation="Exact match"
    )
    
    # Get alignments and match context
    from app.agents.discrepancy import align_line_items, assess_match_context
    alignments, alignment_ratio = align_line_items(sample_state)
    match_context = assess_match_context(sample_state)
    
    discrepancies = detect_price_discrepancies(sample_state, alignments, alignment_ratio, match_context)
    
    # Should detect the price increase
    assert len(discrepancies) > 0
    
    # Should find price mismatch
    price_mismatches = [d for d in discrepancies if d.type == "price_mismatch"]
    assert len(price_mismatches) > 0
    
    # Should have medium or high severity
    assert any(d.severity in ["medium", "high"] for d in price_mismatches)
    
    # High confidence in detection
    assert all(d.confidence > 0.8 for d in price_mismatches)


def test_detect_quantity_mismatch(sample_state):
    """Test detection of quantity discrepancies."""
    # Create invoice with different quantities
    sample_state.extracted_invoice = ExtractedInvoice(
        invoice_number="INV-001",
        invoice_date=datetime(2026, 1, 28),
        supplier_name="Acme Corp",
        po_reference="PO-001",
        subtotal=1000.0,
        tax=100.0,
        total=1100.0,
        line_items=[
            LineItem(
                description="Widget A",
                quantity=110,  # 10% more than PO
                unit_price=8.0,
                total=880.0,
            ),
            LineItem(
                description="Widget B",
                quantity=50,
                unit_price=4.0,
                total=200.0,
            ),
        ],
    )
    
    # Get alignments
    from app.agents.discrepancy import align_line_items
    alignments, alignment_ratio = align_line_items(sample_state)
    
    discrepancies = detect_quantity_discrepancies(sample_state, alignments)
    
    # Should detect the quantity mismatch
    assert len(discrepancies) > 0
    assert any(d.type == "quantity_mismatch" for d in discrepancies)


def test_no_discrepancies_when_perfect_match(sample_state):
    """Test that no discrepancies are found for perfect match."""
    # Create invoice matching PO exactly
    sample_state.extracted_invoice = ExtractedInvoice(
        invoice_number="INV-001",
        invoice_date=datetime(2026, 1, 28),
        supplier_name="Acme Corp",
        po_reference="PO-001",
        subtotal=1000.0,
        tax=100.0,
        total=1100.0,
        line_items=[
            LineItem(
                description="Widget A",
                quantity=100,
                unit_price=8.0,
                total=800.0,
            ),
            LineItem(
                description="Widget B",
                quantity=50,
                unit_price=4.0,
                total=200.0,
            ),
        ],
    )
    
    # Set PO match for context
    sample_state.po_match = POMatch(
        po_number="PO-001",
        match_confidence=0.98,
        match_type="exact_reference",
        explanation="Exact match"
    )
    
    # Get alignments and match context
    from app.agents.discrepancy import align_line_items, assess_match_context
    alignments, alignment_ratio = align_line_items(sample_state)
    match_context = assess_match_context(sample_state)
    
    discrepancies_price = detect_price_discrepancies(sample_state, alignments, alignment_ratio, match_context)
    discrepancies_qty = detect_quantity_discrepancies(sample_state, alignments)
    
    # Should not find any discrepancies
    assert len(discrepancies_price) == 0
    assert len(discrepancies_qty) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
