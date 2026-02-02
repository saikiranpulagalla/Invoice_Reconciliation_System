"""
Tests for matching functionality.
"""

import pytest
from app.agents.matching import (
    match_by_po_reference,
    match_by_supplier_and_products,
    match_by_products_only,
)
from app.schemas.po import PurchaseOrder, POLineItem
from datetime import datetime


@pytest.fixture
def sample_pos():
    """Create sample POs for testing."""
    return [
        PurchaseOrder(
            po_number="PO-001",
            supplier_name="Acme Corp",
            po_date=datetime(2026, 1, 15),
            subtotal=1000.0,
            total=1100.0,
            line_items=[
                POLineItem(line_number="1", description="Widget A", quantity=100, unit_price=8.0, total=800.0),
                POLineItem(line_number="2", description="Widget B", quantity=50, unit_price=4.0, total=200.0),
            ]
        ),
        PurchaseOrder(
            po_number="PO-002",
            supplier_name="TechSupply Inc",
            po_date=datetime(2026, 1, 20),
            subtotal=2500.0,
            total=2750.0,
            line_items=[
                POLineItem(line_number="1", description="Server Component", quantity=5, unit_price=400.0, total=2000.0),
                POLineItem(line_number="2", description="Network Cable", quantity=10, unit_price=50.0, total=500.0),
            ]
        ),
    ]


def test_exact_po_reference_match(sample_pos):
    """Test exact PO reference matching."""
    result = match_by_po_reference("PO-001", sample_pos)
    assert result is not None
    po, confidence, explanation, match_type = result
    assert po.po_number == "PO-001"
    assert confidence > 0.95
    assert match_type == "exact_reference"


def test_no_po_reference_match(sample_pos):
    """Test when PO reference doesn't exist."""
    result = match_by_po_reference("PO-999", sample_pos)
    assert result is None


def test_supplier_and_product_match(sample_pos):
    """Test supplier + product matching."""
    matches = match_by_supplier_and_products(
        "Acme Corp",
        ["Widget A", "Widget B"],
        sample_pos,
    )
    assert len(matches) > 0
    assert matches[0][0].po_number == "PO-001"
    assert matches[0][1] > 0.75  # High confidence


def test_product_only_match(sample_pos):
    """Test product-only matching."""
    matches = match_by_products_only(
        ["Server Component", "Network Cable"],
        sample_pos,
    )
    assert len(matches) > 0
    assert matches[0][0].po_number == "PO-002"


def test_fuzzy_matching_tolerates_variations(sample_pos):
    """Test that fuzzy matching tolerates minor variations."""
    # Should match "Widget A" even with slight variation
    matches = match_by_supplier_and_products(
        "Acme Corporation",  # Slight variation of "Acme Corp"
        ["Widget A Premium"],  # Variation of "Widget A"
        sample_pos,
    )
    assert len(matches) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
