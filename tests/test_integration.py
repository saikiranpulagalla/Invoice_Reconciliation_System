"""
Integration tests for the full invoice reconciliation workflow.
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from app.state import ReconciliationState
from app.schemas.po import PurchaseOrder, POLineItem
from app.schemas.invoice import ExtractedInvoice, InvoiceConfidence, LineItem, LineItemConfidence
from app.agents.document_intelligence import document_intelligence_agent
from app.agents.matching import matching_agent
from app.agents.discrepancy import discrepancy_detection_agent
from app.agents.resolution import resolution_recommendation_agent


@pytest.fixture
def sample_pos():
    """Create sample POs for testing."""
    return [
        PurchaseOrder(
            po_number="PO-2026-001",
            supplier_name="Acme Corp",
            po_date=datetime(2026, 1, 15),
            subtotal=1000.0,
            total=1100.0,
            line_items=[
                POLineItem(line_number="1", description="Widget A", quantity=100, unit_price=8.0, total=800.0),
                POLineItem(line_number="2", description="Widget B", quantity=50, unit_price=4.0, total=200.0),
            ]
        )
    ]


@pytest.fixture
def mock_extracted_invoice():
    """Create a mock extracted invoice."""
    return ExtractedInvoice(
        invoice_number="INV-001",
        invoice_date=datetime(2026, 1, 31),
        supplier_name="Acme Corp",
        po_reference="PO-2026-001",
        subtotal=1000.0,
        tax=100.0,
        total=1100.0,
        line_items=[
            LineItem(
                description="Widget A",
                quantity=100,
                unit_price=8.0,
                total=800.0,
                confidence=LineItemConfidence(description=0.9, quantity=0.9, unit_price=0.9, total=0.9)
            ),
            LineItem(
                description="Widget B", 
                quantity=50,
                unit_price=4.0,
                total=200.0,
                confidence=LineItemConfidence(description=0.9, quantity=0.9, unit_price=0.9, total=0.9)
            )
        ],
        confidence=InvoiceConfidence(
            invoice_number=0.95,
            invoice_date=0.9,
            supplier_name=0.95,
            po_reference=0.9,
            subtotal=0.9,
            tax=0.9,
            total=0.9,
            line_items=0.9
        ),
        document_quality="good",
        extraction_warnings=[]
    )


@pytest.mark.asyncio
async def test_full_workflow_perfect_match(sample_pos, mock_extracted_invoice):
    """Test the full workflow with a perfect invoice-PO match."""
    
    # Create initial state
    state = ReconciliationState(
        invoice_id="test-001",
        processing_timestamp=datetime.now(),
        document_path="test_invoice.pdf",
        available_pos=sample_pos
    )
    
    # Mock the document intelligence agent to return our test invoice
    with patch('app.agents.document_intelligence.extract_text_from_document') as mock_extract:
        mock_extract.return_value = ("Mock invoice text", 0.9)
        
        with patch('app.agents.document_intelligence.get_llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = '''
            {
                "invoice_number": {"value": "INV-001", "confidence": 0.95},
                "invoice_date": {"value": "2026-01-31", "confidence": 0.9},
                "supplier_name": {"value": "Acme Corp", "confidence": 0.95},
                "po_reference": {"value": "PO-2026-001", "confidence": 0.9},
                "subtotal": {"value": 1000.0, "confidence": 0.9},
                "tax": {"value": 100.0, "confidence": 0.9},
                "total": {"value": 1100.0, "confidence": 0.9},
                "line_items": [
                    {
                        "description": "Widget A",
                        "quantity": 100,
                        "unit_price": 8.0,
                        "total": 800.0,
                        "confidence": {"description": 0.9, "quantity": 0.9, "unit_price": 0.9, "total": 0.9}
                    },
                    {
                        "description": "Widget B",
                        "quantity": 50,
                        "unit_price": 4.0,
                        "total": 200.0,
                        "confidence": {"description": 0.9, "quantity": 0.9, "unit_price": 0.9, "total": 0.9}
                    }
                ],
                "extraction_warnings": []
            }
            '''
            
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = mock_response
            mock_llm.return_value = mock_llm_instance
            
            # Step 1: Document Intelligence
            state = await document_intelligence_agent(state)
            
            assert state.extracted_invoice is not None
            assert state.extracted_invoice.invoice_number == "INV-001"
            assert state.extracted_invoice.supplier_name == "Acme Corp"
            assert state.extracted_invoice.total == 1100.0
            assert len(state.extracted_invoice.line_items) == 2
            assert state.extraction_error is None
    
    # Step 2: Matching
    state = await matching_agent(state)
    
    assert state.matched_po is not None
    assert state.matched_po.po_number == "PO-2026-001"
    assert state.po_match is not None
    assert state.po_match.match_confidence > 0.9
    assert state.match_error is None
    
    # Step 3: Discrepancy Detection
    state = await discrepancy_detection_agent(state)
    
    assert state.discrepancies is not None
    assert len(state.discrepancies) == 0  # Perfect match should have no discrepancies
    
    # Step 4: Resolution Recommendation
    state = await resolution_recommendation_agent(state)
    
    assert state.recommended_action == "auto_approve"
    assert state.action_confidence > 0.8
    assert state.human_review_requested is False


@pytest.mark.asyncio
async def test_workflow_with_discrepancies(sample_pos):
    """Test workflow when there are price discrepancies."""
    
    # Create state with mismatched invoice
    state = ReconciliationState(
        invoice_id="test-002",
        processing_timestamp=datetime.now(),
        document_path="test_invoice.pdf",
        available_pos=sample_pos
    )
    
    # Mock extraction with different prices
    with patch('app.agents.document_intelligence.extract_text_from_document') as mock_extract:
        mock_extract.return_value = ("Mock invoice text", 0.9)
        
        with patch('app.agents.document_intelligence.get_llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = '''
            {
                "invoice_number": {"value": "INV-002", "confidence": 0.95},
                "invoice_date": {"value": "2026-01-31", "confidence": 0.9},
                "supplier_name": {"value": "Acme Corp", "confidence": 0.95},
                "po_reference": {"value": "PO-2026-001", "confidence": 0.9},
                "subtotal": {"value": 1200.0, "confidence": 0.9},
                "tax": {"value": 120.0, "confidence": 0.9},
                "total": {"value": 1320.0, "confidence": 0.9},
                "line_items": [
                    {
                        "description": "Widget A",
                        "quantity": 100,
                        "unit_price": 10.0,
                        "total": 1000.0,
                        "confidence": {"description": 0.9, "quantity": 0.9, "unit_price": 0.9, "total": 0.9}
                    }
                ],
                "extraction_warnings": []
            }
            '''
            
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = mock_response
            mock_llm.return_value = mock_llm_instance
            
            # Run through workflow
            state = await document_intelligence_agent(state)
            state = await matching_agent(state)
            state = await discrepancy_detection_agent(state)
            state = await resolution_recommendation_agent(state)
            
            # Should detect price discrepancies
            assert len(state.discrepancies) > 0
            assert any(d.type == "total_variance" for d in state.discrepancies)
            
            # Should recommend human review due to discrepancies
            assert state.recommended_action in ["escalate", "human_review"]
            assert state.human_review_requested is True


@pytest.mark.asyncio
async def test_workflow_no_matching_po(sample_pos):
    """Test workflow when no matching PO is found."""
    
    state = ReconciliationState(
        invoice_id="test-003",
        processing_timestamp=datetime.now(),
        document_path="test_invoice.pdf",
        available_pos=sample_pos
    )
    
    # Mock extraction with different supplier
    with patch('app.agents.document_intelligence.extract_text_from_document') as mock_extract:
        mock_extract.return_value = ("Mock invoice text", 0.9)
        
        with patch('app.agents.document_intelligence.get_llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = '''
            {
                "invoice_number": {"value": "INV-003", "confidence": 0.95},
                "invoice_date": {"value": "2026-01-31", "confidence": 0.9},
                "supplier_name": {"value": "Different Corp", "confidence": 0.95},
                "po_reference": {"value": null, "confidence": 0.0},
                "subtotal": {"value": 500.0, "confidence": 0.9},
                "tax": {"value": 50.0, "confidence": 0.9},
                "total": {"value": 550.0, "confidence": 0.9},
                "line_items": [
                    {
                        "description": "Unknown Product",
                        "quantity": 10,
                        "unit_price": 50.0,
                        "total": 500.0,
                        "confidence": {"description": 0.9, "quantity": 0.9, "unit_price": 0.9, "total": 0.9}
                    }
                ],
                "extraction_warnings": []
            }
            '''
            
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = mock_response
            mock_llm.return_value = mock_llm_instance
            
            # Run through workflow
            state = await document_intelligence_agent(state)
            state = await matching_agent(state)
            state = await discrepancy_detection_agent(state)
            state = await resolution_recommendation_agent(state)
            
            # Should not find matching PO
            assert state.matched_po is None
            assert state.po_match.match_confidence == 0.0
            
            # Should have missing PO discrepancy
            assert len(state.discrepancies) > 0
            assert any(d.type == "missing_po" for d in state.discrepancies)
            
            # Should escalate due to no matching PO
            assert state.recommended_action == "escalate"
            assert state.human_review_requested is True


@pytest.mark.asyncio
async def test_workflow_extraction_failure():
    """Test workflow when document extraction fails."""
    
    state = ReconciliationState(
        invoice_id="test-004",
        processing_timestamp=datetime.now(),
        document_path="invalid_file.pdf",
        available_pos=[]
    )
    
    # Mock extraction failure
    with patch('app.agents.document_intelligence.extract_text_from_document') as mock_extract:
        mock_extract.side_effect = ValueError("Unsupported document format")
        
        # Run document intelligence
        state = await document_intelligence_agent(state)
        
        # Should have extraction error
        assert state.extracted_invoice is None
        assert state.extraction_error is not None
        assert "Unsupported document format" in state.extraction_error
        
        # Subsequent agents should handle gracefully
        state = await matching_agent(state)
        assert state.match_error is not None
        
        state = await discrepancy_detection_agent(state)
        # Should still work even without extracted invoice
        
        state = await resolution_recommendation_agent(state)
        assert state.recommended_action == "escalate"


@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test that workflow handles various error conditions gracefully."""
    
    state = ReconciliationState(
        invoice_id="test-005",
        processing_timestamp=datetime.now(),
        document_path="test.pdf",
        available_pos=[]
    )
    
    # Test with empty PO list
    with patch('app.agents.document_intelligence.extract_text_from_document') as mock_extract:
        mock_extract.return_value = ("Valid text", 0.9)
        
        with patch('app.agents.document_intelligence.get_llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = '{"invoice_number": {"value": "INV-005", "confidence": 0.9}}'
            
            mock_llm_instance = AsyncMock()
            mock_llm_instance.ainvoke.return_value = mock_response
            mock_llm.return_value = mock_llm_instance
            
            # Run workflow
            state = await document_intelligence_agent(state)
            state = await matching_agent(state)
            state = await discrepancy_detection_agent(state)
            state = await resolution_recommendation_agent(state)
            
            # Should handle empty PO list gracefully
            assert state.matched_po is None
            assert state.recommended_action == "escalate"


if __name__ == "__main__":
    pytest.main([__file__])