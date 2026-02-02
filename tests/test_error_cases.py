"""
Tests for error handling and edge cases.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.state import ReconciliationState
from app.schemas.po import PurchaseOrder, POLineItem
from app.agents.document_intelligence import document_intelligence_agent, parse_extracted_json
from app.agents.matching import matching_agent
from app.utils.confidence import combine_confidence_scores


class TestDocumentIntelligenceErrors:
    """Test error handling in document intelligence agent."""
    
    @pytest.mark.asyncio
    async def test_missing_file(self):
        """Test handling of missing document file."""
        state = ReconciliationState(
            invoice_id="test-missing",
            processing_timestamp=datetime.now(),
            document_path="nonexistent.pdf",
            available_pos=[]
        )
        
        with patch('app.agents.document_intelligence.extract_text_from_document') as mock_extract:
            mock_extract.side_effect = FileNotFoundError("File not found")
            
            result = await document_intelligence_agent(state)
            
            assert result.extraction_error is not None
            assert "File not found" in result.extraction_error
            assert result.extracted_invoice is None
    
    @pytest.mark.asyncio
    async def test_empty_document(self):
        """Test handling of empty or unreadable document."""
        state = ReconciliationState(
            invoice_id="test-empty",
            processing_timestamp=datetime.now(),
            document_path="empty.pdf",
            available_pos=[]
        )
        
        with patch('app.agents.document_intelligence.extract_text_from_document') as mock_extract:
            mock_extract.return_value = ("", 0.0)  # Empty text
            
            result = await document_intelligence_agent(state)
            
            assert result.extraction_error is not None
            assert "Insufficient text" in result.extraction_error
            assert result.extracted_invoice is None
    
    @pytest.mark.asyncio
    async def test_llm_api_failure(self):
        """Test handling of LLM API failures."""
        state = ReconciliationState(
            invoice_id="test-llm-fail",
            processing_timestamp=datetime.now(),
            document_path="test.pdf",
            available_pos=[]
        )
        
        with patch('app.agents.document_intelligence.extract_text_from_document') as mock_extract:
            mock_extract.return_value = ("Valid invoice text", 0.9)
            
            with patch('app.agents.document_intelligence.get_llm') as mock_llm:
                mock_llm_instance = AsyncMock()
                mock_llm_instance.ainvoke.side_effect = Exception("API rate limit exceeded")
                mock_llm.return_value = mock_llm_instance
                
                result = await document_intelligence_agent(state)
                
                assert result.extraction_error is not None
                assert "API rate limit exceeded" in result.extraction_error
                assert result.extracted_invoice is None
    
    @pytest.mark.asyncio
    async def test_invalid_llm_response(self):
        """Test handling of invalid LLM responses."""
        state = ReconciliationState(
            invoice_id="test-invalid-response",
            processing_timestamp=datetime.now(),
            document_path="test.pdf",
            available_pos=[]
        )
        
        with patch('app.agents.document_intelligence.extract_text_from_document') as mock_extract:
            mock_extract.return_value = ("Valid invoice text", 0.9)
            
            with patch('app.agents.document_intelligence.get_llm') as mock_llm:
                mock_response = Mock()
                mock_response.content = "This is not JSON at all!"
                
                mock_llm_instance = AsyncMock()
                mock_llm_instance.ainvoke.return_value = mock_response
                mock_llm.return_value = mock_llm_instance
                
                result = await document_intelligence_agent(state)
                
                assert result.extraction_error is not None
                assert "Failed to parse" in result.extraction_error
                assert result.extracted_invoice is None
    
    def test_json_parsing_edge_cases(self):
        """Test JSON parsing with various malformed inputs."""
        
        # Test empty response
        with pytest.raises(ValueError, match="Empty response"):
            parse_extracted_json("")
        
        # Test malformed JSON
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_extracted_json("{ invalid json }")
        
        # Test JSON in markdown that's still invalid
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_extracted_json("```json\n{ invalid: json }\n```")
        
        # Test valid JSON in markdown
        result = parse_extracted_json("```json\n{\"test\": \"value\"}\n```")
        assert result == {"test": "value"}
        
        # Test JSON with extra text
        result = parse_extracted_json("Here's the JSON: {\"test\": \"value\"} and some more text")
        assert result == {"test": "value"}


class TestMatchingErrors:
    """Test error handling in matching agent."""
    
    @pytest.mark.asyncio
    async def test_no_extracted_invoice(self):
        """Test matching when no invoice was extracted."""
        state = ReconciliationState(
            invoice_id="test-no-invoice",
            document_path="test.pdf",
            available_pos=[]
        )
        # Don't set extracted_invoice
        
        result = await matching_agent(state)
        
        assert result.match_error is not None
        assert "No extracted invoice" in result.match_error
        assert result.matched_po is None
    
    @pytest.mark.asyncio
    async def test_empty_po_database(self):
        """Test matching with empty PO database."""
        from app.schemas.invoice import ExtractedInvoice, InvoiceConfidence
        
        state = ReconciliationState(
            invoice_id="test-empty-pos",
            document_path="test.pdf",
            available_pos=[]
        )
        
        state.extracted_invoice = ExtractedInvoice(
            invoice_number="INV-001",
            invoice_date=datetime.now(),
            supplier_name="Test Corp",
            po_reference="PO-001",
            subtotal=100.0,
            tax=10.0,
            total=110.0,
            line_items=[],
            confidence=InvoiceConfidence(),
            document_quality="good",
            extraction_warnings=[]
        )
        
        result = await matching_agent(state)
        
        # Should handle gracefully, not error
        assert result.match_error is None
        assert result.matched_po is None
        assert result.po_match.match_confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_malformed_po_data(self):
        """Test matching with malformed PO data."""
        from app.schemas.invoice import ExtractedInvoice, InvoiceConfidence
        
        # Create PO with missing required fields
        malformed_po = PurchaseOrder(
            po_number="PO-001",
            supplier_name="",  # Empty supplier name
            po_date=datetime.now(),
            subtotal=0.0,  # Zero amounts
            total=0.0,
            line_items=[]  # No line items
        )
        
        state = ReconciliationState(
            invoice_id="test-malformed-po",
            document_path="test.pdf",
            available_pos=[malformed_po]
        )
        
        state.extracted_invoice = ExtractedInvoice(
            invoice_number="INV-001",
            invoice_date=datetime.now(),
            supplier_name="Test Corp",
            po_reference="PO-001",
            subtotal=100.0,
            tax=10.0,
            total=110.0,
            line_items=[],
            confidence=InvoiceConfidence(),
            document_quality="good",
            extraction_warnings=[]
        )
        
        result = await matching_agent(state)
        
        # Should handle gracefully
        assert result.match_error is None
        # Might not match due to empty supplier name
        assert result.po_match.match_confidence < 0.5


class TestConfidenceScoreErrors:
    """Test error handling in confidence scoring."""
    
    def test_invalid_confidence_scores(self):
        """Test handling of invalid confidence scores."""
        
        # Test with out-of-range scores
        result = combine_confidence_scores([-0.5, 1.5, 0.8])
        assert 0.0 <= result <= 1.0
        
        # Test with empty list
        result = combine_confidence_scores([])
        assert result == 0.0
        
        # Test with mismatched weights
        with pytest.raises(ValueError, match="Weights length"):
            combine_confidence_scores([0.8, 0.9], weights=[0.5])
        
        # Test with invalid method
        with pytest.raises(ValueError, match="Unknown confidence combination method"):
            combine_confidence_scores([0.8, 0.9], method="invalid_method")
    
    def test_confidence_score_validation_logging(self):
        """Test that out-of-range scores are logged."""
        
        with patch('app.utils.confidence.setup_logging') as mock_logging:
            mock_logger = Mock()
            mock_logging.return_value = mock_logger
            
            # This should trigger a warning log
            combine_confidence_scores([-0.1, 1.1, 0.5])
            
            # Check that warning was logged
            assert mock_logger.warning.called
            warning_calls = mock_logger.warning.call_args_list
            assert any("out of range" in str(call) for call in warning_calls)


class TestStateManagement:
    """Test error handling in state management."""
    
    def test_invalid_state_transitions(self):
        """Test handling of invalid state transitions."""
        state = ReconciliationState(
            invoice_id="test-state",
            document_path="test.pdf",
            available_pos=[]
        )
        
        # Test adding reasoning with invalid confidence
        state.add_reasoning(
            agent_name="TestAgent",
            message="Test message",
            confidence=-0.5  # Invalid confidence
        )
        
        # Should still work (confidence gets clamped)
        assert len(state.reasoning_log) == 1
        assert 0.0 <= state.reasoning_log[0].confidence <= 1.0
    
    def test_state_serialization_errors(self):
        """Test handling of state serialization errors."""
        from app.schemas.invoice import ExtractedInvoice, InvoiceConfidence
        
        state = ReconciliationState(
            invoice_id="test-serialization",
            document_path="test.pdf",
            available_pos=[]
        )
        
        # Add invoice with potential serialization issues
        state.extracted_invoice = ExtractedInvoice(
            invoice_number="INV-001",
            invoice_date=datetime.now(),
            supplier_name="Test Corp",
            po_reference=None,  # None values
            subtotal=100.0,
            tax=10.0,
            total=110.0,
            line_items=[],
            confidence=InvoiceConfidence(),
            document_quality="good",
            extraction_warnings=[]
        )
        
        # Should be able to convert to dict without errors
        try:
            state_dict = state.model_dump()
            assert isinstance(state_dict, dict)
            assert state_dict["invoice_id"] == "test-serialization"
        except Exception as e:
            pytest.fail(f"State serialization failed: {e}")


class TestEdgeCases:
    """Test various edge cases."""
    
    def test_very_large_invoice(self):
        """Test handling of invoices with many line items."""
        from app.schemas.invoice import ExtractedInvoice, InvoiceConfidence, LineItem, LineItemConfidence
        
        # Create invoice with 1000 line items
        line_items = []
        for i in range(1000):
            line_items.append(LineItem(
                description=f"Item {i}",
                quantity=1.0,
                unit_price=10.0,
                total=10.0,
                confidence=LineItemConfidence()
            ))
        
        invoice = ExtractedInvoice(
            invoice_number="INV-LARGE",
            invoice_date=datetime.now(),
            supplier_name="Test Corp",
            po_reference="PO-001",
            subtotal=10000.0,
            tax=1000.0,
            total=11000.0,
            line_items=line_items,
            confidence=InvoiceConfidence(),
            document_quality="good",
            extraction_warnings=[]
        )
        
        # Should handle large invoices without issues
        assert len(invoice.line_items) == 1000
        assert invoice.total == 11000.0
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        from app.schemas.invoice import ExtractedInvoice, InvoiceConfidence
        
        invoice = ExtractedInvoice(
            invoice_number="INV-ñ001",
            invoice_date=datetime.now(),
            supplier_name="Tëst Cörp™",
            po_reference="PO-001",
            subtotal=100.0,
            tax=10.0,
            total=110.0,
            line_items=[],
            confidence=InvoiceConfidence(),
            document_quality="good",
            extraction_warnings=[]
        )
        
        # Should handle unicode characters
        assert "ñ" in invoice.invoice_number
        assert "ë" in invoice.supplier_name
        assert "™" in invoice.supplier_name
    
    def test_extreme_confidence_values(self):
        """Test handling of extreme confidence values."""
        
        # Test with very small differences
        result = combine_confidence_scores([0.999999, 0.999998, 0.999997])
        assert 0.0 <= result <= 1.0
        
        # Test with all zeros
        result = combine_confidence_scores([0.0, 0.0, 0.0])
        assert result == 0.0
        
        # Test with all ones
        result = combine_confidence_scores([1.0, 1.0, 1.0])
        assert result == 1.0


if __name__ == "__main__":
    pytest.main([__file__])