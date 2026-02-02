"""
Invoice schema and data models.
Represents extracted invoice data with confidence scores.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class LineItemConfidence(BaseModel):
    """Confidence scores for individual line item fields."""
    description: float = Field(default=1.0, ge=0.0, le=1.0)
    quantity: float = Field(default=1.0, ge=0.0, le=1.0)
    unit_price: float = Field(default=1.0, ge=0.0, le=1.0)
    total: float = Field(default=1.0, ge=0.0, le=1.0)
    
    def average(self) -> float:
        """Calculate average confidence."""
        return (self.description + self.quantity + self.unit_price + self.total) / 4


class LineItem(BaseModel):
    """A single line item from an invoice."""
    description: str
    quantity: float
    unit_price: float
    total: float
    po_line_number: Optional[str] = None
    confidence: LineItemConfidence = Field(default_factory=LineItemConfidence)


class InvoiceConfidence(BaseModel):
    """Confidence scores for invoice extraction fields."""
    invoice_number: float = Field(default=1.0, ge=0.0, le=1.0)
    invoice_date: float = Field(default=1.0, ge=0.0, le=1.0)
    supplier_name: float = Field(default=1.0, ge=0.0, le=1.0)
    po_reference: float = Field(default=1.0, ge=0.0, le=1.0)
    subtotal: float = Field(default=1.0, ge=0.0, le=1.0)
    tax: float = Field(default=1.0, ge=0.0, le=1.0)
    total: float = Field(default=1.0, ge=0.0, le=1.0)
    line_items: float = Field(default=1.0, ge=0.0, le=1.0)
    
    def average(self) -> float:
        """Calculate average confidence across all fields."""
        values = [
            self.invoice_number,
            self.invoice_date,
            self.supplier_name,
            self.po_reference,
            self.subtotal,
            self.tax,
            self.total,
            self.line_items,
        ]
        return sum(values) / len(values)


class ExtractedInvoice(BaseModel):
    """Fully extracted invoice with confidence scores."""
    invoice_number: str
    invoice_date: datetime
    supplier_name: str
    po_reference: Optional[str] = None
    subtotal: float
    tax: float
    total: float
    line_items: List[LineItem]
    confidence: InvoiceConfidence = Field(default_factory=InvoiceConfidence)
    
    # OCR metadata
    document_quality: str = "acceptable"  # good, acceptable, poor
    extraction_warnings: List[str] = Field(default_factory=list)
    extraction_errors: List[str] = Field(default_factory=list)
    
    def is_high_quality(self) -> bool:
        """Check if extraction is high quality (>0.85 confidence)."""
        return self.confidence.average() > 0.85
    
    def is_acceptable_quality(self) -> bool:
        """Check if extraction is acceptable quality (>0.7 confidence)."""
        return self.confidence.average() > 0.7
