"""
Purchase Order schema and data models.
Represents POs from the database.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class POLineItem(BaseModel):
    """A single line item in a Purchase Order."""
    line_number: str
    description: str
    quantity: float
    unit_price: float
    total: float


class PurchaseOrder(BaseModel):
    """A Purchase Order record."""
    po_number: str
    supplier_name: str
    po_date: datetime
    expected_delivery: Optional[datetime] = None
    subtotal: float
    tax: Optional[float] = None
    total: float
    line_items: List[POLineItem]
    po_status: str = "open"  # open, partially_received, received, cancelled


class POMatch(BaseModel):
    """Result of matching an invoice to a PO."""
    po_number: str
    match_confidence: float = Field(ge=0.0, le=1.0)
    match_type: str  # exact_reference, supplier_fuzzy, product_fuzzy
    matched_items: List[int] = Field(default_factory=list)  # indices of matched line items
    explanation: str
    alternative_matches: List["POMatch"] = Field(default_factory=list)


POMatch.model_rebuild()
