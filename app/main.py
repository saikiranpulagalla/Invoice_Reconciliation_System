"""
Main entry point for the invoice reconciliation system.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from app.state import ReconciliationState
from app.graph import get_reconciliation_graph
from app.schemas.po import PurchaseOrder, POLineItem
from app.schemas.output import ReconciliationOutput, ProcessingResults, MatchingDetail
from app.utils.logging import setup_logging
from app.utils import dict_to_json_string
from app.config import get_config


logger = setup_logging(__name__)
config = get_config()


def load_purchase_orders_from_file(po_file: str) -> List[PurchaseOrder]:
    """Load POs from JSON file."""
    try:
        with open(po_file, 'r') as f:
            po_data = json.load(f)
        
        pos = []
        for po_dict in po_data:
            line_items = [
                POLineItem(**item) for item in po_dict.get("line_items", [])
            ]
            po = PurchaseOrder(
                po_number=po_dict["po_number"],
                supplier_name=po_dict["supplier_name"],
                po_date=datetime.fromisoformat(po_dict["po_date"]),
                expected_delivery=datetime.fromisoformat(po_dict["expected_delivery"]) if po_dict.get("expected_delivery") else None,
                subtotal=po_dict["subtotal"],
                tax=po_dict.get("tax"),
                total=po_dict["total"],
                line_items=line_items,
                po_status=po_dict.get("po_status", "open"),
            )
            pos.append(po)
        
        logger.info(f"Loaded {len(pos)} purchase orders from {po_file}")
        return pos
    
    except FileNotFoundError:
        logger.warning(f"PO file not found: {po_file}. Using empty PO list.")
        return []
    except Exception as e:
        logger.error(f"Error loading PO file: {e}")
        return []


def build_output(state: ReconciliationState) -> ReconciliationOutput:
    """Build final output from state."""
    
    extracted_data_dict = {}
    if state.extracted_invoice:
        extracted_data_dict = {
            "invoice_number": state.extracted_invoice.invoice_number,
            "invoice_date": state.extracted_invoice.invoice_date.isoformat(),
            "supplier_name": state.extracted_invoice.supplier_name,
            "po_reference": state.extracted_invoice.po_reference,
            "subtotal": state.extracted_invoice.subtotal,
            "tax": state.extracted_invoice.tax,
            "total": state.extracted_invoice.total,
            "line_items_count": len(state.extracted_invoice.line_items),
            "document_quality": state.extracted_invoice.document_quality,
            "extraction_confidence": state.extracted_invoice.confidence.average(),
        }
    
    matching_detail = MatchingDetail(
        matched_po=state.po_match.po_number if state.po_match else None,
        match_confidence=state.po_match.match_confidence if state.po_match else 0.0,
        match_type=state.po_match.match_type if state.po_match else "not_matched",
        explanation=state.po_match.explanation if state.po_match else "No matching attempted",
    )
    
    processing_results = ProcessingResults(
        extraction_confidence=state.extracted_invoice.confidence.average() if state.extracted_invoice else 0.0,
        document_quality=state.extracted_invoice.document_quality if state.extracted_invoice else "poor",
        extracted_data=extracted_data_dict,
        matching_results=matching_detail,
        discrepancies=state.discrepancies,
        recommended_action=state.recommended_action,
        action_reasoning=state.action_reasoning,
        agent_reasoning=state.get_agent_reasoning(),
    )
    
    output = ReconciliationOutput(
        invoice_id=state.invoice_id,
        processing_timestamp=datetime.utcnow(),
        processing_results=processing_results,
    )
    
    return output


async def process_invoice(
    document_path: str,
    invoice_id: str = None,
    po_database_path: str = None,
) -> ReconciliationOutput:
    """
    Process a single invoice through the reconciliation workflow.
    
    Args:
        document_path: Path to invoice PDF or image
        invoice_id: Optional invoice ID (auto-generated if not provided)
        po_database_path: Optional path to PO database JSON file
    
    Returns:
        ReconciliationOutput with full processing results
    """
    
    # Generate invoice ID if needed
    if not invoice_id:
        invoice_id = f"INV-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    # Load PO database
    if not po_database_path:
        po_database_path = config.PO_DATABASE_PATH
    
    pos = load_purchase_orders_from_file(po_database_path)
    
    # Initialize state
    state = ReconciliationState(
        invoice_id=invoice_id,
        processing_timestamp=datetime.utcnow(),
        document_path=document_path,
        available_pos=pos,
    )
    
    logger.info(f"Starting invoice reconciliation for {invoice_id}")
    logger.info(f"Document: {document_path}")
    logger.info(f"Available POs: {len(pos)}")
    
    # Get and run the graph
    graph = get_reconciliation_graph()
    
    try:
        # Run the graph (note: this is an async operation in real LangGraph)
        result = await graph.ainvoke(state, config={"recursion_limit": config.GRAPH_RECURSION_LIMIT})
        # Extract state object if result is dict (LangGraph behavior)
        if isinstance(result, dict):
            final_state = result.get("state", state) if "state" in result else state
            # Update state with result data if dict is the actual state
            if any(key in result for key in ["extracted_invoice", "po_match", "discrepancies"]):
                final_state = ReconciliationState(**result)
            else:
                final_state = state
        else:
            final_state = result
    except AttributeError:
        # If ainvoke is not available, try regular invoke
        try:
            result = graph.invoke(state, config={"recursion_limit": config.GRAPH_RECURSION_LIMIT})
            if isinstance(result, dict):
                if any(key in result for key in ["extracted_invoice", "po_match", "discrepancies"]):
                    final_state = ReconciliationState(**result)
                else:
                    final_state = state
            else:
                final_state = result
        except Exception as e:
            logger.error(f"Error running reconciliation graph: {e}")
            final_state = state
    except Exception as e:
        logger.error(f"Error running reconciliation graph: {e}")
        final_state = state
    
    # Build output
    output = build_output(final_state)
    
    logger.info(f"Invoice processing complete. Recommendation: {output.processing_results.recommended_action}")
    
    return output


async def process_invoices_batch(
    document_paths: List[str],
    po_database_path: str = None,
) -> List[ReconciliationOutput]:
    """
    Process multiple invoices in batch.
    
    Args:
        document_paths: List of paths to invoice documents
        po_database_path: Optional path to PO database JSON file
    
    Returns:
        List of ReconciliationOutput objects
    """
    
    results = []
    
    for idx, document_path in enumerate(document_paths, 1):
        try:
            logger.info(f"Processing invoice {idx}/{len(document_paths)}")
            
            output = await process_invoice(
                document_path=document_path,
                po_database_path=po_database_path,
            )
            
            results.append(output)
        
        except Exception as e:
            logger.error(f"Error processing {document_path}: {e}")
            continue
    
    logger.info(f"Batch processing complete. Processed {len(results)}/{len(document_paths)} invoices.")
    
    return results


def format_output_json(output: ReconciliationOutput) -> str:
    """Format output as JSON string."""
    return dict_to_json_string(output.dict())


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        document_path = sys.argv[1]
        invoice_id = sys.argv[2] if len(sys.argv) > 2 else None
        
        output = asyncio.run(process_invoice(document_path, invoice_id))
        print(format_output_json(output))
    else:
        print("Usage: python main.py <document_path> [invoice_id]")
