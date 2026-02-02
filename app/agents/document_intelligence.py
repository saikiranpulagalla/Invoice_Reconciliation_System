"""
Document Intelligence Agent
Extracts structured invoice data from documents using LLM with OCR.
Assigns confidence scores to each field.
"""

import json
import re
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.state import ReconciliationState
from app.schemas.invoice import (
    ExtractedInvoice,
    InvoiceConfidence,
    LineItem,
    LineItemConfidence,
)
from app.utils.ocr import extract_text_from_pdf, extract_text_from_image
from app.utils.preprocessing import assess_image_quality
from app.utils.logging import setup_logging, log_agent_action
from app.utils.confidence import (
    combine_confidence_scores,
    penalize_confidence,
    boost_confidence,
)
from app.config import get_config


logger = setup_logging(__name__)
config = get_config()


# Initialize LLM
def get_llm(model_name: str = None):
    """Get LLM instance based on provider."""
    model = model_name or config.LLM_MODEL
    
    if config.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_TOKENS,
        )
    else:  # Default to OpenAI for backward compatibility
        return ChatOpenAI(
            model=model,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_BASE,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )


async def try_llm_extraction(text: str, model_name: str = None) -> str:
    """Try LLM extraction with a specific model."""
    
    # Check if mock mode is enabled
    if config.LLM_MOCK_MODE:
        logger.info("Mock mode enabled - returning sample JSON response")
        return '''
        {
            "invoice_number": {"value": "INV-MOCK-001", "confidence": 0.95},
            "invoice_date": {"value": "2026-01-31", "confidence": 0.9},
            "supplier_name": {"value": "Mock Supplier Corp", "confidence": 0.95},
            "po_reference": {"value": "PO-2026-001", "confidence": 0.8},
            "subtotal": {"value": 1000.0, "confidence": 0.9},
            "tax": {"value": 100.0, "confidence": 0.9},
            "total": {"value": 1100.0, "confidence": 0.9},
            "line_items": [
                {
                    "description": "Mock Product A",
                    "quantity": 100,
                    "unit_price": 8.0,
                    "total": 800.0,
                    "confidence": {"description": 0.9, "quantity": 0.9, "unit_price": 0.9, "total": 0.9}
                },
                {
                    "description": "Mock Product B",
                    "quantity": 50,
                    "unit_price": 4.0,
                    "total": 200.0,
                    "confidence": {"description": 0.9, "quantity": 0.9, "unit_price": 0.9, "total": 0.9}
                }
            ],
            "extraction_warnings": ["Mock mode - not real extraction"]
        }
        '''
    
    # Add rate limiting for Gemini (15 requests per minute = 4 second intervals)
    if config.LLM_PROVIDER == "gemini":
        import time
        time.sleep(4)  # Wait 4 seconds between requests to stay within rate limits
    
    llm = get_llm(model_name)
    prompt = PromptTemplate(
        input_variables=["invoice_text"],
        template=EXTRACTION_PROMPT_TEMPLATE,
    )
    
    chain = prompt | llm
    response = await chain.ainvoke({"invoice_text": text})
    return get_llm_response_text(response).strip()


def get_llm_response_text(response) -> str:
    """
    Robustly extract textual content from various LLM response shapes.
    Handles objects from different LangChain/OpenAI wrappers.
    """
    try:
        # Check for content attribute first
        if hasattr(response, "content"):
            content = response.content
            if content and content.strip():
                return str(content)
        
        # Check for reasoning tokens in metadata (for models like gpt-5-nano)
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if isinstance(metadata, dict):
                token_usage = metadata.get("token_usage", {})
                completion_details = token_usage.get("completion_tokens_details", {})
                reasoning_tokens = completion_details.get("reasoning_tokens", 0)
                
                # If we have reasoning tokens but no content, the model might be thinking but not responding
                if reasoning_tokens > 0 and (not hasattr(response, "content") or not response.content):
                    raise ValueError(f"Model used {reasoning_tokens} reasoning tokens but provided no content. This may indicate the prompt needs adjustment or the model is having issues.")

        # LangChain LLMResult: .generations -> list[list[Generation]]
        if hasattr(response, "generations") and response.generations:
            gen0 = response.generations[0]
            if isinstance(gen0, list) and gen0:
                # Generation has .text
                first = gen0[0]
                if hasattr(first, "text") and first.text:
                    return str(first.text)
            elif hasattr(gen0, "text") and gen0.text:
                return str(gen0.text)

        # Sometimes response is a dict
        if isinstance(response, dict):
            for key in ("content", "text", "output_text", "response", "choices"):
                if key in response and response[key]:
                    return str(response[key])
            # choices -> list with text
            if "choices" in response and isinstance(response["choices"], list) and response["choices"]:
                first = response["choices"][0]
                if isinstance(first, dict) and ("text" in first or "message" in first):
                    return str(first.get("text") or first.get("message"))

        # If we get here, we couldn't extract any meaningful content
        response_str = str(response)
        if len(response_str) > 500:
            response_str = response_str[:500] + "..."
        
        raise ValueError(f"Could not extract text content from LLM response. Response type: {type(response)}, Response preview: {response_str}")
    
    except Exception as e:
        # Re-raise ValueError as-is, wrap other exceptions
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error parsing LLM response: {str(e)}")




EXTRACTION_PROMPT_TEMPLATE = """Extract invoice data as JSON from this text:

{invoice_text}

Return ONLY valid JSON in this format:
{{"invoice_number": {{"value": "INV-123", "confidence": 0.9}}, "invoice_date": {{"value": "2026-01-31", "confidence": 0.9}}, "supplier_name": {{"value": "Acme Corp", "confidence": 0.9}}, "po_reference": {{"value": "PO-456", "confidence": 0.8}}, "subtotal": {{"value": 1000.0, "confidence": 0.85}}, "tax": {{"value": 100.0, "confidence": 0.85}}, "total": {{"value": 1100.0, "confidence": 0.85}}, "line_items": [{{"description": "Widget A", "quantity": 100, "unit_price": 8.0, "total": 800.0, "confidence": {{"description": 0.9, "quantity": 0.9, "unit_price": 0.9, "total": 0.9}}}}], "extraction_warnings": []}}

Rules: Return ONLY JSON. Use null for missing fields. Confidence 0.0-1.0."""


def extract_text_from_document(document_path: str) -> Tuple[str, float]:
    """
    Extract text from document (PDF or image).
    
    Returns:
        (text, quality_score)
    """
    path = Path(document_path)
    
    if path.suffix.lower() in ['.pdf']:
        text, confidence = extract_text_from_pdf(document_path, use_ocr=config.ENABLE_OCR)
    elif path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        quality = assess_image_quality(document_path)
        quality_score = {"good": 1.0, "acceptable": 0.75, "poor": 0.5}.get(quality, 0.5)
        
        text, ocr_conf = extract_text_from_image(document_path)
        confidence = combine_confidence_scores(
            [quality_score, ocr_conf],
            weights=[0.3, 0.7],  # OCR confidence is more important
            method="weighted_mean"
        )
    else:
        raise ValueError(f"Unsupported document format: {path.suffix}")
    
    return text, confidence


def parse_extracted_json(json_str: str) -> dict:
    """
    Parse JSON response from LLM, with robust error handling.
    """
    if not json_str or not json_str.strip():
        raise ValueError("Empty response from LLM")
    
    # Clean up the response
    json_str = json_str.strip()
    
    # Try direct parsing first
    try:
        result = json.loads(json_str)
        if isinstance(result, dict) and result:
            return result
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON if it's wrapped in markdown code blocks
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', json_str, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, dict) and result:
                return result
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in the text (curly braces)
    match = re.search(r'\{[\s\S]*\}', json_str)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict) and result:
                return result
        except json.JSONDecodeError:
            pass
    
    # Last resort: try to extract just the object literal
    try:
        # Remove leading/trailing non-JSON characters
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start >= 0 and end > start:
            json_candidate = json_str[start:end+1]
            result = json.loads(json_candidate)
            if isinstance(result, dict) and result:
                return result
    except json.JSONDecodeError:
        pass
    
    # If all parsing fails, provide detailed error
    response_preview = json_str[:300] if len(json_str) > 300 else json_str
    raise ValueError(f"Could not parse JSON from LLM response:\n{response_preview}")


def build_extracted_invoice(extracted_data: dict, ocr_confidence: float) -> ExtractedInvoice:
    """
    Build ExtractedInvoice object from LLM extraction results.
    Handles missing or null values gracefully.
    """
    # Ensure extracted_data has the expected structure
    if not extracted_data:
        extracted_data = {}
    
    # Parse invoice number
    invoice_number_obj = extracted_data.get("invoice_number", {})
    if isinstance(invoice_number_obj, dict):
        invoice_number = invoice_number_obj.get("value") or "UNKNOWN"
        invoice_number_conf = invoice_number_obj.get("confidence", 0.5)
    else:
        invoice_number = "UNKNOWN"
        invoice_number_conf = 0.5
    
    # Parse invoice date
    invoice_date_obj = extracted_data.get("invoice_date", {})
    if isinstance(invoice_date_obj, dict):
        invoice_date_str = invoice_date_obj.get("value")
        invoice_date_conf = invoice_date_obj.get("confidence", 0.5)
    else:
        invoice_date_str = None
        invoice_date_conf = 0.5
    
    try:
        if invoice_date_str:
            invoice_date = datetime.strptime(invoice_date_str, "%Y-%m-%d")
        else:
            invoice_date = datetime.utcnow()
            invoice_date_conf = penalize_confidence(invoice_date_conf, 0.5)
    except (TypeError, ValueError):
        invoice_date = datetime.utcnow()
        invoice_date_conf = penalize_confidence(invoice_date_conf, 0.5)
    
    # Parse supplier name
    supplier_obj = extracted_data.get("supplier_name", {})
    if isinstance(supplier_obj, dict):
        supplier_name = supplier_obj.get("value") or "UNKNOWN"
        supplier_name_conf = supplier_obj.get("confidence", 0.5)
    else:
        supplier_name = "UNKNOWN"
        supplier_name_conf = 0.5
    
    # Parse PO reference
    po_obj = extracted_data.get("po_reference", {})
    if isinstance(po_obj, dict):
        po_reference = po_obj.get("value")
        po_reference_conf = po_obj.get("confidence", 0.5)
    else:
        po_reference = None
        po_reference_conf = 0.5
    
    # Parse amounts
    subtotal_obj = extracted_data.get("subtotal", {})
    if isinstance(subtotal_obj, dict):
        subtotal = float(subtotal_obj.get("value") or 0.0)
        subtotal_conf = subtotal_obj.get("confidence", 0.5)
    else:
        subtotal = 0.0
        subtotal_conf = 0.5
    
    tax_obj = extracted_data.get("tax", {})
    if isinstance(tax_obj, dict):
        tax = float(tax_obj.get("value") or 0.0)
        tax_conf = tax_obj.get("confidence", 0.5)
    else:
        tax = 0.0
        tax_conf = 0.5
    
    total_obj = extracted_data.get("total", {})
    if isinstance(total_obj, dict):
        total = float(total_obj.get("value") or 0.0)
        total_conf = total_obj.get("confidence", 0.5)
    else:
        total = 0.0
        total_conf = 0.5
    
    # Parse line items
    line_items = []
    line_items_conf = 0.9
    for item_data in extracted_data.get("line_items", []):
        try:
            confidence = LineItemConfidence(
                description=item_data.get("confidence", {}).get("description", 0.8),
                quantity=item_data.get("confidence", {}).get("quantity", 0.8),
                unit_price=item_data.get("confidence", {}).get("unit_price", 0.8),
                total=item_data.get("confidence", {}).get("total", 0.8),
            )
            
            item = LineItem(
                description=item_data.get("description", ""),
                quantity=float(item_data.get("quantity", 0)),
                unit_price=float(item_data.get("unit_price", 0)),
                total=float(item_data.get("total", 0)),
                confidence=confidence,
            )
            line_items.append(item)
            line_items_conf = min(line_items_conf, confidence.average())
        except (ValueError, KeyError) as e:
            logger.warning(f"Error parsing line item: {e}")
            continue
    
    # Build invoice object
    confidence = InvoiceConfidence(
        invoice_number=invoice_number_conf,
        invoice_date=invoice_date_conf,
        supplier_name=supplier_name_conf,
        po_reference=po_reference_conf,
        subtotal=subtotal_conf,
        tax=tax_conf,
        total=total_conf,
        line_items=line_items_conf,
    )
    
    # Combine with OCR confidence
    overall_conf = combine_confidence_scores(
        [confidence.average(), ocr_confidence],
        weights=[0.7, 0.3],
    )
    confidence_scores = [
        confidence.invoice_number,
        confidence.invoice_date,
        confidence.supplier_name,
        confidence.po_reference,
        confidence.subtotal,
        confidence.tax,
        confidence.total,
        confidence.line_items,
    ]
    
    document_quality = "good" if overall_conf > 0.85 else "acceptable" if overall_conf > 0.70 else "poor"
    
    invoice = ExtractedInvoice(
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        supplier_name=supplier_name,
        po_reference=po_reference,
        subtotal=subtotal,
        tax=tax,
        total=total,
        line_items=line_items,
        confidence=confidence,
        document_quality=document_quality,
        extraction_warnings=extracted_data.get("extraction_warnings", []),
    )
    
    return invoice


async def document_intelligence_agent(state: ReconciliationState) -> ReconciliationState:
    """
    Document Intelligence Agent node.
    
    Responsibilities:
    1. Extract text from PDF/image
    2. Use LLM to structure extraction
    3. Assign confidence scores
    4. Detect quality issues
    
    Updates state:
    - extracted_invoice
    - extraction_error (if applicable)
    
    Adds reasoning log entry.
    """
    logger.info(f"[DocumentIntelligenceAgent] Processing invoice: {state.invoice_id}")
    
    try:
        # Step 1: Extract text from document
        logger.debug(f"Extracting text from document: {state.document_path}")
        text, ocr_confidence = extract_text_from_document(state.document_path)
        
        if not text or len(text.strip()) < 50:
            error_msg = "Insufficient text extracted from document"
            logger.error(f"[DocumentIntelligenceAgent] {error_msg}")
            state.extraction_error = error_msg
            state.add_reasoning(
                agent_name="DocumentIntelligenceAgent",
                message=f"Document extraction failed: {error_msg}",
                confidence=0.0,
            )
            return state
        
        logger.debug(f"Extracted {len(text)} characters from document (OCR confidence: {ocr_confidence:.2f})")
        
        # Step 2: Use LLM to structure extraction
        logger.debug("Calling LLM for invoice extraction...")
        
        response_text = None
        extraction_error = None
        
        # Try primary model first
        try:
            response_text = await try_llm_extraction(text, config.LLM_MODEL)
            logger.debug(f"Primary model ({config.LLM_MODEL}) response (first 500 chars): {response_text[:500]}")
        except Exception as primary_error:
            logger.warning(f"Primary model ({config.LLM_MODEL}) failed: {primary_error}")
            extraction_error = str(primary_error)
            
            # Try fallback model
            try:
                logger.info(f"Trying fallback model: {config.LLM_FALLBACK_MODEL}")
                response_text = await try_llm_extraction(text, config.LLM_FALLBACK_MODEL)
                logger.debug(f"Fallback model ({config.LLM_FALLBACK_MODEL}) response (first 500 chars): {response_text[:500]}")
                extraction_error = None  # Clear error since fallback worked
            except Exception as fallback_error:
                logger.error(f"Fallback model ({config.LLM_FALLBACK_MODEL}) also failed: {fallback_error}")
                extraction_error = f"Both models failed. Primary: {primary_error}. Fallback: {fallback_error}"
        
        # If both models failed, return error
        if not response_text and extraction_error:
            logger.error(f"All LLM attempts failed: {extraction_error}")
            state.extraction_error = f"LLM extraction failed: {extraction_error}"
            state.add_reasoning(
                agent_name="DocumentIntelligenceAgent",
                message=f"LLM extraction failed: {extraction_error}",
                confidence=0.0,
            )
            return state
        
        if not response_text:
            logger.error("All LLM attempts returned empty responses")
            state.extraction_error = "All LLM attempts returned empty responses"
            state.add_reasoning(
                agent_name="DocumentIntelligenceAgent",
                message="All LLM attempts returned empty responses",
                confidence=0.0,
            )
            return state

        # Parse JSON from response
        try:
            extracted_data = parse_extracted_json(response_text)
            if not extracted_data:
                raise ValueError("Parsed JSON is empty")
        except ValueError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response was: {response_text[:300]}")
            state.extraction_error = f"Failed to parse extraction response: {str(e)}"
            state.add_reasoning(
                agent_name="DocumentIntelligenceAgent",
                message=f"LLM response parsing failed: {str(e)}",
                confidence=0.0,
            )
            return state
        
        # Step 3: Build invoice object with confidence scores
        state.extracted_invoice = build_extracted_invoice(extracted_data, ocr_confidence)
        
        log_agent_action(
            logger,
            "DocumentIntelligenceAgent",
            "Invoice extracted",
            {
                "invoice_number": state.extracted_invoice.invoice_number,
                "supplier": state.extracted_invoice.supplier_name,
                "total": state.extracted_invoice.total,
                "line_items_count": len(state.extracted_invoice.line_items),
            },
            state.extracted_invoice.confidence.average(),
        )
        
        state.add_reasoning(
            agent_name="DocumentIntelligenceAgent",
            message=f"Successfully extracted invoice {state.extracted_invoice.invoice_number} "
                    f"from {state.extracted_invoice.supplier_name}. "
                    f"Document quality: {state.extracted_invoice.document_quality}. "
                    f"Extracted {len(state.extracted_invoice.line_items)} line items.",
            confidence=state.extracted_invoice.confidence.average(),
            action="extraction_complete",
        )
        
        if state.extracted_invoice.extraction_warnings:
            logger.warning(f"Extraction warnings: {state.extracted_invoice.extraction_warnings}")
            state.add_reasoning(
                agent_name="DocumentIntelligenceAgent",
                message=f"Extraction warnings: {', '.join(state.extracted_invoice.extraction_warnings)}",
                confidence=state.extracted_invoice.confidence.average() * 0.9,
            )
    
    except Exception as e:
        logger.exception(f"[DocumentIntelligenceAgent] Unexpected error: {e}")
        state.extraction_error = str(e)
        state.add_reasoning(
            agent_name="DocumentIntelligenceAgent",
            message=f"Unexpected error during extraction: {str(e)}",
            confidence=0.0,
        )
    
    return state
