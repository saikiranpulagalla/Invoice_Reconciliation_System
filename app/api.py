"""
Optional FastAPI REST endpoint for invoice reconciliation.
Can be run with: uvicorn app.api:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import asyncio
from pathlib import Path

from app.main import process_invoice, format_output_json
from app.config import get_config

app = FastAPI(
    title="Invoice Reconciliation API",
    description="Multi-agent invoice reconciliation system",
    version="1.0.0",
)

config = get_config()


@app.post("/process")
async def process_invoice_endpoint(
    file: UploadFile = File(...),
    invoice_id: str = Form(None),
):
    """
    Process an invoice through the reconciliation workflow.
    
    Args:
        file: Invoice PDF or image file
        invoice_id: Optional invoice ID
    
    Returns:
        JSON with reconciliation results
    """
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            # Process the invoice
            output = await process_invoice(
                document_path=tmp_path,
                invoice_id=invoice_id,
            )
            
            return JSONResponse(
                content=output.dict(default=str),
                status_code=200,
            )
        
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e),
                "message": "Failed to process invoice",
            },
            status_code=500,
        )


@app.post("/process-batch")
async def process_batch_endpoint(
    files: list = File(...),
):
    """
    Process multiple invoices in batch.
    
    Args:
        files: List of invoice files
    
    Returns:
        JSON with results for each invoice
    """
    
    try:
        # Save all files temporarily
        tmp_paths = []
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                contents = await file.read()
                tmp.write(contents)
                tmp_paths.append(tmp.name)
        
        try:
            # Process all invoices
            from app.main import process_invoices_batch
            outputs = await process_invoices_batch(tmp_paths)
            
            return JSONResponse(
                content=[output.dict(default=str) for output in outputs],
                status_code=200,
            )
        
        finally:
            # Clean up temp files
            for tmp_path in tmp_paths:
                Path(tmp_path).unlink(missing_ok=True)
    
    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e),
                "message": "Failed to process batch",
            },
            status_code=500,
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/config")
async def get_config_endpoint():
    """Get current configuration (sanitized)."""
    return {
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL,
        "ocr_enabled": config.ENABLE_OCR,
        "extraction_confidence_high": config.EXTRACTION_CONFIDENCE_HIGH,
        "matching_confidence_high": config.MATCHING_CONFIDENCE_HIGH,
        "price_variance_tolerance": config.PRICE_VARIANCE_TOLERANCE,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )
