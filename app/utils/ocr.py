"""
OCR utilities for document processing.
Handles text extraction from PDFs and images.
"""

import os
from typing import Optional, List, Tuple
from PIL import Image
import pdfplumber
from pathlib import Path

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from app.config import get_config
from app.utils.logging import setup_logging


logger = setup_logging(__name__)
config = get_config()

import shutil

# Detect whether the tesseract binary is available on the system PATH.
# pytesseract Python package may be installed but the external tesseract
# executable must also be installed for OCR to work.
TESSERACT_INSTALLED = shutil.which("tesseract") is not None


def extract_text_from_pdf(pdf_path: str, use_ocr: bool = True) -> Tuple[str, float]:
    """
    Extract text from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to use OCR on scanned pages
    
    Returns:
        (extracted_text, confidence_score)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    text_parts = []
    confidence = 1.0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Try to extract text directly
                page_text = page.extract_text()
                
                if page_text and len(page_text.strip()) > 100:
                    # Text extraction succeeded
                    text_parts.append(page_text)
                    logger.debug(f"Extracted text from PDF page {page_num + 1}")
                elif use_ocr and config.ENABLE_OCR:
                    # Fall back to OCR or attempt character-level reconstruction
                    logger.debug(f"Using OCR for PDF page {page_num + 1}")
                    try:
                        # Convert page to PIL image and extract text if tesseract is available
                        pil_image = page.to_image().original
                        if PYTESSERACT_AVAILABLE and TESSERACT_INSTALLED:
                            ocr_text, ocr_conf = extract_text_from_pil_image(pil_image)
                            if ocr_text and len(ocr_text.strip()) > 20:
                                text_parts.append(ocr_text)
                                confidence = min(confidence, ocr_conf)
                            else:
                                logger.warning(f"OCR failed to extract text from PDF page {page_num + 1}")
                                confidence = min(confidence, 0.3)
                        else:
                            # If pytesseract or the tesseract binary is not available,
                            # try reconstructing text from pdfplumber's character objects
                            if not PYTESSERACT_AVAILABLE:
                                logger.warning("pytesseract Python package not installed; skipping OCR")
                            if not TESSERACT_INSTALLED:
                                logger.warning("Tesseract binary not found in PATH; skipping OCR")

                            # Attempt to reconstruct text from page.chars
                            try:
                                chars = page.chars
                                if chars:
                                    # Sort by top (y) then x0 and rebuild lines
                                    chars_sorted = sorted(chars, key=lambda c: (int(round(c.get("top", 0))), int(round(c.get("x0", 0)))))
                                    lines = []
                                    current_top = None
                                    current_line = []
                                    for ch in chars_sorted:
                                        top = int(round(ch.get("top", 0)))
                                        ch_text = ch.get("text", "")
                                        if current_top is None:
                                            current_top = top
                                        # new line when vertical gap is large
                                        if abs(top - current_top) > 3:
                                            lines.append("".join(current_line))
                                            current_line = [ch_text]
                                            current_top = top
                                        else:
                                            current_line.append(ch_text)
                                    if current_line:
                                        lines.append("".join(current_line))
                                    page_reconstructed = "\n".join([l for l in lines if l.strip()])
                                    if page_reconstructed:
                                        logger.debug(f"Reconstructed text from PDF page {page_num + 1} using chars (length={len(page_reconstructed)})")
                                        text_parts.append(page_reconstructed)
                                        confidence = min(confidence, 0.4)
                                    else:
                                        logger.warning(f"No text reconstructed from PDF page {page_num + 1}")
                                        confidence = min(confidence, 0.2)
                                else:
                                    logger.warning(f"No character data available for PDF page {page_num + 1}")
                                    confidence = min(confidence, 0.2)
                            except Exception as recon_err:
                                logger.error(f"Character-level reconstruction failed: {recon_err}")
                                confidence = min(confidence, 0.2)
                    except Exception as ocr_error:
                        logger.error(f"OCR error on page {page_num + 1}: {ocr_error}")
                        confidence = min(confidence, 0.3)
                else:
                    logger.warning(f"Could not extract text from PDF page {page_num + 1}")
                    confidence = min(confidence, 0.5)
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        confidence = 0.3
    
    return "\n".join(text_parts), confidence


def extract_text_from_pil_image(pil_image: Image.Image) -> Tuple[str, float]:
    """
    Extract text from PIL Image object using OCR.
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        (extracted_text, confidence_score)
    """
    if not PYTESSERACT_AVAILABLE:
        logger.error("pytesseract not available. Install with: pip install pytesseract")
        return "", 0.0
    
    try:
        # Extract text
        text = pytesseract.image_to_string(pil_image)
        
        # Get detailed data with confidence scores
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) / 100 for conf in data['conf'] if int(conf) > -1]
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.5 if text else 0.0
        
        logger.debug(f"Extracted text from PIL image with confidence {avg_confidence:.2f}")
        
        return text, avg_confidence
    
    except Exception as e:
        logger.error(f"Error extracting text from PIL image: {e}")
        return "", 0.0


def extract_text_from_image(image_path: str) -> Tuple[str, float]:
    """
    Extract text from image using OCR.
    
    Args:
        image_path: Path to image file
    
    Returns:
        (extracted_text, confidence_score)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not PYTESSERACT_AVAILABLE:
        logger.error("pytesseract not available. Install with: pip install pytesseract")
        return "", 0.0
    
    try:
        # Open image
        image = Image.open(image_path)
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        # Get detailed data with confidence scores
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) / 100 for conf in data['conf'] if int(conf) > -1]
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.5 if text else 0.0
        
        logger.debug(f"Extracted text from image with confidence {avg_confidence:.2f}")
        
        return text, avg_confidence
    
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return "", 0.0


def preprocess_image(image_path: str, save_path: Optional[str] = None) -> Optional[str]:
    """
    Preprocess image for better OCR results.
    Includes rotation detection, grayscale conversion, and denoising.
    
    Args:
        image_path: Path to input image
        save_path: Optional path to save preprocessed image
    
    Returns:
        Path to preprocessed image (or save_path if provided)
    """
    if not CV2_AVAILABLE:
        logger.warning("OpenCV not available. Skipping image preprocessing.")
        return image_path
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return image_path
        
        # Convert to grayscale if configured
        if config.PREPROCESS_GRAYSCALE:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        image = cv2.fastNlMeansDenoising(image, h=10)
        
        # Apply thresholding for better contrast
        _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        
        # Rotate if configured and image seems rotated
        if config.PREPROCESS_ROTATE:
            angle = detect_rotation(image)
            if abs(angle) > 5:  # Only rotate if angle is significant
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, matrix, (w, h))
                logger.debug(f"Rotated image by {angle:.1f} degrees")
        
        # Save if save_path provided
        if save_path:
            cv2.imwrite(save_path, image)
            logger.debug(f"Saved preprocessed image to {save_path}")
            return save_path
        else:
            # Return original (preprocessing was done in-memory for analysis)
            return image_path
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image_path


def detect_rotation(image) -> float:
    """
    Detect image rotation angle using edge detection.
    
    Args:
        image: OpenCV image
    
    Returns:
        Rotation angle in degrees
    """
    if not CV2_AVAILABLE:
        return 0.0
    
    try:
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return 0.0
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        # Return median angle close to 0 or 90
        if angles:
            median_angle = np.median(angles)
            if median_angle > 45:
                return median_angle - 90
            else:
                return median_angle
        
        return 0.0
    
    except Exception as e:
        logger.error(f"Error detecting rotation: {e}")
        return 0.0
