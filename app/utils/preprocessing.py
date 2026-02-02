"""
Image preprocessing utilities.
"""

from PIL import Image
import os


def validate_image_format(file_path: str) -> bool:
    """Validate that file is a readable image."""
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if not os.path.exists(file_path):
        return False
    
    # Check extension
    if not any(file_path.lower().endswith(fmt) for fmt in valid_formats):
        return False
    
    # Try to open and validate
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_dimensions(file_path: str) -> tuple:
    """Get image dimensions (width, height)."""
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception:
        return None


def assess_image_quality(file_path: str) -> str:
    """
    Quick assessment of image quality for OCR.
    
    Returns: "good", "acceptable", or "poor"
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            
            # Very small images are problematic
            if width < 400 or height < 300:
                return "poor"
            
            # High resolution is good
            if width > 2000 and height > 1500:
                return "good"
            
            # Medium resolution is acceptable
            if width > 800 and height > 600:
                return "acceptable"
            
            return "acceptable"
    
    except Exception:
        return "poor"
