"""
Configuration for the invoice reconciliation system.
"""

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional


class Config:
    """Base configuration."""
    
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")  # Free Gemini model
    LLM_FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", "gemini-1.5-pro")  # Fallback model
    LLM_MOCK_MODE: bool = os.getenv("LLM_MOCK_MODE", "false").lower() == "true"  # Mock mode for testing
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")  # For backward compatibility
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", os.getenv("LLM_API_KEY", ""))  # Google API key
    LLM_API_BASE: Optional[str] = os.getenv("LLM_API_BASE", None)
    LLM_TEMPERATURE: float = 0.1  # Lower temperature for more deterministic extraction
    LLM_MAX_TOKENS: int = 1000  # Reduced for token efficiency
    LLM_TIMEOUT: int = 30
    
    # OCR Configuration
    ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "true").lower() == "true"
    OCR_LANGUAGE: str = "eng"
    OCR_ENGINE: str = "pytesseract"  # pytesseract or paddleocr
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    
    # Image Preprocessing
    PREPROCESS_GRAYSCALE: bool = True
    PREPROCESS_ROTATE: bool = True
    PREPROCESS_DESKEW: bool = True
    
    # Confidence Thresholds
    EXTRACTION_CONFIDENCE_HIGH: float = 0.85
    EXTRACTION_CONFIDENCE_ACCEPTABLE: float = 0.70
    MATCHING_CONFIDENCE_HIGH: float = 0.90
    MATCHING_CONFIDENCE_ACCEPTABLE: float = 0.75
    FUZZY_MATCHING_THRESHOLD: float = 0.80
    
    # Discrepancy Thresholds
    PRICE_VARIANCE_TOLERANCE: float = 0.02  # 2% tolerance
    QUANTITY_VARIANCE_TOLERANCE: float = 0.05  # 5% tolerance
    
    # Decision Logic
    AUTO_APPROVE_EXTRACTION_THRESHOLD: float = 0.85
    AUTO_APPROVE_MATCH_THRESHOLD: float = 0.90
    AUTO_APPROVE_DISCREPANCY_THRESHOLD: float = 0.05  # 5% max variance
    
    ESCALATE_EXTRACTION_THRESHOLD: float = 0.50
    ESCALATE_MATCH_THRESHOLD: float = 0.60
    ESCALATE_DISCREPANCY_THRESHOLD: float = 0.20  # 20% variance
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = os.getenv("LOG_FILE", "invoice_reconciliation.log")
    
    # Data Paths
    INVOICES_DIR: str = os.path.join(os.path.dirname(__file__), "data", "invoices")
    PO_DATABASE_PATH: str = os.path.join(os.path.dirname(__file__), "data", "purchase_orders.json")
    
    # API Configuration (if using FastAPI)
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    
    # Workflow Configuration
    ENABLE_HUMAN_REVIEW: bool = os.getenv("ENABLE_HUMAN_REVIEW", "true").lower() == "true"
    GRAPH_RECURSION_LIMIT: int = 100
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        if cls.LLM_PROVIDER == "openai" and not cls.LLM_API_KEY:
            raise ValueError("LLM_API_KEY must be set for OpenAI provider")
        
        if cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY must be set for Gemini provider")
        
        if cls.LLM_PROVIDER not in ["openai", "azure", "custom", "gemini", "mock"]:
            raise ValueError(f"Invalid LLM_PROVIDER: {cls.LLM_PROVIDER}")


class DevelopmentConfig(Config):
    """Development configuration."""
    LLM_TEMPERATURE = 0.5
    LOG_LEVEL = "DEBUG"
    API_DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    LLM_TEMPERATURE = 0.2
    LOG_LEVEL = "INFO"
    API_DEBUG = False


class TestConfig(Config):
    """Test configuration."""
    LLM_TEMPERATURE = 0.0
    LOG_LEVEL = "DEBUG"
    ENABLE_HUMAN_REVIEW = False


def get_config(env: str = None) -> Config:
    """Get configuration based on environment."""
    if env is None:
        env = os.getenv("ENV", "development").lower()
    
    if env == "production":
        config = ProductionConfig()
    elif env == "test":
        config = TestConfig()
    else:
        config = DevelopmentConfig()
    
    # Validate configuration on creation
    config.validate()
    return config
