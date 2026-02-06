"""
Entry point for Streamlit Cloud deployment.

This file is in the root directory so Streamlit Cloud can find it easily.
It simply imports and runs the actual Streamlit app from app/ui/streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Set Streamlit environment variable FIRST
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the actual app
from app.ui.streamlit_app import *  # noqa: F401, F403
S
