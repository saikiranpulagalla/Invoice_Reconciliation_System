#!/usr/bin/env python
"""
Convenience script to run the Streamlit UI for Invoice Reconciliation System.

Usage:
    python run_ui.py

This script:
1. Loads environment variables from .env
2. Starts the Streamlit app
3. Opens it in your default browser
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit UI."""
    
    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "app" / "ui" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting Invoice Reconciliation UI...")
    print(f"📍 App: {app_path}")
    print("")
    print("The UI will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("")
    
    # Run Streamlit
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=False
        )
    except KeyboardInterrupt:
        print("\n✅ Streamlit server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
