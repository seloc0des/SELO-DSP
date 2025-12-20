#!/usr/bin/env python3
"""
Launcher script for SELO DSP Backend

This script provides an easy way to run the backend server without needing
to use the -m flag or understand package structure. Simply run:

    python run_backend.py

Or make it executable and run directly:

    chmod +x run_backend.py
    ./run_backend.py

Environment variables can be configured in backend/.env
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path to enable relative imports
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir.parent))

def main():
    """Start the SELO DSP backend server"""
    try:
        import uvicorn
        from backend.main import app
        
        # Get configuration from environment or use defaults
        host = os.getenv("BACKEND_HOST", "0.0.0.0")
        port = int(os.getenv("BACKEND_PORT", "8000"))
        reload = os.getenv("BACKEND_RELOAD", "false").lower() in ("true", "1", "yes")
        
        print("=" * 60)
        print("🚀 Starting SELO DSP Backend Server")
        print("=" * 60)
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Reload: {reload}")
        print(f"Backend Directory: {backend_dir}")
        print("=" * 60)
        print()
        
        # Run the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ Error: Failed to import required modules: {e}")
        print("\nMake sure you have installed the requirements:")
        print("  pip install -r backend/requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
