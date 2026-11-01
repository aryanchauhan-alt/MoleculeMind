"""
MoleculeMind Startup Script.
Launches the FastAPI backend and serves the bundled React frontend at http://127.0.0.1:8000.
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    print("=" * 70)
    print("Starting MoleculeMind Platform...")
    print("Frontend & Backend URL: http://127.0.0.1:8000")
    print("API Documentation:      http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
